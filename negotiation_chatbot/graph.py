import os
import uuid
import logging
import re
import time
from typing import Dict, List, Any, Callable
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection parameters
ENABLE_NEO4J = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "6xlBSIDu8Nc8gjXrpt3kNuwM7AZHGI3WJrfpN2fFDXE")

# Initialize Neo4j driver
driver = None
if ENABLE_NEO4J:
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASS),
            max_connection_lifetime=3600,  # 1 hour (was 5 seconds) - reduces reconnection overhead
            connection_acquisition_timeout=30,  # 30 seconds (was 5) - allows for temporary spikes
            max_connection_pool_size=50,  # Connection pool for concurrent requests
            connection_timeout=10  # Socket connection timeout
        )
        logger.info("Connected to Neo4j database with optimized connection pool settings")
    except Exception as e:
        logger.warning(f"Neo4j connection failed (continuing without graph features): {e}")
        driver = None
else:
    logger.info("Neo4j disabled (set ENABLE_NEO4J=true to enable graph features)")


def execute_with_retry(session_func: Callable, max_retries: int = 3, initial_delay: float = 0.1):
    """
    Execute Neo4j operation with exponential backoff retry.

    Args:
        session_func: Callable that performs the database operation
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 0.1)

    Returns:
        Result of session_func execution

    Raises:
        Exception: If all retries fail, raises the last exception
    """
    for attempt in range(max_retries):
        try:
            return session_func()
        except (ServiceUnavailable, SessionExpired) as e:
            if attempt == max_retries - 1:
                # Last attempt failed, re-raise the exception
                logger.error(f"Neo4j operation failed after {max_retries} attempts: {e}")
                raise

            # Calculate exponential backoff delay
            delay = initial_delay * (2 ** attempt)
            logger.warning(
                f"Neo4j operation failed (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {delay:.2f}s: {e}"
            )
            time.sleep(delay)

    # This should never be reached, but just in case
    raise RuntimeError("Unexpected error in execute_with_retry")


def upsert_turn(conv_id: str, speaker: str, text: str, move: str, pd: str):
    """
    Upsert a conversation turn into Neo4j.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker name
        text: Turn text
        move: Move type
        pd: Power dynamic
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    # Add debugging for text parameter
    logger.info(f"upsert_turn called with: conv_id={conv_id}, speaker={speaker}, text='{text[:50] if text else 'EMPTY'}...', move={move}, pd={pd}")
    
    def create_turn(tx, conv_id, speaker, text, move, pd):
        try:
            # Generate unique turn ID
            turn_id = str(uuid.uuid4())
            logger.info(f"Creating turn with ID: {turn_id}")
            
            # MERGE Person node
            result = tx.run("""
                MERGE (p:Person {name: $speaker})
                RETURN p
                """, speaker=speaker)
            logger.info(f"Person node created/merged: {result.single()}")
            
            # MERGE Conversation node
            result = tx.run("""
                MERGE (c:Conv {id: $conv_id})
                RETURN c
                """, conv_id=conv_id)
            logger.info(f"Conversation node created/merged: {result.single()}")
            
            # CREATE Turn node
            result = tx.run("""
                CREATE (t:Turn {
                    id: $turn_id,
                    text: $text,
                    move: $move,
                    pd: $pd,
                    ts: datetime()
                })
                RETURN t
                """, turn_id=turn_id, text=text, move=move, pd=pd)
            logger.info(f"Turn node created: {result.single()}")
            
            # MERGE relationships
            result = tx.run("""
                MATCH (p:Person {name: $speaker})
                MATCH (t:Turn {id: $turn_id})
                MERGE (p)-[:SPOKE]->(t)
                RETURN count(*) as relationships_created
                """, speaker=speaker, turn_id=turn_id)
            logger.info(f"SPOKE relationship created: {result.single()}")
            
            result = tx.run("""
                MATCH (c:Conv {id: $conv_id})
                MATCH (t:Turn {id: $turn_id})
                MERGE (c)-[:HAS_TURN]->(t)
                RETURN count(*) as relationships_created
                """, conv_id=conv_id, turn_id=turn_id)
            logger.info(f"HAS_TURN relationship created: {result.single()}")
            
            # Link to previous turn via NEXT relationship
            result = tx.run("""
                MATCH (c:Conv {id: $conv_id})
                MATCH (t:Turn {id: $turn_id})
                WITH c, t
                OPTIONAL MATCH (c)-[:HAS_TURN]->(prev:Turn)
                WHERE NOT EXISTS((prev)-[:NEXT]->())
                WITH t, prev
                WHERE prev IS NOT NULL
                MERGE (prev)-[:NEXT]->(t)
                RETURN count(*) as next_relationships_created
                """, conv_id=conv_id, turn_id=turn_id)
            logger.info(f"NEXT relationship created: {result.single()}")
            
        except Exception as e:
            logger.error(f"Error in create_turn transaction: {e}")
            raise
    
    try:
        def upsert_operation():
            with driver.session() as session:
                session.execute_write(create_turn, conv_id, speaker, text, move, pd)

        execute_with_retry(upsert_operation)
        logger.info(f"Upserted turn for conversation {conv_id}, speaker {speaker}")

        # Extract and store offers from the message
        # Get the current turn number by counting turns in this conversation
        turn_number = get_turn_number(conv_id)
        process_message_for_offers(conv_id, speaker, text, turn_number)

    except Exception as e:
        logger.error(f"Error upserting turn: {e}")
        raise

def fetch_last_n(conv_id: str, n: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch the last n turns for a conversation.

    Args:
        conv_id: Conversation identifier
        n: Number of turns to fetch (default: 5)

    Returns:
        List of dictionaries containing turn data, ordered ascending by timestamp
    """
    if not driver:
        logger.info("Neo4j driver not available, falling back to JSON file")
        # Fallback: read from JSON file
        import json
        json_path = f"./negotiation_chatbot/data/{conv_id}.json"
        if not os.path.exists(json_path):
            json_path = f"./data/{conv_id}.json"
        if not os.path.exists(json_path):
            logger.warning(f"Conversation file not found: {json_path}")
            return []

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Filter out Coach/Bot messages, keep only A and B
            history = data.get('history', [])
            turns = []
            for entry in history:
                speaker = entry.get('speaker')
                if speaker in ['A', 'B']:  # Only include negotiating parties
                    turns.append({
                        "id": entry.get('ts', str(uuid.uuid4())),
                        "text": entry.get('text', ''),
                        "move": entry.get('move'),
                        "pd": entry.get('pd'),
                        "ts": entry.get('ts', ''),
                        "speaker": speaker
                    })

            # Return last n turns
            logger.info(f"Fetched {len(turns[-n:])} turns from JSON file (total {len(turns)} A/B turns)")
            return turns[-n:] if len(turns) > n else turns
        except Exception as e:
            logger.error(f"Error reading conversation from JSON: {e}")
            return []
    
    def get_last_turns(tx, conv_id, n):
        result = tx.run("""
            MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person)
            RETURN t.id as id, t.text as text, t.move as move, t.pd as pd, t.ts as ts, p.name as speaker
            ORDER BY t.ts DESC
            LIMIT $n
            """, conv_id=conv_id, n=n)
        
        turns = []
        for record in result:
            turn_data = {
                "id": record["id"],
                "text": record["text"],
                "move": record["move"],
                "pd": record["pd"],
                "ts": record["ts"],
                "speaker": record["speaker"]
            }
            logger.info(f"Retrieved turn: {turn_data}")
            turns.append(turn_data)
        
        # Reverse to get ascending order
        return list(reversed(turns))
    
    try:
        def fetch_operation():
            with driver.session() as session:
                return session.execute_read(get_last_turns, conv_id, n)

        turns = execute_with_retry(fetch_operation)
        logger.info(f"Fetched {len(turns)} turns for conversation {conv_id}")
        return turns
    except Exception as e:
        logger.error(f"Error fetching turns: {e}")
        return []

def get_conversation_graph_data(conv_id: str) -> Dict[str, Any]:
    """
    Get conversation graph data for visualization.
    
    Args:
        conv_id: Conversation identifier
        
    Returns:
        Dictionary with nodes and edges for visualization
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return {"error": "Neo4j driver not available"}
    
    def get_graph_data(tx, conv_id):
        # Get all nodes and relationships for the conversation
        result = tx.run("""
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person)
            OPTIONAL MATCH (t)-[:NEXT]->(next:Turn)
            RETURN c, t, p, next
            ORDER BY t.ts
            """, conv_id=conv_id)
        
        nodes = []
        edges = []
        node_ids = set()
        
        for record in result:
            # Add conversation node
            conv_node = {
                "id": record["c"]["id"],
                "type": "Conv",
                "name": record["c"]["id"]
            }
            if conv_node["id"] not in node_ids:
                nodes.append(conv_node)
                node_ids.add(conv_node["id"])
            
            # Add person node
            if record["p"]:
                person_node = {
                    "id": record["p"]["name"],
                    "type": "Person",
                    "name": record["p"]["name"]
                }
                if person_node["id"] not in node_ids:
                    nodes.append(person_node)
                    node_ids.add(person_node["id"])
                
                # Add SPOKE relationship
                if record["t"]:
                    edges.append({
                        "source": person_node["id"],
                        "target": record["t"]["id"],
                        "type": "SPOKE"
                    })
            
            # Add turn node
            if record["t"]:
                turn_node = {
                    "id": record["t"]["id"],
                    "type": "Turn",
                    "name": f"{record['t']['move']} ({record['t']['pd']})"
                }
                if turn_node["id"] not in node_ids:
                    nodes.append(turn_node)
                    node_ids.add(turn_node["id"])
                
                # Add HAS_TURN relationship
                edges.append({
                    "source": conv_node["id"],
                    "target": turn_node["id"],
                    "type": "HAS_TURN"
                })
                
                # Add NEXT relationship
                if record["next"]:
                    edges.append({
                        "source": turn_node["id"],
                        "target": record["next"]["id"],
                        "type": "NEXT"
                    })
        
        return {
            "conv_id": conv_id,
            "nodes": nodes,
            "edges": edges
        }
    
    try:
        with driver.session() as session:
            graph_data = session.execute_read(get_graph_data, conv_id)
        logger.info(f"Retrieved graph data for conversation {conv_id}")
        return graph_data
    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        return {"error": str(e)}

def get_conversation_stats(conv_id: str) -> Dict[str, Any]:
    """
    Get conversation statistics for visualization.
    
    Args:
        conv_id: Conversation identifier
        
    Returns:
        Dictionary with conversation statistics
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return {"error": "Neo4j driver not available"}
    
    def get_stats(tx, conv_id):
        # Get basic conversation stats
        result = tx.run("""
            MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)
            RETURN 
                count(t) as turn_count,
                collect(DISTINCT t.move) as moves,
                collect(DISTINCT t.pd) as power_dynamics
            """, conv_id=conv_id)
        
        basic_stats = result.single()
        
        # Get speaker statistics
        speaker_result = tx.run("""
            MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person)
            RETURN p.name as speaker, count(t) as turn_count,
                   collect(t.move) as moves, collect(t.pd) as power_dynamics
            """, conv_id=conv_id)
        
        speaker_stats = {}
        for record in speaker_result:
            speaker_stats[record["speaker"]] = {
                "turn_count": record["turn_count"],
                "moves": record["moves"],
                "power_dynamics": record["power_dynamics"]
            }
        
        return {
            "conv_id": conv_id,
            "turn_count": basic_stats["turn_count"] if basic_stats else 0,
            "moves": basic_stats["moves"] if basic_stats else [],
            "power_dynamics": basic_stats["power_dynamics"] if basic_stats else [],
            "speaker_stats": speaker_stats
        }
    
    try:
        with driver.session() as session:
            stats = session.execute_read(get_stats, conv_id)
        logger.info(f"Retrieved stats for conversation {conv_id}")
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

def get_speaker_turns(conv_id: str, speaker: str) -> List[Dict[str, Any]]:
    """
    Get all turns for a specific speaker in a conversation.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker name
        
    Returns:
        List of dictionaries containing turn data
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return []
    
    def get_speaker_turns(tx, conv_id, speaker):
        result = tx.run("""
            MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)<-[:SPOKE]-(p:Person {name: $speaker})
            RETURN t.id as id, t.text as text, t.move as move, t.pd as pd, t.ts as ts
            ORDER BY t.ts ASC
            """, conv_id=conv_id, speaker=speaker)
        
        turns = []
        for record in result:
            turns.append({
                "id": record["id"],
                "text": record["text"],
                "move": record["move"],
                "pd": record["pd"],
                "ts": record["ts"]
            })
        
        return turns
    
    try:
        with driver.session() as session:
            turns = session.execute_read(get_speaker_turns, conv_id, speaker)
        logger.info(f"Fetched {len(turns)} turns for speaker {speaker} in conversation {conv_id}")
        return turns
    except Exception as e:
        logger.error(f"Error fetching speaker turns: {e}")
        return []

def cleanup_conversation(conv_id: str):
    """
    Clean up all data for a conversation.
    
    Args:
        conv_id: Conversation identifier
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    def delete_conversation(tx, conv_id):
        tx.run("""
            MATCH (c:Conv {id: $conv_id})
            OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
            OPTIONAL MATCH (p:Person)-[:SPOKE]->(t)
            OPTIONAL MATCH (t)-[:NEXT]->(next:Turn)
            DETACH DELETE t, next
            DETACH DELETE c
            """, conv_id=conv_id)
    
    try:
        with driver.session() as session:
            session.execute_write(delete_conversation, conv_id)
        logger.info(f"Cleaned up conversation {conv_id}")
    except Exception as e:
        logger.error(f"Error cleaning up conversation: {e}")
        raise

def upsert_rag_usage(conv_id: str, speaker: str, rag_used: bool):
    """
    Log RAG usage in Neo4j for analytics.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker identifier
        rag_used: Whether RAG context was used
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    def log_rag_usage(tx, conv_id, speaker, rag_used):
        # Create or merge the conversation
        tx.run("""
            MERGE (c:Conv {id: $conv_id})
            """, conv_id=conv_id)
        
        # Create or merge the speaker
        tx.run("""
            MERGE (p:Person {name: $speaker})
            """, speaker=speaker)
        
        # Create RAG usage record
        tx.run("""
            MATCH (c:Conv {id: $conv_id})
            MATCH (p:Person {name: $speaker})
            CREATE (r:RAGUsage {
                id: randomUUID(),
                rag_used: $rag_used,
                timestamp: datetime()
            })
            CREATE (c)-[:HAS_RAG_USAGE]->(r)
            CREATE (p)-[:USED_RAG]->(r)
            """, conv_id=conv_id, speaker=speaker, rag_used=rag_used)
    
    try:
        with driver.session() as session:
            session.execute_write(log_rag_usage, conv_id, speaker, rag_used)
        logger.info(f"RAG usage logged: conv_id={conv_id}, speaker={speaker}, rag_used={rag_used}")
    except Exception as e:
        logger.error(f"Error logging RAG usage: {e}")

def create_deal_outcome(conv_id: str, deal_reached: bool = True, status: str = "accepted", details: str = None):
    """
    Create an Outcome node when a deal is reached.
    
    Args:
        conv_id: Conversation identifier
        deal_reached: Boolean indicating if deal was reached
        status: Status of the outcome (accepted, rejected, pending)
        details: Additional details about the outcome
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    def create_outcome(tx, conv_id, deal_reached, status, details):
        # Create or merge the conversation
        tx.run("""
            MERGE (c:Conv {id: $conv_id})
            """, conv_id=conv_id)
        
        # Create Outcome node
        tx.run("""
            MATCH (c:Conv {id: $conv_id})
            CREATE (o:Outcome {
                deal_reached: $deal_reached,
                status: $status,
                details: $details,
                ts: datetime()
            })
            CREATE (c)-[:OF_CONV]->(o)
            """, conv_id=conv_id, deal_reached=deal_reached, status=status, details=details)
    
    try:
        with driver.session() as session:
            session.execute_write(create_outcome, conv_id, deal_reached, status, details)
        logger.info(f"Deal outcome created: conv_id={conv_id}, deal_reached={deal_reached}, status={status}")
    except Exception as e:
        logger.error(f"Error creating deal outcome: {e}")

def mark_turn_as_accepted(conv_id: str, turn_id: str = None):
    """
    Mark a turn as accepted (indicating deal reached).
    
    Args:
        conv_id: Conversation identifier
        turn_id: Specific turn ID to mark (if None, marks the last turn)
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    def mark_accepted(tx, conv_id, turn_id):
        if turn_id:
            # Mark specific turn
            tx.run("""
                MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn {id: $turn_id})
                SET t.annotation = 'accept', t.status = 'accepted'
                """, conv_id=conv_id, turn_id=turn_id)
        else:
            # Mark the last turn
            tx.run("""
                MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)
                WITH t ORDER BY t.ts DESC
                LIMIT 1
                SET t.annotation = 'accept', t.status = 'accepted'
                """, conv_id=conv_id)
    
    try:
        with driver.session() as session:
            session.execute_write(mark_accepted, conv_id, turn_id)
        logger.info(f"Turn marked as accepted: conv_id={conv_id}, turn_id={turn_id}")
    except Exception as e:
        logger.error(f"Error marking turn as accepted: {e}")

def upsert_coalition(conv_id: str, coalition_id: str, member_names: List[str]):
    """Create a Coalition node and link members to it.

    Graceful no-op if Neo4j is disabled.

    Args:
        conv_id: Conversation identifier
        coalition_id: Unique coalition identifier
        member_names: List of party names in the coalition
    """
    if not driver:
        logger.info("Neo4j disabled – skipping coalition upsert")
        return

    def _create_coalition(tx, conv_id, coalition_id, member_names):
        # MERGE Coalition node
        tx.run("""
            MERGE (co:Coalition {id: $coalition_id})
            """, coalition_id=coalition_id)
        # Link to conversation
        tx.run("""
            MATCH (c:Conv {id: $conv_id})
            MATCH (co:Coalition {id: $coalition_id})
            MERGE (co)-[:IN_CONV]->(c)
            """, conv_id=conv_id, coalition_id=coalition_id)
        # Link members
        for name in member_names:
            tx.run("""
                MERGE (p:Person {name: $name})
                WITH p
                MATCH (co:Coalition {id: $coalition_id})
                MERGE (p)-[:MEMBER_OF]->(co)
                """, name=name, coalition_id=coalition_id)

    try:
        def op():
            with driver.session() as session:
                session.execute_write(_create_coalition, conv_id, coalition_id, member_names)
        execute_with_retry(op)
        logger.info("Coalition %s upserted for conversation %s", coalition_id, conv_id)
    except Exception as e:
        logger.error("Error upserting coalition: %s", e)


def fetch_coalitions(conv_id: str) -> List[Dict[str, Any]]:
    """Fetch all coalitions and their members for a conversation.

    Args:
        conv_id: Conversation identifier

    Returns:
        List of dicts with keys: coalition_id, members
    """
    if not driver:
        logger.info("Neo4j disabled – returning empty coalitions")
        return []

    def _get_coalitions(tx, conv_id):
        result = tx.run("""
            MATCH (co:Coalition)-[:IN_CONV]->(c:Conv {id: $conv_id})
            OPTIONAL MATCH (p:Person)-[:MEMBER_OF]->(co)
            RETURN co.id AS coalition_id, collect(p.name) AS members
            """, conv_id=conv_id)
        return [{"coalition_id": r["coalition_id"], "members": r["members"]} for r in result]

    try:
        with driver.session() as session:
            return session.execute_read(_get_coalitions, conv_id)
    except Exception as e:
        logger.error("Error fetching coalitions: %s", e)
        return []


def store_offer(conv_id: str, speaker: str, issue: str, value: str, turn_number: int):
    """
    Store an Offer node in Neo4j with the specified relationship structure.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker name
        issue: The issue being negotiated (e.g., "salary", "price", "delivery_time")
        value: The numeric value of the offer
        turn_number: The turn number when the offer was made
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return
    
    def create_offer(tx, conv_id, speaker, issue, value, turn_number):
        try:
            # MERGE Person node
            result = tx.run("""
                MERGE (p:Person {name: $speaker})
                RETURN p
                """, speaker=speaker)
            logger.info(f"Person node created/merged: {result.single()}")
            
            # MERGE Conversation node
            result = tx.run("""
                MERGE (c:Conv {id: $conv_id})
                RETURN c
                """, conv_id=conv_id)
            logger.info(f"Conversation node created/merged: {result.single()}")
            
            # CREATE Offer node
            result = tx.run("""
                CREATE (o:Offer {
                    issue: $issue,
                    value: $value,
                    turn_number: $turn_number,
                    ts: datetime()
                })
                RETURN o
                """, issue=issue, value=value, turn_number=turn_number)
            logger.info(f"Offer node created: {result.single()}")
            
            # MERGE relationships: (p:Person)-[:MADE]->(o:Offer)-[:IN]->(c:Conv)
            result = tx.run("""
                MATCH (p:Person {name: $speaker})
                MATCH (o:Offer {issue: $issue, value: $value, turn_number: $turn_number})
                MATCH (c:Conv {id: $conv_id})
                MERGE (p)-[:MADE]->(o)
                MERGE (o)-[:IN]->(c)
                RETURN count(*) as relationships_created
                """, speaker=speaker, issue=issue, value=value, turn_number=turn_number, conv_id=conv_id)
            logger.info(f"Offer relationships created: {result.single()}")
            
        except Exception as e:
            logger.error(f"Error in create_offer transaction: {e}")
            raise
    
    try:
        with driver.session() as session:
            session.execute_write(create_offer, conv_id, speaker, issue, value, turn_number)
        logger.info(f"Stored offer for conversation {conv_id}, speaker {speaker}, issue {issue}, value {value}")
    except Exception as e:
        logger.error(f"Error storing offer: {e}")
        raise

def extract_offers_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract numeric offers from text using regex patterns.
    
    Args:
        text: The text to analyze
        
    Returns:
        List of dictionaries containing extracted offers with issue and value
    """
    offers = []
    
    # Regex patterns for different types of offers
    patterns = [
        # Currency amounts: $100, $1,000, $1.50, etc.
        (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', 'price'),
        # Percentages: 10%, 5.5%, etc.
        (r'(\d+(?:\.\d+)?)%', 'percentage'),
        # Quantities: 100 units, 5 pieces, 3 items, etc.
        (r'(\d+)\s+(units?|pieces?|items?|boxes?|cases?)', 'quantity'),
        # Time periods: 30 days, 2 weeks, 3 months, etc.
        (r'(\d+)\s+(days?|weeks?|months?|years?)', 'time'),
        # Plain numbers that might be offers (context dependent)
        (r'\b(\d+(?:\.\d+)?)\b', 'numeric_value')
    ]
    
    for pattern, issue_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            value = match.group(1) if issue_type != 'price' else match.group(0)
            offers.append({
                'issue': issue_type,
                'value': value,
                'start': match.start(),
                'end': match.end(),
                'full_match': match.group(0)
            })
    
    return offers

def get_turn_number(conv_id: str) -> int:
    """
    Get the current turn number for a conversation.
    
    Args:
        conv_id: Conversation identifier
        
    Returns:
        The current turn number (1-based)
    """
    if not driver:
        logger.error("Neo4j driver not available")
        return 1
    
    def count_turns(tx, conv_id):
        result = tx.run("""
            MATCH (c:Conv {id: $conv_id})-[:HAS_TURN]->(t:Turn)
            RETURN count(t) as turn_count
            """, conv_id=conv_id)
        
        record = result.single()
        return record["turn_count"] if record else 0
    
    try:
        with driver.session() as session:
            turn_count = session.execute_read(count_turns, conv_id)
        return turn_count
    except Exception as e:
        logger.error(f"Error getting turn number: {e}")
        return 1

def process_message_for_offers(conv_id: str, speaker: str, text: str, turn_number: int):
    """
    Process a message to extract and store any offers found.
    
    Args:
        conv_id: Conversation identifier
        speaker: Speaker name
        text: Message text to analyze
        turn_number: Current turn number
    """
    if not text:
        return
    
    # Extract offers from the text
    offers = extract_offers_from_text(text)
    
    # Store each offer found
    for offer in offers:
        try:
            store_offer(conv_id, speaker, offer['issue'], offer['value'], turn_number)
            logger.info(f"Stored offer: {offer['issue']} = {offer['value']} from {speaker}")
        except Exception as e:
            logger.error(f"Failed to store offer {offer}: {e}")

if __name__ == "__main__":
    # Test the module
    try:
        # Test upsert_turn
        test_conv_id = "test_conv_001"
        test_speaker = "Alice"
        test_text = "I would like to propose a 10% increase in salary."
        test_move = "offer"
        test_pd = "high_power"
        
        print("Testing upsert_turn...")
        upsert_turn(test_conv_id, test_speaker, test_text, test_move, test_pd)
        
        # Test fetch_last_n
        print("Testing fetch_last_n...")
        turns = fetch_last_n(test_conv_id, 5)
        print(f"Fetched {len(turns)} turns")
        
        # Test conversation stats
        print("Testing get_conversation_stats...")
        stats = get_conversation_stats(test_conv_id)
        print(f"Conversation stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # Clean up test data
        cleanup_conversation(test_conv_id)
        # Close driver
        if driver:
            driver.close()
            print("Neo4j driver closed")
