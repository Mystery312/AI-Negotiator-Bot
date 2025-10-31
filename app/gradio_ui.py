import gradio as gr
import requests
import json
import uuid
import os
import logging
import time
import sys
from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import asyncio

# Add parent directory to sys.path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
def resolve_api_base_url() -> str:
    """Resolve API base URL with fallbacks for local/dev vs docker."""
    env_base = os.getenv("API_BASE_URL")
    candidates = []
    if env_base:
        candidates.append(env_base)
    candidates.extend([
        "http://api:8000",            # docker compose service name
        "http://localhost:8000",      # local dev default
        "http://127.0.0.1:8000",      # explicit loopback
    ])
    for base in candidates:
        try:
            r = requests.get(f"{base}/health", timeout=0.8)
            if r.status_code == 200:
                logger.info(f"Resolved API_BASE_URL: {base}")
                return base
        except Exception:
            continue
    logger.warning("Could not reach any API candidate; defaulting to http://localhost:8000")
    return "http://localhost:8000"

API_BASE_URL = resolve_api_base_url()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Ensure data directory exists
os.makedirs("./data", exist_ok=True)

def get_conversation_graph_data(conv_id: str) -> dict:
    """
    Fetch conversation graph data from the API for visualization.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/graph/{conv_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to fetch graph data: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching graph data: {str(e)}"}

def create_graph_visualization(graph_data: dict) -> go.Figure:
    """
    Create a Plotly visualization of the conversation graph.
    """
    if "error" in graph_data:
        # Create empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=graph_data["error"],
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Graph Visualization Error",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    # Extract nodes and edges from graph data
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    # Create node positions (simple layout)
    node_positions = {}
    for i, node in enumerate(nodes):
        node_positions[node["id"]] = {
            "x": i * 2,
            "y": 0 if node["type"] == "Person" else 1 if node["type"] == "Conv" else 2
        }
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in edges:
        source_pos = node_positions.get(edge["source"], {"x": 0, "y": 0})
        target_pos = node_positions.get(edge["target"], {"x": 0, "y": 0})
        edge_x.extend([source_pos["x"], target_pos["x"], None])
        edge_y.extend([source_pos["y"], target_pos["y"], None])
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    for node in nodes:
        pos = node_positions.get(node["id"], {"x": 0, "y": 0})
        color = "blue" if node["type"] == "Person" else "green" if node["type"] == "Conv" else "orange"
        
        fig.add_trace(go.Scatter(
            x=[pos["x"]], y=[pos["y"]],
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=[node.get("name", node.get("id", ""))],
            textposition="middle center",
            hoverinfo='text',
            name=node["type"]
        ))
    
    fig.update_layout(
        title=f"Conversation Graph: {graph_data.get('conv_id', 'Unknown')}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        height=400
    )
    
    return fig

def create_conversation_stats(conv_id: str) -> go.Figure:
    """
    Create a statistics visualization for the conversation.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/stats/{conv_id}")
        if response.status_code == 200:
            stats = response.json()
            
            # Create subplots for different statistics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Move Distribution", "Power Dynamics", "Speaker Activity", "Timeline"),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Move distribution pie chart
            if "moves" in stats:
                move_counts = {}
                for move in stats["moves"]:
                    move_counts[move] = move_counts.get(move, 0) + 1
                
                fig.add_trace(
                    go.Pie(labels=list(move_counts.keys()), values=list(move_counts.values())),
                    row=1, col=1
                )
            
            # Power dynamics bar chart
            if "power_dynamics" in stats:
                pd_counts = {}
                for pd in stats["power_dynamics"]:
                    pd_counts[pd] = pd_counts.get(pd, 0) + 1
                
                fig.add_trace(
                    go.Bar(x=list(pd_counts.keys()), y=list(pd_counts.values())),
                    row=1, col=2
                )
            
            # Speaker activity
            if "speaker_stats" in stats:
                speakers = list(stats["speaker_stats"].keys())
                turn_counts = [stats["speaker_stats"][s]["turn_count"] for s in speakers]
                
                fig.add_trace(
                    go.Bar(x=speakers, y=turn_counts),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f"Conversation Statistics: {conv_id}",
                height=600
            )
            
            return fig
        else:
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Failed to fetch stats: {response.status_code}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig
            
    except Exception as e:
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating stats: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

def test_ollama_connection() -> str:
    """
    Test connection to Ollama API and return status message.
    """
    try:
        logger.info(f"Testing connection to {OLLAMA_BASE_URL}")
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        logger.info(f"Ollama response status: {response.status_code}")
        if response.status_code == 200:
                return f"Connected to Ollama at {OLLAMA_BASE_URL}"
        else:
            return f"❌ Ollama returned status {response.status_code}"
    except Exception as e:
        return f"❌ Cannot connect to Ollama: {str(e)}"

def get_default_models() -> Dict[str, List[str]]:
    """
    Return a dictionary of default models by provider as fallback.
    """
    return {
        "ollama": ["qwen3:latest", "llama3.2:latest", "mistral:latest", "codellama:latest", "phi3:latest"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    }

def get_available_models() -> Dict[str, List[str]]:
    """
    Fetch available models from all providers.
    """
    try:
        try:
            from app.llm_client import get_available_providers
        except ImportError:
            from llm_client import get_available_providers
        providers = get_available_providers()
        logger.info(f"Available providers: {providers}")
        return providers
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return get_default_models()

def get_all_models_flat() -> List[str]:
    """
    Get all available models as a flat list for the dropdown.
    """
    providers = get_available_models()
    all_models = []
    
    for provider, models in providers.items():
        for model in models:
            # Add provider prefix to distinguish models
            if provider == "gemini":
                all_models.append(f"gemini:{model}")
            else:
                all_models.append(f"ollama:{model}")
    
    return all_models

def create_conversation() -> str:
    """Generate a new conversation ID"""
    return f"conv_{uuid.uuid4().hex[:8]}"

def render_chat(history: List[Dict]) -> List[Dict]:
    """
    Convert history to Gradio Chatbot format with role chips and meta badges.
    Returns list of dicts with 'role' and 'content' keys.
    """
    chat_messages = []
    
    for msg in history:
        # Debug print
        print(f"DEBUG: Rendering message: role={msg.get('role')}, speaker={msg.get('speaker')}, display_name={msg.get('display_name')}, text={msg.get('text')}")
        role = msg.get("role", "")
        display_name = msg.get("display_name", msg.get("speaker", ""))
        text = msg.get("text", "")
        move = msg.get("move")
        pd = msg.get("pd")
        
        # Determine chip class and emoji
        if role == "A":
            chip_class = "chip-a"
            emoji = ""
        elif role == "B":
            chip_class = "chip-b"
            emoji = ""
        elif role == "Coach":
            chip_class = "chip-bot"
            emoji = ""
        else:
            chip_class = "chip-bot"
            emoji = ""
        
        # Build meta badges
        meta_parts = []
        if move:
            meta_parts.append(f"move: {move}")
        if pd:
            meta_parts.append(f"PD: {pd}")
        
        meta_html = ""
        if meta_parts:
            meta_html = f' <span class="meta">· {" · ".join(meta_parts)}</span>'
        
        # Create the HTML bubble and row alignment
        if role == "Coach":
            bubble_class = "coach-bubble"
            row_class = "coach"
        elif role == "A":
            bubble_class = ""
            row_class = "you"  # 'You' (A) should be right-aligned
        else:
            bubble_class = ""
            row_class = "them"   # 'Other Party' (B) should be left-aligned

        html_content = (
            f'<div class="chat-row {row_class}">' \
            f'<span class="chip {chip_class}">{display_name}</span>' \
            f'<div class="bubble {bubble_class}">{text}{meta_html}</div>' \
            f'</div>'
        )
        
        # Return in the correct format for Gradio Chatbot
        chat_messages.append({
            "role": "user" if role in ["A", "B"] else "assistant",
            "content": html_content
        })
    
    return chat_messages

def save_conversation(conv_id: str, history: List[Dict]) -> None:
    """Auto-save conversation to JSON file"""
    try:
        filepath = f"./data/{conv_id}.json"
        with open(filepath, 'w') as f:
            json.dump({
                "conv_id": conv_id,
                "history": history,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Saved conversation to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

def load_conversation(conv_id: str) -> List[Dict]:
    """Load conversation from JSON file"""
    try:
        filepath = f"./data/{conv_id}.json"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data.get("history", [])
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
    return []

try:
    from app.dond_data import load_dond
except ImportError:
    from dond_data import load_dond

# Try to load validation samples
def verify_jsonl_format(file_path: str) -> tuple[bool, str]:
    """
    Verify that a file is in valid JSONL format with expected structure.
    Returns (is_valid, error_message).
    """
    try:
        import json
        from pathlib import Path
        
        if not Path(file_path).exists():
            return False, f"File not found: {file_path}"
            
        # Read first few lines to check format
        valid_lines = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 lines
                    break
                    
                try:
                    data = json.loads(line.strip())
                    # Verify expected structure for DoND data format
                    if not isinstance(data, dict):
                        return False, f"Invalid line {i+1}: not a JSON object"
                    if 'dialogue' not in data:
                        return False, f"Invalid line {i+1}: missing 'dialogue' field"
                    if not isinstance(data['dialogue'], str):
                        return False, f"Invalid line {i+1}: 'dialogue' is not a string"
                    if 'input' not in data or 'count' not in data['input']:
                        return False, f"Invalid line {i+1}: missing 'input.count' field"
                    if not isinstance(data['input']['count'], list):
                        return False, f"Invalid line {i+1}: 'input.count' is not a list"
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    return False, f"Invalid JSON on line {i+1}: {str(e)}"
                
        if valid_lines == 0:
            return False, "File is empty"
            
        # Get file size
        size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        if size_mb < 0.001:
            return False, f"File seems too small ({size_mb:.3f} MB)"
            
        return True, f"Valid JSONL format with {valid_lines} samples checked"
        
    except Exception as e:
        return False, f"Error verifying file: {str(e)}"

def load_validation_samples():
    """Try to load validation samples with directory checks."""
    try:
        # Check DOND_DATA_DIR environment variable, default to the actual location
        dond_dir = os.getenv("DOND_DATA_DIR", "deal_or_no_dialog/exported")
        logger.info(f"DOND_DATA_DIR is set to: {dond_dir}")
        
        # Check if directory exists
        if os.path.exists(dond_dir):
            logger.info(f"Data directory found at: {os.path.abspath(dond_dir)}")
            validation_file = os.path.join(dond_dir, "validation.jsonl")
            if os.path.exists(validation_file):
                logger.info(f"Found validation.jsonl at: {os.path.abspath(validation_file)}")
                
                # Verify file format before loading
                is_valid, msg = verify_jsonl_format(validation_file)
                if not is_valid:
                    logger.error(f"Invalid validation.jsonl: {msg}")
                    return []
                else:
                    logger.info(f"Verified validation.jsonl: {msg}")
                    
                samples = load_dond("validation")
                sample_count = len(samples) if samples else 0
                if sample_count == 0:
                    logger.error("No samples loaded from validation.jsonl")
                    return []
                    
                logger.info(f"Successfully loaded {sample_count} validation samples")
                
                # Validate sample structure
                for i, sample in enumerate(samples[:5]):  # Check first 5 samples
                    if not hasattr(sample, 'turns') or not isinstance(sample.turns, list):
                        logger.error(f"Sample {i} missing 'turns' list")
                        return []
                    if not hasattr(sample, 'counts') or not isinstance(sample.counts, list):
                        logger.error(f"Sample {i} missing 'counts' list")
                        return []
                    if not sample.turns:
                        logger.error(f"Sample {i} has empty turns")
                        return []
                    if not sample.counts:
                        logger.error(f"Sample {i} has empty counts")
                        return []
                
                logger.info("Sample structure validation passed")
                return samples
            else:
                logger.error(f"validation.jsonl not found in {os.path.abspath(dond_dir)}")
        else:
            logger.error(f"Data directory not found at: {os.path.abspath(dond_dir)}")
            
            # Try alternative paths
            alt_paths = [
                "deal_or_no_dialog/exported",     # Primary path (actual location)
                "app/deal_or_no_dialog/exported", # Alternative location
                "../deal_or_no_dialog/exported",  # Fallback paths
                "data",
                "app/data"
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    logger.info(f"Found alternative data directory: {os.path.abspath(path)}")
                    validation_file = os.path.join(path, "validation.jsonl")
                    if os.path.exists(validation_file):
                        logger.info(f"Found validation.jsonl in alternative path: {os.path.abspath(validation_file)}")
                        
                        # Verify file format before loading
                        is_valid, msg = verify_jsonl_format(validation_file)
                        if not is_valid:
                            logger.error(f"Invalid validation.jsonl in alternative path: {msg}")
                            continue
                        else:
                            logger.info(f"Verified validation.jsonl in alternative path: {msg}")
                            
                        os.environ["DOND_DATA_DIR"] = path
                        samples = load_dond("validation")
                        
                        # Validate samples
                        sample_count = len(samples) if samples else 0
                        if sample_count == 0:
                            logger.error("No samples loaded from alternative validation.jsonl")
                            continue
                            
                        logger.info(f"Successfully loaded {sample_count} validation samples from alternative path")
                        return samples
        
        return []
    except Exception as e:
        logger.error(f"Failed to load validation samples: {e}")
        return []

VAL_SAMPLES = load_validation_samples()

def load_dond_sample(idx, conv_id):
    """Load a DoND sample and convert it to chat history format."""
    try:
        if not VAL_SAMPLES:
            logger.error("No validation samples available")
            return [], render_chat([]), "*No samples available*"
        
        sample = VAL_SAMPLES[int(idx)]
        history = []
        
        # Debug: Print sample info
        logger.info(f"Loading sample {idx}: {len(sample.turns)} turns, counts: {sample.counts}")
        
        # Parse the dialogue turns - they are strings, not dicts
        for i, turn in enumerate(sample.turns):
            if not turn.strip():  # Skip empty turns
                continue
                
            # Simple parsing: assume alternating speakers
            # First turn is "you", second is "them", etc.
            if i % 2 == 0:
                role = "A"
                speaker = "You"
            else:
                role = "B" 
                speaker = "Them"
                
            history.append({
                "role": role,
                "speaker": speaker,
                "text": turn.strip(),
                "move": None,
                "pd": None,
                "ts": ""
            })
        
        save_conversation(conv_id, history)
        item_md = "\n".join([f"- **{k}**: {q}" for k,q in enumerate(sample.counts)])
        return history, render_chat(history), item_md
        
    except Exception as e:
        logger.error(f"Error loading DoND sample: {e}")
        return [], render_chat([]), f"*Error loading sample: {str(e)}*"

def label_turn_async(text: str) -> Dict:
    """Label a turn with move and PD asynchronously"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/label",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"move": None, "pd": None}
    except Exception as e:
        logger.error(f"Failed to label turn: {e}")
        return {"move": None, "pd": None}

def get_coach_advice(conv_id: str, speaker: str, model: str, text: Optional[str] = None) -> Tuple[str, str, str]:
    """Get advice from coach API. If text is provided, the API will label and upsert this turn before advising."""
    try:
        # Parse model to extract provider and model name
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            # Default to ollama if no provider specified
            provider = "ollama"
            model_name = model
        
        payload = {
            "conv_id": conv_id,
            "speaker": speaker,
            "text": text or "",  # If provided, upsert this message before advising
            "model": model_name,
            "provider": provider
        }
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            # Strategic advice and provenance/context
            advice = result.get("advice", "No advice available") or "No advice available"
            rag_source = result.get("rag_source", "")
            rag_context = result.get("rag_context", "")
            return advice, rag_source, rag_context
        else:
            return (f"Error getting advice: {response.status_code}", "", "")
    except Exception as e:
        return (f"Connection error: {str(e)}", "", "")

def export_conversation_md(conv_id: str, history: List[Dict]) -> str:
    """Export conversation as Markdown"""
    try:
        filename = f"conversation_{conv_id}_{int(time.time())}.md"
        filepath = f"./data/{filename}"
        
        with open(filepath, 'w') as f:
            f.write(f"# Negotiation Transcript: {conv_id}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for msg in history:
                role = msg.get("role", "")
                speaker = msg.get("speaker", "")
                text = msg.get("text", "")
                move = msg.get("move")
                pd = msg.get("pd")
                
                # Role prefix
                if role == "A":
                    prefix = "**Alice:**"
                elif role == "B":
                    prefix = "**Bob:**"
                elif role == "Coach":
                    prefix = "**Coach:**"
                else:
                    prefix = f"**{speaker}:**"
                
                # Meta info
                meta = []
                if move:
                    meta.append(f"move: {move}")
                if pd:
                    meta.append(f"PD: {pd}")
                
                meta_str = f" *({', '.join(meta)})*" if meta else ""
                
                f.write(f"{prefix} {text}{meta_str}\n\n")
        
        return f"Exported to {filepath}"
    except Exception as e:
        return f"Export error: {str(e)}"

def check_api_status() -> str:
    """Check API status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            return "API is running and healthy"
        else:
            return "API returned error status"
    except Exception as e:
        return f"Cannot connect to API: {str(e)}"

def get_conversation_summary(conv_id: str) -> str:
    """Get conversation summary for debug panel"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats/{conv_id}")
        if response.status_code == 200:
            stats = response.json()
            
            summary = f"**Conversation Summary:**\n"
            summary += f"• Total turns: {stats.get('total_turns', 0)}\n"
            
            if "speaker_stats" in stats:
                summary += f"• Active speakers: {len(stats['speaker_stats'])}\n"
                for speaker, data in stats["speaker_stats"].items():
                    summary += f"  - {speaker}: {data.get('turn_count', 0)} turns\n"
            
            if "moves" in stats:
                move_counts = {}
                for move in stats["moves"]:
                    move_counts[move] = move_counts.get(move, 0) + 1
                summary += f"• Recent moves: {', '.join([f'{move}({count})' for move, count in move_counts.items()])}\n"
            
            return summary
        else:
            return f"Error fetching summary: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

# Item analysis helper
def analyze_items_in_message(text: str, all_items: List[str]) -> List[str]:
    """Find item mentions in message text."""
    found_items = []
    text_lower = text.lower()
    
    # Map common item names to their item IDs
    item_mapping = {
        "book": "item0",
        "books": "item0", 
        "hat": "item1",
        "hats": "item1",
        "ball": "item2", 
        "balls": "item2",
        "basketball": "item2",
        "basketballs": "item2"
    }
    
    # Check for mapped item names
    for item_name, item_id in item_mapping.items():
        if item_name in text_lower:
            found_items.append(item_id)
    
    # Also check for the original item IDs (item0, item1, etc.)
    for item in all_items:
        if item in text_lower:
            found_items.append(item)
    
    return list(set(found_items))  # Remove duplicates

# Simple sentiment analysis helper
def analyze_sentiment(text: str) -> str:
    """Basic sentiment analysis based on keywords."""
    positive = ["agree", "accept", "good", "great", "yes", "deal", "happy", "like", "fair"]
    negative = ["disagree", "reject", "bad", "no", "unfair", "not", "don't", "cannot"]
    
    text = text.lower()
    pos_count = sum(1 for word in positive if word in text)
    neg_count = sum(1 for word in negative if word in text)
    
    if pos_count > neg_count:
        return "Positive 👍"
    elif neg_count > pos_count:
        return "Negative 👎"
    else:
        return "Neutral 💭"

# Deal/No-Deal detection helper
def detect_deal_outcome(turns: List[str]) -> str:
    """Detect if the conversation ends in a deal or no-deal using keyword matching."""
    if not turns:
        return "Unknown ❓"
    
    # Look at the last few turns for deal indicators
    last_turns = turns[-3:]  # Check last 3 turns
    text = " ".join(last_turns).lower()
    
    # Deal indicators (positive phrases) - expanded list
    deal_indicators = [
        "agreed", "accept", "yes", "okay", "ok", "fine", "sure", 
        "that works", "sounds good", "i agree", "we have a deal",
        "let's do it", "that's fair", "i accept", "agreement",
        "deal", "alright", "all right", "good", "great", "perfect",
        "sounds good", "works for me", "i'm good", "i'm fine"
    ]
    
    # No-deal indicators (negative phrases)
    no_deal_indicators = [
        "no deal", "no thanks", "i decline", "not interested", 
        "walk away", "no agreement", "can't agree", "not acceptable",
        "i refuse", "that's not fair", "unfair", "no way",
        "no", "nope", "nah", "not", "don't", "won't"
    ]
    
    # Check for explicit "deal" vs "no deal" first
    if "no deal" in text:
        return "No Deal ❌"
    elif "deal" in text and "no deal" not in text:
        return "Deal ✅"
    
    # Check for other indicators with more weight for recent turns
    deal_score = 0
    no_deal_score = 0
    
    # Give more weight to the last turn
    for i, turn in enumerate(last_turns):
        turn_lower = turn.lower()
        weight = 3 if i == len(last_turns) - 1 else 1  # Last turn gets 3x weight
        
        for indicator in deal_indicators:
            if indicator in turn_lower:
                deal_score += weight
                break  # Only count each indicator once per turn
        
        for indicator in no_deal_indicators:
            if indicator in turn_lower:
                no_deal_score += weight
                break  # Only count each indicator once per turn
    
    if deal_score > no_deal_score:
        return "Deal ✅"
    elif no_deal_score > deal_score:
        return "No Deal ❌"
    else:
        # Check if conversation ends with <selection> (indicates deal in DoND format)
        if turns and turns[-1].strip() == "<selection>":
            return "Deal ✅"
        return "Unknown ❓"

def detect_deal_outcome_llm(turns: List[str], model: str = "qwen3:latest") -> str:
    """Detect if the conversation ends in a deal or no-deal using LLM analysis."""
    if not turns:
        return "Unknown ❓"
    
    try:
        # Try to import LLM client
        try:
            from app.llm_client import create_llm_client, get_provider_from_model
        except ImportError:
            from llm_client import create_llm_client, get_provider_from_model
        
        # Use the provided model (from UI). It may include a provider prefix like "ollama:qwen3:latest".
        chosen_model = model or "qwen3:latest"
        if ":" in chosen_model:
            provider_prefix, actual_model = chosen_model.split(":", 1)
            provider = provider_prefix
            model_name = actual_model
        else:
            provider = get_provider_from_model(chosen_model)
            model_name = chosen_model
        llm_client = create_llm_client(provider, model_name)
        
        # Create the analysis prompt
        conversation_text = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(turns[-5:])])  # Last 5 turns
        
        prompt = f"""Analyze this negotiation conversation and determine if it ended in a DEAL or NO DEAL.

Conversation:
{conversation_text}

Instructions:
- Look for clear indicators of agreement or disagreement
- Consider context, tone, and explicit statements
- Pay special attention to the final turns
- If there's ambiguity, err on the side of "NO DEAL"

Respond with ONLY one of these exact phrases:
- "DEAL" (if both parties clearly agreed to terms)
- "NO DEAL" (if parties disagreed, walked away, or outcome is unclear)
- "UNKNOWN" (if impossible to determine)

Your analysis:"""
        
        # Get LLM response
        response = llm_client.generate_response([
            {"role": "user", "content": prompt}
        ], temperature=0.1)  # Low temperature for consistent results
        
        # Parse the response
        response = response.strip().upper()
        if "DEAL" in response and "NO DEAL" not in response:
            return "Deal ✅"
        elif "NO DEAL" in response:
            return "No Deal ❌"
        else:
            return "Unknown ❓"
            
    except Exception as e:
        logger.warning(f"LLM deal detection failed: {e}, falling back to keyword detection")
        return detect_deal_outcome(turns)  # Fall back to keyword method

#  Early detection helpers 
def detect_first_no_deal_turn(turns: List[str]) -> int | None:
    """Return 0-based index of the earliest turn that strongly signals no-deal."""
    if not turns:
        return None
    no_deal_indicators = [
        "no deal", "no thanks", "i decline", "not interested",
        "walk away", "no agreement", "can't agree", "not acceptable",
        "i refuse", "unfair", "no way", "nope", "nah"
    ]
    for i, t in enumerate(turns):
        tl = (t or "").lower()
        if any(ind in tl for ind in no_deal_indicators):
            return i
    return None

def detect_first_nash_like_turn(turns: List[str]) -> int | None:
    """Heuristic detection of the first turn proposing an equal/50-50 style split.

    We approximate Nash bargaining proposal with common phrases: 50-50, equally,
    evenly, equal split, half-and-half. This is a UI hint, not a proof.
    """
    import re
    nash_patterns = [
        r"50\s*[-–]?\s*50",         # 50-50 or 50 50
        r"\b50%\b.*\b50%\b",      # 50% ... 50%
        r"\bequal(?:ly)?\s+split\b",
        r"\bsplit\b.*\b(equally|evenly)\b",
        r"\bhalf(?:\s*(and|&)\s*half)?\b",
    ]
    if not turns:
        return None
    for i, t in enumerate(turns):
        tl = (t or "").lower()
        if any(re.search(p, tl) for p in nash_patterns):
            return i
    return None

def load_dond_sample_viz(idx, use_llm_detection=True, filter_no_deal=False, enable_coach=False, model: str = "qwen3:latest"):
    """Load and analyze a DoND sample for visualization with optional coach advice."""
    try:
        if not VAL_SAMPLES:
            error_msg = (
                "### ⚠️ No samples available\n\n"
                "Please check:\n"
                "1. The `deal_or_no_dialog/exported` directory exists\n"
                "2. It contains `validation.jsonl`\n"
                "3. The `DOND_DATA_DIR` environment variable points to the correct location\n\n"
                "Current search paths:\n"
                "- deal_or_no_dialog/exported/  (primary)\n"
                "- app/deal_or_no_dialog/exported/\n"
                "- ../deal_or_no_dialog/exported/\n"
                "- data/\n"
                "- app/data/"
            )
            logger.error("No validation samples available")
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            empty_df = [["", "", "No samples available", "", "", ""]]
            return (error_msg,
                   error_msg,
                   "*No coach advice available*",
                   empty_df, 
                   empty_fig, 
                   empty_fig)
        
        # Filter for no-deal conversations if requested
        if filter_no_deal:
            # Find all no-deal samples
            no_deal_samples = []
            for i, s in enumerate(VAL_SAMPLES):
                outcome = detect_deal_outcome(s.turns)
                if "No Deal" in outcome:
                    no_deal_samples.append((i, s))
            
            if not no_deal_samples:
                error_msg = "### ⚠️ No 'No Deal' conversations found\n\nAll conversations in the dataset ended in deals."
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="No no-deal samples", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                empty_df = [["", "", "No no-deal samples available", "", "", ""]]
                return (error_msg, error_msg, "*No coach advice available*", empty_df, empty_fig, empty_fig)
            
            # Select the no-deal sample whose original ID is closest to the requested idx
            try:
                target = int(idx)
            except Exception:
                target = 0
            try:
                min_dist = min(abs(i - target) for i, _ in no_deal_samples)
                candidates = [(i, s) for i, s in no_deal_samples if abs(i - target) == min_dist]
                # Break ties randomly to avoid always showing the same sample
                import random as _rnd
                actual_idx, sample = _rnd.choice(candidates)
            except Exception:
                # Fallback to a deterministic first no-deal sample
                actual_idx, sample = no_deal_samples[0]
        else:
            sample = VAL_SAMPLES[int(idx)]
        
        # Item counts analysis
        item_counts_md = "#### Item Counts\n"
        item_counts_md += "\n".join([f"- **Item {k}**: {q}" for k, q in enumerate(sample.counts)])
        
        # Deal outcome analysis - use LLM if enabled, otherwise use keyword detection
        if use_llm_detection:
            try:
                deal_outcome = detect_deal_outcome_llm(sample.turns, model)
                detection_method = "LLM Analysis"
            except Exception as e:
                logger.warning(f"LLM detection failed for sample {idx}: {e}")
                deal_outcome = detect_deal_outcome(sample.turns)
                detection_method = "Keyword Detection (LLM failed)"
        else:
            deal_outcome = detect_deal_outcome(sample.turns)
            detection_method = "Keyword Detection"
        
        item_counts_md += f"\n\n#### Deal Outcome\n"
        item_counts_md += f"- **Result**: {deal_outcome}\n"
        item_counts_md += f"- **Detection Method**: {detection_method}"
        
        # Early outcome markers (Nash-like or No Deal)
        first_no_deal_idx = detect_first_no_deal_turn(sample.turns)
        first_nash_idx = detect_first_nash_like_turn(sample.turns)

        # Coach advice functionality summary (kept minimal; per-turn advice handled below)
        coach_advice_md = "*Coach advice available inline in the timeline when enabled.*" if enable_coach else "*Coach advice disabled*"
        
        # Speaker statistics initialization (no sentiment)
        speakers = {"You": {"turns": 0, "items_mentioned": set()},
                   "Them": {"turns": 0, "items_mentioned": set()}}
        
        # Timeline data
        timeline_data = []
        all_items = [f"item{i}" for i in range(len(sample.counts))]
        
        # Create a temp conversation ID and process each turn
        temp_conv_id = f"dond_viz_{int(time.time())}"
        # Save local history for export/debug (not used by API)
        history = []
        # Process each turn
        for i, turn in enumerate(sample.turns):
            if not turn.strip():
                continue
            
            # Determine speaker
            speaker = "You" if i % 2 == 0 else "Them"
            
            # Analyze items mentioned
            items_mentioned = analyze_items_in_message(turn.lower(), all_items)
            
            # Update speaker statistics
            speakers[speaker]["turns"] += 1
            speakers[speaker]["items_mentioned"].update(items_mentioned)
            history.append({
                "role": "A" if speaker == "You" else "B",
                "speaker": speaker,
                "text": turn.strip(),
                "move": None,
                "pd": None,
                "ts": datetime.now().isoformat()
            })
            
            # Deal markers for this row
            outcome_marker = ""
            if first_nash_idx is not None and i == first_nash_idx:
                outcome_marker = "Nash-like"
            if first_no_deal_idx is not None and i == first_no_deal_idx:
                outcome_marker = "No-Deal Signal"
            # Add to timeline
            timeline_data.append([
                i + 1,  # Turn number
                speaker,
                turn.strip(),
                ", ".join(items_mentioned) if items_mentioned else "-",
                outcome_marker if outcome_marker else (deal_outcome if i == len(sample.turns) - 1 else ""),
                ""  # RAG Source (only shown on coach advice rows)
            ])

            # If enabled, upsert this turn via API and append coach advice (for both parties)
            if enable_coach:
                try:
                    advice_text, rag_src, rag_ctx = get_coach_advice(temp_conv_id, speaker, model, text=turn.strip())
                    if advice_text and not advice_text.startswith(("Error", "Connection error", "Waiting for both parties")):
                        timeline_data.append([
                            i + 1,              # same turn index to appear right after
                            "Coach",
                            f"Advice: {advice_text}",
                            "-",
                            "",  # deal outcome (blank on advice rows)
                            (rag_src or "none")
                        ])
                except Exception as e:
                    logger.warning(f"Per-turn coach advice failed at turn {i+1}: {e}")
        
        # Persist local history (for export/debug)
        save_conversation(temp_conv_id, history)

        # Create speaker statistics markdown
        speaker_stats_md = "#### Speaker Statistics\n\n"
        speaker_stats_md += f"**Deal Outcome**: {deal_outcome}\n"
        speaker_stats_md += f"**Detection Method**: {detection_method}\n\n"
        for speaker, stats in speakers.items():
            speaker_stats_md += f"**{speaker}**:\n"
            speaker_stats_md += f"- Turns: {stats['turns']}\n"
            speaker_stats_md += f"- Items mentioned: {', '.join(sorted(stats['items_mentioned'])) if stats['items_mentioned'] else 'None'}\n"
        
        # Speaker activity plot (simple turn counts)
        fig_speaker = go.Figure()
        fig_speaker.add_bar(x=list(speakers.keys()), y=[speakers['You']['turns'], speakers['Them']['turns']], name="Turns")
        fig_speaker.update_layout(title="Speaker Activity", xaxis_title="Speaker", yaxis_title="Turns")
        
        # Create content analysis plot
        item_mentions = {item: [] for item in all_items}
        for i, turn in enumerate(sample.turns):
            items = analyze_items_in_message(turn.lower(), all_items)
            for item in all_items:
                item_mentions[item].append(1 if item in items else 0)
        
        fig_content = go.Figure()
        for item, mentions in item_mentions.items():
            fig_content.add_trace(go.Scatter(
                x=list(range(1, len(mentions) + 1)),
                y=mentions,
                name=item,
                mode='lines+markers'
            ))
        fig_content.update_layout(
            title="Item Mentions Over Time",
            xaxis_title="Turn",
            yaxis_title="Mentioned",
            yaxis=dict(ticktext=["No", "Yes"], tickvals=[0, 1])
        )
        
        return item_counts_md, speaker_stats_md, coach_advice_md, timeline_data, fig_speaker, fig_content
        
    except Exception as e:
        logger.error(f"Error in load_dond_sample_viz: {e}")
        return ("*Error loading item counts*",
                "*Error loading speaker stats*",
                "*Error loading coach advice*",
                [], 
                go.Figure(), 
                go.Figure())

def create_unified_interface():
    """Create the unified chat interface"""
    # Get initial list of models
    available_models = get_all_models_flat()
    logger.info(f"Initial models for dropdown: {available_models}")
    
    # Ensure we have at least one model
    if not available_models:
        default_models = get_default_models()
        available_models = []
        for provider, models in default_models.items():
            for model in models:
                if provider == "gemini":
                    available_models.append(f"gemini:{model}")
                else:
                    available_models.append(f"ollama:{model}")
        logger.warning("No models found, using defaults")
    
    # Ensure the default model is in the list
    default_model = "ollama:qwen3:latest"
    if default_model not in available_models:
        available_models.insert(0, default_model)
        logger.info(f"Added default model to list: {available_models}")
    
    logger.info(f"Final models for dropdown: {available_models}")
    logger.info(f"Default model selected: {default_model}")
    
    with gr.Blocks(title="AI Chat Negotiator", theme=gr.themes.Soft(), css="app/style.css") as demo:
        gr.Markdown("# AI Chat Negotiator")
        gr.Markdown("Multi-party negotiation with AI assistance and real-time analysis.")
        
        # Top row: Conversation ID and Model selection
        with gr.Row():
            conversation_id = gr.Textbox(
                label="Conversation ID", 
                value=create_conversation(), 
                placeholder="Enter conversation ID or leave empty for auto-generated"
            )
            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=default_model,
                label="AI Model",
                info="Choose which AI model to use for advice generation (Ollama or Gemini)",
                allow_custom_value=True
            )
        
        # DoND sample selector (removed per request)
        # sample_idx = gr.Slider(
        #     minimum=0, maximum=1499, step=1, value=0,
        #     label="DoND sample ( validation split )",
        #     interactive=True
        # )
        
        # Negotiator names row
        with gr.Row():
            speaker_a = gr.Textbox(
                label="Your Name", 
                value="You", 
                placeholder="Enter your name"
            )
            speaker_b = gr.Textbox(
                label="Other Party Name", 
                value="The other party", 
                placeholder="Enter the other party's name"
            )
        
        # Main chat area
        with gr.Row():
            with gr.Column(scale=1):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Negotiation Chat", 
                    height=500, 
                    show_label=False, 
                    type="messages",
                    sanitize_html=False
                )
                
                # Input area
                with gr.Row():
                    role_radio = gr.Radio(
                        choices=["You", "Other Party"],
                        value="You",
                        label="Role",
                        scale=1
                    )
                    message_input = gr.Textbox(
                        label="Message", 
                        placeholder="Type your message... (Enter to send, Shift+Enter for newline)", 
                        lines=2,
                        scale=3
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Bottom tool area with accordions
        with gr.Accordion("DoND Conversation Visualizer", open=False):
            # Sample selector row
            with gr.Row():
                # Sample slider - disable if no samples available
                max_samples = len(VAL_SAMPLES) - 1 if VAL_SAMPLES else 0
                viz_sample_idx = gr.Slider(
                    minimum=0, maximum=max_samples, step=1, value=0,
                    label=f"DoND Sample ID (0-{max_samples})" if VAL_SAMPLES else "DoND Sample ID (No samples available)",
                    interactive=bool(VAL_SAMPLES)
                )
                # Load button - disable if no samples available
                load_viz_btn = gr.Button(
                    "Load Sample" if VAL_SAMPLES else "No Samples Available",
                    variant="primary" if VAL_SAMPLES else "secondary",
                    interactive=bool(VAL_SAMPLES)
                )
                # LLM detection toggle
                use_llm_detection = gr.Checkbox(
                    label="Use LLM for Deal Detection",
                    value=True,
                    interactive=bool(VAL_SAMPLES)
                )
                # Filter for no-deal conversations
                filter_no_deal = gr.Checkbox(
                    label="Show Only No-Deal Conversations",
                    value=False,
                    interactive=bool(VAL_SAMPLES)
                )
                # Enable coach advice
                enable_coach = gr.Checkbox(
                    label="Enable Coach Advice",
                    value=False,
                    interactive=bool(VAL_SAMPLES)
                )
            
            # Conversation display area
            with gr.Row():
                # Left side: Conversation details
                with gr.Column(scale=1):
                    gr.Markdown("#### Conversation Details")
                    item_counts = gr.Markdown("*Item counts will appear here*")
                    speaker_stats = gr.Markdown("*Speaker statistics will appear here*")
                    coach_advice = gr.Markdown("*Coach advice will appear here*")
                
                # Right side: Message timeline
                with gr.Column(scale=2):
                    gr.Markdown("#### Message Timeline")
                    message_timeline = gr.Dataframe(
                        headers=["Turn", "Speaker", "Message", "Items Mentioned", "Deal Outcome", "RAG Source"],
                        datatype=["number", "string", "string", "string", "string", "string"],
                        row_count=10,
                        col_count=(6, "fixed"),
                        interactive=False,
                        wrap=True
                    )
            
            # Analysis plots
            with gr.Row():
                # Left plot: Speaker Activity
                speaker_plot = gr.Plot(label="Speaker Activity")
                # Right plot: Content Analysis
                content_plot = gr.Plot(label="Content Analysis")

        #  Pareto Coach Effectiveness Simulator 
        with gr.Accordion("Pareto Coach Effectiveness Simulator", open=False):
            with gr.Row():
                sim_n = gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Number of samples (validation)")
                sim_baseline = gr.Dropdown(choices=["equal", "greedy", "walkaway", "statusquo"], value="equal", label="Baseline")
                sim_ratio = gr.Slider(minimum=0.7, maximum=1.0, step=0.05, value=1.0, label="Success threshold ratio")
                run_sim_btn = gr.Button("Run Simulation", variant="primary")

            sim_results_md = gr.Markdown("*No results yet*")
            sim_transcripts_html = gr.HTML("<i>No transcripts yet</i>")

            def run_pareto_sim(n, baseline, ratio, model):
                try:
                    # Prefer absolute import when running within app package
                    from app.simulate_dond import simulate_with_coach  # type: ignore
                except Exception:
                    # Fallback for script mode
                    from simulate_dond import simulate_with_coach  # type: ignore
                summary = simulate_with_coach(int(n), baseline, float(ratio))
                lines = [
                    "### Coach Rescue Summary",
                    f"- Total evaluated: {summary['total']}",
                    f"- Successes without coach: {summary['success_without_coach']}",
                    f"- Rescued by coach: {summary['rescued_by_coach']}",
                    f"- Rescue rate among previous failures: {summary['rescue_rate']:.1%}",
                    f"- Overall success with coach: {summary['overall_success_with_coach']:.1%}",
                ]
                transcripts = summary.get("transcripts", [])
                # Build full HTML for all transcripts, including an AI advice line
                html_parts = ["<div>"]
                for i, sess in enumerate(transcripts):
                    html_parts.append(f"<div style='margin:12px 0;padding:8px;border:1px solid #444;border-radius:8px;'>")
                    html_parts.append(f"<div><b>Transcript {i}</b> - {'RESCUED' if sess.get('rescued') else 'NOT RESCUED'}</div>")
                    conv_id = f"simui_{int(time.time())}_{i}"
                    last_advice = None
                    # replay messages to API to fetch the last AI advice in context
                    for m in sess.get("messages", []):
                        role = m.get("role", "")
                        text = m.get("text", "")
                        if role in ("You", "Them"):
                            try:
                                advice, rag_src, rag_ctx = get_coach_advice(conv_id, role, model, text=text)
                                if advice and not advice.lower().startswith(("waiting for both parties to speak", "error")):
                                    last_advice = advice
                            except Exception:
                                pass
                        align = "left" if role == "You" else ("right" if role == "Them" else "center")
                        html_parts.append(f"<div style='text-align:{align};margin:4px 0;'><span style='font-weight:600'>{role}:</span> {text}</div>")
                    if last_advice:
                        html_parts.append(f"<div style='text-align:center;margin-top:6px;'><span style='font-weight:600'>Coach AI advice:</span> {last_advice}</div>")
                    html_parts.append("</div>")
                html_parts.append("</div>")
                return "\n".join(lines), "\n".join(html_parts)

            run_sim_btn.click(fn=run_pareto_sim, inputs=[sim_n, sim_baseline, sim_ratio, model_dropdown], outputs=[sim_results_md, sim_transcripts_html])

        # (Hidden) Tools & Exports panel removed entirely
        # We still need placeholders for graph/stats/table used by events later
        graph_plot = gr.Plot(visible=False)
        stats_plot = gr.Plot(visible=False)
        item_table = gr.Markdown(visible=False)
        
        # (Hidden) Model & API panel removed
        
        # (Hidden) Conversation Inspector removed
        
        # (Hidden) Debug / Status removed
        status_display = gr.Textbox(visible=False)
        history_json = gr.JSON(visible=False)
        summary_json = gr.JSON(visible=False)
        summary_display = gr.Markdown(visible=False)
        
        # Hidden state for history
        history_state = gr.State([])
        
        def on_send(role, text, model, conv_id, history, speaker_a, speaker_b):
            """Handle sending messages"""
            if not text.strip():
                return history, render_chat(history)
            
            # Determine role and speaker
            if role == "You":
                msg_role = "A"
                speaker = "A"
                display_name = speaker_a or "You"
            elif role == "Other Party":
                msg_role = "B"
                speaker = "B"
                display_name = speaker_b or "The other party"
            else:
                msg_role = "A"
                speaker = "A"
                display_name = speaker_a or "You"
            
            # Create new message
            new_msg = {
                "role": msg_role,
                "speaker": speaker,  # Always 'A' or 'B' for backend
                "display_name": display_name,  # For UI display
                "text": text,
                "move": None,
                "pd": None,
                "ts": datetime.now().isoformat()
            }
            # Debug print
            print(f"DEBUG: Added message: role={msg_role}, speaker={speaker}, display_name={display_name}, text={text}")
            
            # Add to history
            history.append(new_msg)
            
            # Auto-save
            save_conversation(conv_id, history)
            
            # Helper: should we render this advice bubble?
            def _usable_advice(text: str) -> bool:
                if not text:
                    return False
                t = text.strip().lower()
                if t.startswith("waiting for both parties to speak"):
                    return False
                if "waiting for both parties to speak" in t:
                    return False
                return True

            # Get advice after messages from either party (A or B)
            try:
                # Pass role ("A" or "B") instead of speaker name to coach
                coach_advice, rag_source, rag_context = get_coach_advice(conv_id, msg_role, model, text=text)
                if coach_advice and not coach_advice.startswith(("Error", "Connection error")):
                    coach_msg = {
                        "role": "Coach",
                        "speaker": "Coach",
                        "text": coach_advice,
                        "move": None,
                        "pd": None,
                        "ts": datetime.now().isoformat()
                    }
                    history.append(coach_msg)
                    save_conversation(conv_id, history)
            except Exception as e:
                logger.error(f"Failed to get coach advice: {e}")
            
            # Auto-generate bot proposal for role B (other party) messages
            if msg_role == "B":
                try:
                    from app.autoplay import generate_bot_proposal, format_proposal_message
                    
                    # Extract conversation turns for preference estimation
                    turns = []
                    for msg in history:
                        if msg["role"] in ["A", "B"]:
                            turns.append(msg["text"])
                    
                    # Default item counts (can be made configurable)
                    counts = {"item0": 3, "item1": 2, "item2": 1}
                    
                    # Generate bot proposal
                    proposal = generate_bot_proposal(turns, counts)
                    if proposal:
                        # Format and add bot proposal message
                        proposal_text = format_proposal_message(proposal, counts)
                        proposal_msg = {
                            "role": "Coach",
                            "speaker": "Bot",
                            "text": proposal_text,
                            "move": None,
                            "pd": None,
                            "ts": datetime.now().isoformat()
                        }
                        history.append(proposal_msg)
                        save_conversation(conv_id, history)
                except Exception as e:
                    logger.error(f"Failed to generate bot proposal: {e}")
            
            # After sending, auto-switch role for next message
            next_role = "Other Party" if role == "You" else "You"
            return history, render_chat(history), next_role
        
        def new_conversation():
            """Start a new conversation"""
            new_conv_id = create_conversation()
            return new_conv_id, [], render_chat([])
        
        def export_md(conv_id, history):
            """Export conversation as Markdown"""
            return export_conversation_md(conv_id, history)
        
        def update_visualizations(conv_id):
            """Update visualizations"""
            graph_data = get_conversation_graph_data(conv_id)
            graph_fig = create_graph_visualization(graph_data)
            stats_fig = create_conversation_stats(conv_id)
            return graph_fig, stats_fig
        
        def update_model_list():
            """Update the model dropdown with fresh list"""
            models = get_all_models_flat()
            logger.info(f"Refreshed models: {models}")
            if not models:
                default_models = get_default_models()
                models = []
                for provider, provider_models in default_models.items():
                    for model in provider_models:
                        if provider == "gemini":
                            models.append(f"gemini:{model}")
                        else:
                            models.append(f"ollama:{model}")
                logger.warning("No models found on refresh, using defaults")
            
            # Ensure default model is in the list
            default_model = "ollama:qwen3:latest"
            if default_model not in models:
                models.insert(0, default_model)
                logger.info(f"Added default model to refreshed list: {models}")
            
            return gr.Dropdown(choices=models, value=default_model, allow_custom_value=True)
        
        def _update_inspector(conv_id):
            """Update conversation inspector with history and stats"""
            try:
                # Load history
                history = load_conversation(conv_id)
                
                # Fetch stats from API
                stats_response = requests.get(f"{API_BASE_URL}/stats/{conv_id}")
                stats = stats_response.json() if stats_response.status_code == 200 else {"error": "Failed to fetch stats"}
                
                return history, stats
            except Exception as e:
                logger.error(f"Error updating inspector: {e}")
                return [], {"error": str(e)}
        
        def update_debug_info(conv_id, history):
            """Update debug information"""
            summary = get_conversation_summary(conv_id)
            return summary
        
        # Event handlers
        send_btn.click(
            fn=on_send,
            inputs=[role_radio, message_input, model_dropdown, conversation_id, history_state, speaker_a, speaker_b],
            outputs=[history_state, chatbot, role_radio]
        )
        
        message_input.submit(
            fn=on_send,
            inputs=[role_radio, message_input, model_dropdown, conversation_id, history_state, speaker_a, speaker_b],
            outputs=[history_state, chatbot, role_radio]
        )
        
        # Removed: new_conv_btn, export_btn, update_viz_btn UI controls
        
        # Removed: model/API test buttons
        
        # Update debug info when conversation changes
        # Removed: debug info auto-update
        
        # Update conversation inspector when conversation changes
        # Removed: conversation inspector updates
        
        # Load conversation when ID changes
        def load_conv(conv_id):
            history = load_conversation(conv_id)
            return history, render_chat(history)
        
        conversation_id.change(
            fn=load_conv,
            inputs=conversation_id,
            outputs=[history_state, chatbot]
        )
        
        # Removed: top DoND sample slider bindings
        
        # Load DoND visualization sample
        load_viz_btn.click(
            fn=load_dond_sample_viz,
            inputs=[viz_sample_idx, use_llm_detection, filter_no_deal, enable_coach, model_dropdown],
            outputs=[item_counts, speaker_stats, coach_advice, message_timeline, speaker_plot, content_plot]
        )
        
        # Load initial status
        # Removed: initial status check
        
    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-port", type=int, default=7860, help="Port to run the server on")
    args = parser.parse_args()
    
    demo = create_unified_interface()
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=False, show_error=True) 