import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample negotiation tactics data"""
    return [
        {
            "text": "When dealing with aggressive negotiators, maintain calm and professional demeanor. Use active listening to understand their concerns. Avoid matching their aggression - instead, redirect the conversation toward mutual interests and shared goals. Document all agreements in writing to prevent misunderstandings.",
            "category": "aggressive_negotiators"
        },
        {
            "text": "For information sharing in negotiations, start with low-risk information to build trust. Gradually increase the depth of information shared as reciprocity develops. Always verify information received from the other party before making decisions based on it.",
            "category": "information_sharing"
        },
        {
            "text": "When making concessions, start with small, low-cost concessions to test the other party's willingness to reciprocate. Never make unilateral concessions without getting something in return. Keep track of all concessions made and received.",
            "category": "concessions"
        },
        {
            "text": "To build trust in negotiations, be consistent in your behavior and follow through on commitments. Show respect for the other party's interests and concerns. Use objective criteria when possible to justify your positions.",
            "category": "trust_building"
        },
        {
            "text": "When facing threats or ultimatums, remain calm and ask for clarification. Explore the underlying interests behind the threat. Consider whether the threat is credible and what alternatives you have if the threat is carried out.",
            "category": "threats_ultimatums"
        },
        {
            "text": "For cooperative negotiation strategies, focus on expanding the pie rather than dividing it. Look for creative solutions that satisfy both parties' interests. Use brainstorming techniques to generate multiple options before deciding.",
            "category": "cooperative_strategies"
        },
        {
            "text": "When the other party defects or acts uncooperatively, resist the urge to retaliate immediately. Consider whether this is a temporary setback or a fundamental change in their approach. Use tit-for-tat strategy: cooperate initially, then mirror their last move.",
            "category": "defection_response"
        },
        {
            "text": "To handle deadlocks in negotiations, take a break to allow emotions to cool. Bring in a neutral third party if necessary. Focus on underlying interests rather than positions. Consider whether the current negotiation is the right forum for resolving the issue.",
            "category": "deadlock_resolution"
        }
    ]

def initialize_rag_system():
    """Initialize RAG system with automatic setup"""
    try:
        print("Loading embedding model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("Embedding model loaded successfully.")

        print("Connecting to vector database...")
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use get_or_create_collection instead of try-catch
        collection = chroma_client.get_or_create_collection(name="negotiation_tactics")
        print("Collection 'negotiation_tactics' ready.")
        
        # Check if collection has data
        count = collection.count()
        if count == 0:
            print("Collection is empty. Adding sample data...")
            add_sample_data_to_collection(collection, embedding_model)
        else:
            print(f"Collection has {count} documents.")
        
        print("Successfully connected to vector database.")
        return embedding_model, collection
        
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        return None, None

def add_sample_data_to_collection(collection, embedding_model):
    """Add sample negotiation tactics to the collection"""
    negotiation_tactics = create_sample_data()
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for tactic in negotiation_tactics:
        documents.append(tactic["text"])
        metadatas.append({"category": tactic["category"]})
        ids.append(str(uuid.uuid4()))
    
    # Create embeddings
    print("Creating embeddings for negotiation tactics...")
    embeddings = embedding_model.encode(documents).tolist()
    
    # Add documents to collection
    print("Adding documents to collection...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Successfully added {len(documents)} negotiation tactics to the RAG system!")

#  RAG System Initialization 
# This section loads the embedding model and connects to the vector store.
# It's done globally to avoid reloading the model on every request, which would be very slow.
embedding_model, negotiation_collection = initialize_rag_system()

def retrieve_rag_context(query: str, n_results: int = 5) -> str:
    """
    Retrieves relevant context from the vector database based on the user's query.
    
    Args:
        query (str): The user's question.
        n_results (int): The number of relevant chunks to retrieve.
        
    Returns:
        str: A formatted string containing the retrieved context, or an empty string if retrieval fails.
    """
    if not embedding_model or not negotiation_collection:
        print("RAG system not initialized. Skipping retrieval.")
        return ""
    
    try:
        # 1. Embed the user's query using the same model we used for the documents.
        query_embedding = embedding_model.encode(query).tolist()
        
        # 2. Query the collection to find the most similar document chunks.
        results = negotiation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # 3. Format the results into a single string to be used as context.
        context_chunks = results.get('documents', [[]])[0]
        if not context_chunks:
            return ""
        
        # Each chunk is separated for clarity in the final prompt.
        formatted_context = "\n\n\n\n".join(context_chunks)
        return formatted_context
        
    except Exception as e:
        print(f"Error during RAG retrieval: {e}")
        return "" 