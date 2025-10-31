#!/usr/bin/env python3
"""
Setup script for RAG system with sample negotiation data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chromadb
from sentence_transformers import SentenceTransformer
import uuid

def setup_rag():
    print("Setting up RAG system with sample negotiation data...")
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully.")
    
    # Connect to ChromaDB
    print("Connecting to vector database...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    try:
        collection = chroma_client.get_collection(name="negotiation_tactics")
        print("Using existing collection: negotiation_tactics")
    except:
        print("Creating new collection: negotiation_tactics")
        collection = chroma_client.create_collection(name="negotiation_tactics")
    
    # Sample negotiation tactics data
    negotiation_tactics = [
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
    print("Available categories:")
    for tactic in negotiation_tactics:
        print(f"  - {tactic['category']}")
    
    # Test the RAG system
    print("\nTesting RAG system...")
    from app.rag import retrieve_rag_context
    
    test_queries = [
        "How do I handle aggressive negotiators?",
        "What should I do when someone makes threats?",
        "How do I build trust in negotiations?",
        "What if the other party doesn't cooperate?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = retrieve_rag_context(query)
        if context:
            print(f"✅ Found relevant context: {context[:100]}...")
        else:
            print("❌ No relevant context found")

if __name__ == "__main__":
    setup_rag() 