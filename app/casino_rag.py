import os
from typing import Dict, List, Optional, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from convokit import Corpus, download
import pickle
from pathlib import Path
import random
import logging
import json

logger = logging.getLogger(__name__)

class CasinoRAG:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.corpus = None
        self.eval_conversations = None
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.initialize_system()
    
    def initialize_system(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="casino_negotiations",
                metadata={"hnsw:space": "cosine"}
            )
            if self.collection.count() == 0:
                self.load_or_download_corpus()
                if self.corpus:
                    self.populate_casino_data()
        except Exception:
            self.embedding_model = None
            self.chroma_client = None
            self.collection = None
            self.corpus = None
    
    def load_or_download_corpus(self):
        """Load cached corpus or download and cache it. Also partition for eval."""
        corpus_cache_file = self.cache_dir / "casino_corpus.pkl"
        eval_ids_file = self.cache_dir / "casino_eval_ids.json"
        # --- Reserved dialogue IDs for evaluation ---
        reserved_eval_ids = {
            "0", "4", "22", "32", "35",  # Category 1
            "1", "3", "11", "16", "29",  # Category 2
            "5", "19", "20", "25", "41"  # Category 3
        }
        try:
            if corpus_cache_file.exists() and eval_ids_file.exists():
                logger.info("Loading casino corpus and eval split from cache")
                with open(corpus_cache_file, 'rb') as f:
                    self.corpus = pickle.load(f)
                with open(eval_ids_file, 'r') as f:
                    eval_ids = json.load(f)
                self.eval_conversations = set(eval_ids)
                logger.info("Successfully loaded corpus and eval split from cache")
                return

            self.corpus = Corpus(filename=download("casino-corpus"))

            all_conv_ids = [conv.id for conv in self.corpus.iter_conversations()]
            remaining_ids = [cid for cid in all_conv_ids if cid not in reserved_eval_ids]
            random.seed(42)  # For reproducibility
            random.shuffle(remaining_ids)
            split_idx = int(0.8 * len(remaining_ids))
            rag_ids = set(remaining_ids[:split_idx])
            eval_ids = set(remaining_ids[split_idx:]) | reserved_eval_ids
            self.eval_conversations = eval_ids
            
            with open(corpus_cache_file, 'wb') as f:
                pickle.dump(self.corpus, f)
            with open(eval_ids_file, 'w') as f:
                json.dump(list(eval_ids), f)
            logger.info("Successfully cached casino corpus and eval split")

        except Exception as corpus_error:
            logger.error(f"Error loading/downloading casino corpus: {corpus_error}")
            logger.info("casino RAG will use fallback mode - no corpus data available")
            self.corpus = None
            self.eval_conversations = None
    
    def clear_cache(self):
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
        except Exception:
            pass
    
    def get_cache_info(self):
        try:
            corpus_cache_file = self.cache_dir / "casino_corpus.pkl"
            cache_size = corpus_cache_file.stat().st_size if corpus_cache_file.exists() else 0
            return {
                "cache_dir": str(self.cache_dir),
                "corpus_cached": corpus_cache_file.exists(),
                "cache_size_mb": round(cache_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def chunk_casino_conversations(self) -> List[Dict]:
        documents = []
        try:
            for conversation in self.corpus.iter_conversations():
                conv_id = conversation.id
                conv_meta = conversation.meta
                strategy = conv_meta.get('strategy', 'unknown')
                personality_pair = conv_meta.get('personality_pair', 'unknown')
                outcome_points = conv_meta.get('outcome_points', 0)
                turns = list(conversation.iter_utterances())
                for i in range(len(turns) - 1):
                    speaker_utterance = turns[i]
                    reply = turns[i + 1]
                    speaker_text = speaker_utterance.text
                    reply_text = reply.text
                    pd = speaker_utterance.meta.get('pd', 'C')
                    document_text = f"Speaker: {speaker_text}\nReply: {reply_text}"
                    metadata = {
                        "strategy": strategy,
                        "pd": pd,
                        "personality_pair": personality_pair,
                        "outcome_points": outcome_points,
                        "conv_id": conv_id,
                        "turn_index": i
                    }
                    documents.append({
                        "text": document_text,
                        "metadata": metadata
                    })
            return documents
        except Exception:
            return []
    
    def populate_casino_data(self):
        try:
            documents = self.chunk_casino_conversations()
            if not documents:
                return
            batch_size = 1000
            total_docs = len(documents)
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                texts = [doc["text"] for doc in batch_docs]
                metadatas = [doc["metadata"] for doc in batch_docs]
                ids = [f"casino_{j}" for j in range(i, batch_end)]
                embeddings = self.embedding_model.encode(texts).tolist()
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
        except Exception:
            try:
                documents = self.chunk_casino_conversations()
                self.populate_casino_data_small_batches(documents)
            except Exception:
                pass
    
    def populate_casino_data_small_batches(self, documents):
        batch_size = 100
        total_docs = len(documents)
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = documents[i:batch_end]
            texts = [doc["text"] for doc in batch_docs]
            metadatas = [doc["metadata"] for doc in batch_docs]
            ids = [f"casino_{j}" for j in range(i, batch_end)]
            embeddings = self.embedding_model.encode(texts).tolist()
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
    
    def window_pd(self, moves: List[str], window_size: int = 4) -> str:
        if not moves:
            return 'C' * window_size
        padded_moves = (moves + ['C'] * window_size)[-window_size:]
        return ''.join(padded_moves)
    
    def create_casino_query(self, strategy: str, pd_seq: str, pref_gap: float = 0.0) -> str:
        query_parts = []
        if strategy:
            query_parts.append(f"strategy: {strategy}")
        if pd_seq:
            query_parts.append(f"power dynamics: {pd_seq}")
        if pref_gap > 0:
            query_parts.append(f"preference gap: {pref_gap}")
        return " ".join(query_parts) if query_parts else "negotiation conversation"
    
    def retrieve_casino_context(self, query: str, k: int = 4) -> str:
        try:
            if not self.collection or not self.embedding_model:
                return ""
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            if not results['documents'] or not results['documents'][0]:
                return ""
            context_parts = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                strategy = metadata.get('strategy', 'unknown')
                pd = metadata.get('pd', 'C')
                personality = metadata.get('personality_pair', 'unknown')
                context_parts.append(
                    f"Example {i+1} (Strategy: {strategy}, PD: {pd}, Personality: {personality}):\n{doc}\n"
                )
            return "\n".join(context_parts)
        except Exception:
            return ""
    
    def get_casino_context_for_negotiation(self, 
                                         strategy: str, 
                                         recent_moves: List[str], 
                                         pref_gap: float = 0.0) -> str:
        pd_seq = self.window_pd(recent_moves, 4)
        query = self.create_casino_query(strategy, pd_seq, pref_gap)
        context = self.retrieve_casino_context(query, k=3)
        if context:
            return f"casino Negotiation Examples:\n{context}"
        else:
            return "No relevant casino examples found."

casino_rag = None

def initialize_casino_rag():
    global casino_rag
    if casino_rag is None:
        casino_rag = CasinoRAG()
    return casino_rag

def get_casino_context(strategy: str, recent_moves: List[str], pref_gap: float = 0.0) -> str:
    global casino_rag
    if casino_rag is None:
        initialize_casino_rag()
    return casino_rag.get_casino_context_for_negotiation(strategy, recent_moves, pref_gap)

def preload_casino_rag():
    try:
        initialize_casino_rag()
        return True
    except Exception:
        return False