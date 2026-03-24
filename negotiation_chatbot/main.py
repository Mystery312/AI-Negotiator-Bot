#python provided web framework:FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from typing import Dict, Any
import os
import uuid
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from .ingest import label_text
from .graph import upsert_turn, create_deal_outcome, mark_turn_as_accepted
from .coach import get_advice, get_advice_async, record_strategy_outcome, _strategy_learner  # Phase 3: Added async version
from .rag import retrieve_rag_context
from .casino_rag import preload_casino_rag, initialize_casino_rag

# Load environment variables from project root
import pathlib
_env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

#  OpenAI-compatible request/response  
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionReq(BaseModel):
    model: str
    messages: list[Message] = Field(..., min_items=1)
    stream: bool | None = False

class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResp(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: list[Choice]
    created: int
    model: str

# ========== existing FastAPI app ==========
app = FastAPI(title="Chat Negotiator", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on app startup"""
    logger.info("Starting Chat Negotiator API...")

    # Skip preloading for faster startup - will lazy-load on first use
    # Preloading takes 8-10 seconds due to loading Sentence Transformer models
    # Set PRELOAD_RAG=true environment variable to enable preloading
    if os.getenv("PRELOAD_RAG", "false").lower() == "true":
        logger.info("Preloading RAG systems (CaSiNo + Generic)...")
        try:
            # Preload CaSiNo RAG
            preload_casino_rag()
            logger.info("✅ CaSiNo RAG system preloaded successfully")

            # Preload Generic RAG
            from negotiation_chatbot.rag import _ensure_initialized
            _ensure_initialized()
            logger.info("✅ Generic RAG system preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload RAG systems: {e}")
            logger.info("Continuing without RAG preload - will lazy-load on first use")
    else:
        logger.info("Skipping RAG preload for fast startup (set PRELOAD_RAG=true to enable)")

    logger.info("Chat Negotiator API startup complete")

class ChatMessage(BaseModel):
    conv_id: str
    speaker: str
    text: str
    model: str | None = "qwen3:latest"  # Add optional model parameter
    provider: str | None = "ollama"  # Add optional provider parameter
    item_names: dict[str, str] | None = None  # Custom item names {"item0": "Engineers", ...}
    item_counts: dict[str, int] | None = None  # Custom item counts {"item0": 3, ...}
    parties: list[dict] | None = None  # N-party: [{"id": "p1", "name": "Alice", "weights": [0.5, 0.3, 0.2]}, ...]
    coalitions: list[dict] | None = None  # [{"id": "c1", "member_ids": ["p1","p2"], "joint_weights": [...]}]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/chat")
async def chat(message: ChatMessage):
    """Process a chat message and return advice"""
    import time
    request_start = time.time()
    logger.info(f"Processing chat message: conv_id={message.conv_id}, speaker={message.speaker}, text={message.text[:50]}..., model={message.model}")

    try:
        # Step 1: Label the text to get move and pd (ASYNC for better performance)
        step1_start = time.time()
        logger.info("Step 1: Labeling text (async)...")
        from negotiation_chatbot.ingest import label_text_async
        labels = await label_text_async(message.text)
        logger.info(f"✅ Step 1 completed in {time.time() - step1_start:.2f}s - Labels: {labels}")
        
        # Ensure we have the required fields
        if "move" not in labels:
            logger.warning(f"Missing 'move' field in labels: {labels}")
            if "move_type" in labels:
                labels["move"] = labels["move_type"]
            else:
                labels["move"] = "info_share"
        
        if "pd" not in labels:
            logger.warning(f"Missing 'pd' field in labels: {labels}")
            labels["pd"] = "C"
        
        logger.info(f"Final labels: {labels}")
        
        # Step 2: Upsert the turn into Neo4j (optional - graceful failure)
        logger.info("Step 2: Upserting turn to Neo4j...")
        try:
            upsert_turn(
                conv_id=message.conv_id,
                speaker=message.speaker,
                text=message.text,
                move=labels["move"],
                pd=labels["pd"]
            )
            logger.info("Turn upserted successfully")
        except Exception as e:
            logger.warning(f"Failed to upsert turn to Neo4j (continuing without graph): {e}")
        
        # Step 3: Get advice from the coach with selected model and provider
        step3_start = time.time()
        logger.info(f"Step 3: Getting advice from coach using model {message.model} and provider {message.provider}...")
        # If provider is specified, use it; otherwise, determine from model name
        if message.provider:
            provider = message.provider
            model_name = message.model
        else:
            # Determine provider from model name
            if message.model and message.model.startswith(("gemini-", "gemini")):
                provider = "gemini"
                model_name = message.model
            else:
                provider = "ollama"
                model_name = message.model

        # Extract item config or use defaults
        item_names = message.item_names or {}
        item_counts = message.item_counts or {"item0": 3, "item1": 2, "item2": 1}

        # Pass custom item names and counts to coach (ASYNC - Phase 3 optimization)
        advice_result = await get_advice_async(
            message.conv_id,
            message.speaker,
            model_name,
            provider=provider,
            item_names=item_names,
            item_counts=item_counts,
            parties=message.parties,
        )
        logger.info(f"✅ Step 3 completed in {time.time() - step3_start:.2f}s")
        
        # Step 4: Log RAG usage in Neo4j (optional)
        try:
            from negotiation_chatbot.graph import upsert_rag_usage
            # Prefer explicit provenance flag from coach
            rag_src = advice_result.get("rag_source", "none")
            rag_used = rag_src in ("casino", "generic")
            upsert_rag_usage(message.conv_id, message.speaker, rag_used)
            logger.info(f"RAG usage logged: {rag_used} (source={rag_src})")
        except Exception as e:
            logger.warning(f"Failed to log RAG usage: {e}")
        
        # Step 5: Return the advice and reply
        total_time = time.time() - request_start
        logger.info(f"✅ REQUEST COMPLETE - Total time: {total_time:.2f}s")
        return {
            "advice": advice_result["advice"],
            "reply": advice_result["reply"],
            "rag_source": advice_result.get("rag_source", "none"),
            "rag_context": advice_result.get("rag_context", "")
        }
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#  new /v1/chat/completions 
@app.post("/v1/chat/completions", response_model=ChatCompletionResp)
async def openai_compat(req: ChatCompletionReq):
    """OpenAI-compatible endpoint for OpenWebUI integration"""
    logger.info(f"OpenAI-compatible endpoint called with model={req.model}")
    logger.info(f"Number of messages: {len(req.messages)}")
    
    try:
        # very first user message drives the logic
        user_msg = next(m for m in req.messages if m.role == "user")
        text = user_msg.content
        logger.info(f"User message extracted: {text[:50]}...")

        # simple session logic: model name ⇒ conv_id, first system msg ⇒ speaker
        conv_id = req.model    # e.g. "demo"
        speaker = "User"
        logger.info(f"Using conv_id={conv_id}, speaker={speaker}")

        # Step 1: Label the text to get move and pd
        logger.info("Step 1: Labeling text for OpenAI endpoint...")
        labels = label_text(text)
        logger.info(f"Text labeled for OpenAI endpoint: {labels}")
        
        # Ensure we have the required fields
        if "move" not in labels:
            logger.warning(f"Missing 'move' field in labels: {labels}")
            if "move_type" in labels:
                labels["move"] = labels["move_type"]
            else:
                labels["move"] = "info_share"
        
        if "pd" not in labels:
            logger.warning(f"Missing 'pd' field in labels: {labels}")
            labels["pd"] = "C"
        
        logger.info(f"Final labels for OpenAI endpoint: {labels}")
        
        # Step 2: Upsert the turn into Neo4j
        logger.info("Step 2: Upserting turn to Neo4j for OpenAI endpoint...")
        upsert_turn(
            conv_id=conv_id,
            speaker=speaker,
            text=text,
            move=labels["move"],
            pd=labels["pd"]
        )
        logger.info("Turn upserted successfully for OpenAI endpoint")
        
        # Step 3: Get advice from the coach
        logger.info("Step 3: Getting advice from coach for OpenAI endpoint...")
        result = get_advice(conv_id, speaker)
        logger.info(f"Advice received for OpenAI endpoint: {result}")

        reply_msg = Message(role="assistant", content=result["reply"])
        response = ChatCompletionResp(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            choices=[Choice(message=reply_msg)],
            created=int(time.time()),
            model=req.model,
        )
        logger.info("OpenAI-compatible response created successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in OpenAI-compatible endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/{conv_id}")
async def get_conversation_graph(conv_id: str):
    """Get conversation graph data for visualization"""
    try:
        from negotiation_chatbot.graph import get_conversation_graph_data
        graph_data = get_conversation_graph_data(conv_id)
        return graph_data
    except Exception as e:
        logger.error(f"Error getting graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{conv_id}")
async def get_conversation_stats(conv_id: str):
    """Get conversation statistics for visualization"""
    try:
        from negotiation_chatbot.graph import get_conversation_stats
        stats = get_conversation_stats(conv_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/casino/cache")
async def get_casino_cache_info():
    """Get CaSiNo cache information"""
    try:
        casino_rag = initialize_casino_rag()
        cache_info = casino_rag.get_cache_info()
        return cache_info
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/casino/cache/clear")
async def clear_casino_cache():
    """Clear CaSiNo cache"""
    try:
        casino_rag = initialize_casino_rag()
        casino_rag.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/casino/reload")
async def reload_casino_rag():
    """Reload CaSiNo RAG system"""
    try:
        # Clear cache and reload
        casino_rag = initialize_casino_rag()
        casino_rag.clear_cache()
        preload_casino_rag()
        return {"message": "CaSiNo RAG system reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading CaSiNo RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class LabelRequest(BaseModel):
    text: str

class DealOutcome(BaseModel):
    conv_id: str
    deal_reached: bool = True
    status: str = "accepted"
    details: str | None = None

@app.post("/label")
async def label_text_endpoint(request: LabelRequest):
    """Label text with move and power dynamics"""
    try:
        from negotiation_chatbot.ingest import label_text
        labels = label_text(request.text)
        return labels
    except Exception as e:
        logger.error(f"Error labeling text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deal/outcome")
async def create_deal_outcome_endpoint(deal: DealOutcome):
    """Create a deal outcome for a conversation"""
    try:
        create_deal_outcome(
            conv_id=deal.conv_id,
            deal_reached=deal.deal_reached,
            status=deal.status,
            details=deal.details
        )
        record_strategy_outcome(deal.conv_id, accepted=deal.deal_reached)
        return {"message": f"Deal outcome created for conversation {deal.conv_id}"}
    except Exception as e:
        logger.error(f"Error creating deal outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deal/mark-accepted/{conv_id}")
async def mark_conversation_accepted(conv_id: str, turn_id: str | None = None):
    """Mark a conversation turn as accepted (deal reached)"""
    try:
        mark_turn_as_accepted(conv_id=conv_id, turn_id=turn_id)
        record_strategy_outcome(conv_id, accepted=True)
        return {"message": f"Turn marked as accepted for conversation {conv_id}"}
    except Exception as e:
        logger.error(f"Error marking turn as accepted: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategy/weights")
async def get_strategy_weights():
    """Return current Thompson Sampling strategy weights for monitoring."""
    return _strategy_learner.get_weights()

class CoalitionCreate(BaseModel):
    conv_id: str
    coalition_id: str
    member_names: list[str]

@app.post("/coalition/create")
async def create_coalition(req: CoalitionCreate):
    """Create a coalition for an N-party negotiation."""
    try:
        from negotiation_chatbot.graph import upsert_coalition
        upsert_coalition(req.conv_id, req.coalition_id, req.member_names)
        return {"message": f"Coalition {req.coalition_id} created", "conv_id": req.conv_id}
    except Exception as e:
        logger.error(f"Error creating coalition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/coalition/{conv_id}")
async def get_coalitions(conv_id: str):
    """Get all coalitions for a conversation."""
    try:
        from negotiation_chatbot.graph import fetch_coalitions
        return fetch_coalitions(conv_id)
    except Exception as e:
        logger.error(f"Error fetching coalitions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Changed default to 8001
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
