"""
Async wrappers for blocking I/O operations.

This module provides async wrappers using thread pools for CPU-bound and I/O-bound
operations that don't have native async support. This is part of Phase 3 performance
optimization to enable concurrent request handling.

Architecture:
- Uses ThreadPoolExecutor for running blocking operations
- Integrates with asyncio event loop via run_in_executor
- Preserves existing sync function signatures and behavior
- Enables parallel execution of independent operations via asyncio.gather()
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound and blocking I/O operations
# Size = 4 workers balances concurrency vs memory usage for single-server deployment
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_helper")


async def run_in_thread(func, *args, **kwargs):
    """
    Run a blocking function in the thread pool.

    Args:
        func: Blocking function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result of func(*args, **kwargs)

    Usage:
        result = await run_in_thread(blocking_function, arg1, arg2, key=value)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))


# ============================================================================
# Neo4j Graph Database Operations
# ============================================================================

async def fetch_last_n_async(conv_id: str, n: int) -> List[Dict[str, Any]]:
    """
    Async wrapper for fetching last N conversation turns from Neo4j.

    Args:
        conv_id: Conversation ID
        n: Number of recent turns to fetch

    Returns:
        List of turn dictionaries with keys: speaker, text, move, pd, timestamp
    """
    from negotiation_chatbot.graph import fetch_last_n
    logger.debug(f"Async fetching last {n} turns for conversation {conv_id}")
    return await run_in_thread(fetch_last_n, conv_id, n)


async def upsert_turn_async(
    conv_id: str,
    speaker: str,
    text: str,
    move: Optional[str] = None,
    pd: Optional[str] = None
) -> None:
    """
    Async wrapper for upserting a conversation turn to Neo4j.

    Args:
        conv_id: Conversation ID
        speaker: Speaker name
        text: Message text
        move: Negotiation move classification
        pd: Power dynamic classification
    """
    from negotiation_chatbot.graph import upsert_turn
    logger.debug(f"Async upserting turn for {speaker} in conversation {conv_id}")
    await run_in_thread(upsert_turn, conv_id, speaker, text, move, pd)


async def upsert_coalition_async(conv_id: str, coalition_id: str, member_names: List[str]) -> None:
    """Async wrapper for upserting a coalition to Neo4j."""
    from negotiation_chatbot.graph import upsert_coalition
    logger.debug(f"Async upserting coalition {coalition_id} for conversation {conv_id}")
    await run_in_thread(upsert_coalition, conv_id, coalition_id, member_names)


async def estimate_n_party_preferences_async(turn_texts: List[str], n_parties: int) -> List[List[float]]:
    """Async wrapper for N-party preference estimation."""
    from negotiation_chatbot.n_party_preference import estimate_n_party_preferences
    logger.debug(f"Async estimating {n_parties}-party preferences for {len(turn_texts)} turns")
    return await run_in_thread(estimate_n_party_preferences, turn_texts, n_parties)


async def check_deal_reached_async(conv_id: str) -> bool:
    """
    Async wrapper for checking if a deal has been reached.

    Args:
        conv_id: Conversation ID

    Returns:
        True if deal reached, False otherwise
    """
    from negotiation_chatbot.coach import check_deal_reached
    return await run_in_thread(check_deal_reached, conv_id)


# ============================================================================
# Machine Learning - Preference Estimation
# ============================================================================

async def estimate_preferences_async(turn_texts: List[str]) -> Tuple[List[float], List[float]]:
    """
    Async wrapper for DistilBERT-based preference estimation.

    Args:
        turn_texts: List of conversation turn texts

    Returns:
        Tuple of (my_preferences, opponent_preferences) as weight lists
    """
    from negotiation_chatbot.coach import estimate_preferences_cached
    logger.debug(f"Async estimating preferences for {len(turn_texts)} turns")
    # Use cached version to benefit from TTL cache
    turn_texts_tuple = tuple(turn_texts)
    return await run_in_thread(estimate_preferences_cached, turn_texts_tuple)


# ============================================================================
# LLM-Based Analysis Functions
# ============================================================================

async def analyze_item_priorities_async(
    turns: List[Dict[str, Any]],
    model: str,
    item_names: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Async wrapper for LLM-based item priority analysis.

    Args:
        turns: List of conversation turns
        model: LLM model name
        item_names: Optional custom item names mapping

    Returns:
        Dictionary with priority analysis results
    """
    from negotiation_chatbot.coach import analyze_item_priorities_cached
    import json

    logger.debug(f"Async analyzing item priorities with model {model}")
    turns_json = json.dumps(turns, default=str)
    item_names_json = json.dumps(item_names) if item_names else None
    return await run_in_thread(analyze_item_priorities_cached, turns_json, model, item_names_json)


async def extract_current_offers_async(
    turns: List[Dict[str, Any]],
    model: str,
    item_names: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Async wrapper for LLM-based current offers extraction.

    Args:
        turns: List of conversation turns
        model: LLM model name
        item_names: Optional custom item names mapping

    Returns:
        Dictionary with extracted offers
    """
    from negotiation_chatbot.coach import extract_current_offers_cached
    import json

    logger.debug(f"Async extracting current offers with model {model}")
    turns_json = json.dumps(turns, default=str)
    item_names_json = json.dumps(item_names) if item_names else None
    return await run_in_thread(extract_current_offers_cached, turns_json, model, item_names_json)


async def retrieve_rag_context_async(hint: str, turns: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Async wrapper for RAG (Retrieval-Augmented Generation) context retrieval.

    Args:
        hint: Search hint/query
        turns: List of conversation turns

    Returns:
        Tuple of (rag_context, rag_source)
    """
    from negotiation_chatbot.coach import retrieve_rag_context_cached
    import json

    logger.debug(f"Async retrieving RAG context for hint: {hint[:50]}...")
    turns_json = json.dumps(turns, default=str)
    return await run_in_thread(retrieve_rag_context_cached, hint, turns_json)


async def llm_generate_reply_async(
    hint: str,
    summary: str,
    rag_context: str,
    last_turn: Dict[str, Any],
    model: str,
    repeated: bool = False
) -> str:
    """
    Async wrapper for LLM-based reply generation.

    Args:
        hint: Strategic hint/advice
        summary: Conversation summary
        rag_context: Retrieved context from RAG
        last_turn: Last conversation turn
        model: LLM model name
        repeated: Whether this is a repeated generation attempt

    Returns:
        Generated reply text
    """
    from negotiation_chatbot.coach import _llm_generate_reply

    logger.debug(f"Async generating LLM reply with model {model}")
    return await run_in_thread(
        _llm_generate_reply,
        hint,
        summary,
        rag_context,
        last_turn,
        repeated=repeated,
        model=model
    )


async def llm_closure_reply_async(advice_text: str, model: str) -> str:
    """
    Async wrapper for LLM-based closure reply generation.

    Args:
        advice_text: Advice text for deal closure
        model: LLM model name

    Returns:
        Generated closure reply
    """
    from negotiation_chatbot.coach import _llm_closure_reply

    logger.debug(f"Async generating closure reply with model {model}")
    return await run_in_thread(_llm_closure_reply, advice_text, model)


# ============================================================================
# Utility Functions
# ============================================================================

async def parallel_gather(*awaitables):
    """
    Execute multiple async operations in parallel and return all results.

    This is a convenience wrapper around asyncio.gather() that logs execution.

    Args:
        *awaitables: Async functions/coroutines to execute in parallel

    Returns:
        List of results in the same order as input awaitables

    Usage:
        results = await parallel_gather(
            fetch_last_n_async(conv_id, 5),
            estimate_preferences_async(texts),
            analyze_item_priorities_async(turns, model)
        )
        turns, preferences, priorities = results
    """
    logger.debug(f"Executing {len(awaitables)} operations in parallel")
    start_time = asyncio.get_event_loop().time()

    results = await asyncio.gather(*awaitables, return_exceptions=False)

    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"Parallel execution of {len(awaitables)} operations completed in {elapsed:.2f}s")

    return results


def cleanup_thread_pool():
    """
    Shutdown the thread pool gracefully.

    Call this during application shutdown to ensure all threads complete.
    """
    logger.info("Shutting down async helper thread pool")
    _thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")
