from redis.asyncio import Redis
import json
import hashlib
import os
import time
from document_assistant import logger


redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)


async def get_cached_search(query: str, user_id: int):
    """Get cached search results; returns parsed JSON or None."""
    start = time.perf_counter()
    try:
        version = await get_user_version(user_id)
        cache_key = f"search:{user_id}:{version}:{hashlib.sha256(query.lower().encode()).hexdigest()}"
        cached = await redis_client.get(cache_key)
        if cached:
            payload = json.loads(cached)
            logger.debug("Cache hit: user=%s key=%s matches=%s time=%.0fms", user_id, cache_key, payload.get("match_count") if isinstance(payload, dict) else None, (time.perf_counter() - start) * 1000)
            return payload
        logger.debug("Cache miss: user=%s key=%s time=%.0fms", user_id, cache_key, (time.perf_counter() - start) * 1000)
        return None
    except Exception:
        logger.error("Error accessing cache for user=%s query_hash=%s", user_id, hashlib.sha256(query.lower().encode()).hexdigest(), exc_info=True)
        return None


async def cache_search(query: str, user_id: int, results: list, ttl: int = 1800):
    """Cache search results for 30 minutes"""
    start = time.perf_counter()
    try:
        version = await get_user_version(user_id)
        cache_key = f"search:{user_id}:{version}:{hashlib.sha256(query.lower().encode()).hexdigest()}"
        await redis_client.setex(cache_key, ttl, json.dumps(results))
        logger.debug("Cached search: user=%s key=%s ttl=%s time=%.0fms", user_id, cache_key, ttl, (time.perf_counter() - start) * 1000)
    except Exception:
        logger.error("Failed to cache search for user=%s", user_id, exc_info=True)


async def get_user_version(user_id: int) -> int:
    try:
        version = await redis_client.get(f"user_version:{user_id}")
        v = int(version) if version else 1
        logger.debug("User cache version: user=%s version=%s", user_id, v)
        return v
    except Exception:
        logger.error("Failed to get user version for user=%s, defaulting to 1", user_id, exc_info=True)
        return 1


async def invalidate_user_cache(user_id: int):
    try:
        new_version = await redis_client.incr(f"user_version:{user_id}")
        logger.info("Invalidated cache for user=%s new_version=%s", user_id, new_version)
    except Exception:
        logger.error("Failed to invalidate cache for user=%s", user_id, exc_info=True)
