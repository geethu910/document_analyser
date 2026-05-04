from redis.asyncio import Redis
import json
import hashlib
import os


redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

async def get_cached_search(query: str, user_id: int):
    """Get cached search results"""
    version = await get_user_version(user_id)
    cache_key = f"search:{user_id}:{version}:{hashlib.sha256(query.lower().encode()).hexdigest()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    return None


async def cache_search(query: str, user_id: int, results: list, ttl: int = 1800):
    """Cache search results for 30 minutes"""
    version = await get_user_version(user_id)
    cache_key = f"search:{user_id}:{version}:{hashlib.sha256(query.lower().encode()).hexdigest()}"
    await redis_client.setex(cache_key, ttl, json.dumps(results))

async def get_user_version(user_id: int) -> int:
    version = await redis_client.get(f"user_version:{user_id}")
    return int(version) if version else 1

async def invalidate_user_cache(user_id: int):
    await redis_client.incr(f"user_version:{user_id}")
