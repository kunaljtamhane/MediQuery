"""
Person C — Session Deduplication Engine (Weeks 5-6)
Tracks passages already served to a user this session.
Before including a passage, checks cosine similarity > 0.85 against the cache.
If duplicate, it is skipped so the user sees fresh content each turn.
"""
import os
import json
import numpy as np
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", 0.85))
SESSION_TTL = 3600  # 1 hour in seconds

_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _cache_key(session_id: str) -> str:
    return f"dedup:{session_id}"


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def is_duplicate(session_id: str, embedding: list[float]) -> bool:
    """Return True if this passage is too similar to something already served."""
    key = _cache_key(session_id)
    cached_raw = _redis.lrange(key, 0, -1)

    for entry in cached_raw:
        stored_emb = json.loads(entry)
        if _cosine(embedding, stored_emb) >= DEDUP_THRESHOLD:
            return True
    return False


def mark_served(session_id: str, embedding: list[float]):
    """Record that this passage's embedding has been served in this session."""
    key = _cache_key(session_id)
    _redis.rpush(key, json.dumps(embedding))
    _redis.expire(key, SESSION_TTL)


def filter_duplicates(passages: list[dict], session_id: str) -> list[dict]:
    """
    Filter out passages already seen this session.
    passages must include an 'embedding' field.
    Call mark_served() for each passage you actually return.
    """
    fresh = []
    for p in passages:
        emb = p.get("embedding")
        if emb is None:
            fresh.append(p)  # no embedding available, include anyway
            continue
        if not is_duplicate(session_id, emb):
            fresh.append(p)
            mark_served(session_id, emb)
    return fresh
