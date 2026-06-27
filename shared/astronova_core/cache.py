import json
from typing import Any

import redis.asyncio as redis

from astronova_core.config import get_settings

settings = get_settings()

class RedisCache:
    def __init__(self):
        self.client = redis.from_url(settings.redis.redis_url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, expire_seconds: int = 3600) -> None:
        await self.client.set(key, json.dumps(value), ex=expire_seconds)

    async def delete(self, key: str) -> None:
        await self.client.delete(key)
