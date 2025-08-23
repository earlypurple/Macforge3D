"""
Advanced Caching System with Redis Integration
Provides intelligent caching for AI models, 3D assets, and computation results
"""

import asyncio
import json
import hashlib
import pickle
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import uuid
import time

logger = logging.getLogger(__name__)

class SmartCache:
    """Advanced caching system with multiple storage backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'redis_url': 'redis://localhost:6379',
            'memory_cache_size': 1000,
            'disk_cache_size': 10000,
            'ttl_default': 3600,  # 1 hour
            'compression': True,
            'ai_model_ttl': 86400,  # 24 hours for AI models
            'mesh_ttl': 7200,  # 2 hours for meshes
            'texture_ttl': 1800  # 30 minutes for textures
        }
        
        # Multiple cache layers
        self.memory_cache: Dict[str, Tuple[Any, datetime, Dict[str, Any]]] = {}
        self.redis_client = None
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Cache policies
        self.policies = {
            'ai_models': {'ttl': self.config['ai_model_ttl'], 'priority': 'high'},
            'meshes': {'ttl': self.config['mesh_ttl'], 'priority': 'medium'},
            'textures': {'ttl': self.config['texture_ttl'], 'priority': 'low'},
            'computations': {'ttl': self.config['ttl_default'], 'priority': 'medium'}
        }
        
    async def initialize(self):
        """Initialize cache backends."""
        try:
            # Try to import and initialize Redis
            try:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(
                    self.config['redis_url'],
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("✅ Redis cache backend initialized")
            except ImportError:
                logger.warning("Redis not available, using memory cache only")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache only")
                self.redis_client = None
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            return False
    
    def _generate_key(self, namespace: str, key: Union[str, Dict[str, Any]]) -> str:
        """Generate a cache key with namespace."""
        if isinstance(key, dict):
            # Create deterministic key from dict
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)
        
        # Hash long keys
        if len(key_str) > 200:
            key_str = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"macforge3d:{namespace}:{key_str}"
    
    def _get_ttl(self, namespace: str) -> int:
        """Get TTL for a namespace."""
        policy = self.policies.get(namespace, {})
        return policy.get('ttl', self.config['ttl_default'])
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.config['compression']:
            return pickle.dumps(data)
        
        try:
            import lz4.frame
            return lz4.frame.compress(pickle.dumps(data))
        except ImportError:
            import gzip
            return gzip.compress(pickle.dumps(data))
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data from storage."""
        if not self.config['compression']:
            return pickle.loads(data)
        
        try:
            import lz4.frame
            return pickle.loads(lz4.frame.decompress(data))
        except ImportError:
            import gzip
            return pickle.loads(gzip.decompress(data))
    
    async def get(self, namespace: str, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._generate_key(namespace, key)
        self.cache_stats['total_requests'] += 1
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            value, expiry, metadata = self.memory_cache[cache_key]
            if datetime.now() < expiry:
                self.cache_stats['hits'] += 1
                metadata['last_accessed'] = datetime.now().isoformat()
                return value
            else:
                # Expired, remove from memory cache
                del self.memory_cache[cache_key]
        
        # Try Redis cache
        if self.redis_client:
            try:
                data = await self.redis_client.get(cache_key)
                if data:
                    value = self._decompress_data(data)
                    
                    # Store in memory cache for faster access
                    expiry = datetime.now() + timedelta(seconds=self._get_ttl(namespace))
                    metadata = {
                        'cached_at': datetime.now().isoformat(),
                        'last_accessed': datetime.now().isoformat(),
                        'source': 'redis'
                    }
                    self._store_in_memory(cache_key, value, expiry, metadata)
                    
                    self.cache_stats['hits'] += 1
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, namespace: str, key: Union[str, Dict[str, Any]], value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        cache_key = self._generate_key(namespace, key)
        ttl = ttl or self._get_ttl(namespace)
        
        try:
            # Store in memory cache
            expiry = datetime.now() + timedelta(seconds=ttl)
            metadata = {
                'cached_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'namespace': namespace,
                'ttl': ttl
            }
            self._store_in_memory(cache_key, value, expiry, metadata)
            
            # Store in Redis cache
            if self.redis_client:
                try:
                    compressed_data = self._compress_data(value)
                    await self.redis_client.setex(cache_key, ttl, compressed_data)
                except Exception as e:
                    logger.warning(f"Redis set error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def _store_in_memory(self, key: str, value: Any, expiry: datetime, metadata: Dict[str, Any]):
        """Store value in memory cache with LRU eviction."""
        # Check memory cache size
        if len(self.memory_cache) >= self.config['memory_cache_size']:
            self._evict_lru()
        
        self.memory_cache[key] = (value, expiry, metadata)
    
    def _evict_lru(self):
        """Evict least recently used items from memory cache."""
        if not self.memory_cache:
            return
        
        # Find oldest last_accessed item
        oldest_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k][2].get('last_accessed', '1970-01-01')
        )
        
        del self.memory_cache[oldest_key]
        self.cache_stats['evictions'] += 1
    
    async def delete(self, namespace: str, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete value from cache."""
        cache_key = self._generate_key(namespace, key)
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from Redis cache
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        return True
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace."""
        pattern = f"macforge3d:{namespace}:*"
        count = 0
        
        # Clear from memory cache
        keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"macforge3d:{namespace}:")]
        for key in keys_to_remove:
            del self.memory_cache[key]
            count += 1
        
        # Clear from Redis cache
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    count += len(keys)
            except Exception as e:
                logger.warning(f"Redis clear namespace error: {e}")
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_size = len(self.memory_cache)
        redis_info = {}
        
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info('memory')
            except Exception as e:
                logger.warning(f"Redis info error: {e}")
        
        hit_rate = 0
        if self.cache_stats['total_requests'] > 0:
            hit_rate = self.cache_stats['hits'] / self.cache_stats['total_requests']
        
        return {
            'hit_rate': hit_rate,
            'memory_cache_size': memory_size,
            'memory_cache_limit': self.config['memory_cache_size'],
            'redis_connected': self.redis_client is not None,
            'redis_memory': redis_info.get('used_memory_human', 'N/A'),
            'stats': self.cache_stats,
            'policies': self.policies
        }
    
    async def preload_ai_models(self, model_configs: List[Dict[str, Any]]):
        """Preload AI model configurations and weights."""
        for config in model_configs:
            try:
                model_key = {
                    'model_name': config['name'],
                    'version': config['version'],
                    'config_hash': hashlib.sha256(str(config).encode()).hexdigest()[:16]
                }
                
                # Check if already cached
                cached_model = await self.get('ai_models', model_key)
                if cached_model:
                    logger.info(f"AI model {config['name']} already cached")
                    continue
                
                # Simulate model loading (in real implementation, load actual model)
                model_data = {
                    'config': config,
                    'weights': f"weights_placeholder_{config['name']}",
                    'loaded_at': datetime.now().isoformat(),
                    'size_mb': config.get('size_mb', 100)
                }
                
                await self.set('ai_models', model_key, model_data, ttl=self.config['ai_model_ttl'])
                logger.info(f"Preloaded AI model: {config['name']}")
                
            except Exception as e:
                logger.error(f"Failed to preload model {config.get('name', 'unknown')}: {e}")

class CacheManager:
    """High-level cache manager with intelligent caching strategies."""
    
    def __init__(self):
        self.cache = SmartCache()
        self.cache_warming_tasks = {}
        
    async def initialize(self):
        """Initialize the cache manager."""
        return await self.cache.initialize()
    
    async def cache_model_result(self, model_name: str, input_hash: str, result: Any) -> bool:
        """Cache AI model inference result."""
        cache_key = {
            'model': model_name,
            'input_hash': input_hash,
            'type': 'inference_result'
        }
        return await self.cache.set('ai_models', cache_key, result)
    
    async def get_cached_model_result(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached AI model inference result."""
        cache_key = {
            'model': model_name,
            'input_hash': input_hash,
            'type': 'inference_result'
        }
        return await self.cache.get('ai_models', cache_key)
    
    async def cache_mesh_optimization(self, mesh_hash: str, optimization_params: Dict[str, Any], result: Any) -> bool:
        """Cache mesh optimization result."""
        cache_key = {
            'mesh_hash': mesh_hash,
            'params': optimization_params,
            'type': 'optimization'
        }
        return await self.cache.set('meshes', cache_key, result)
    
    async def cache_texture_generation(self, prompt: str, style_params: Dict[str, Any], texture_data: bytes) -> bool:
        """Cache generated texture."""
        cache_key = {
            'prompt': prompt,
            'style': style_params,
            'type': 'generated_texture'
        }
        return await self.cache.set('textures', cache_key, texture_data)
    
    async def warm_cache(self, namespace: str, items: List[Tuple[Any, Any]]):
        """Warm up cache with frequently used items."""
        task_id = str(uuid.uuid4())
        
        async def warming_task():
            for key, value in items:
                try:
                    await self.cache.set(namespace, key, value)
                    await asyncio.sleep(0.1)  # Don't overwhelm the cache
                except Exception as e:
                    logger.error(f"Cache warming error for {key}: {e}")
        
        self.cache_warming_tasks[task_id] = asyncio.create_task(warming_task())
        return task_id
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        stats = await self.cache.get_stats()
        
        # Add additional metrics
        stats['cache_warming_tasks'] = len(self.cache_warming_tasks)
        stats['namespaces'] = list(self.cache.policies.keys())
        
        return stats

# Global cache manager instance
cache_manager = CacheManager()

async def initialize_cache():
    """Initialize the global cache manager."""
    return await cache_manager.initialize()

async def get_cached_result(namespace: str, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
    """Get cached result."""
    return await cache_manager.cache.get(namespace, key)

async def cache_result(namespace: str, key: Union[str, Dict[str, Any]], value: Any, ttl: Optional[int] = None) -> bool:
    """Cache a result."""
    return await cache_manager.cache.set(namespace, key, value, ttl)

if __name__ == "__main__":
    # Test the caching system
    async def test_cache():
        # Initialize cache
        success = await initialize_cache()
        print(f"Cache initialized: {success}")
        
        # Test basic caching
        await cache_result('test', 'key1', {'data': 'test_value'})
        result = await get_cached_result('test', 'key1')
        print(f"Basic cache test: {result}")
        
        # Test AI model caching
        model_key = {'model': 'text-to-3d', 'input': 'test prompt'}
        await cache_result('ai_models', model_key, {'result': 'generated_mesh'})
        cached_model = await get_cached_result('ai_models', model_key)
        print(f"AI model cache test: {cached_model}")
        
        # Test performance metrics
        metrics = await cache_manager.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Test cache warming
        items_to_warm = [
            ({'prompt': f'test_{i}'}, {'mesh': f'mesh_data_{i}'})
            for i in range(5)
        ]
        task_id = await cache_manager.warm_cache('textures', items_to_warm)
        print(f"Cache warming task started: {task_id}")
        
        # Wait a bit and check stats again
        await asyncio.sleep(2)
        final_stats = await cache_manager.get_performance_metrics()
        print(f"Final stats: {final_stats}")
    
    asyncio.run(test_cache())