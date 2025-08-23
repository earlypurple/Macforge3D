"""
Progressive Web App (PWA) Manager for MacForge3D
Enables offline functionality, push notifications, and native-like experience
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
import hashlib

logger = logging.getLogger(__name__)

class ServiceWorkerManager:
    """Manages service worker functionality for offline capabilities."""
    
    def __init__(self):
        self.cache_version = "macforge3d-v1.0.0"
        self.cached_resources = [
            "/",
            "/index.html",
            "/manifest.json",
            "/css/main.css",
            "/js/app.js",
            "/js/three.min.js",
            "/js/webxr-polyfill.js",
            "/icons/icon-192x192.png",
            "/icons/icon-512x512.png",
            "/offline.html"
        ]
        self.dynamic_cache = "macforge3d-dynamic"
        self.strategies = {
            "api": "network_first",
            "models": "cache_first",
            "textures": "cache_first",
            "static": "cache_first"
        }
    
    def generate_service_worker_code(self) -> str:
        """Generate service worker JavaScript code."""
        return f"""
// MacForge3D Service Worker - Generated {datetime.now().isoformat()}
const CACHE_VERSION = '{self.cache_version}';
const STATIC_CACHE = 'macforge3d-static-' + CACHE_VERSION;
const DYNAMIC_CACHE = '{self.dynamic_cache}';

const STATIC_FILES = {json.dumps(self.cached_resources)};

// Install event - cache static resources
self.addEventListener('install', event => {{
    console.log('🔧 Service Worker installing...');
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => {{
                console.log('📦 Caching static resources');
                return cache.addAll(STATIC_FILES);
            }})
            .then(() => self.skipWaiting())
    );
}});

// Activate event - clean up old caches
self.addEventListener('activate', event => {{
    console.log('✅ Service Worker activating...');
    event.waitUntil(
        caches.keys().then(cacheNames => {{
            return Promise.all(
                cacheNames.map(cacheName => {{
                    if (cacheName.startsWith('macforge3d-static-') && cacheName !== STATIC_CACHE) {{
                        console.log('🗑️ Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }}
                }})
            );
        }}).then(() => self.clients.claim())
    );
}});

// Fetch event - implement caching strategies
self.addEventListener('fetch', event => {{
    const url = new URL(event.request.url);
    
    // Skip non-GET requests
    if (event.request.method !== 'GET') {{
        return;
    }}
    
    // API requests - Network First strategy
    if (url.pathname.startsWith('/api/')) {{
        event.respondWith(networkFirst(event.request));
    }}
    // 3D Models and textures - Cache First strategy
    else if (url.pathname.includes('/models/') || url.pathname.includes('/textures/')) {{
        event.respondWith(cacheFirst(event.request));
    }}
    // Static resources - Cache First strategy
    else if (STATIC_FILES.includes(url.pathname)) {{
        event.respondWith(cacheFirst(event.request));
    }}
    // Everything else - Network First with fallback
    else {{
        event.respondWith(networkFirst(event.request));
    }}
}});

// Network First strategy
async function networkFirst(request) {{
    try {{
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {{
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }}
        
        return networkResponse;
    }} catch (error) {{
        console.log('🌐 Network failed, trying cache:', request.url);
        const cachedResponse = await caches.match(request);
        
        if (cachedResponse) {{
            return cachedResponse;
        }}
        
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {{
            return caches.match('/offline.html');
        }}
        
        throw error;
    }}
}}

// Cache First strategy
async function cacheFirst(request) {{
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {{
        // Update cache in background
        fetch(request).then(response => {{
            if (response.ok) {{
                const cache = caches.open(DYNAMIC_CACHE);
                cache.then(c => c.put(request, response));
            }}
        }}).catch(() => {{}});
        
        return cachedResponse;
    }}
    
    try {{
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {{
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }}
        
        return networkResponse;
    }} catch (error) {{
        console.error('❌ Failed to fetch:', request.url);
        throw error;
    }}
}}

// Background sync for model uploads
self.addEventListener('sync', event => {{
    if (event.tag === 'upload-model') {{
        console.log('🔄 Background sync: uploading models');
        event.waitUntil(uploadPendingModels());
    }}
}});

// Push notifications
self.addEventListener('push', event => {{
    console.log('📬 Push notification received');
    
    const options = {{
        body: event.data ? event.data.text() : 'New update available!',
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        tag: 'macforge3d-notification',
        actions: [
            {{
                action: 'open',
                title: 'Open App',
                icon: '/icons/action-open.png'
            }},
            {{
                action: 'dismiss',
                title: 'Dismiss',
                icon: '/icons/action-dismiss.png'
            }}
        ],
        data: {{
            url: '/',
            timestamp: Date.now()
        }}
    }};
    
    event.waitUntil(
        self.registration.showNotification('MacForge3D', options)
    );
}});

// Notification click handling
self.addEventListener('notificationclick', event => {{
    console.log('🔔 Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'open' || !event.action) {{
        event.waitUntil(
            clients.openWindow(event.notification.data.url || '/')
        );
    }}
}});

// Upload pending models function
async function uploadPendingModels() {{
    const db = await openDB();
    const tx = db.transaction('pending_uploads', 'readonly');
    const store = tx.objectStore('pending_uploads');
    const pendingUploads = await store.getAll();
    
    for (const upload of pendingUploads) {{
        try {{
            await fetch('/api/models/upload', {{
                method: 'POST',
                body: upload.data
            }});
            
            // Remove from pending uploads
            const deleteTx = db.transaction('pending_uploads', 'readwrite');
            const deleteStore = deleteTx.objectStore('pending_uploads');
            await deleteStore.delete(upload.id);
            
            console.log('✅ Model uploaded:', upload.name);
        }} catch (error) {{
            console.error('❌ Failed to upload model:', upload.name, error);
        }}
    }}
}}

// IndexedDB helper
function openDB() {{
    return new Promise((resolve, reject) => {{
        const request = indexedDB.open('MacForge3DDB', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = event => {{
            const db = event.target.result;
            
            if (!db.objectStoreNames.contains('pending_uploads')) {{
                const store = db.createObjectStore('pending_uploads', {{ keyPath: 'id' }});
                store.createIndex('timestamp', 'timestamp');
            }}
            
            if (!db.objectStoreNames.contains('cached_models')) {{
                const store = db.createObjectStore('cached_models', {{ keyPath: 'id' }});
                store.createIndex('lastAccessed', 'lastAccessed');
            }}
        }};
    }});
}}

console.log('🚀 MacForge3D Service Worker loaded');
"""

class PWAManifestManager:
    """Manages PWA manifest generation and updates."""
    
    def __init__(self):
        self.app_info = {
            "name": "MacForge3D",
            "short_name": "MacForge3D",
            "description": "Professional 3D modeling and AI-powered creation platform",
            "version": "1.0.0",
            "author": "MacForge3D Team"
        }
        
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate PWA manifest.json."""
        return {
            "name": self.app_info["name"],
            "short_name": self.app_info["short_name"],
            "description": self.app_info["description"],
            "start_url": "/",
            "display": "standalone",
            "orientation": "landscape",
            "theme_color": "#1a1a1a",
            "background_color": "#000000",
            "scope": "/",
            "lang": "en",
            "dir": "ltr",
            "categories": ["productivity", "graphics", "3d", "design"],
            "icons": [
                {
                    "src": "/icons/icon-72x72.png",
                    "sizes": "72x72",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-96x96.png",
                    "sizes": "96x96",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-128x128.png",
                    "sizes": "128x128",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-144x144.png",
                    "sizes": "144x144",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-152x152.png",
                    "sizes": "152x152",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/icons/icon-384x384.png",
                    "sizes": "384x384",
                    "type": "image/png"
                },
                {
                    "src": "/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable"
                }
            ],
            "screenshots": [
                {
                    "src": "/screenshots/desktop-1.png",
                    "sizes": "1280x720",
                    "type": "image/png",
                    "form_factor": "wide",
                    "label": "3D Modeling Interface"
                },
                {
                    "src": "/screenshots/mobile-1.png",
                    "sizes": "360x640",
                    "type": "image/png",
                    "form_factor": "narrow",
                    "label": "Mobile 3D Viewer"
                }
            ],
            "shortcuts": [
                {
                    "name": "New Model",
                    "short_name": "New",
                    "description": "Create a new 3D model",
                    "url": "/new",
                    "icons": [
                        {
                            "src": "/icons/shortcut-new.png",
                            "sizes": "192x192"
                        }
                    ]
                },
                {
                    "name": "AI Generator",
                    "short_name": "AI",
                    "description": "Generate 3D models with AI",
                    "url": "/ai-generator",
                    "icons": [
                        {
                            "src": "/icons/shortcut-ai.png",
                            "sizes": "192x192"
                        }
                    ]
                },
                {
                    "name": "VR Mode",
                    "short_name": "VR",
                    "description": "Enter VR modeling mode",
                    "url": "/vr",
                    "icons": [
                        {
                            "src": "/icons/shortcut-vr.png",
                            "sizes": "192x192"
                        }
                    ]
                }
            ],
            "share_target": {
                "action": "/share",
                "method": "POST",
                "enctype": "multipart/form-data",
                "params": {
                    "title": "title",
                    "text": "text",
                    "url": "url",
                    "files": [
                        {
                            "name": "model_files",
                            "accept": [".stl", ".obj", ".ply", ".gltf", ".fbx"]
                        }
                    ]
                }
            },
            "file_handlers": [
                {
                    "action": "/open-model",
                    "accept": {
                        "model/stl": [".stl"],
                        "model/obj": [".obj"],
                        "model/ply": [".ply"],
                        "model/gltf+json": [".gltf"],
                        "model/gltf-binary": [".glb"]
                    }
                }
            ],
            "protocol_handlers": [
                {
                    "protocol": "web+macforge3d",
                    "url": "/handle-protocol?url=%s"
                }
            ],
            "edge_side_panel": {
                "preferred_width": 400
            },
            "launch_handler": {
                "client_mode": "focus-existing"
            }
        }

class PushNotificationManager:
    """Manages push notifications for PWA."""
    
    def __init__(self):
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.notification_types = {
            "model_ready": "Your 3D model is ready for download",
            "collaboration_invite": "You've been invited to collaborate on a project",
            "ai_generation_complete": "AI model generation completed",
            "render_complete": "Your render job is finished",
            "update_available": "New features available - update now!"
        }
    
    async def subscribe_user(self, user_id: str, subscription_data: Dict[str, Any]) -> bool:
        """Subscribe user to push notifications."""
        try:
            self.subscriptions[user_id] = {
                "endpoint": subscription_data["endpoint"],
                "keys": subscription_data["keys"],
                "subscribed_at": datetime.now().isoformat(),
                "active": True
            }
            
            logger.info(f"📱 User subscribed to push notifications: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe user {user_id}: {e}")
            return False
    
    async def send_notification(self, user_id: str, notification_type: str, custom_message: Optional[str] = None) -> bool:
        """Send push notification to user."""
        if user_id not in self.subscriptions:
            return False
        
        try:
            subscription = self.subscriptions[user_id]
            if not subscription["active"]:
                return False
            
            message = custom_message or self.notification_types.get(notification_type, "New update!")
            
            # In a real implementation, you would use a service like Firebase Cloud Messaging
            # or web-push library to send actual push notifications
            
            notification_data = {
                "title": "MacForge3D",
                "body": message,
                "icon": "/icons/icon-192x192.png",
                "badge": "/icons/badge-72x72.png",
                "tag": notification_type,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "type": notification_type,
                    "url": "/"
                }
            }
            
            logger.info(f"📬 Push notification sent to {user_id}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification to {user_id}: {e}")
            return False
    
    async def unsubscribe_user(self, user_id: str) -> bool:
        """Unsubscribe user from push notifications."""
        if user_id in self.subscriptions:
            self.subscriptions[user_id]["active"] = False
            logger.info(f"🔕 User unsubscribed from notifications: {user_id}")
            return True
        return False

class OfflineManager:
    """Manages offline functionality and data synchronization."""
    
    def __init__(self):
        self.offline_storage: Dict[str, Any] = {}
        self.pending_operations: List[Dict[str, Any]] = []
        self.sync_strategies = {
            "models": "bidirectional",
            "projects": "bidirectional", 
            "settings": "client_wins",
            "cache": "server_wins"
        }
    
    async def store_offline_data(self, key: str, data: Any, strategy: str = "bidirectional") -> bool:
        """Store data for offline access."""
        try:
            self.offline_storage[key] = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "dirty": False
            }
            
            logger.info(f"💾 Data stored offline: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store offline data: {e}")
            return False
    
    async def get_offline_data(self, key: str) -> Optional[Any]:
        """Retrieve offline data."""
        if key in self.offline_storage:
            self.offline_storage[key]["last_accessed"] = datetime.now().isoformat()
            return self.offline_storage[key]["data"]
        return None
    
    async def queue_operation(self, operation: Dict[str, Any]) -> str:
        """Queue operation for when connection is restored."""
        operation_id = str(uuid.uuid4())
        operation["id"] = operation_id
        operation["queued_at"] = datetime.now().isoformat()
        operation["status"] = "pending"
        
        self.pending_operations.append(operation)
        logger.info(f"⏳ Operation queued for sync: {operation['type']}")
        return operation_id
    
    async def sync_pending_operations(self) -> int:
        """Sync all pending operations when online."""
        synced_count = 0
        
        for operation in self.pending_operations:
            if operation["status"] != "pending":
                continue
            
            try:
                # Simulate API call
                success = await self._execute_operation(operation)
                
                if success:
                    operation["status"] = "completed"
                    operation["synced_at"] = datetime.now().isoformat()
                    synced_count += 1
                    logger.info(f"✅ Operation synced: {operation['type']}")
                else:
                    operation["status"] = "failed"
                    logger.error(f"❌ Operation sync failed: {operation['type']}")
                    
            except Exception as e:
                operation["status"] = "failed"
                operation["error"] = str(e)
                logger.error(f"Operation sync error: {e}")
        
        # Remove completed operations
        self.pending_operations = [op for op in self.pending_operations if op["status"] == "pending"]
        
        return synced_count
    
    async def _execute_operation(self, operation: Dict[str, Any]) -> bool:
        """Execute a pending operation."""
        # Simulate operation execution
        await asyncio.sleep(0.1)
        return True  # Simulate success
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get offline storage statistics."""
        total_items = len(self.offline_storage)
        total_size = sum(len(str(item["data"])) for item in self.offline_storage.values())
        pending_ops = len(self.pending_operations)
        
        return {
            "total_items": total_items,
            "total_size_bytes": total_size,
            "pending_operations": pending_ops,
            "storage_used_mb": total_size / (1024 * 1024),
            "last_sync": datetime.now().isoformat()
        }

class PWAManager:
    """Main PWA manager coordinating all PWA functionality."""
    
    def __init__(self):
        self.service_worker = ServiceWorkerManager()
        self.manifest = PWAManifestManager()
        self.notifications = PushNotificationManager()
        self.offline = OfflineManager()
        self.install_events: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize PWA functionality."""
        try:
            logger.info("🚀 Initializing PWA Manager")
            
            # Generate and cache service worker
            sw_code = self.service_worker.generate_service_worker_code()
            await self.offline.store_offline_data("service_worker_code", sw_code)
            
            # Generate and cache manifest
            manifest_data = self.manifest.generate_manifest()
            await self.offline.store_offline_data("manifest", manifest_data)
            
            logger.info("✅ PWA Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PWA Manager: {e}")
            return False
    
    async def handle_install_event(self, user_id: str, user_agent: str) -> str:
        """Handle PWA installation event."""
        install_id = str(uuid.uuid4())
        
        install_event = {
            "id": install_id,
            "user_id": user_id,
            "user_agent": user_agent,
            "installed_at": datetime.now().isoformat(),
            "platform": self._detect_platform(user_agent)
        }
        
        self.install_events.append(install_event)
        
        # Send welcome notification
        await self.notifications.send_notification(
            user_id,
            "update_available",
            "Welcome to MacForge3D! Your app is now installed and ready to use offline."
        )
        
        logger.info(f"📱 PWA installed by user: {user_id}")
        return install_id
    
    def _detect_platform(self, user_agent: str) -> str:
        """Detect platform from user agent."""
        user_agent = user_agent.lower()
        
        if "mobile" in user_agent or "android" in user_agent:
            return "mobile"
        elif "ipad" in user_agent or "tablet" in user_agent:
            return "tablet"
        elif "mac" in user_agent:
            return "desktop_mac"
        elif "windows" in user_agent:
            return "desktop_windows"
        elif "linux" in user_agent:
            return "desktop_linux"
        else:
            return "unknown"
    
    async def get_pwa_stats(self) -> Dict[str, Any]:
        """Get PWA usage statistics."""
        total_installs = len(self.install_events)
        active_subscriptions = len([s for s in self.notifications.subscriptions.values() if s["active"]])
        storage_stats = await self.offline.get_storage_stats()
        
        platform_breakdown = {}
        for event in self.install_events:
            platform = event["platform"]
            platform_breakdown[platform] = platform_breakdown.get(platform, 0) + 1
        
        return {
            "total_installs": total_installs,
            "active_push_subscriptions": active_subscriptions,
            "platform_breakdown": platform_breakdown,
            "offline_storage": storage_stats,
            "service_worker_version": self.service_worker.cache_version,
            "last_updated": datetime.now().isoformat()
        }

# Global PWA manager instance
pwa_manager = PWAManager()

async def initialize_pwa():
    """Initialize PWA functionality."""
    return await pwa_manager.initialize()

async def get_service_worker_code() -> str:
    """Get service worker JavaScript code."""
    return pwa_manager.service_worker.generate_service_worker_code()

async def get_manifest() -> Dict[str, Any]:
    """Get PWA manifest."""
    return pwa_manager.manifest.generate_manifest()

if __name__ == "__main__":
    # Test PWA functionality
    async def test_pwa():
        # Initialize PWA
        success = await initialize_pwa()
        print(f"PWA initialized: {success}")
        
        # Test service worker generation
        sw_code = await get_service_worker_code()
        print(f"Service worker generated: {len(sw_code)} characters")
        
        # Test manifest generation
        manifest = await get_manifest()
        print(f"Manifest generated: {json.dumps(manifest, indent=2)}")
        
        # Test push notifications
        user_id = "test_user_123"
        subscription_data = {
            "endpoint": "https://fcm.googleapis.com/fcm/send/test",
            "keys": {"auth": "test_auth", "p256dh": "test_key"}
        }
        
        await pwa_manager.notifications.subscribe_user(user_id, subscription_data)
        await pwa_manager.notifications.send_notification(user_id, "ai_generation_complete")
        
        # Test offline storage
        await pwa_manager.offline.store_offline_data("test_model", {"vertices": 1000, "faces": 2000})
        stored_data = await pwa_manager.offline.get_offline_data("test_model")
        print(f"Offline data test: {stored_data}")
        
        # Test install event
        install_id = await pwa_manager.handle_install_event(user_id, "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)")
        print(f"Install event: {install_id}")
        
        # Get PWA stats
        stats = await pwa_manager.get_pwa_stats()
        print(f"PWA stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_pwa())