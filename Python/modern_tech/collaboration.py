"""
Real-time Collaboration System for MacForge3D
Enables real-time collaborative 3D modeling using WebRTC and WebSockets
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
import uuid
import weakref

logger = logging.getLogger(__name__)

class CollaborationEvent:
    """Represents a collaboration event."""
    
    def __init__(self, event_type: str, user_id: str, data: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.user_id = user_id
        self.data = data
        self.timestamp = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'user_id': self.user_id,
            'data': self.data,
            'timestamp': self.timestamp
        }

class CollaborationSession:
    """Manages a real-time collaboration session."""
    
    def __init__(self, session_id: str, project_id: str):
        self.id = session_id
        self.project_id = project_id
        self.participants: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.history: List[CollaborationEvent] = []
        self.cursors: Dict[str, Dict[str, Any]] = {}
        self.selections: Dict[str, List[str]] = {}  # user_id -> selected object ids
        self.locks: Dict[str, str] = {}  # object_id -> user_id
        self.created_at = datetime.now()
        
    def add_participant(self, user_id: str, user_info: Dict[str, Any]):
        """Add a participant to the session."""
        self.participants[user_id] = {
            **user_info,
            'joined_at': datetime.now().isoformat(),
            'active': True,
            'permissions': user_info.get('permissions', ['view', 'edit'])
        }
        
        event = CollaborationEvent('user_joined', user_id, {
            'user_info': user_info,
            'participant_count': len(self.participants)
        })
        self.history.append(event)
        self._emit_event(event)
        
    def remove_participant(self, user_id: str):
        """Remove a participant from the session."""
        if user_id in self.participants:
            self.participants[user_id]['active'] = False
            
            # Release all locks held by this user
            locks_to_release = [obj_id for obj_id, lock_user in self.locks.items() if lock_user == user_id]
            for obj_id in locks_to_release:
                del self.locks[obj_id]
            
            event = CollaborationEvent('user_left', user_id, {
                'released_locks': locks_to_release,
                'participant_count': len([p for p in self.participants.values() if p['active']])
            })
            self.history.append(event)
            self._emit_event(event)
    
    def update_cursor(self, user_id: str, position: Dict[str, float], target_object: Optional[str] = None):
        """Update user's 3D cursor position."""
        self.cursors[user_id] = {
            'position': position,
            'target_object': target_object,
            'timestamp': datetime.now().isoformat()
        }
        
        event = CollaborationEvent('cursor_moved', user_id, {
            'position': position,
            'target_object': target_object
        })
        self._emit_event(event)
    
    def update_selection(self, user_id: str, selected_objects: List[str]):
        """Update user's selection."""
        self.selections[user_id] = selected_objects
        
        event = CollaborationEvent('selection_changed', user_id, {
            'selected_objects': selected_objects
        })
        self._emit_event(event)
    
    def request_lock(self, user_id: str, object_id: str) -> bool:
        """Request exclusive lock on an object."""
        if object_id in self.locks and self.locks[object_id] != user_id:
            return False  # Already locked by another user
        
        self.locks[object_id] = user_id
        event = CollaborationEvent('object_locked', user_id, {
            'object_id': object_id
        })
        self.history.append(event)
        self._emit_event(event)
        return True
    
    def release_lock(self, user_id: str, object_id: str) -> bool:
        """Release lock on an object."""
        if object_id not in self.locks or self.locks[object_id] != user_id:
            return False  # Not locked by this user
        
        del self.locks[object_id]
        event = CollaborationEvent('object_unlocked', user_id, {
            'object_id': object_id
        })
        self.history.append(event)
        self._emit_event(event)
        return True
    
    def apply_transformation(self, user_id: str, object_id: str, transformation: Dict[str, Any]) -> bool:
        """Apply transformation to an object."""
        # Check if user has permission and lock
        if object_id in self.locks and self.locks[object_id] != user_id:
            return False
        
        if 'edit' not in self.participants.get(user_id, {}).get('permissions', []):
            return False
        
        event = CollaborationEvent('object_transformed', user_id, {
            'object_id': object_id,
            'transformation': transformation
        })
        self.history.append(event)
        self._emit_event(event)
        return True
    
    def add_object(self, user_id: str, object_data: Dict[str, Any]) -> Optional[str]:
        """Add a new object to the scene."""
        if 'edit' not in self.participants.get(user_id, {}).get('permissions', []):
            return None
        
        object_id = str(uuid.uuid4())
        event = CollaborationEvent('object_added', user_id, {
            'object_id': object_id,
            'object_data': object_data
        })
        self.history.append(event)
        self._emit_event(event)
        return object_id
    
    def delete_object(self, user_id: str, object_id: str) -> bool:
        """Delete an object from the scene."""
        if 'edit' not in self.participants.get(user_id, {}).get('permissions', []):
            return False
        
        # Release lock if held
        if object_id in self.locks:
            del self.locks[object_id]
        
        event = CollaborationEvent('object_deleted', user_id, {
            'object_id': object_id
        })
        self.history.append(event)
        self._emit_event(event)
        return True
    
    def send_message(self, user_id: str, message: str, message_type: str = 'text'):
        """Send a chat message."""
        event = CollaborationEvent('message', user_id, {
            'message': message,
            'message_type': message_type
        })
        self.history.append(event)
        self._emit_event(event)
    
    def on(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event: CollaborationEvent):
        """Emit event to all handlers."""
        handlers = self.event_handlers.get(event.type, []) + self.event_handlers.get('*', [])
        for handler in handlers:
            try:
                asyncio.create_task(handler(event))
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        active_participants = {uid: info for uid, info in self.participants.items() if info['active']}
        return {
            'id': self.id,
            'project_id': self.project_id,
            'participants': active_participants,
            'cursors': self.cursors,
            'selections': self.selections,
            'locks': self.locks,
            'created_at': self.created_at.isoformat(),
            'event_count': len(self.history)
        }

class CollaborationManager:
    """Manages all collaboration sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
    def create_session(self, project_id: str, creator_id: str) -> CollaborationSession:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        session = CollaborationSession(session_id, project_id)
        self.sessions[session_id] = session
        
        # Add creator as first participant
        creator_info = {
            'username': f'user_{creator_id[:8]}',
            'permissions': ['view', 'edit', 'admin'],
            'avatar': f'/avatars/{creator_id}.jpg'
        }
        session.add_participant(creator_id, creator_info)
        
        if creator_id not in self.user_sessions:
            self.user_sessions[creator_id] = set()
        self.user_sessions[creator_id].add(session_id)
        
        logger.info(f"Created collaboration session {session_id} for project {project_id}")
        return session
    
    def join_session(self, session_id: str, user_id: str, user_info: Dict[str, Any]) -> Optional[CollaborationSession]:
        """Join an existing collaboration session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session.add_participant(user_id, user_info)
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        return session
    
    def leave_session(self, session_id: str, user_id: str):
        """Leave a collaboration session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.remove_participant(user_id)
            
            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get a collaboration session."""
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[CollaborationSession]:
        """Get all sessions for a user."""
        session_ids = self.user_sessions.get(user_id, set())
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def cleanup_inactive_sessions(self):
        """Clean up sessions with no active participants."""
        to_remove = []
        for session_id, session in self.sessions.items():
            active_count = len([p for p in session.participants.values() if p['active']])
            if active_count == 0:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up inactive session {session_id}")

class WebRTCSignaling:
    """WebRTC signaling server for peer-to-peer communication."""
    
    def __init__(self):
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.rooms: Dict[str, Set[str]] = {}
        
    async def join_room(self, room_id: str, peer_id: str, peer_info: Dict[str, Any]):
        """Join a WebRTC room."""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(peer_id)
        self.peers[peer_id] = {
            'room_id': room_id,
            'info': peer_info,
            'connected_at': datetime.now().isoformat()
        }
        
        # Notify other peers
        await self._broadcast_to_room(room_id, {
            'type': 'peer_joined',
            'peer_id': peer_id,
            'peer_info': peer_info
        }, exclude=peer_id)
    
    async def leave_room(self, peer_id: str):
        """Leave a WebRTC room."""
        if peer_id not in self.peers:
            return
        
        room_id = self.peers[peer_id]['room_id']
        self.rooms[room_id].discard(peer_id)
        del self.peers[peer_id]
        
        # Notify other peers
        await self._broadcast_to_room(room_id, {
            'type': 'peer_left',
            'peer_id': peer_id
        })
    
    async def handle_offer(self, from_peer: str, to_peer: str, offer: Dict[str, Any]):
        """Handle WebRTC offer."""
        await self._send_to_peer(to_peer, {
            'type': 'offer',
            'from_peer': from_peer,
            'offer': offer
        })
    
    async def handle_answer(self, from_peer: str, to_peer: str, answer: Dict[str, Any]):
        """Handle WebRTC answer."""
        await self._send_to_peer(to_peer, {
            'type': 'answer',
            'from_peer': from_peer,
            'answer': answer
        })
    
    async def handle_ice_candidate(self, from_peer: str, to_peer: str, candidate: Dict[str, Any]):
        """Handle ICE candidate."""
        await self._send_to_peer(to_peer, {
            'type': 'ice_candidate',
            'from_peer': from_peer,
            'candidate': candidate
        })
    
    async def _broadcast_to_room(self, room_id: str, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all peers in a room."""
        if room_id not in self.rooms:
            return
        
        for peer_id in self.rooms[room_id]:
            if peer_id != exclude:
                await self._send_to_peer(peer_id, message)
    
    async def _send_to_peer(self, peer_id: str, message: Dict[str, Any]):
        """Send message to a specific peer."""
        # In a real implementation, this would send via WebSocket
        logger.info(f"Sending to peer {peer_id}: {message['type']}")

# Global instances
collaboration_manager = CollaborationManager()
webrtc_signaling = WebRTCSignaling()

async def create_collaboration_session(project_id: str, creator_id: str) -> str:
    """Create a new collaboration session."""
    session = collaboration_manager.create_session(project_id, creator_id)
    return session.id

async def join_collaboration(session_id: str, user_id: str, user_info: Dict[str, Any]) -> bool:
    """Join a collaboration session."""
    session = collaboration_manager.join_session(session_id, user_id, user_info)
    return session is not None

if __name__ == "__main__":
    # Test the collaboration system
    async def test_collaboration():
        # Create session
        session_id = await create_collaboration_session("project_1", "user_1")
        print(f"Created session: {session_id}")
        
        # Join another user
        success = await join_collaboration(session_id, "user_2", {
            'username': 'Alice',
            'permissions': ['view', 'edit']
        })
        print(f"User 2 joined: {success}")
        
        # Get session and test operations
        session = collaboration_manager.get_session(session_id)
        if session:
            # Test cursor update
            session.update_cursor("user_1", {"x": 1.0, "y": 2.0, "z": 3.0})
            
            # Test object operations
            obj_id = session.add_object("user_2", {"type": "cube", "size": 1.0})
            print(f"Added object: {obj_id}")
            
            # Test lock
            locked = session.request_lock("user_1", obj_id)
            print(f"Lock acquired: {locked}")
            
            # Test transformation
            transformed = session.apply_transformation("user_1", obj_id, {
                "translation": [1, 0, 0],
                "rotation": [0, 45, 0]
            })
            print(f"Transformation applied: {transformed}")
            
            print(f"Session state: {session.get_state()}")
    
    asyncio.run(test_collaboration())