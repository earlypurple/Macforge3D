"""
Real-Time Collaborative 3D Design Module
Enable multiple users to collaborate on 3D projects simultaneously.
"""

import asyncio
import websockets
import json
import time
import threading
from typing import Dict, Any, List, Set, Optional, Callable, Tuple, Union
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class User:
    """Represents a collaborative user."""
    user_id: str
    username: str
    session_id: str
    connected_at: float
    last_active: float
    cursor_position: List[float]
    selected_objects: List[str]
    permissions: Dict[str, bool]
    color: str

@dataclass
class CollaborationEvent:
    """Represents a collaboration event."""
    event_id: str
    event_type: str
    user_id: str
    timestamp: float
    data: Dict[str, Any]
    session_id: str

@dataclass
class ProjectSnapshot:
    """Represents a project state snapshot."""
    snapshot_id: str
    project_id: str
    timestamp: float
    created_by: str
    mesh_data: Dict[str, Any]
    metadata: Dict[str, Any]
    version: int

class ConflictResolver:
    """Resolve conflicts in collaborative editing."""
    
    def __init__(self):
        self.resolution_strategies = {
            'last_writer_wins': self._last_writer_wins,
            'merge_changes': self._merge_changes,
            'user_priority': self._user_priority_resolution,
            'semantic_merge': self._semantic_merge
        }
    
    def resolve_conflict(self, events: List[CollaborationEvent], strategy: str = 'semantic_merge') -> CollaborationEvent:
        """
        Resolve conflicts between simultaneous edits.
        
        Args:
            events: Conflicting events
            strategy: Resolution strategy
            
        Returns:
            Resolved event
        """
        if not events:
            return None
        
        if len(events) == 1:
            return events[0]
        
        resolver = self.resolution_strategies.get(strategy, self._semantic_merge)
        return resolver(events)
    
    def _last_writer_wins(self, events: List[CollaborationEvent]) -> CollaborationEvent:
        """Simple last-writer-wins strategy."""
        return max(events, key=lambda e: e.timestamp)
    
    def _merge_changes(self, events: List[CollaborationEvent]) -> CollaborationEvent:
        """Merge changes when possible."""
        if not events:
            return None
        
        base_event = events[0]
        merged_data = base_event.data.copy()
        
        for event in events[1:]:
            # Simple merge - could be much more sophisticated
            for key, value in event.data.items():
                if key not in merged_data:
                    merged_data[key] = value
                elif isinstance(value, dict) and isinstance(merged_data[key], dict):
                    merged_data[key].update(value)
                elif isinstance(value, list) and isinstance(merged_data[key], list):
                    merged_data[key].extend(value)
                else:
                    merged_data[key] = value  # Last writer wins for this field
        
        return CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type='merged_edit',
            user_id='system',
            timestamp=time.time(),
            data=merged_data,
            session_id=base_event.session_id
        )
    
    def _user_priority_resolution(self, events: List[CollaborationEvent]) -> CollaborationEvent:
        """Resolve based on user priority."""
        # For now, just return the event from the first user (could implement real priority)
        return events[0]
    
    def _semantic_merge(self, events: List[CollaborationEvent]) -> CollaborationEvent:
        """Intelligent semantic merge of changes."""
        if not events:
            return None
        
        # Group events by type
        by_type = defaultdict(list)
        for event in events:
            by_type[event.event_type].append(event)
        
        # Handle different event types differently
        merged_data = {}
        latest_timestamp = max(e.timestamp for e in events)
        
        for event_type, type_events in by_type.items():
            if event_type == 'mesh_edit':
                # Merge mesh edits intelligently
                merged_data.update(self._merge_mesh_edits(type_events))
            elif event_type == 'material_change':
                # Material changes can often be merged
                merged_data.update(self._merge_material_changes(type_events))
            elif event_type == 'transform':
                # Transform operations might conflict
                merged_data.update(self._resolve_transform_conflicts(type_events))
            else:
                # Default to last writer wins
                latest_event = max(type_events, key=lambda e: e.timestamp)
                merged_data.update(latest_event.data)
        
        return CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type='semantic_merge',
            user_id='system',
            timestamp=latest_timestamp,
            data=merged_data,
            session_id=events[0].session_id
        )
    
    def _merge_mesh_edits(self, events: List[CollaborationEvent]) -> Dict[str, Any]:
        """Merge mesh editing operations."""
        merged = {}
        
        # Collect all vertex modifications
        vertex_changes = {}
        face_changes = {}
        
        for event in events:
            data = event.data
            if 'vertex_changes' in data:
                vertex_changes.update(data['vertex_changes'])
            if 'face_changes' in data:
                face_changes.update(data['face_changes'])
            if 'new_vertices' in data:
                merged.setdefault('new_vertices', []).extend(data['new_vertices'])
            if 'new_faces' in data:
                merged.setdefault('new_faces', []).extend(data['new_faces'])
        
        if vertex_changes:
            merged['vertex_changes'] = vertex_changes
        if face_changes:
            merged['face_changes'] = face_changes
        
        return merged
    
    def _merge_material_changes(self, events: List[CollaborationEvent]) -> Dict[str, Any]:
        """Merge material change operations."""
        merged_materials = {}
        
        for event in events:
            data = event.data
            if 'materials' in data:
                merged_materials.update(data['materials'])
        
        return {'materials': merged_materials} if merged_materials else {}
    
    def _resolve_transform_conflicts(self, events: List[CollaborationEvent]) -> Dict[str, Any]:
        """Resolve conflicting transform operations."""
        # For transforms, we might want to compose them or pick the most recent
        latest_transform = max(events, key=lambda e: e.timestamp)
        return latest_transform.data

class CollaborationSession:
    """Manages a collaborative editing session."""
    
    def __init__(self, session_id: str, project_id: str):
        self.session_id = session_id
        self.project_id = project_id
        self.users: Dict[str, User] = {}
        self.event_history: List[CollaborationEvent] = []
        self.current_state: Dict[str, Any] = {}
        self.conflict_resolver = ConflictResolver()
        self.version = 0
        self.snapshots: List[ProjectSnapshot] = []
        self.locks: Dict[str, str] = {}  # object_id -> user_id
        self.created_at = time.time()
        self.last_activity = time.time()
        
    def add_user(self, user: User) -> bool:
        """Add a user to the session."""
        if user.user_id in self.users:
            logger.warning(f"User {user.user_id} already in session")
            return False
        
        self.users[user.user_id] = user
        self.last_activity = time.time()
        
        # Broadcast user joined event
        join_event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type='user_joined',
            user_id=user.user_id,
            timestamp=time.time(),
            data={'username': user.username, 'color': user.color},
            session_id=self.session_id
        )
        self.add_event(join_event)
        
        logger.info(f"User {user.username} joined session {self.session_id}")
        return True
    
    def remove_user(self, user_id: str) -> bool:
        """Remove a user from the session."""
        if user_id not in self.users:
            return False
        
        user = self.users.pop(user_id)
        self.last_activity = time.time()
        
        # Release any locks held by this user
        locks_to_release = [obj_id for obj_id, lock_user in self.locks.items() if lock_user == user_id]
        for obj_id in locks_to_release:
            del self.locks[obj_id]
        
        # Broadcast user left event
        leave_event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type='user_left',
            user_id=user_id,
            timestamp=time.time(),
            data={'username': user.username},
            session_id=self.session_id
        )
        self.add_event(leave_event)
        
        logger.info(f"User {user.username} left session {self.session_id}")
        return True
    
    def add_event(self, event: CollaborationEvent) -> bool:
        """Add an event to the session."""
        self.event_history.append(event)
        self.last_activity = time.time()
        
        # Apply event to current state
        self._apply_event_to_state(event)
        
        # Check for conflicts with recent events
        recent_events = [e for e in self.event_history[-10:] if abs(e.timestamp - event.timestamp) < 1.0]
        if len(recent_events) > 1:
            conflicts = self._detect_conflicts(recent_events)
            if conflicts:
                resolved_event = self.conflict_resolver.resolve_conflict(conflicts)
                if resolved_event:
                    self._apply_event_to_state(resolved_event)
        
        return True
    
    def _apply_event_to_state(self, event: CollaborationEvent):
        """Apply an event to the current project state."""
        if event.event_type == 'mesh_edit':
            self._apply_mesh_edit(event.data)
        elif event.event_type == 'material_change':
            self._apply_material_change(event.data)
        elif event.event_type == 'transform':
            self._apply_transform(event.data)
        elif event.event_type == 'user_cursor':
            self._update_user_cursor(event.user_id, event.data)
        
        self.version += 1
    
    def _apply_mesh_edit(self, data: Dict[str, Any]):
        """Apply mesh editing changes to current state."""
        if 'mesh_data' not in self.current_state:
            self.current_state['mesh_data'] = {'vertices': [], 'faces': []}
        
        mesh_data = self.current_state['mesh_data']
        
        if 'vertex_changes' in data:
            for vertex_id, new_pos in data['vertex_changes'].items():
                if int(vertex_id) < len(mesh_data['vertices']):
                    mesh_data['vertices'][int(vertex_id)] = new_pos
        
        if 'new_vertices' in data:
            mesh_data['vertices'].extend(data['new_vertices'])
        
        if 'new_faces' in data:
            mesh_data['faces'].extend(data['new_faces'])
    
    def _apply_material_change(self, data: Dict[str, Any]):
        """Apply material changes to current state."""
        if 'materials' not in self.current_state:
            self.current_state['materials'] = {}
        
        if 'materials' in data:
            self.current_state['materials'].update(data['materials'])
    
    def _apply_transform(self, data: Dict[str, Any]):
        """Apply transform changes to current state."""
        if 'transforms' not in self.current_state:
            self.current_state['transforms'] = {}
        
        if 'object_id' in data and 'transform' in data:
            self.current_state['transforms'][data['object_id']] = data['transform']
    
    def _update_user_cursor(self, user_id: str, data: Dict[str, Any]):
        """Update user cursor position."""
        if user_id in self.users:
            if 'position' in data:
                self.users[user_id].cursor_position = data['position']
            if 'selected_objects' in data:
                self.users[user_id].selected_objects = data['selected_objects']
    
    def _detect_conflicts(self, events: List[CollaborationEvent]) -> List[CollaborationEvent]:
        """Detect conflicting events."""
        conflicts = []
        
        # Group events by type and affected objects
        object_events = defaultdict(list)
        
        for event in events:
            if event.event_type in ['mesh_edit', 'transform', 'material_change']:
                # Determine affected objects
                affected_objects = self._get_affected_objects(event)
                for obj_id in affected_objects:
                    object_events[obj_id].append(event)
        
        # Find conflicts (multiple events affecting same object)
        for obj_id, obj_events in object_events.items():
            if len(obj_events) > 1:
                conflicts.extend(obj_events)
        
        return conflicts
    
    def _get_affected_objects(self, event: CollaborationEvent) -> List[str]:
        """Get list of objects affected by an event."""
        affected = []
        
        if event.event_type == 'mesh_edit':
            affected.append('main_mesh')  # Could be more specific
        elif event.event_type == 'transform':
            if 'object_id' in event.data:
                affected.append(event.data['object_id'])
        elif event.event_type == 'material_change':
            if 'object_id' in event.data:
                affected.append(event.data['object_id'])
        
        return affected
    
    def create_snapshot(self, user_id: str) -> str:
        """Create a snapshot of the current project state."""
        snapshot = ProjectSnapshot(
            snapshot_id=str(uuid.uuid4()),
            project_id=self.project_id,
            timestamp=time.time(),
            created_by=user_id,
            mesh_data=self.current_state.copy(),
            metadata={
                'version': self.version,
                'active_users': list(self.users.keys()),
                'event_count': len(self.event_history)
            },
            version=self.version
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only last 10 snapshots
        if len(self.snapshots) > 10:
            self.snapshots = self.snapshots[-10:]
        
        return snapshot.snapshot_id
    
    def get_state_for_user(self, user_id: str) -> Dict[str, Any]:
        """Get the current state formatted for a specific user."""
        user_state = {
            'session_info': {
                'session_id': self.session_id,
                'project_id': self.project_id,
                'version': self.version,
                'your_user_id': user_id
            },
            'project_state': self.current_state.copy(),
            'active_users': {
                uid: {
                    'username': user.username,
                    'cursor_position': user.cursor_position,
                    'selected_objects': user.selected_objects,
                    'color': user.color,
                    'last_active': user.last_active
                }
                for uid, user in self.users.items() if uid != user_id
            },
            'recent_events': [
                {
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'timestamp': event.timestamp,
                    'data': event.data
                }
                for event in self.event_history[-20:]  # Last 20 events
            ],
            'object_locks': self.locks.copy()
        }
        
        return user_state

class CollaborationServer:
    """WebSocket server for real-time collaboration."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = False
        
    async def start_server(self):
        """Start the collaboration server."""
        self.running = True
        logger.info(f"Starting collaboration server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    async def handle_connection(self, websocket, path):
        """Handle a new WebSocket connection."""
        user_id = None
        session_id = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'join_session':
                        user_id, session_id = await self._handle_join_session(websocket, data)
                    elif message_type == 'leave_session':
                        await self._handle_leave_session(websocket, data)
                    elif message_type == 'collaboration_event':
                        await self._handle_collaboration_event(websocket, data)
                    elif message_type == 'request_state':
                        await self._handle_state_request(websocket, data)
                    elif message_type == 'ping':
                        await self._handle_ping(websocket, data)
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            # Clean up on disconnect
            if user_id and session_id:
                await self._cleanup_user(user_id, session_id)
    
    async def _handle_join_session(self, websocket, data: Dict[str, Any]) -> Tuple[str, str]:
        """Handle user joining a session."""
        session_id = data.get('session_id')
        project_id = data.get('project_id')
        user_info = data.get('user_info', {})
        
        user_id = user_info.get('user_id', str(uuid.uuid4()))
        username = user_info.get('username', f'User_{user_id[:8]}')
        
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = CollaborationSession(session_id, project_id)
        
        session = self.sessions[session_id]
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            session_id=session_id,
            connected_at=time.time(),
            last_active=time.time(),
            cursor_position=[0, 0, 0],
            selected_objects=[],
            permissions={'edit': True, 'view': True},
            color=self._generate_user_color(user_id)
        )
        
        # Add user to session
        session.add_user(user)
        self.user_connections[user_id] = websocket
        
        # Send current state to user
        state = session.get_state_for_user(user_id)
        await websocket.send(json.dumps({
            'type': 'session_joined',
            'data': state
        }))
        
        # Broadcast to other users
        await self._broadcast_to_session(session_id, {
            'type': 'user_joined',
            'data': {
                'user_id': user_id,
                'username': username,
                'color': user.color
            }
        }, exclude_user=user_id)
        
        logger.info(f"User {username} joined session {session_id}")
        return user_id, session_id
    
    async def _handle_leave_session(self, websocket, data: Dict[str, Any]):
        """Handle user leaving a session."""
        user_id = data.get('user_id')
        session_id = data.get('session_id')
        
        await self._cleanup_user(user_id, session_id)
    
    async def _handle_collaboration_event(self, websocket, data: Dict[str, Any]):
        """Handle a collaboration event."""
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        event_data = data.get('event_data', {})
        
        if session_id not in self.sessions:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Session not found'
            }))
            return
        
        session = self.sessions[session_id]
        
        # Create collaboration event
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_data.get('type', 'unknown'),
            user_id=user_id,
            timestamp=time.time(),
            data=event_data.get('data', {}),
            session_id=session_id
        )
        
        # Add event to session
        session.add_event(event)
        
        # Broadcast event to other users
        await self._broadcast_to_session(session_id, {
            'type': 'collaboration_event',
            'data': {
                'event': asdict(event),
                'current_version': session.version
            }
        }, exclude_user=user_id)
    
    async def _handle_state_request(self, websocket, data: Dict[str, Any]):
        """Handle request for current state."""
        session_id = data.get('session_id')
        user_id = data.get('user_id')
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            state = session.get_state_for_user(user_id)
            
            await websocket.send(json.dumps({
                'type': 'state_update',
                'data': state
            }))
    
    async def _handle_ping(self, websocket, data: Dict[str, Any]):
        """Handle ping message."""
        await websocket.send(json.dumps({'type': 'pong'}))
    
    async def _cleanup_user(self, user_id: str, session_id: str):
        """Clean up user on disconnect."""
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            username = session.users.get(user_id, User('', '', '', 0, 0, [], [], {}, '')).username
            session.remove_user(user_id)
            
            # Broadcast user left
            await self._broadcast_to_session(session_id, {
                'type': 'user_left',
                'data': {
                    'user_id': user_id,
                    'username': username
                }
            })
            
            # Remove empty sessions
            if not session.users:
                del self.sessions[session_id]
                logger.info(f"Removed empty session {session_id}")
    
    async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: str = None):
        """Broadcast message to all users in a session."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        message_json = json.dumps(message)
        
        for user_id in session.users:
            if user_id != exclude_user and user_id in self.user_connections:
                try:
                    await self.user_connections[user_id].send(message_json)
                except websockets.ConnectionClosed:
                    # Connection closed, will be cleaned up
                    pass
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")
    
    def _generate_user_color(self, user_id: str) -> str:
        """Generate a unique color for a user."""
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        return colors[hash(user_id) % len(colors)]

# Global collaboration manager
collaboration_server = None

def start_collaboration_server(host: str = 'localhost', port: int = 8765):
    """Start the collaboration server."""
    global collaboration_server
    collaboration_server = CollaborationServer(host, port)
    
    # Run in background thread
    def run_server():
        asyncio.run(collaboration_server.start_server())
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    logger.info(f"Collaboration server started on {host}:{port}")
    return collaboration_server

def get_collaboration_stats() -> Dict[str, Any]:
    """Get collaboration server statistics."""
    if not collaboration_server:
        return {'error': 'Server not running'}
    
    total_users = sum(len(session.users) for session in collaboration_server.sessions.values())
    
    return {
        'server_running': collaboration_server.running,
        'active_sessions': len(collaboration_server.sessions),
        'total_users': total_users,
        'sessions': {
            session_id: {
                'project_id': session.project_id,
                'user_count': len(session.users),
                'version': session.version,
                'created_at': session.created_at,
                'last_activity': session.last_activity
            }
            for session_id, session in collaboration_server.sessions.items()
        }
    }