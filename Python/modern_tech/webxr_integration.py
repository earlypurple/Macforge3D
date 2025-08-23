"""
WebXR Integration for MacForge3D
Enables immersive VR/AR experiences in web browsers and native apps
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import uuid
import math

logger = logging.getLogger(__name__)

class XRController:
    """Represents a VR/AR controller."""
    
    def __init__(self, controller_id: str, hand: str):
        self.id = controller_id
        self.hand = hand  # 'left' or 'right'
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.buttons = {}
        self.triggers = {}
        self.thumbsticks = {}
        self.connected = True
        self.last_update = datetime.now()
    
    def update_pose(self, position: List[float], rotation: List[float]):
        """Update controller pose."""
        self.position = position
        self.rotation = rotation
        self.last_update = datetime.now()
    
    def update_input(self, buttons: Dict[str, bool], triggers: Dict[str, float], thumbsticks: Dict[str, List[float]]):
        """Update controller input state."""
        self.buttons = buttons
        self.triggers = triggers
        self.thumbsticks = thumbsticks
        self.last_update = datetime.now()

class XRSession:
    """Manages an XR session."""
    
    def __init__(self, session_id: str, mode: str, user_id: str):
        self.id = session_id
        self.mode = mode  # 'immersive-vr', 'immersive-ar', 'inline'
        self.user_id = user_id
        self.active = False
        self.controllers: Dict[str, XRController] = {}
        self.head_pose = {
            'position': [0.0, 1.6, 0.0],  # Default head height
            'rotation': [0.0, 0.0, 0.0, 1.0]
        }
        self.eye_tracking = {
            'supported': False,
            'gaze_point': [0.0, 0.0, -1.0],
            'confidence': 0.0
        }
        self.hand_tracking = {
            'supported': False,
            'left_hand': None,
            'right_hand': None
        }
        self.room_scale = {
            'bounds': [],
            'center': [0.0, 0.0, 0.0]
        }
        self.created_at = datetime.now()
        
    def add_controller(self, hand: str) -> str:
        """Add a new controller."""
        controller_id = str(uuid.uuid4())
        controller = XRController(controller_id, hand)
        self.controllers[controller_id] = controller
        return controller_id
    
    def update_head_pose(self, position: List[float], rotation: List[float]):
        """Update head pose."""
        self.head_pose['position'] = position
        self.head_pose['rotation'] = rotation
    
    def update_eye_tracking(self, gaze_point: List[float], confidence: float):
        """Update eye tracking data."""
        self.eye_tracking['gaze_point'] = gaze_point
        self.eye_tracking['confidence'] = confidence
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            'id': self.id,
            'mode': self.mode,
            'user_id': self.user_id,
            'active': self.active,
            'head_pose': self.head_pose,
            'controllers': {cid: {
                'hand': ctrl.hand,
                'position': ctrl.position,
                'rotation': ctrl.rotation,
                'buttons': ctrl.buttons,
                'connected': ctrl.connected
            } for cid, ctrl in self.controllers.items()},
            'eye_tracking': self.eye_tracking,
            'hand_tracking': self.hand_tracking,
            'room_scale': self.room_scale
        }

class WebXRManager:
    """Manages WebXR sessions and interactions."""
    
    def __init__(self):
        self.sessions: Dict[str, XRSession] = {}
        self.supported_modes = ['immersive-vr', 'immersive-ar', 'inline']
        self.device_capabilities = {
            'position_tracking': True,
            'rotation_tracking': True,
            'room_scale': True,
            'hand_tracking': True,
            'eye_tracking': False,
            'haptic_feedback': True
        }
        self.scene_objects: Dict[str, Dict[str, Any]] = {}
        
    async def check_xr_support(self) -> Dict[str, bool]:
        """Check XR support for different modes."""
        # Simulate XR capability detection
        return {
            'immersive-vr': True,
            'immersive-ar': True,
            'inline': True,
            'hand-tracking': True,
            'hit-test': True,
            'dom-overlay': True,
            'light-estimation': True,
            'anchors': True
        }
    
    async def create_session(self, mode: str, user_id: str, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new XR session."""
        try:
            if mode not in self.supported_modes:
                raise ValueError(f"Unsupported XR mode: {mode}")
            
            session_id = str(uuid.uuid4())
            session = XRSession(session_id, mode, user_id)
            
            # Apply session options
            if options:
                if 'required_features' in options:
                    # Check if required features are supported
                    for feature in options['required_features']:
                        if feature not in self.device_capabilities:
                            raise ValueError(f"Required feature not supported: {feature}")
                
                if 'room_scale' in options:
                    session.room_scale = options['room_scale']
            
            # Add default controllers for VR mode
            if mode == 'immersive-vr':
                session.add_controller('left')
                session.add_controller('right')
            
            self.sessions[session_id] = session
            session.active = True
            
            logger.info(f"🥽 XR session created: {session_id} (mode: {mode})")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create XR session: {e}")
            return None
    
    async def end_session(self, session_id: str) -> bool:
        """End an XR session."""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            del self.sessions[session_id]
            logger.info(f"XR session ended: {session_id}")
            return True
        return False
    
    async def update_pose(self, session_id: str, pose_data: Dict[str, Any]) -> bool:
        """Update pose data for a session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Update head pose
        if 'head' in pose_data:
            head_data = pose_data['head']
            session.update_head_pose(
                head_data.get('position', session.head_pose['position']),
                head_data.get('rotation', session.head_pose['rotation'])
            )
        
        # Update controller poses
        if 'controllers' in pose_data:
            for controller_id, controller_data in pose_data['controllers'].items():
                if controller_id in session.controllers:
                    controller = session.controllers[controller_id]
                    controller.update_pose(
                        controller_data.get('position', controller.position),
                        controller_data.get('rotation', controller.rotation)
                    )
                    
                    # Update input state
                    if 'input' in controller_data:
                        input_data = controller_data['input']
                        controller.update_input(
                            input_data.get('buttons', {}),
                            input_data.get('triggers', {}),
                            input_data.get('thumbsticks', {})
                        )
        
        # Update eye tracking
        if 'eye_tracking' in pose_data:
            eye_data = pose_data['eye_tracking']
            session.update_eye_tracking(
                eye_data.get('gaze_point', [0, 0, -1]),
                eye_data.get('confidence', 0.0)
            )
        
        return True
    
    async def add_scene_object(self, session_id: str, object_data: Dict[str, Any]) -> Optional[str]:
        """Add a 3D object to the XR scene."""
        if session_id not in self.sessions:
            return None
        
        object_id = str(uuid.uuid4())
        
        # Default object properties
        scene_object = {
            'id': object_id,
            'type': object_data.get('type', 'mesh'),
            'position': object_data.get('position', [0, 0, 0]),
            'rotation': object_data.get('rotation', [0, 0, 0, 1]),
            'scale': object_data.get('scale', [1, 1, 1]),
            'mesh_data': object_data.get('mesh_data', {}),
            'material': object_data.get('material', {'color': [1, 1, 1, 1]}),
            'physics': object_data.get('physics', {'enabled': False}),
            'interactive': object_data.get('interactive', False),
            'visible': object_data.get('visible', True),
            'session_id': session_id,
            'created_at': datetime.now().isoformat()
        }
        
        self.scene_objects[object_id] = scene_object
        logger.info(f"🎯 Scene object added: {object_id}")
        return object_id
    
    async def update_scene_object(self, object_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scene object."""
        if object_id not in self.scene_objects:
            return False
        
        scene_object = self.scene_objects[object_id]
        scene_object.update(updates)
        return True
    
    async def remove_scene_object(self, object_id: str) -> bool:
        """Remove a scene object."""
        if object_id in self.scene_objects:
            del self.scene_objects[object_id]
            return True
        return False
    
    async def handle_controller_interaction(self, session_id: str, controller_id: str, interaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle controller interaction with scene objects."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if controller_id not in session.controllers:
            return None
        
        controller = session.controllers[controller_id]
        interaction_type = interaction.get('type', 'select')
        
        # Perform ray casting to find intersected objects
        ray_origin = controller.position
        ray_direction = self._get_controller_forward_vector(controller.rotation)
        
        intersected_objects = await self._raycast_scene_objects(session_id, ray_origin, ray_direction)
        
        if intersected_objects:
            closest_object = intersected_objects[0]
            
            # Handle different interaction types
            if interaction_type == 'select':
                return await self._handle_object_selection(closest_object['id'], controller_id)
            elif interaction_type == 'grab':
                return await self._handle_object_grab(closest_object['id'], controller_id)
            elif interaction_type == 'teleport':
                return await self._handle_teleport(session_id, closest_object['position'])
        
        return None
    
    async def handle_hand_tracking(self, session_id: str, hand_data: Dict[str, Any]) -> bool:
        """Handle hand tracking data."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.hand_tracking['supported'] = True
        
        for hand in ['left_hand', 'right_hand']:
            if hand in hand_data:
                session.hand_tracking[hand] = {
                    'joints': hand_data[hand].get('joints', {}),
                    'gestures': hand_data[hand].get('gestures', []),
                    'confidence': hand_data[hand].get('confidence', 0.0)
                }
        
        return True
    
    async def enable_haptic_feedback(self, session_id: str, controller_id: str, intensity: float, duration: float) -> bool:
        """Trigger haptic feedback on a controller."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if controller_id not in session.controllers:
            return False
        
        # Simulate haptic feedback
        logger.info(f"🎮 Haptic feedback: intensity={intensity}, duration={duration}ms")
        return True
    
    async def perform_hit_test(self, session_id: str, ray_origin: List[float], ray_direction: List[float]) -> List[Dict[str, Any]]:
        """Perform hit test for AR placement."""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        if session.mode != 'immersive-ar':
            return []
        
        # Simulate hit test results
        hit_results = []
        
        # Simulate floor plane detection
        if ray_direction[1] < 0:  # Ray pointing downward
            t = -ray_origin[1] / ray_direction[1]  # Intersection with y=0 plane
            if t > 0:
                hit_point = [
                    ray_origin[0] + ray_direction[0] * t,
                    0.0,
                    ray_origin[2] + ray_direction[2] * t
                ]
                
                hit_results.append({
                    'position': hit_point,
                    'normal': [0, 1, 0],
                    'distance': t,
                    'type': 'plane'
                })
        
        return hit_results
    
    def _get_controller_forward_vector(self, rotation: List[float]) -> List[float]:
        """Get forward vector from controller rotation quaternion."""
        # Convert quaternion to forward vector (simplified)
        x, y, z, w = rotation
        forward = [
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ]
        return forward
    
    async def _raycast_scene_objects(self, session_id: str, ray_origin: List[float], ray_direction: List[float]) -> List[Dict[str, Any]]:
        """Perform raycast against scene objects."""
        intersections = []
        
        for object_id, obj in self.scene_objects.items():
            if obj['session_id'] != session_id or not obj['visible']:
                continue
            
            # Simplified sphere intersection test
            obj_pos = obj['position']
            obj_scale = obj['scale'][0]  # Assume uniform scale
            
            # Vector from ray origin to object center
            oc = [obj_pos[i] - ray_origin[i] for i in range(3)]
            
            # Project onto ray direction
            t = sum(oc[i] * ray_direction[i] for i in range(3))
            
            if t > 0:  # Object in front of ray
                # Closest point on ray to object center
                closest_point = [ray_origin[i] + t * ray_direction[i] for i in range(3)]
                
                # Distance from object center to closest point
                distance_sq = sum((obj_pos[i] - closest_point[i]) ** 2 for i in range(3))
                
                if distance_sq <= obj_scale ** 2:  # Intersection
                    intersections.append({
                        'id': object_id,
                        'object': obj,
                        'distance': t,
                        'position': closest_point
                    })
        
        # Sort by distance
        intersections.sort(key=lambda x: x['distance'])
        return intersections
    
    async def _handle_object_selection(self, object_id: str, controller_id: str) -> Dict[str, Any]:
        """Handle object selection."""
        return {
            'type': 'selection',
            'object_id': object_id,
            'controller_id': controller_id,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_object_grab(self, object_id: str, controller_id: str) -> Dict[str, Any]:
        """Handle object grab interaction."""
        return {
            'type': 'grab',
            'object_id': object_id,
            'controller_id': controller_id,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_teleport(self, session_id: str, target_position: List[float]) -> Dict[str, Any]:
        """Handle teleportation."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Update head position for teleportation
            session.head_pose['position'] = [
                target_position[0],
                session.head_pose['position'][1],  # Keep same height
                target_position[2]
            ]
        
        return {
            'type': 'teleport',
            'target_position': target_position,
            'timestamp': datetime.now().isoformat()
        }

class ARFeatures:
    """Advanced AR features for MacForge3D."""
    
    def __init__(self):
        self.anchors: Dict[str, Dict[str, Any]] = {}
        self.light_estimation = {
            'ambient_intensity': 1.0,
            'light_direction': [0, -1, 0],
            'color_temperature': 6500
        }
        self.plane_detection = {
            'horizontal_planes': [],
            'vertical_planes': []
        }
    
    async def create_anchor(self, position: List[float], rotation: List[float]) -> str:
        """Create a persistent AR anchor."""
        anchor_id = str(uuid.uuid4())
        
        anchor = {
            'id': anchor_id,
            'position': position,
            'rotation': rotation,
            'persistent': True,
            'created_at': datetime.now().isoformat()
        }
        
        self.anchors[anchor_id] = anchor
        logger.info(f"⚓ AR anchor created: {anchor_id}")
        return anchor_id
    
    async def update_light_estimation(self, ambient_intensity: float, light_direction: List[float], color_temperature: float):
        """Update environmental light estimation."""
        self.light_estimation = {
            'ambient_intensity': ambient_intensity,
            'light_direction': light_direction,
            'color_temperature': color_temperature
        }
    
    async def detect_planes(self, session_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Detect horizontal and vertical planes in AR."""
        # Simulate plane detection
        horizontal_planes = [
            {
                'id': str(uuid.uuid4()),
                'center': [0, 0, 0],
                'extents': [2.0, 2.0],
                'orientation': 'horizontal'
            }
        ]
        
        vertical_planes = [
            {
                'id': str(uuid.uuid4()),
                'center': [0, 1, -2],
                'extents': [3.0, 2.0],
                'orientation': 'vertical'
            }
        ]
        
        self.plane_detection = {
            'horizontal_planes': horizontal_planes,
            'vertical_planes': vertical_planes
        }
        
        return self.plane_detection

# Global instances
webxr_manager = WebXRManager()
ar_features = ARFeatures()

async def initialize_webxr():
    """Initialize WebXR system."""
    support = await webxr_manager.check_xr_support()
    logger.info(f"WebXR support: {support}")
    return True

async def create_xr_session(mode: str, user_id: str, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Create a new XR session."""
    return await webxr_manager.create_session(mode, user_id, options)

if __name__ == "__main__":
    # Test WebXR functionality
    async def test_webxr():
        # Initialize WebXR
        await initialize_webxr()
        
        # Create VR session
        session_id = await create_xr_session('immersive-vr', 'test_user', {
            'required_features': ['local-floor', 'bounded-floor']
        })
        print(f"VR session created: {session_id}")
        
        if session_id:
            # Add scene object
            object_id = await webxr_manager.add_scene_object(session_id, {
                'type': 'mesh',
                'position': [0, 1, -2],
                'scale': [0.5, 0.5, 0.5],
                'material': {'color': [1, 0, 0, 1]},
                'interactive': True
            })
            print(f"Scene object added: {object_id}")
            
            # Simulate pose update
            await webxr_manager.update_pose(session_id, {
                'head': {
                    'position': [0, 1.6, 0],
                    'rotation': [0, 0, 0, 1]
                },
                'controllers': {
                    list(webxr_manager.sessions[session_id].controllers.keys())[0]: {
                        'position': [0.3, 1.2, -0.5],
                        'rotation': [0, 0, 0, 1],
                        'input': {
                            'buttons': {'trigger': True},
                            'triggers': {'main': 0.8}
                        }
                    }
                }
            })
            
            # Test controller interaction
            controller_id = list(webxr_manager.sessions[session_id].controllers.keys())[0]
            interaction_result = await webxr_manager.handle_controller_interaction(
                session_id, controller_id, {'type': 'select'}
            )
            print(f"Interaction result: {interaction_result}")
            
            # Get session state
            state = webxr_manager.sessions[session_id].get_session_state()
            print(f"Session state: {json.dumps(state, indent=2)}")
        
        # Test AR features
        anchor_id = await ar_features.create_anchor([0, 0, -1], [0, 0, 0, 1])
        print(f"AR anchor created: {anchor_id}")
        
        planes = await ar_features.detect_planes(session_id or 'test')
        print(f"Detected planes: {planes}")
    
    asyncio.run(test_webxr())