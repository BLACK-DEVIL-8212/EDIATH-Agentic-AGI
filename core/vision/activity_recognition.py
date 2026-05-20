# activity_recognition.py

import cv2
import numpy as np
import asyncio
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import time
import json
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from scipy.spatial.distance import cosine
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class ActivityType(Enum):
    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    JUMPING = "jumping"
    SITTING = "sitting"
    STANDING = "standing"
    FALLING = "falling"
    WAVING = "waving"
    POINTING = "pointing"
    CROUCHING = "crouching"
    CLAPPING = "clapping"
    DANCING = "dancing"
    SUSPICIOUS = "suspicious"


class ActivityDetectionMode(Enum):
    MOTION_BASED = "motion_based"
    POSE_BASED = "pose_based"
    HYBRID = "hybrid"
    DEEP_LEARNING = "deep_learning"


class PoseModelType(Enum):
    MEDIAPIPE = "mediapipe"
    YOLO_POSE = "yolo_pose"
    OPENPOSE = "openpose"
    MOVENET = "movenet"


@dataclass
class ActivityConfig:
    mode: ActivityDetectionMode = ActivityDetectionMode.HYBRID
    pose_model: PoseModelType = PoseModelType.YOLO_POSE
    confidence_threshold: float = 0.5
    smoothing_window: int = 10
    motion_sensitivity: float = 1.0
    use_gpu: bool = True
    enable_pose: bool = True
    enable_alerting: bool = True
    alert_cooldown: float = 2.0  # seconds
    history_size: int = 100
    fps_target: int = 30
    min_activity_duration: float = 0.5  # seconds
    fall_detection_enabled: bool = True
    suspicious_activities: List[str] = None
    
    def __post_init__(self):
        if self.suspicious_activities is None:
            self.suspicious_activities = ["running", "falling", "jumping"]


# ------------------------
# POSE ESTIMATION ENGINE
# ------------------------
class PoseEstimationEngine:
    def __init__(self, config: ActivityConfig):
        self.config = config
        self.model = None
        self.pose_landmarks = {}
        self.keypoints = {}
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize pose estimation model"""
        if YOLO_AVAILABLE and self.config.pose_model == PoseModelType.YOLO_POSE:
            try:
                self.model = YOLO('yolov8n-pose.pt')
                logger.info("✅ YOLOv8 Pose model loaded")
            except Exception as e:
                logger.error(f"Failed to load YOLO pose model: {e}")
                self.model = None
        else:
            logger.warning("⚠️ Advanced pose model not available")
            self.model = None
    
    def estimate_pose(self, frame: np.ndarray) -> Dict:
        """Estimate human pose from frame"""
        if self.model is None:
            return {}
        
        try:
            results = self.model(frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    
                    if len(keypoints) > 0:
                        keypoint_data = self._extract_keypoints(keypoints[0])
                        return keypoint_data
            
            return {}
            
        except Exception as e:
            logger.debug(f"Pose estimation error: {e}")
            return {}
    
    def _extract_keypoints(self, keypoints_array: np.ndarray) -> Dict:
        """Extract and organize keypoints"""
        # YOLOv8 pose has 17 keypoints
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        keypoints_dict = {}
        for i, (x, y, conf) in enumerate(keypoints_array):
            if i < len(keypoint_names) and conf > self.config.confidence_threshold:
                keypoints_dict[keypoint_names[i]] = {
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf)
                }
        
        return keypoints_dict
    
    def calculate_angles(self, keypoints: Dict) -> Dict:
        """Calculate joint angles from keypoints"""
        angles = {}
        
        # Define joint connections
        joint_pairs = [
            ('left_shoulder', 'left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee', 'left_ankle'),
            ('right_hip', 'right_knee', 'right_ankle'),
            ('left_shoulder', 'left_hip', 'left_knee'),
            ('right_shoulder', 'right_hip', 'right_knee')
        ]
        
        for p1, p2, p3 in joint_pairs:
            if p1 in keypoints and p2 in keypoints and p3 in keypoints:
                angle = self._calculate_angle(
                    (keypoints[p1]['x'], keypoints[p1]['y']),
                    (keypoints[p2]['x'], keypoints[p2]['y']),
                    (keypoints[p3]['x'], keypoints[p3]['y'])
                )
                angles[f"{p1}_{p2}_{p3}"] = angle
        
        return angles
    
    def _calculate_angle(self, a: Tuple, b: Tuple, c: Tuple) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)


# ------------------------
# ADVANCED ACTIVITY RECOGNITION ENGINE
# ------------------------
class ActivityRecognitionEngine:
    def __init__(self, config: Optional[ActivityConfig] = None):
        self.config = config or ActivityConfig()
        
        # State tracking
        self.last_activity: Optional[str] = None
        self.last_detected_time: Optional[datetime] = None
        self.prev_frame = None
        self._running = False
        
        # Motion tracking
        self.motion_history = deque(maxlen=self.config.history_size)
        self.velocity_history = deque(maxlen=self.config.history_size)
        self.acceleration_history = deque(maxlen=self.config.history_size)
        
        # Activity smoothing
        self.activity_buffer = deque(maxlen=self.config.smoothing_window)
        self.activity_timestamps = deque(maxlen=self.config.smoothing_window)
        
        # Alert system
        self.last_alert_time: Optional[datetime] = None
        self.alert_history = deque(maxlen=50)
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Advanced components
        self.pose_engine = PoseEstimationEngine(self.config) if self.config.enable_pose else None
        
        # Temporal features
        self.keypoint_histories = {}
        self.gesture_buffer = deque(maxlen=30)
        
        # ML fallback (simple rule-based)
        self.activity_thresholds = {
            ActivityType.IDLE: (0, 500),
            ActivityType.WALKING: (500, 3000),
            ActivityType.RUNNING: (3000, 10000),
            ActivityType.JUMPING: (10000, float('inf')),
        }
        
        logger.info(f"✅ Advanced Activity Recognition Engine Initialized - Mode: {self.config.mode.value}")
    
    # --------------------------------------------------
    # MOTION ANALYSIS (ADVANCED)
    # --------------------------------------------------
    def _compute_motion_features(self, gray: np.ndarray) -> Dict:
        """Extract advanced motion features"""
        if self.prev_frame is None:
            self.prev_frame = gray
            return {"motion_score": 0, "velocity": 0, "acceleration": 0, "direction": 0}
        
        # Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        
        # Multiple scales of motion
        motion_pyramid = []
        for scale in [1, 0.5, 0.25]:
            scaled = cv2.resize(diff, None, fx=scale, fy=scale)
            motion_pyramid.append(np.sum(scaled) / 255)
        
        # Optical flow (dense)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Flow magnitude and direction
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        
        motion_score = np.mean(magnitude) * 1000
        
        # Velocity and acceleration
        current_velocity = motion_score
        if len(self.velocity_history) > 0:
            acceleration = current_velocity - self.velocity_history[-1]
        else:
            acceleration = 0
        
        self.velocity_history.append(current_velocity)
        self.acceleration_history.append(acceleration)
        
        # Motion direction histogram
        direction_hist, _ = np.histogram(direction, bins=8)
        dominant_direction = np.argmax(direction_hist) * 45 if len(direction_hist) > 0 else 0
        
        # Update state
        self.prev_frame = gray
        
        return {
            "motion_score": float(motion_score),
            "velocity": float(current_velocity),
            "acceleration": float(acceleration),
            "direction": float(dominant_direction),
            "pyramid_scores": [float(s) for s in motion_pyramid],
            "mean_magnitude": float(np.mean(magnitude)),
            "std_magnitude": float(np.std(magnitude))
        }
    
    def _smooth_motion(self, motion_score: float) -> float:
        """Apply temporal smoothing to motion"""
        self.motion_history.append(motion_score)
        
        if len(self.motion_history) >= 3:
            # Apply Gaussian weights
            weights = np.exp(-0.5 * (np.arange(len(self.motion_history)) - len(self.motion_history) + 1) ** 2 / (len(self.motion_history) / 2) ** 2)
            weights /= weights.sum()
            smoothed = np.sum(np.array(self.motion_history) * weights)
        else:
            smoothed = motion_score
        
        return smoothed
    
    # --------------------------------------------------
    # POSE-BASED ACTIVITY CLASSIFICATION
    # --------------------------------------------------
    def _classify_by_pose(self, keypoints: Dict) -> Tuple[str, float]:
        """Classify activity using pose keypoints"""
        if not keypoints or len(keypoints) < 5:
            return "unknown", 0.0
        
        # Calculate joint angles
        angles = self.pose_engine.calculate_angles(keypoints) if self.pose_engine else {}
        
        # Features for classification
        features = {
            'arm_angle_left': angles.get('left_shoulder_left_elbow_left_wrist', 0),
            'arm_angle_right': angles.get('right_shoulder_right_elbow_right_wrist', 0),
            'leg_angle_left': angles.get('left_hip_left_knee_left_ankle', 0),
            'leg_angle_right': angles.get('right_hip_right_knee_right_ankle', 0),
            'body_angle': angles.get('left_shoulder_left_hip_left_knee', 0),
        }
        
        # Simple rule-based classification
        confidence = 0.0
        activity = "unknown"
        
        # Check for sitting (knees bent ~90 degrees)
        if (70 < features['leg_angle_left'] < 110 or 70 < features['leg_angle_right'] < 110):
            activity = "sitting"
            confidence = 0.7
        
        # Check for standing (legs straight)
        elif (features['leg_angle_left'] > 160 or features['leg_angle_right'] > 160):
            # Check arm position for waving
            if features['arm_angle_left'] > 120 or features['arm_angle_right'] > 120:
                activity = "waving"
                confidence = 0.8
            else:
                activity = "standing"
                confidence = 0.6
        
        # Check for crouching
        elif (120 < features['leg_angle_left'] < 160 and 120 < features['leg_angle_right'] < 160):
            activity = "crouching"
            confidence = 0.7
        
        return activity, confidence
    
    # --------------------------------------------------
    # HYBRID ACTIVITY CLASSIFICATION
    # --------------------------------------------------
    def _classify_activity_hybrid(self, motion_features: Dict, pose_data: Dict) -> Tuple[str, float]:
        """Combine motion and pose for better classification"""
        motion_score = motion_features['motion_score']
        velocity = motion_features['velocity']
        acceleration = motion_features['acceleration']
        
        # Motion-based classification
        if motion_score < 800:
            base_activity = "idle"
            base_confidence = 0.8
        elif motion_score < 3000:
            base_activity = "walking"
            base_confidence = 0.7
        elif motion_score < 8000:
            base_activity = "running"
            base_confidence = 0.75
        else:
            base_activity = "jumping"
            base_confidence = 0.7
        
        # Refine with pose data
        if pose_data and pose_data.get('keypoints'):
            pose_activity, pose_confidence = self._classify_by_pose(pose_data['keypoints'])
            
            # Combine confidences
            if pose_activity != "unknown" and pose_confidence > 0.6:
                # Weighted average
                if base_activity == "idle" and pose_activity in ["sitting", "standing"]:
                    final_activity = pose_activity
                    final_confidence = (base_confidence + pose_confidence) / 2
                elif base_activity == "walking" and pose_activity == "standing":
                    final_activity = base_activity
                    final_confidence = base_confidence
                else:
                    final_activity = base_activity
                    final_confidence = base_confidence
            else:
                final_activity = base_activity
                final_confidence = base_confidence
        else:
            final_activity = base_activity
            final_confidence = base_confidence
        
        # Special case: falling detection
        if self.config.fall_detection_enabled:
            if self._detect_fall(motion_features, pose_data):
                final_activity = "falling"
                final_confidence = 0.85
        
        return final_activity, final_confidence
    
    def _detect_fall(self, motion_features: Dict, pose_data: Dict) -> bool:
        """Detect falling motion pattern"""
        # Rapid acceleration followed by low motion
        if len(self.acceleration_history) > 5:
            recent_acc = list(self.acceleration_history)[-5:]
            peak_acc = max(recent_acc)
            
            # Sudden high acceleration
            if peak_acc > 5000 and motion_features['motion_score'] < 1000:
                # Check for horizontal orientation in pose
                if pose_data and pose_data.get('keypoints'):
                    # Simple check: compare shoulder and hip heights
                    if 'left_shoulder' in pose_data['keypoints'] and 'left_hip' in pose_data['keypoints']:
                        shoulder_y = pose_data['keypoints']['left_shoulder']['y']
                        hip_y = pose_data['keypoints']['left_hip']['y']
                        
                        # Person is horizontal if shoulder and hip are at similar height
                        if abs(shoulder_y - hip_y) < 50:  # Threshold for orientation
                            return True
                
                return True
        
        return False
    
    # --------------------------------------------------
    # ACTIVITY SMOOTHING AND CONSISTENCY
    # --------------------------------------------------
    def _apply_temporal_smoothing(self, current_activity: str, confidence: float) -> str:
        """Apply temporal smoothing to avoid rapid activity changes"""
        current_time = time.time()
        
        # Add to buffer
        self.activity_buffer.append(current_activity)
        self.activity_timestamps.append(current_time)
        
        if len(self.activity_buffer) < self.config.smoothing_window:
            return current_activity
        
        # Get most common activity in window
        from collections import Counter
        activity_counts = Counter(self.activity_buffer)
        most_common = activity_counts.most_common(1)[0]
        
        # Check if activity has persisted long enough
        if most_common[1] >= self.config.smoothing_window * 0.6:
            return most_common[0]
        
        return self.last_activity or current_activity
    
    # --------------------------------------------------
    # GESTURE RECOGNITION
    # --------------------------------------------------
    def _recognize_gestures(self, keypoints_history: deque) -> List[str]:
        """Recognize gestures from keypoint sequence"""
        gestures = []
        
        if len(keypoints_history) < 10:
            return gestures
        
        # Extract wrist positions
        left_wrist_history = []
        right_wrist_history = []
        
        for frame_data in keypoints_history:
            if 'left_wrist' in frame_data:
                left_wrist_history.append((frame_data['left_wrist']['x'], frame_data['left_wrist']['y']))
            if 'right_wrist' in frame_data:
                right_wrist_history.append((frame_data['right_wrist']['x'], frame_data['right_wrist']['y']))
        
        # Detect waving (horizontal motion of wrists)
        if len(left_wrist_history) > 5:
            left_movement = np.std([p[0] for p in left_wrist_history])
            if left_movement > 50:
                gestures.append("left_wave")
        
        if len(right_wrist_history) > 5:
            right_movement = np.std([p[0] for p in right_wrist_history])
            if right_movement > 50:
                gestures.append("right_wave")
        
        # Detect clapping (wrists moving towards each other)
        if len(left_wrist_history) > 5 and len(right_wrist_history) > 5:
            distances = []
            for lw, rw in zip(left_wrist_history[-10:], right_wrist_history[-10:]):
                dist = np.sqrt((lw[0] - rw[0])**2 + (lw[1] - rw[1])**2)
                distances.append(dist)
            
            if np.std(distances) > 20 and np.min(distances) < 50:
                gestures.append("clapping")
        
        return gestures
    
    # --------------------------------------------------
    # ALERT SYSTEM
    # --------------------------------------------------
    def _check_alerts(self, activity: str, confidence: float) -> Optional[Dict]:
        """Generate alerts for suspicious activities"""
        current_time = time.time()
        
        # Check cooldown
        if self.last_alert_time:
            if current_time - self.last_alert_time < self.config.alert_cooldown:
                return None
        
        # Check if activity is suspicious
        if activity in self.config.suspicious_activities and confidence > 0.6:
            self.last_alert_time = datetime.now()
            
            alert = {
                "type": "suspicious_activity",
                "activity": activity,
                "confidence": confidence,
                "timestamp": self.last_alert_time.isoformat(),
                "message": f"Suspicious activity detected: {activity}"
            }
            
            self.alert_history.append(alert)
            return alert
        
        return None
    
    # --------------------------------------------------
    # MAIN PROCESSING FRAME
    # --------------------------------------------------
    async def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with advanced activity recognition"""
        if frame is None:
            return {}
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Convert to grayscale for motion analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract motion features
            motion_features = self._compute_motion_features(gray)
            smoothed_motion = self._smooth_motion(motion_features['motion_score'])
            motion_features['motion_score'] = smoothed_motion
            
            # Pose estimation (if enabled)
            pose_data = {}
            if self.config.enable_pose and self.pose_engine:
                keypoints = self.pose_engine.estimate_pose(frame)
                if keypoints:
                    pose_data = {
                        'keypoints': keypoints,
                        'angles': self.pose_engine.calculate_angles(keypoints)
                    }
                    
                    # Store for gesture recognition
                    self.keypoint_histories[self.frame_count % 100] = keypoints
                    if len(self.keypoint_histories) > 50:
                        # Convert to deque for gesture recognition
                        recent_keypoints = deque(list(self.keypoint_histories.values())[-30:], maxlen=30)
                        gestures = self._recognize_gestures(recent_keypoints)
                        if gestures:
                            pose_data['gestures'] = gestures
            
            # Classify activity based on mode
            if self.config.mode == ActivityDetectionMode.MOTION_BASED:
                activity, confidence = self._classify_activity_basic(smoothed_motion)
            elif self.config.mode == ActivityDetectionMode.POSE_BASED and pose_data:
                activity, confidence = self._classify_by_pose(pose_data.get('keypoints', {}))
            else:  # HYBRID or default
                activity, confidence = self._classify_activity_hybrid(motion_features, pose_data)
            
            # Apply temporal smoothing
            smoothed_activity = self._apply_temporal_smoothing(activity, confidence)
            
            # Update state
            self.last_activity = smoothed_activity
            self.last_detected_time = datetime.now()
            
            # Check for alerts
            alert = None
            if self.config.enable_alerting:
                alert = self._check_alerts(smoothed_activity, confidence)
            
            # Prepare result
            result = {
                "activity": smoothed_activity,
                "raw_activity": activity,
                "confidence": float(confidence),
                "motion_level": int(smoothed_motion),
                "velocity": float(motion_features['velocity']),
                "acceleration": float(motion_features['acceleration']),
                "timestamp": self.last_detected_time.isoformat(),
                "frame_id": self.frame_count,
                "pose_data": pose_data,
                "alert": alert
            }
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Activity recognition error: {e}")
            return {}
    
    def _classify_activity_basic(self, motion_score: float) -> Tuple[str, float]:
        """Basic motion-based classification"""
        if motion_score < 800:
            return "idle", 0.8
        elif motion_score < 3000:
            return "walking", 0.7
        elif motion_score < 8000:
            return "running", 0.75
        else:
            return "jumping", 0.7
    
    # --------------------------------------------------
    # MONITOR LOOP (ADVANCED)
    # --------------------------------------------------
    async def monitor(self, frame_source_callable, callback: Optional[callable] = None):
        """Advanced monitoring loop with callbacks"""
        self._running = True
        logger.info("🎥 Advanced Activity Monitoring Started")
        
        frame_times = deque(maxlen=30)
        
        try:
            while self._running:
                frame_start = time.time()
                
                # Get frame
                frame = await frame_source_callable()
                
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process frame
                result = await self.process_frame(frame)
                
                # Execute callback if provided
                if callback and result:
                    await callback(result) if asyncio.iscoroutinefunction(callback) else callback(result)
                
                # Calculate and log FPS periodically
                frame_times.append(time.time() - frame_start)
                if len(frame_times) == 30 and self.frame_count % 30 == 0:
                    avg_fps = 30.0 / sum(frame_times) if sum(frame_times) > 0 else 0
                    logger.debug(f"Activity Recognition FPS: {avg_fps:.1f}")
                
                # Adaptive sleep
                await asyncio.sleep(max(0, 1.0/self.config.fps_target - (time.time() - frame_start)))
                
        except asyncio.CancelledError:
            logger.info("Activity monitor cancelled")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        finally:
            self._running = False
            logger.info("🛑 Activity monitoring stopped")
    
    # --------------------------------------------------
    # UTILITY METHODS
    # --------------------------------------------------
    def stop(self):
        """Stop monitoring"""
        self._running = False
    
    def get_current_activity(self) -> Dict:
        """Get current activity state"""
        return {
            "activity": self.last_activity,
            "last_detected": self.last_detected_time.isoformat() if self.last_detected_time else None,
            "frame_count": self.frame_count,
            "buffer_size": len(self.activity_buffer)
        }
    
    def is_suspicious(self) -> bool:
        """Check if current activity is suspicious"""
        return self.last_activity in self.config.suspicious_activities if self.last_activity else False
    
    def draw_activity(self, frame: np.ndarray, activity_data: Dict) -> np.ndarray:
        """Draw advanced activity overlay"""
        if frame is None or cv2 is None:
            return frame
        
        try:
            activity = activity_data.get("activity", "unknown")
            confidence = activity_data.get("confidence", 0)
            motion_level = activity_data.get("motion_level", 0)
            
            # Main activity text
            color = (0, 255, 0) if not self.is_suspicious() else (0, 0, 255)
            cv2.putText(
                frame,
                f"Activity: {activity} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            
            # Motion level bar
            bar_width = 200
            bar_height = 10
            motion_percent = min(1.0, motion_level / 10000)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 50 + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 50), (10 + int(bar_width * motion_percent), 50 + bar_height), (0, 255, 0), -1)
            
            # Velocity and acceleration
            velocity = activity_data.get("velocity", 0)
            acceleration = activity_data.get("acceleration", 0)
            
            cv2.putText(
                frame,
                f"Vel: {velocity:.0f} | Acc: {acceleration:.0f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # Gestures if present
            if activity_data.get("pose_data", {}).get("gestures"):
                gestures = ", ".join(activity_data["pose_data"]["gestures"])
                cv2.putText(
                    frame,
                    f"Gesture: {gestures}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 0),
                    1
                )
            
            # Alert indicator
            if activity_data.get("alert"):
                cv2.putText(
                    frame,
                    "⚠️ ALERT!",
                    (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            
        except Exception as e:
            logger.debug(f"Draw error: {e}")
        
        return frame
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing if avg_processing > 0 else 0
        
        return {
            "mode": self.config.mode.value,
            "pose_enabled": self.config.enable_pose,
            "frame_count": self.frame_count,
            "last_activity": self.last_activity,
            "history_size": len(self.motion_history),
            "buffer_size": len(self.activity_buffer),
            "avg_processing_time_ms": avg_processing * 1000,
            "fps": fps,
            "alerts_generated": len(self.alert_history),
            "alert_rate": len(self.alert_history) / max(1, self.frame_count) * 100,
            "config": {
                "smoothing_window": self.config.smoothing_window,
                "fall_detection": self.config.fall_detection_enabled,
                "suspicious_activities": self.config.suspicious_activities
            }
        }
    
    def export_activity_log(self, filepath: str):
        """Export activity history to JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "last_activity": self.last_activity,
            "activity_history": list(self.activity_buffer),
            "alert_history": list(self.alert_history),
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Activity log exported to {filepath}")
    
    def reset(self):
        """Reset all state and history"""
        self.motion_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.activity_buffer.clear()
        self.activity_timestamps.clear()
        self.alert_history.clear()
        self.keypoint_histories.clear()
        self.processing_times.clear()
        self.prev_frame = None
        self.last_activity = None
        self.frame_count = 0
        logger.info("🔄 Activity recognition engine reset")


# --------------------------------------------------
# ASYNC CALLBACK EXAMPLE
# --------------------------------------------------
async def alert_callback(activity_data: Dict):
    """Example callback for alerts"""
    if activity_data.get("alert"):
        print(f"🚨 ALERT: {activity_data['alert']['message']}")


# --------------------------------------------------
# USAGE EXAMPLE
# --------------------------------------------------
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure advanced activity recognition
    config = ActivityConfig(
        mode=ActivityDetectionMode.HYBRID,
        enable_pose=True,
        enable_alerting=True,
        smoothing_window=10,
        fall_detection_enabled=True,
        suspicious_activities=["running", "falling", "jumping"]
    )
    
    # Initialize engine
    engine = ActivityRecognitionEngine(config)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    async def frame_source():
        ret, frame = cap.read()
        return frame if ret else None
    
    async def main():
        print("🎥 Starting activity recognition (press 'q' to quit)")
        
        # Start monitoring with callback
        monitor_task = asyncio.create_task(
            engine.monitor(frame_source, callback=alert_callback)
        )
        
        # Main display loop
        while True:
            frame = await frame_source()
            if frame is None:
                break
            
            # Process frame
            result = await engine.process_frame(frame)
            
            # Draw overlay
            annotated = engine.draw_activity(frame, result)
            
            # Show stats periodically
            if engine.frame_count % 60 == 0:
                stats = engine.get_stats()
                print(f"📊 FPS: {stats['fps']:.1f}, Activity: {result.get('activity', 'unknown')}")
            
            cv2.imshow('Advanced Activity Recognition', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        engine.stop()
        monitor_task.cancel()
        cap.release()
        cv2.destroyAllWindows()
        
        # Export logs
        engine.export_activity_log("activity_log.json")
        print(json.dumps(engine.get_stats(), indent=2))
    
    # Run async main
    asyncio.run(main())