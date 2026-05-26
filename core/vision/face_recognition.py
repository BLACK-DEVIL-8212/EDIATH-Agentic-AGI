# face_recognition.py

import cv2
import numpy as np
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False


# ------------------------
# ENUMS & CONFIGURATIONS
# ------------------------
class FaceRecognitionModel(Enum):
    HOG = "hog"  # face_recognition default
    CNN = "cnn"  # face_recognition CNN
    FACENET = "facenet"
    ARC_FACE = "arcface"
    DEEP_FACE = "deepface"
    RETINAFACE = "retinaface"


class AntiSpoofingMethod(Enum):
    NONE = "none"
    BLINK = "blink"
    HEAD_POSE = "head_pose"
    TEXTURE = "texture"
    DEPTH = "depth"
    MULTI = "multi"


@dataclass
class FaceRecognitionConfig:
    model: FaceRecognitionModel = FaceRecognitionModel.HOG
    anti_spoofing: AntiSpoofingMethod = AntiSpoofingMethod.MULTI
    confidence_threshold: float = 0.6
    distance_threshold: float = 0.5
    min_face_size: int = 40
    max_faces: int = 10
    enable_landmarks: bool = True
    enable_emotion: bool = True
    enable_age_gender: bool = True
    enable_tracking: bool = True
    track_lifetime: int = 30  # frames
    detection_interval: int = 5  # detect every N frames
    use_gpu: bool = True
    cache_size: int = 100
    data_dir: str = "./faces"


# ------------------------
# FACIAL LANDMARKS
# ------------------------
class FacialLandmarks:
    def __init__(self, landmarks: Dict[str, Tuple[int, int]]):
        self.landmarks = landmarks
        
    def get_eye_centers(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get left and right eye centers"""
        left_eye = self.landmarks.get('left_eye', (0, 0))
        right_eye = self.landmarks.get('right_eye', (0, 0))
        return left_eye, right_eye
    
    def get_nose_tip(self) -> Tuple[int, int]:
        return self.landmarks.get('nose_tip', (0, 0))
    
    def get_mouth_center(self) -> Tuple[int, int]:
        return self.landmarks.get('mouth_center', (0, 0))
    
    def calculate_eye_aspect_ratio(self) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        left_eye = self.landmarks.get('left_eye', None)
        if left_eye:
            # Simplified EAR calculation
            return 0.3  # Placeholder
        return 0.0
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw landmarks on frame"""
        if cv2 is None:
            return frame
        
        for name, (x, y) in self.landmarks.items():
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(frame, name[:3], (x + 5, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame


# ------------------------
# FACE TRACKER
# ------------------------
class FaceTracker:
    def __init__(self, lifetime: int = 30):
        self.tracks = {}
        self.next_id = 0
        self.lifetime = lifetime
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        current_time = time.time()
        
        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.lifetime:
                del self.tracks[track_id]
        
        # Match detections to existing tracks (simple IoU matching)
        matched = set()
        for det in detections:
            best_iou = 0
            best_id = None
            
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(det['location'], track['last_location'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                det['track_id'] = best_id
                self.tracks[best_id]['last_location'] = det['location']
                self.tracks[best_id]['age'] = 0
                matched.add(best_id)
            else:
                det['track_id'] = self.next_id
                self.tracks[self.next_id] = {
                    'id': self.next_id,
                    'last_location': det['location'],
                    'age': 0,
                    'first_seen': current_time
                }
                self.next_id += 1
        
        return detections
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2
        
        x1 = max(left1, left2)
        y1 = max(top1, top2)
        x2 = min(right1, right2)
        y2 = min(bottom1, bottom2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


# ------------------------
# ANTI-SPOOFING ENGINE
# ------------------------
class AntiSpoofingEngine:
    def __init__(self, method: AntiSpoofingMethod):
        self.method = method
        self.blink_history = deque(maxlen=10)
        self.head_pose_history = deque(maxlen=10)
        
    def detect_spoof(self, frame: np.ndarray, landmarks: Optional[FacialLandmarks]) -> Tuple[bool, float]:
        """Detect if face is real or spoof"""
        if self.method == AntiSpoofingMethod.NONE:
            return True, 1.0
        
        if self.method == AntiSpoofingMethod.BLINK:
            return self._detect_blink_spoof(landmarks)
        
        elif self.method == AntiSpoofingMethod.TEXTURE:
            return self._detect_texture_spoof(frame)
        
        elif self.method == AntiSpoofingMethod.MULTI:
            return self._multi_modal_detection(frame, landmarks)
        
        return True, 0.8
    
    def _detect_blink_spoof(self, landmarks: Optional[FacialLandmarks]) -> Tuple[bool, float]:
        """Detect spoof by checking for blinks"""
        if not landmarks:
            return True, 0.5
        
        ear = landmarks.calculate_eye_aspect_ratio()
        self.blink_history.append(ear)
        
        # Check for blink pattern in history
        if len(self.blink_history) > 5:
            blink_variance = np.var(self.blink_history)
            has_blinks = blink_variance > 0.01
            return has_blinks, 0.7 if has_blinks else 0.3
        
        return True, 0.5
    
    def _detect_texture_spoof(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Detect spoof using texture analysis"""
        if frame is None:
            return True, 0.5
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Calculate LBP-like texture features
        texture_score = np.std(gray) / 255.0
        
        # Real faces have medium texture variation
        is_real = 0.1 < texture_score < 0.8
        confidence = 0.6 if is_real else 0.4
        
        return is_real, confidence
    
    def _multi_modal_detection(self, frame: np.ndarray, landmarks: Optional[FacialLandmarks]) -> Tuple[bool, float]:
        """Combine multiple spoof detection methods"""
        blink_real, blink_conf = self._detect_blink_spoof(landmarks)
        texture_real, texture_conf = self._detect_texture_spoof(frame)
        
        # Weighted combination
        final_conf = (blink_conf * 0.6 + texture_conf * 0.4)
        is_real = blink_real and texture_real
        
        return is_real, final_conf


# ------------------------
# EMOTION, AGE, GENDER
# ------------------------
class FaceAttributes:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.last_analysis = {}
        
    def analyze(self, face_img: np.ndarray) -> Dict:
        """Analyze face for emotion, age, gender"""
        if not DEEPFACE_AVAILABLE or face_img is None:
            return {}
        
        try:
            # Analyze with DeepFace
            results = DeepFace.analyze(
                face_img,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            if results:
                result = results[0] if isinstance(results, list) else results
                return {
                    'emotion': result.get('dominant_emotion', 'unknown'),
                    'emotion_scores': result.get('emotion', {}),
                    'age': result.get('age', 0),
                    'gender': result.get('gender', 'unknown'),
                    'gender_confidence': result.get('gender_confidence', 0)
                }
        except Exception as e:
            logger.debug(f"Attribute analysis error: {e}")
        
        return {}
    
    def draw_attributes(self, frame: np.ndarray, attributes: Dict, position: Tuple[int, int]) -> np.ndarray:
        """Draw attributes on frame"""
        if cv2 is None:
            return frame
        
        y_offset = position[1]
        for key, value in attributes.items():
            if key not in ['emotion_scores']:
                text = f"{key}: {value}"
                cv2.putText(frame, text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 20
        
        return frame


# ------------------------
# ENHANCED FACE RECOGNITION ENGINE
# ------------------------
class FaceRecognitionEngine:
    def __init__(self, config: Optional[FaceRecognitionConfig] = None):
        self.config = config or FaceRecognitionConfig()
        
        # Data storage
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Known faces
        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.known_metadata: List[Dict] = []
        
        # State
        self._loaded = False
        self._lock = asyncio.Lock()
        self.frame_count = 0
        
        # Advanced components
        self.tracker = FaceTracker(self.config.track_lifetime) if self.config.enable_tracking else None
        self.anti_spoofing = AntiSpoofingEngine(self.config.anti_spoofing)
        self.attribute_analyzer = FaceAttributes() if (self.config.enable_emotion or self.config.enable_age_gender) else None
        
        # Performance
        self.processing_times = deque(maxlen=100)
        self.recognition_cache = {}
        
        # Statistics
        self.total_detections = 0
        self.recognized_faces = 0
        self.unknown_faces = 0
        
        logger.info(f"✅ Advanced Face Recognition Engine Initialized - Model: {self.config.model.value}")
    
    # ------------------------
    # FACE DETECTION (MULTI-MODEL)
    # ------------------------
    async def _detect_faces(self, rgb_frame: np.ndarray) -> Tuple[List, List]:
        """Detect faces using configured model"""
        if self.config.model == FaceRecognitionModel.HOG:
            return await self._detect_hog(rgb_frame)
        elif self.config.model == FaceRecognitionModel.CNN:
            return await self._detect_cnn(rgb_frame)
        elif self.config.model == FaceRecognitionModel.RETINAFACE and RETINAFACE_AVAILABLE:
            return await self._detect_retinaface(rgb_frame)
        else:
            return await self._detect_hog(rgb_frame)
    
    async def _detect_hog(self, rgb_frame: np.ndarray) -> Tuple[List, List]:
        """HOG-based face detection"""
        if not FACE_RECOGNITION_AVAILABLE:
            return [], []
        
        def detect():
            locations = face_recognition.face_locations(rgb_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            return locations, encodings
        
        return await asyncio.to_thread(detect)
    
    async def _detect_cnn(self, rgb_frame: np.ndarray) -> Tuple[List, List]:
        """CNN-based face detection (more accurate)"""
        if not FACE_RECOGNITION_AVAILABLE:
            return [], []
        
        def detect():
            locations = face_recognition.face_locations(rgb_frame, model="cnn")
            encodings = face_recognition.face_encodings(rgb_frame, locations)
            return locations, encodings
        
        return await asyncio.to_thread(detect)
    
    async def _detect_retinaface(self, rgb_frame: np.ndarray) -> Tuple[List, List]:
        """RetinaFace detection with landmarks"""
        if not RETINAFACE_AVAILABLE:
            return [], []
        
        def detect():
            faces = RetinaFace.detect_faces(rgb_frame)
            locations = []
            encodings = []
            
            for face_id, face_data in faces.items():
                area = face_data['area']
                x1, y1, x2, y2 = area.astype(int)
                locations.append((y1, x2, y2, x1))  # Convert to face_recognition format
                
                # Extract face encoding (simplified)
                face_img = rgb_frame[y1:y2, x1:x2]
                if FACE_RECOGNITION_AVAILABLE and face_img.size > 0:
                    enc = face_recognition.face_encodings(face_img)
                    if enc:
                        encodings.append(enc[0])
                    else:
                        encodings.append(np.zeros(128))
                else:
                    encodings.append(np.zeros(128))
            
            return locations, encodings
        
        return await asyncio.to_thread(detect)
    
    # ------------------------
    # FACIAL LANDMARKS
    # ------------------------
    async def _extract_landmarks(self, rgb_frame: np.ndarray, location: Tuple) -> Optional[FacialLandmarks]:
        """Extract facial landmarks"""
        if not self.config.enable_landmarks or not FACE_RECOGNITION_AVAILABLE:
            return None
        
        try:
            def extract():
                landmarks = face_recognition.face_landmarks(rgb_frame, [location])
                if landmarks and len(landmarks) > 0:
                    return landmarks[0]
                return {}
            
            landmarks_dict = await asyncio.to_thread(extract)
            
            if landmarks_dict:
                # Convert to standard format
                formatted = {}
                for key, points in landmarks_dict.items():
                    if points:
                        # Use first point for simplicity
                        formatted[key] = (points[0][0], points[0][1])
                return FacialLandmarks(formatted)
            
        except Exception as e:
            logger.debug(f"Landmark extraction error: {e}")
        
        return None
    
    # ------------------------
    # FACE RECOGNITION
    # ------------------------
    async def _recognize_face(self, encoding: np.ndarray) -> Tuple[str, float]:
        """Recognize face against known encodings"""
        if not self.known_encodings:
            return "Unknown", 0.0
        
        # Check cache first
        cache_key = hashlib.md5(encoding.tobytes()).hexdigest()
        if cache_key in self.recognition_cache:
            cached_name, cached_conf = self.recognition_cache[cache_key]
            if time.time() - self.recognition_cache.get(f"{cache_key}_time", 0) < 60:
                return cached_name, cached_conf
        
        def compute_distances():
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            return distances
        
        distances = await asyncio.to_thread(compute_distances)
        
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        
        if best_distance < self.config.distance_threshold:
            name = self.known_names[best_match_idx]
            confidence = 1 - best_distance
            
            # Update cache
            if len(self.recognition_cache) > self.config.cache_size:
                self.recognition_cache.clear()
            self.recognition_cache[cache_key] = (name, confidence)
            self.recognition_cache[f"{cache_key}_time"] = time.time()
            
            return name, float(confidence)
        
        return "Unknown", float(max(0, 1 - best_distance))
    
    # ------------------------
    # MAIN RECOGNITION PIPELINE
    # ------------------------
    async def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        """Advanced face recognition with all features"""
        if frame is None:
            return []
        
        start_time = time.time()
        self.frame_count += 1
        
        if not self._loaded:
            await self.load_known_faces()
        
        try:
            # Skip detection every N frames for performance
            if self.config.detection_interval > 1 and self.frame_count % self.config.detection_interval != 0:
                return []
            
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            locations, encodings = await self._detect_faces(rgb)
            
            results = []
            
            for (top, right, bottom, left), encoding in zip(locations, encodings):
                # Skip tiny faces
                height = bottom - top
                if height < self.config.min_face_size:
                    continue
                
                # Extract face region
                face_region = rgb[top:bottom, left:right] if top >= 0 and bottom > top and left >= 0 and right > left else None
                
                # Anti-spoofing check
                landmarks = await self._extract_landmarks(rgb, (top, right, bottom, left))
                is_real, spoof_confidence = self.anti_spoofing.detect_spoof(frame, landmarks)
                
                if not is_real and spoof_confidence < 0.5:
                    continue  # Skip spoof faces
                
                # Recognize face
                name, confidence = await self._recognize_face(encoding)
                
                # Extract attributes
                attributes = {}
                if self.attribute_analyzer and face_region is not None:
                    attributes = await asyncio.to_thread(
                        self.attribute_analyzer.analyze, 
                        face_region
                    )
                
                face_data = {
                    "name": name,
                    "confidence": round(confidence, 2),
                    "location": (top, right, bottom, left),
                    "spoof_score": round(spoof_confidence, 2),
                    "attributes": attributes,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add landmarks if available
                if landmarks:
                    face_data["landmarks"] = landmarks.landmarks
                
                results.append(face_data)
                
                # Update statistics
                self.total_detections += 1
                if name != "Unknown":
                    self.recognized_faces += 1
                else:
                    self.unknown_faces += 1
            
            # Apply tracking
            if self.tracker and self.config.enable_tracking:
                results = self.tracker.update(results)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return []
    
    # ------------------------
    # FACE MANAGEMENT
    # ------------------------
    async def add_face(self, name: str, frame: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Add new face to database with metadata"""
        if frame is None:
            return False
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            locations, encodings = await self._detect_faces(rgb)
            
            if not encodings:
                logger.warning(f"No face detected for {name}")
                return False
            
            # Use first detected face
            encoding = encodings[0]
            location = locations[0]
            
            # Extract face region for attribute analysis
            top, right, bottom, left = location
            face_region = rgb[top:bottom, left:right]
            
            # Analyze attributes if available
            face_metadata = metadata or {}
            if self.attribute_analyzer and face_region is not None:
                attributes = await asyncio.to_thread(
                    self.attribute_analyzer.analyze,
                    face_region
                )
                face_metadata.update(attributes)
            
            # Save to database
            async with self._lock:
                self.known_names.append(name)
                self.known_encodings.append(encoding)
                self.known_metadata.append(face_metadata)
                
                await self._save_face_data(name, encoding, face_metadata, face_region)
            
            logger.info(f"✅ Face added: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Add face error: {e}")
            return False
    
    async def _save_face_data(self, name: str, encoding: np.ndarray, metadata: Dict, face_image: np.ndarray):
        """Save face data to disk"""
        data = {
            "name": name,
            "encoding": encoding.tolist(),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        # Save encoding
        encoding_path = self.data_dir / f"{name}_encoding.json"
        
        def write_encoding():
            with open(encoding_path, "w") as f:
                json.dump(data, f, indent=2)
        
        await asyncio.to_thread(write_encoding)
        
        # Save face image
        if face_image is not None:
            image_path = self.data_dir / f"{name}_face.jpg"
            
            def write_image():
                cv2.imwrite(str(image_path), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            
            await asyncio.to_thread(write_image)
    
    async def load_known_faces(self):
        """Load all known faces from database"""
        async with self._lock:
            self.known_encodings.clear()
            self.known_names.clear()
            self.known_metadata.clear()
            
            for file in self.data_dir.glob("*_encoding.json"):
                try:
                    def load():
                        with open(file, "r") as f:
                            return json.load(f)
                    
                    data = await asyncio.to_thread(load)
                    encoding = np.array(data["encoding"])
                    
                    if encoding.shape != (128,):
                        continue
                    
                    self.known_names.append(data["name"])
                    self.known_encodings.append(encoding)
                    self.known_metadata.append(data.get("metadata", {}))
                    
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
            
            self._loaded = True
            logger.info(f"📚 Loaded {len(self.known_names)} known faces")
    
    async def remove_face(self, name: str) -> bool:
        """Remove face from database"""
        async with self._lock:
            if name in self.known_names:
                idx = self.known_names.index(name)
                self.known_names.pop(idx)
                self.known_encodings.pop(idx)
                self.known_metadata.pop(idx)
                
                # Delete files
                encoding_path = self.data_dir / f"{name}_encoding.json"
                image_path = self.data_dir / f"{name}_face.jpg"
                
                def delete_files():
                    if encoding_path.exists():
                        encoding_path.unlink()
                    if image_path.exists():
                        image_path.unlink()
                
                await asyncio.to_thread(delete_files)
                logger.info(f"🗑️ Removed face: {name}")
                return True
            
            return False
    
    async def verify_face(self, frame: np.ndarray, expected_name: str) -> Tuple[bool, float]:
        """Verify if face in frame matches expected person"""
        faces = await self.recognize_faces(frame)
        
        for face in faces:
            if face["name"] == expected_name:
                return True, face["confidence"]
        
        return False, 0.0
    
    # ------------------------
    # VISUALIZATION
    # ------------------------
    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Draw advanced face visualization"""
        if cv2 is None:
            return frame
        
        for face in faces:
            top, right, bottom, left = face["location"]
            name = face["name"]
            confidence = face.get("confidence", 0)
            track_id = face.get("track_id")
            attributes = face.get("attributes", {})
            spoof_score = face.get("spoof_score", 1.0)
            
            # Color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw track ID if available
            if track_id is not None:
                cv2.putText(frame, f"ID:{track_id}", (left, top - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw spoof indicator
            if spoof_score < 0.5:
                cv2.putText(frame, "⚠️ SPOOF", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw attributes
            if attributes:
                y_offset = bottom + 20
                if "emotion" in attributes:
                    cv2.putText(frame, f"😊 {attributes['emotion']}", (left, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                if "age" in attributes:
                    cv2.putText(frame, f"🎂 {int(attributes['age'])}", (left, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                if "gender" in attributes:
                    gender_symbol = "♂️" if attributes['gender'] == 'Male' else "♀️"
                    cv2.putText(frame, f"{gender_symbol} {attributes['gender']}", (left, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw landmarks if available
            if "landmarks" in face:
                for point_name, (x, y) in face["landmarks"].items():
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        
        return frame
    
    # ------------------------
    # STATISTICS & EXPORT
    # ------------------------
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            "known_faces": len(self.known_names),
            "names": self.known_names,
            "loaded": self._loaded,
            "total_detections": self.total_detections,
            "recognized_faces": self.recognized_faces,
            "unknown_faces": self.unknown_faces,
            "recognition_rate": self.recognized_faces / max(1, self.total_detections),
            "avg_processing_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "cache_size": len(self.recognition_cache),
            "config": {
                "model": self.config.model.value,
                "anti_spoofing": self.config.anti_spoofing.value,
                "enable_emotion": self.config.enable_emotion,
                "enable_age_gender": self.config.enable_age_gender,
                "enable_tracking": self.config.enable_tracking
            }
        }
    
    def export_database(self, filepath: str):
        """Export face database to JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_faces": len(self.known_names),
            "faces": [
                {
                    "name": name,
                    "metadata": metadata,
                    "encoding_shape": encoding.shape
                }
                for name, encoding, metadata in zip(self.known_names, self.known_encodings, self.known_metadata)
            ],
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📁 Database exported to {filepath}")
    
    async def train_model(self, method: str = "svm"):
        """Train a custom classifier for better recognition"""
        if len(self.known_encodings) < 2:
            logger.warning("Need at least 2 faces to train")
            return False
        
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import LabelEncoder
            
            X = np.array(self.known_encodings)
            y = self.known_names
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Train SVM classifier
            clf = SVC(kernel='rbf', probability=True, C=1.0)
            clf.fit(X, y_encoded)
            
            # Store classifier
            self.classifier = clf
            self.label_encoder = le
            
            logger.info("✅ Classifier trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def reset_stats(self):
        """Reset all statistics"""
        self.total_detections = 0
        self.recognized_faces = 0
        self.unknown_faces = 0
        self.processing_times.clear()
        logger.info("Statistics reset")


# ------------------------
# USAGE EXAMPLE
# ------------------------
async def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure face recognition
    config = FaceRecognitionConfig(
        model=FaceRecognitionModel.HOG,
        anti_spoofing=AntiSpoofingMethod.MULTI,
        confidence_threshold=0.6,
        distance_threshold=0.5,
        enable_landmarks=True,
        enable_emotion=True,
        enable_age_gender=True,
        enable_tracking=True
    )
    
    # Initialize engine
    engine = FaceRecognitionEngine(config)
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    print("🎥 Face Recognition Started (press 'q' to quit)")
    print("   'a' - Add face, 'r' - Remove face, 's' - Show stats")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Recognize faces
        faces = await engine.recognize_faces(frame)
        
        # Draw results
        annotated = engine.draw_faces(frame.copy(), faces)
        
        # Show FPS and stats
        stats = engine.get_stats()
        cv2.putText(annotated, f"FPS: {stats['fps']:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Known: {stats['known_faces']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Advanced Face Recognition', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Add current face
            name = input("Enter name: ")
            if await engine.add_face(name, frame):
                print(f"✅ Added {name} to database")
        elif key == ord('r'):
            # Remove face
            name = input("Enter name to remove: ")
            if await engine.remove_face(name):
                print(f"🗑️ Removed {name}")
        elif key == ord('s'):
            # Show stats
            print(json.dumps(engine.get_stats(), indent=2))
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Export database
    engine.export_database("face_database_export.json")
    print("\nFinal Statistics:")
    print(json.dumps(engine.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())