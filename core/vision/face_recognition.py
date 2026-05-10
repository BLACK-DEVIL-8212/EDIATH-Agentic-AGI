# face_recognition.py

import cv2
import numpy as np
import asyncio
import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionEngine:
    def __init__(self, data_dir: str = "./faces"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []

        self._loaded = False
        self._lock = asyncio.Lock()

        logger.info("✅ Face Recognition Engine Initialized")

    # ------------------------
    # LOAD FACES
    # ------------------------
    async def load_known_faces(self):
        async with self._lock:
            self.known_encodings.clear()
            self.known_names.clear()

            for file in self.data_dir.glob("*.json"):
                try:
                    with open(file, "r") as f:
                        data = json.load(f)

                    enc = np.array(data["encoding"])

                    if enc.shape != (128,):
                        continue  # skip invalid encoding

                    self.known_names.append(data["name"])
                    self.known_encodings.append(enc)

                except Exception as e:
                    logger.exception("Error loading face file %s: %s", file, e)
                    continue

            self._loaded = True
            logger.info("Loaded %d faces", len(self.known_names))

    # ------------------------
    # ADD FACE
    # ------------------------
    async def add_face(self, name: str, frame: np.ndarray) -> bool:
        if frame is None:
            return False

        try:
            import face_recognition

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            encodings = await asyncio.to_thread(face_recognition.face_encodings, rgb)

            if not encodings:
                return False

            encoding = encodings[0]

            async with self._lock:
                self.known_names.append(name)
                self.known_encodings.append(encoding)

                await self._save_encoding(name, encoding)

            return True

        except Exception as e:
            logger.exception("Add face error: %s", e)
            return False

    # ------------------------
    # SAVE
    # ------------------------
    async def _save_encoding(self, name, encoding):
        data = {"name": name, "encoding": encoding.tolist()}
        path = self.data_dir / f"{name}.json"

        def write():
            with open(path, "w") as f:
                json.dump(data, f)

        await asyncio.to_thread(write)

    # ------------------------
    # RECOGNITION (IMPROVED)
    # ------------------------
    async def recognize_faces(self, frame: np.ndarray) -> List[Dict]:
        if frame is None:
            return []

        if not self._loaded:
            await self.load_known_faces()

        try:
            import face_recognition

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            locations, encodings = await asyncio.to_thread(self._detect, rgb)

            results = []

            for (top, right, bottom, left), encoding in zip(locations, encodings):

                # 🔥 Skip tiny faces (noise)
                if (bottom - top) < 40:
                    continue

                name = "Unknown"
                confidence = 0.0

                if self.known_encodings:
                    distances = await asyncio.to_thread(
                        face_recognition.face_distance, self.known_encodings, encoding
                    )

                    best_match_idx = np.argmin(distances)
                    best_distance = distances[best_match_idx]

                    # 🔥 Better threshold
                    if best_distance < 0.5:
                        name = self.known_names[best_match_idx]
                        confidence = 1 - best_distance

                results.append(
                    {
                        "name": name,
                        "confidence": round(confidence, 2),
                        "location": (top, right, bottom, left),
                    }
                )

            return results

        except Exception as e:
            logger.exception("Recognition error: %s", e)
            return []

    def _detect(self, rgb):
        import face_recognition

        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        return locs, encs

    # ------------------------
    # DRAW
    # ------------------------
    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        for face in faces:
            top, right, bottom, left = face["location"]
            name = face["name"]
            conf = face.get("confidence", 0)

            label = f"{name} ({conf:.2f})"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame

    # ------------------------
    # VERIFY
    # ------------------------
    async def verify_face(self, frame: np.ndarray, expected_name: str) -> bool:
        faces = await self.recognize_faces(frame)
        return any(f["name"] == expected_name for f in faces)

    # ------------------------
    # REMOVE
    # ------------------------
    async def remove_face(self, name: str) -> bool:
        async with self._lock:
            if name in self.known_names:
                idx = self.known_names.index(name)
                self.known_names.pop(idx)
                self.known_encodings.pop(idx)

                path = self.data_dir / f"{name}.json"
                if path.exists():
                    await asyncio.to_thread(path.unlink)

                return True
            return False

    # ------------------------
    # STATS
    # ------------------------
    def get_stats(self) -> Dict:
        return {
            "known_faces": len(self.known_names),
            "loaded": self._loaded,
            "storage": "encoding_only",
        }
