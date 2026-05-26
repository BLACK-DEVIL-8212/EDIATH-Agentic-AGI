"""Vision modules for visual perception and AI processing.

Vision components can import heavy CV/ML dependencies, so package-level exports
are resolved lazily instead of loading every vision subsystem on import.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "Webcam": (".webcam", "Webcam"),
    "FrameBuffer": (".webcam", "FrameBuffer"),
    "ObjectDetector": (".object_detector", "ObjectDetector"),
    "Detection": (".object_detector", "Detection"),
    "Captioner": (".captioner", "Captioner"),
    "ImageCaption": (".captioner", "ImageCaption"),
    "PerceptionLoop": (".perception_loop", "PerceptionLoop"),
    "PerceptionStep": (".perception_loop", "PerceptionStep"),
    "VisionEngine": (".vision_engine", "VisionEngine"),
    "ActivityRecognitionEngine": (
        ".activity_recognition",
        "ActivityRecognitionEngine",
    ),
    "FaceRecognitionEngine": (".face_recognition", "FaceRecognitionEngine"),
    "SharedVisionMemory": (".shared_memory", "SharedVisionMemory"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "Webcam",
    "FrameBuffer",
    "ObjectDetector",
    "Detection",
    "Captioner",
    "ImageCaption",
    "PerceptionLoop",
    "PerceptionStep",
    "VisionEngine",
    "ActivityRecognitionEngine",
    "FaceRecognitionEngine",
    "SharedVisionMemory",
]
