"""System modules for EDIATH core."""

from enum import Enum


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    NOT_RESPONDING = "not_responding"


# Keep package init lightweight to avoid circular imports.
__all__ = [
    "ComponentStatus",
]
