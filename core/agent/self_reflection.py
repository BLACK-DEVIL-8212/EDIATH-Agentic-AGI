"""
Advanced Self Reflection - Learning Engine for EDIATH
With Shallow and Deep Introspection Capabilities
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..utils.logger import logger

# #==================== Enums #====================


class ReflectionDepth(Enum):
    """Depth of reflection analysis"""

    SHALLOW = "shallow"  # Quick surface-level reflection
    MODERATE = "moderate"  # Balanced reflection
    DEEP = "deep"  # Comprehensive deep analysis
    CRITICAL = "critical"  # Maximum depth for critical events


class ReflectionType(Enum):
    """Types of reflection"""

    ACTION = "action"
    DECISION = "decision"
    ERROR = "error"
    SUCCESS = "success"
    LEARNING = "learning"
    PATTERN = "pattern"
    STRATEGY = "strategy"


# #==================== Data Classes #====================


@dataclass
class ReflectionMetadata:
    """Metadata for reflection entries"""

    depth: ReflectionDepth = ReflectionDepth.MODERATE
    reflection_type: ReflectionType = ReflectionType.ACTION
    processing_time_ms: float = 0.0
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class ReflectionEntry:
    def __init__(
        self,
        topic: str,
        observation: str,
        result: Optional[Any] = None,
        success: bool = True,
        depth: ReflectionDepth = ReflectionDepth.MODERATE,
        reflection_type: ReflectionType = ReflectionType.ACTION,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            # ------------------------
            # 🔥 VALIDATION + NORMALIZATION
            # ------------------------
            self.topic = self._safe_str(topic, "general")
            self.observation = self._safe_str(observation, "")

            # result can be any type → safely store as string (limited)
            if result is not None:
                self.result = self._safe_str(result, "", limit=500)
            else:
                self.result = None

            self.success = bool(success)

            # ------------------------
            # 🔥 DEPTH AND METADATA
            # ------------------------
            self.depth = depth
            self.reflection_type = reflection_type
            self.metadata = ReflectionMetadata(
                depth=depth,
                reflection_type=reflection_type,
                tags=metadata.get("tags", []) if metadata else [],
                context=metadata.get("context", {}) if metadata else {},
            )

            # ------------------------
            # 🔥 REFLECTION DATA
            # ------------------------
            self.insight: Optional[str] = None
            self.value_score: float = 0.0
            self.shallow_insight: Optional[str] = None  # NEW: Quick insight
            self.deep_analysis: Optional[Dict[str, Any]] = None  # NEW: Deep analysis

            # ------------------------
            # 🔥 TIMESTAMP (ISO SAFE)
            # ------------------------
            self.timestamp = datetime.now()

        except Exception:
            # 🔥 HARD FAILSAFE
            self.topic = "error"
            self.observation = ""
            self.result = None
            self.success = False
            self.insight = None
            self.value_score = 0.0
            self.shallow_insight = None
            self.deep_analysis = None
            self.depth = ReflectionDepth.SHALLOW
            self.reflection_type = ReflectionType.ERROR
            self.metadata = ReflectionMetadata()
            self.timestamp = datetime.now()

    # ------------------------
    # 🔥 SAFE STRING CONVERTER
    # ------------------------
    def _safe_str(self, value: Any, default: str = "", limit: int = 200) -> str:
        try:
            if value is None:
                return default

            text = str(value).strip()

            if not text:
                return default

            return text[:limit]

        except Exception:
            return default

    # ------------------------
    # 🔥 SERIALIZATION (IMPORTANT)
    # ------------------------
    def to_dict(self):
        return {
            "topic": self.topic,
            "observation": self.observation,
            "result": self.result,
            "success": self.success,
            "insight": self.insight,
            "shallow_insight": self.shallow_insight,
            "deep_analysis": self.deep_analysis,
            "value_score": self.value_score,
            "depth": self.depth.value,
            "reflection_type": self.reflection_type.value,
            "metadata": {
                "tags": self.metadata.tags,
                "processing_time_ms": self.metadata.processing_time_ms,
                "confidence": self.metadata.confidence,
            },
            "timestamp": self.timestamp.isoformat(),
        }


class SelfReflection:
    def __init__(self, max_reflections: int = 1000, max_improvements: int = 200):
        # 🔥 storage
        self.reflections: List[ReflectionEntry] = []
        self.improvements: List[str] = []
        self.deep_insights: List[Dict[str, Any]] = []  # NEW: Store deep insights
        self.patterns: Dict[str, Any] = {}  # NEW: Pattern recognition storage

        # 🔥 limits (prevent memory bloat)
        self.max_reflections = max_reflections
        self.max_improvements = max_improvements
        self.max_deep_insights = 100

        self.reflection_count = 0
        self.debug = False

        # Performance tracking
        self._processing_times: List[float] = []

    # ------------------------
    # AUTO REFLECTION 🔥 WITH DEPTH CONTROL
    # ------------------------
    def reflect(
        self,
        topic: str,
        observation: str,
        result: Optional[Any] = None,
        success: bool = True,
        depth: ReflectionDepth = ReflectionDepth.MODERATE,
        reflection_type: ReflectionType = ReflectionType.ACTION,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReflectionEntry:

        start_time = datetime.now()

        try:
            entry = ReflectionEntry(
                topic, observation, result, success, depth, reflection_type, metadata
            )

            # 🔥 scoring based on depth
            entry.value_score = self._evaluate(success, result, depth)

            # 🔥 generate insights based on depth
            if depth == ReflectionDepth.SHALLOW:
                entry.shallow_insight = self._generate_shallow_insight(entry)
                entry.insight = entry.shallow_insight

            elif depth == ReflectionDepth.MODERATE:
                entry.insight = self._generate_moderate_insight(entry)
                entry.shallow_insight = self._generate_shallow_insight(entry)

            elif depth == ReflectionDepth.DEEP:
                entry.insight = self._generate_deep_insight(entry)
                entry.shallow_insight = self._generate_shallow_insight(entry)
                entry.deep_analysis = self._generate_deep_analysis(entry)

            elif depth == ReflectionDepth.CRITICAL:
                entry.insight = self._generate_critical_insight(entry)
                entry.shallow_insight = self._generate_shallow_insight(entry)
                entry.deep_analysis = self._generate_comprehensive_analysis(entry)

            # Record processing time
            processing_ms = (datetime.now() - start_time).total_seconds() * 1000
            entry.metadata.processing_time_ms = processing_ms
            self._processing_times.append(processing_ms)

            # 🔥 bounded storage
            if len(self.reflections) >= self.max_reflections:
                self.reflections.pop(0)

            self.reflections.append(entry)
            self.reflection_count += 1

            # Store deep insights separately
            if (
                depth in [ReflectionDepth.DEEP, ReflectionDepth.CRITICAL]
                and entry.deep_analysis
            ):
                if len(self.deep_insights) >= self.max_deep_insights:
                    self.deep_insights.pop(0)
                self.deep_insights.append(
                    {
                        "topic": entry.topic,
                        "insight": entry.insight,
                        "analysis": entry.deep_analysis,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                )

            if logger:
                logger.info(
                    f"[Reflection] {entry.topic} | depth={depth.value} | score={entry.value_score:.2f} | time={processing_ms:.1f}ms"
                )

            # Identify patterns
            self._identify_patterns(entry)

            return entry

        except Exception as e:
            if logger:
                logger.warning(f"[Reflection Error] {e}")

            return ReflectionEntry(
                "error", observation, None, False, ReflectionDepth.SHALLOW
            )

    # ------------------------
    # 🔥 SHALLOW INTROSPECTION (NEW)
    # ------------------------
    def shallow_reflect(
        self, topic: str, observation: str, result: Optional[Any] = None
    ) -> ReflectionEntry:
        """Quick surface-level reflection without heavy processing"""
        return self.reflect(
            topic=topic,
            observation=observation,
            result=result,
            depth=ReflectionDepth.SHALLOW,
            reflection_type=ReflectionType.ACTION,
        )

    def quick_scan(self, action: str, outcome: str) -> str:
        """Ultra-fast reflection for real-time decisions"""
        try:
            success = "success" in outcome.lower() or "good" in outcome.lower()
            insight = f"Quick scan: {action} → {'effective' if success else 'needs improvement'}"

            # Store minimal reflection
            self.shallow_reflect(action, outcome, success)

            return insight
        except Exception:
            return "Scan unavailable"

    # ------------------------
    # 🔥 DEEP INTROSPECTION (NEW)
    # ------------------------
    async def deep_reflect_async(
        self, topic: str, observation: str, result: Optional[Any] = None
    ) -> ReflectionEntry:
        """Asynchronous deep reflection for complex analysis"""
        import asyncio

        # Simulate heavy processing for deep reflection
        await asyncio.sleep(0.01)  # Allow other tasks to run

        return self.reflect(
            topic=topic,
            observation=observation,
            result=result,
            depth=ReflectionDepth.DEEP,
            reflection_type=ReflectionType.LEARNING,
        )

    def critical_reflect(
        self, topic: str, observation: str, error: Exception
    ) -> ReflectionEntry:
        """Critical reflection for error analysis"""
        return self.reflect(
            topic=topic,
            observation=observation,
            result=str(error),
            success=False,
            depth=ReflectionDepth.CRITICAL,
            reflection_type=ReflectionType.ERROR,
            metadata={"error_type": type(error).__name__},
        )

    # ------------------------
    # EVALUATION 🔥 WITH DEPTH
    # ------------------------
    def _evaluate(
        self,
        success: bool,
        result: Any,
        depth: ReflectionDepth = ReflectionDepth.MODERATE,
    ) -> float:
        try:
            base = 0.2

            if success:
                base = 0.6

                if result:
                    length = len(str(result))
                    if length > 50:
                        base = 0.85
                    else:
                        base = 0.7

            # Adjust based on depth (deeper analysis gives more accurate scoring)
            if depth == ReflectionDepth.SHALLOW:
                base = base * 0.9
            elif depth == ReflectionDepth.DEEP:
                base = base * 1.05
            elif depth == ReflectionDepth.CRITICAL:
                base = base * 1.1

            return round(min(max(base, 0.0), 1.0), 2)

        except:
            return 0.0

    # ------------------------
    # 🔥 SHALLOW INSIGHT GENERATION (NEW)
    # ------------------------
    def _generate_shallow_insight(self, entry: ReflectionEntry) -> str:
        """Quick surface-level insight"""
        try:
            if not entry.success:
                return f"{entry.topic}: needs adjustment"

            if entry.value_score >= 0.8:
                return f"{entry.topic}: good result"

            return f"{entry.topic}: processed"
        except:
            return "insight unavailable"

    # ------------------------
    # 🔥 MODERATE INSIGHT GENERATION
    # ------------------------
    def _generate_moderate_insight(self, entry: ReflectionEntry) -> str:
        """Balanced insight generation"""
        try:
            if not entry.success:
                return f"{entry.topic}: failure → adjust strategy"

            if entry.value_score >= 0.8:
                return f"{entry.topic}: strong success → reuse pattern"

            if entry.value_score >= 0.6:
                return f"{entry.topic}: moderate success → refine approach"

            return f"{entry.topic}: weak outcome → improve execution"
        except:
            return "insight unavailable"

    # ------------------------
    # 🔥 DEEP INSIGHT GENERATION (NEW)
    # ------------------------
    def _generate_deep_insight(self, entry: ReflectionEntry) -> str:
        """Comprehensive deep insight with analysis"""
        try:
            factors = []

            if entry.value_score > 0.8:
                factors.append("high effectiveness")
            elif entry.value_score > 0.6:
                factors.append("moderate effectiveness")
            else:
                factors.append("low effectiveness")

            if len(entry.observation) > 100:
                factors.append("detailed context")

            if entry.success:
                recommendation = "Document and reuse this pattern"
            else:
                recommendation = "Analyze root cause and adjust"

            return f"{entry.topic}: {', '.join(factors)} → {recommendation}"
        except:
            return "Deep insight unavailable"

    def _generate_deep_analysis(self, entry: ReflectionEntry) -> Dict[str, Any]:
        """Generate detailed analysis for deep reflection"""
        try:
            return {
                "root_cause_analysis": self._analyze_root_cause(entry),
                "impact_assessment": self._assess_impact(entry),
                "recommendations": self._generate_recommendations(entry),
                "confidence_score": entry.value_score,
                "related_patterns": self._find_related_patterns(entry),
            }
        except Exception as e:
            return {"error": str(e), "analysis_unavailable": True}

    def _generate_critical_insight(self, entry: ReflectionEntry) -> str:
        """Maximum depth insight for critical events"""
        try:
            severity = "HIGH" if not entry.success else "MEDIUM"
            action_item = (
                "Immediate review required" if not entry.success else "Monitor closely"
            )

            return f"[CRITICAL] {entry.topic}: {severity} impact | {action_item}"
        except:
            return "Critical insight unavailable"

    def _generate_comprehensive_analysis(
        self, entry: ReflectionEntry
    ) -> Dict[str, Any]:
        """Comprehensive analysis for critical reflections"""
        try:
            return {
                "root_cause": self._deep_root_cause_analysis(entry),
                "impact_analysis": self._full_impact_analysis(entry),
                "mitigation_strategy": self._generate_mitigation(entry),
                "prevention_measures": self._generate_prevention(entry),
                "confidence": entry.value_score,
                "requires_immediate_action": not entry.success,
                "priority": "HIGH" if not entry.success else "MEDIUM",
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_root_cause(self, entry: ReflectionEntry) -> str:
        """Analyze root cause of outcome"""
        if not entry.success:
            return f"Failure in {entry.topic}: {entry.observation[:100]}"
        return f"Success achieved through effective {entry.topic}"

    def _deep_root_cause_analysis(self, entry: ReflectionEntry) -> Dict[str, Any]:
        """Deep root cause analysis for critical events"""
        return {
            "primary_cause": (
                entry.observation[:200] if not entry.success else "No failure detected"
            ),
            "contributing_factors": ["execution", "context", "timing"],
            "severity": "critical" if not entry.success else "normal",
        }

    def _assess_impact(self, entry: ReflectionEntry) -> str:
        """Assess impact of the reflection"""
        if entry.value_score > 0.8:
            return "High positive impact"
        elif entry.value_score > 0.6:
            return "Moderate positive impact"
        elif entry.success:
            return "Low positive impact"
        else:
            return "Negative impact requiring attention"

    def _full_impact_analysis(self, entry: ReflectionEntry) -> Dict[str, Any]:
        """Full impact analysis for critical events"""
        return {
            "immediate_impact": "system_affected" if not entry.success else "normal",
            "long_term_impact": "needs_monitoring" if not entry.success else "minimal",
            "risk_level": "high" if not entry.success else "low",
        }

    def _generate_recommendations(self, entry: ReflectionEntry) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if entry.success and entry.value_score > 0.8:
            recommendations.append("Document successful pattern")
            recommendations.append("Apply to similar scenarios")
        elif entry.success:
            recommendations.append("Fine-tune for better results")
        else:
            recommendations.append("Review execution process")
            recommendations.append("Adjust parameters")

        return recommendations

    def _generate_mitigation(self, entry: ReflectionEntry) -> str:
        """Generate mitigation strategy"""
        if not entry.success:
            return (
                "Immediate: Isolate issue | Short-term: Debug | Long-term: Prevention"
            )
        return "Monitor and maintain current approach"

    def _generate_prevention(self, entry: ReflectionEntry) -> List[str]:
        """Generate prevention measures"""
        if not entry.success:
            return ["Add validation", "Improve error handling", "Enhance testing"]
        return ["Continue current practices"]

    def _find_related_patterns(self, entry: ReflectionEntry) -> List[str]:
        """Find related patterns in reflection history"""
        related = []
        for ref in self.reflections[-50:]:
            if ref.topic == entry.topic and ref.id != entry.id:
                related.append(f"Similar {ref.topic} with score {ref.value_score}")
                if len(related) >= 3:
                    break
        return related

    def _identify_patterns(self, entry: ReflectionEntry):
        """Identify recurring patterns in reflections"""
        try:
            key = f"{entry.topic}:{entry.success}"
            if key not in self.patterns:
                self.patterns[key] = 0
            self.patterns[key] += 1

            # Trim patterns dict
            if len(self.patterns) > 100:
                # Remove least frequent patterns
                min_key = min(self.patterns, key=self.patterns.get)
                del self.patterns[min_key]
        except Exception:
            pass

    # ------------------------
    # IMPROVEMENTS 🔥
    # ------------------------
    def identify_improvement(self, improvement_area: str, priority: str = "normal"):
        try:
            if not improvement_area:
                return

            improvement_area = str(improvement_area).strip()

            if not improvement_area:
                return

            # 🔥 bounded + deduplicated
            if improvement_area not in self.improvements:
                if len(self.improvements) >= self.max_improvements:
                    self.improvements.pop(0)

                self.improvements.append(improvement_area)

                if logger:
                    logger.info(
                        f"[Improvement] {improvement_area} (priority: {priority})"
                    )

        except Exception as e:
            if logger:
                logger.warning(f"[Improvement Error] {e}")

    # ------------------------
    # ANALYSIS 🔥 ENHANCED
    # ------------------------
    def analyze_patterns(
        self, depth: ReflectionDepth = ReflectionDepth.MODERATE
    ) -> Dict[str, Any]:
        try:
            total = len(self.reflections)

            if total == 0:
                return {
                    "total": 0,
                    "success": 0,
                    "failures": 0,
                    "avg_score": 0.0,
                    "by_depth": {},
                    "by_type": {},
                }

            success_count = sum(1 for r in self.reflections if r.success)
            fail_count = total - success_count

            avg_score = sum(r.value_score for r in self.reflections) / total

            # Analyze by depth
            by_depth = {}
            for depth_val in ReflectionDepth:
                depth_reflections = [
                    r for r in self.reflections if r.depth == depth_val
                ]
                if depth_reflections:
                    by_depth[depth_val.value] = {
                        "count": len(depth_reflections),
                        "avg_score": sum(r.value_score for r in depth_reflections)
                        / len(depth_reflections),
                    }

            # Analyze by type
            by_type = {}
            for type_val in ReflectionType:
                type_reflections = [
                    r for r in self.reflections if r.reflection_type == type_val
                ]
                if type_reflections:
                    by_type[type_val.value] = {
                        "count": len(type_reflections),
                        "avg_score": sum(r.value_score for r in type_reflections)
                        / len(type_reflections),
                    }

            result = {
                "total": total,
                "success": success_count,
                "failures": fail_count,
                "avg_score": round(avg_score, 3),
                "success_rate": round(success_count / max(1, total), 3),
                "by_depth": by_depth,
                "by_type": by_type,
                "patterns": self.patterns,
            }

            # Add deep analysis if requested
            if depth == ReflectionDepth.DEEP:
                result["trend_analysis"] = self._analyze_trends()
                result["recommendations"] = self._generate_system_recommendations()

            return result

        except Exception as e:
            if logger:
                logger.warning(f"[Analysis Error] {e}")

            return {
                "total": 0,
                "success": 0,
                "failures": 0,
                "avg_score": 0.0,
                "success_rate": 0.0,
                "by_depth": {},
                "by_type": {},
            }

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends over time"""
        try:
            if len(self.reflections) < 10:
                return {"trend": "insufficient_data"}

            recent = self.reflections[-10:]
            older = self.reflections[-20:-10]

            recent_avg = sum(r.value_score for r in recent) / len(recent)
            older_avg = (
                sum(r.value_score for r in older) / len(older) if older else recent_avg
            )

            trend = (
                "improving"
                if recent_avg > older_avg
                else "declining" if recent_avg < older_avg else "stable"
            )

            return {
                "trend": trend,
                "recent_avg": round(recent_avg, 3),
                "previous_avg": round(older_avg, 3),
                "change": round(recent_avg - older_avg, 3),
            }
        except Exception:
            return {"trend": "unknown"}

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []

        if self.reflection_count > 100:
            recommendations.append(
                "High reflection volume - consider optimizing storage"
            )

        success_rate = self.analyze_patterns()["success_rate"]
        if success_rate < 0.6:
            recommendations.append("Low success rate - review system strategies")
        elif success_rate > 0.9:
            recommendations.append("Excellent performance - document best practices")

        avg_processing = sum(self._processing_times) / max(
            1, len(self._processing_times)
        )
        if avg_processing > 100:
            recommendations.append("Slow reflection processing - consider optimization")

        return recommendations[:5]

    # ------------------------
    # GETTERS 🔥 ENHANCED
    # ------------------------
    def get_recent_reflections(
        self, limit: int = 5, depth_filter: Optional[ReflectionDepth] = None
    ):
        try:
            limit = max(1, min(limit, 50))
            reflections = self.reflections[-limit:]

            if depth_filter:
                reflections = [r for r in reflections if r.depth == depth_filter]

            return reflections
        except:
            return []

    def get_deep_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get stored deep insights"""
        return self.deep_insights[-limit:]

    def get_best_strategies(self, min_score: float = 0.8):
        try:
            return [r for r in self.reflections if r.value_score >= min_score]
        except:
            return []

    def get_failure_patterns(self) -> List[Dict[str, Any]]:
        """Get analysis of failure patterns"""
        failures = [r for r in self.reflections if not r.success]

        if not failures:
            return []

        # Group by topic
        by_topic = {}
        for f in failures:
            by_topic[f.topic] = by_topic.get(f.topic, 0) + 1

        return [
            {
                "topic": topic,
                "failure_count": count,
                "percentage": round(count / len(failures) * 100, 1),
            }
            for topic, count in sorted(
                by_topic.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        analysis = self.analyze_patterns(ReflectionDepth.DEEP)

        return {
            "total_reflections": self.reflection_count,
            "stored": len(self.reflections),
            "improvements": len(self.improvements),
            "deep_insights": len(self.deep_insights),
            "avg_score": analysis["avg_score"],
            "success_rate": analysis["success_rate"],
            "patterns_identified": len(self.patterns),
            "avg_processing_time_ms": sum(self._processing_times)
            / max(1, len(self._processing_times)),
            "by_depth": analysis.get("by_depth", {}),
            "by_type": analysis.get("by_type", {}),
        }

    def get_stats(self):
        """Backward compatible stats method"""
        try:
            analysis = self.analyze_patterns()

            return {
                "total_reflections": self.reflection_count,
                "stored": len(self.reflections),
                "improvements": len(self.improvements),
                "avg_score": analysis["avg_score"],
                "success_rate": analysis["success_rate"],
                "deep_insights_count": len(self.deep_insights),
            }

        except Exception as e:
            if logger:
                logger.warning(f"[Stats Error] {e}")

            return {
                "total_reflections": 0,
                "stored": 0,
                "improvements": 0,
                "avg_score": 0.0,
                "success_rate": 0.0,
                "deep_insights_count": 0,
            }

    # ------------------------
    # CLEANUP
    # ------------------------
    def clear_old_reflections(self, days: int = 30):
        """Clear reflections older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        self.reflections = [
            r for r in self.reflections if r.timestamp.timestamp() > cutoff
        ]
        logger.info(
            f"Cleared reflections older than {days} days, {len(self.reflections)} remaining"
        )


# #==================== Export #====================

__all__ = ["SelfReflection", "ReflectionEntry", "ReflectionDepth", "ReflectionType"]
