"""Audit Logger - production-grade action audit trail."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from collections import Counter

import hashlib
import json
import os

from core.utils.logger import logger

try:
    from security.encryption import EncryptionManager
except:
    EncryptionManager = None


# ------------------------
# 🔥 LOG LEVELS
# ------------------------
class AuditLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"

    # ------------------------
    # 🔥 STRING CONVERSION
    # ------------------------
    @classmethod
    def from_string(cls, value: str):
        try:
            if not value:
                return cls.INFO

            value = str(value).lower().strip()

            # ------------------------
            # 🔥 DIRECT MAP (FAST PATH)
            # ------------------------
            direct_map = {
                "info": cls.INFO,
                "warning": cls.WARNING,
                "warn": cls.WARNING,
                "w": cls.WARNING,
                "error": cls.ERROR,
                "err": cls.ERROR,
                "e": cls.ERROR,
                "critical": cls.CRITICAL,
                "crit": cls.CRITICAL,
                "c": cls.CRITICAL,
                "security": cls.SECURITY,
                "sec": cls.SECURITY,
                "s": cls.SECURITY,
            }

            if value in direct_map:
                return direct_map[value]

            # ------------------------
            # 🔥 PARTIAL MATCH (SMART)
            # ------------------------
            for level in cls:
                if value in level.value:
                    return level

            # ------------------------
            # 🔥 FALLBACK
            # ------------------------
            return cls.INFO

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return cls.INFO

    # ------------------------
    # 🔥 PRIORITY SCORE
    # ------------------------
    def priority(self) -> int:
        try:
            # ------------------------
            # 🔥 STATIC MAP (FAST)
            # ------------------------
            if not hasattr(AuditLevel, "_priority_map"):
                AuditLevel._priority_map = {
                    AuditLevel.INFO: 1,
                    AuditLevel.WARNING: 2,
                    AuditLevel.ERROR: 3,
                    AuditLevel.CRITICAL: 4,
                    AuditLevel.SECURITY: 5,
                }

            return AuditLevel._priority_map.get(self, 1)

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return 1

    # ------------------------
    # 🔥 IS SEVERE
    # ------------------------
    def is_severe(self) -> bool:
        try:
            # ------------------------
            # 🔥 STATIC SET (CACHED)
            # ------------------------
            if not hasattr(AuditLevel, "_severe_levels"):
                AuditLevel._severe_levels = {
                    AuditLevel.ERROR,
                    AuditLevel.CRITICAL,
                    AuditLevel.SECURITY,
                }

            return self in AuditLevel._severe_levels

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False

    # ------------------------
    # 🔥 IS SECURITY RELATED
    # ------------------------
    def is_security(self) -> bool:
        try:
            # ------------------------
            # 🔥 STATIC SET (EXTENSIBLE)
            # ------------------------
            if not hasattr(AuditLevel, "_security_levels"):
                AuditLevel._security_levels = {
                    AuditLevel.SECURITY,
                }

            return self in AuditLevel._security_levels

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return False

    # ------------------------
    # 🔥 COLOR TAG (FOR LOGGING/UI)
    # ------------------------
    def color(self) -> str:
        try:
            # ------------------------
            # 🔥 STATIC MAP (CACHED)
            # ------------------------
            if not hasattr(AuditLevel, "_color_map"):
                AuditLevel._color_map = {
                    AuditLevel.INFO: "green",
                    AuditLevel.WARNING: "yellow",
                    AuditLevel.ERROR: "red",
                    AuditLevel.CRITICAL: "magenta",
                    AuditLevel.SECURITY: "cyan",
                }

            return AuditLevel._color_map.get(self, "white")

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return "white"

    # ------------------------
    # 🔥 TO DICT
    # ------------------------
    def to_dict(self) -> Dict[str, Any]:
        try:
            # ------------------------
            # 🔥 BASE DATA
            # ------------------------
            data = {
                "level": self.value,
                "name": self.name,
                "priority": self.priority(),
                "is_severe": self.is_severe(),
                "is_security": self.is_security(),
                "color": self.color(),
            }

            # ------------------------
            # 🔥 OPTIONAL FLAGS
            # ------------------------
            data["severity_tag"] = "high" if data["is_severe"] else "low"
            data["category"] = "security" if data["is_security"] else "general"

            return data

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return {
                "level": "info",
                "name": "INFO",
                "priority": 1,
                "is_severe": False,
                "is_security": False,
                "color": "white",
                "severity_tag": "low",
                "category": "general",
            }

    # ------------------------
    # 🔥 STRING REPRESENTATION
    # ------------------------
    def __str__(self) -> str:
        try:
            # ------------------------
            # 🔥 HUMAN READABLE
            # ------------------------
            return self.value.upper()
        except Exception:
            return "UNKNOWN"

    def __repr__(self) -> str:
        try:
            # ------------------------
            # 🔥 DEBUG FRIENDLY
            # ------------------------
            return (
                f"<AuditLevel name={self.name} "
                f"value={self.value} "
                f"priority={self.priority()}>"
            )
        except Exception:
            return "<AuditLevel UNKNOWN>"


# ------------------------
# 🔥 AUDIT ENTRY
# ------------------------
class AuditEntry:
    def __init__(
        self,
        action: str,
        actor: str,
        level: AuditLevel = AuditLevel.INFO,
        details: Optional[Dict[str, Any]] = None,
        result: str = "success",
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            import uuid

            # ------------------------
            # 🔥 CORE FIELDS
            # ------------------------
            self.id = str(uuid.uuid4())  # unique ID
            self.action = str(action).strip()
            self.actor = str(actor).strip()

            # ------------------------
            # 🔥 LEVEL NORMALIZATION
            # ------------------------
            if isinstance(level, AuditLevel):
                self.level = level
            else:
                self.level = AuditLevel.from_string(str(level))

            # ------------------------
            # 🔥 DETAILS (SAFE COPY)
            # ------------------------
            self.details = dict(details) if details else {}

            # ------------------------
            # 🔥 RESULT NORMALIZATION
            # ------------------------
            self.result = str(result).lower().strip()

            # ------------------------
            # 🔥 TIMESTAMP
            # ------------------------
            self.timestamp = timestamp or datetime.utcnow()

            # ------------------------
            # 🔥 TRACE ID (FOR CORRELATION)
            # ------------------------
            self.trace_id = trace_id or str(uuid.uuid4())

            # ------------------------
            # 🔥 METADATA
            # ------------------------
            self.metadata = metadata or {}

            # ------------------------
            # 🔥 DERIVED FIELDS
            # ------------------------
            self.priority = self.level.priority()
            self.is_severe = self.level.is_severe()
            self.is_security = self.level.is_security()

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE INIT
            # ------------------------
            self.id = "unknown"
            self.action = "unknown"
            self.actor = "system"
            self.level = AuditLevel.INFO
            self.details = {}
            self.result = "error"
            self.timestamp = datetime.utcnow()
            self.trace_id = "unknown"
            self.metadata = {}
            self.priority = 1
            self.is_severe = False
            self.is_security = False

    def to_dict(self) -> Dict[str, Any]:
        try:

            # ------------------------
            # 🔥 BASE DATA
            # ------------------------
            data = {
                "id": getattr(self, "id", None),
                "trace_id": getattr(self, "trace_id", None),
                "action": self.action,
                "actor": self.actor,
                "level": self.level.value,
                "level_name": self.level.name,
                "priority": getattr(self, "priority", self.level.priority()),
                "is_severe": getattr(self, "is_severe", self.level.is_severe()),
                "is_security": getattr(self, "is_security", self.level.is_security()),
                "details": self.details,
                "result": self.result,
                "timestamp": self.timestamp.isoformat(),
                "metadata": getattr(self, "metadata", {}),
            }

            # ------------------------
            # 🔥 SIGNATURE (TAMPER DETECTION)
            # ------------------------
            try:
                raw = json.dumps(data, sort_keys=True)
                signature = hashlib.sha256(raw.encode()).hexdigest()
                data["signature"] = signature
            except Exception:
                data["signature"] = None

            return data

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            return {
                "id": "unknown",
                "action": "unknown",
                "actor": "system",
                "level": "info",
                "priority": 1,
                "details": {},
                "result": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "signature": None,
            }


# ------------------------
# 🔥 AUDIT LOGGER
# ------------------------
class AuditLogger:
    def __init__(
        self,
        max_entries: int = 10000,
        log_file: str = "logs/audit_log.json",
        encrypt: bool = False,
        auto_flush: bool = True,
        flush_interval: int = 10,
    ):
        try:
            import os
            import threading
            from datetime import datetime

            # ------------------------
            # 🔥 CORE CONFIG
            # ------------------------
            self.max_entries = max(100, int(max_entries))
            self.log_file = log_file
            self.auto_flush = auto_flush
            self.flush_interval = max(1, flush_interval)

            # ------------------------
            # 🔥 STORAGE
            # ------------------------
            self.audit_log: List[AuditEntry] = []

            # ------------------------
            # 🔥 LOCK (THREAD SAFE)
            # ------------------------
            self.lock = threading.RLock()

            # ------------------------
            # 🔥 METRICS
            # ------------------------
            self.total_logged = 0
            self.last_flush_time = datetime.utcnow()
            self._pending_writes = 0

            # ------------------------
            # 🔥 FILE SYSTEM SETUP
            # ------------------------
            try:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            except:
                pass

            # ------------------------
            # 🔥 ENCRYPTION SETUP
            # ------------------------
            self.encrypt_enabled = bool(encrypt and EncryptionManager is not None)

            if self.encrypt_enabled:
                try:
                    self.encryptor = EncryptionManager()
                except Exception:
                    self.encrypt_enabled = False
                    self.encryptor = None
            else:
                self.encryptor = None

            # ------------------------
            # 🔥 WORKER FLAGS
            # ------------------------
            self._initialized = True
            self._closed = False

        except Exception:
            # ------------------------
            # 🔥 FAILSAFE INIT
            # ------------------------
            self.audit_log = []
            self.max_entries = 1000
            self.log_file = "audit_fallback.log"
            self.lock = threading.Lock()
            self.encrypt_enabled = False
            self.encryptor = None
            self.total_logged = 0
            self._pending_writes = 0
            self._closed = False

    # ------------------------
    # 🔥 CORE LOG FUNCTION
    # ------------------------
    def log_action(
        self,
        action: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        level: AuditLevel = AuditLevel.INFO,
        result: str = "success",
    ) -> AuditEntry:

        try:

            with self.lock:

                # ------------------------
                # 🔥 NORMALIZE INPUT
                # ------------------------
                action = str(action).strip()
                actor = str(actor).strip()
                result = str(result).lower().strip()

                if not isinstance(level, AuditLevel):
                    level = AuditLevel.from_string(str(level))

                # ------------------------
                # 🔥 SAFE DETAILS COPY
                # ------------------------
                safe_details = dict(details) if details else {}

                # ------------------------
                # 🔥 ENCRYPT DETAILS
                # ------------------------
                if self.encrypt_enabled and safe_details:
                    try:
                        safe_details = {
                            "encrypted": self.encryptor.encrypt(
                                json.dumps(safe_details)
                            )
                        }
                    except Exception:
                        safe_details = {"error": "encryption_failed"}

                # ------------------------
                # 🔥 CREATE ENTRY
                # ------------------------
                entry = AuditEntry(
                    action=action,
                    actor=actor,
                    level=level,
                    details=safe_details,
                    result=result,
                    timestamp=datetime.utcnow(),
                )

                # ------------------------
                # 🔥 APPEND MEMORY
                # ------------------------
                self.audit_log.append(entry)
                self.total_logged += 1
                self._pending_writes += 1

                # ------------------------
                # 🔥 MEMORY LIMIT (RING BUFFER)
                # ------------------------
                if len(self.audit_log) > self.max_entries:
                    overflow = len(self.audit_log) - self.max_entries
                    del self.audit_log[:overflow]

                # ------------------------
                # 🔥 FILE WRITE (BATCH / AUTO FLUSH)
                # ------------------------
                if self.auto_flush:
                    try:
                        if self._pending_writes >= self.flush_interval:
                            self._flush_to_file()
                            self._pending_writes = 0
                    except Exception:
                        pass
                else:
                    # fallback single write
                    try:
                        self._append_to_file(entry)
                    except Exception:
                        pass

                return entry

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE ENTRY
            # ------------------------
            try:
                fallback = AuditEntry(
                    action="log_failure",
                    actor="system",
                    level=AuditLevel.ERROR,
                    details={"error": str(e)},
                    result="error",
                )
                return fallback
            except Exception:
                # ultimate fallback
                return None

    # ------------------------
    # 🔥 SECURITY EVENTS
    # ------------------------
    def log_security_event(
        self,
        event: str,
        actor: str,
        details: Dict[str, Any],
        severity: Optional[str] = None,
        alert: bool = False,
    ) -> AuditEntry:

        try:
            from datetime import datetime

            # ------------------------
            # 🔥 NORMALIZE INPUT
            # ------------------------
            event = str(event).strip()
            actor = str(actor).strip()

            safe_details = dict(details) if details else {}

            # ------------------------
            # 🔥 ADD SECURITY METADATA
            # ------------------------
            safe_details.update(
                {
                    "security_event": True,
                    "severity": severity or "medium",
                    "alert": bool(alert),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # ------------------------
            # 🔥 AUTO ESCALATION
            # ------------------------
            level = AuditLevel.SECURITY

            if severity:
                severity = severity.lower()

                if severity in ["high", "critical"]:
                    level = AuditLevel.CRITICAL

            # ------------------------
            # 🔥 LOG USING CORE PIPELINE
            # ------------------------
            entry = self.log_action(
                action=event,
                actor=actor,
                details=safe_details,
                level=level,
                result="security_event",
            )

            # ------------------------
            # 🔥 OPTIONAL ALERT HOOK
            # ------------------------
            if alert:
                try:
                    self._trigger_alert(entry)
                except:
                    pass

            return entry

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            try:
                return self.log_action(
                    action="security_log_failure",
                    actor="system",
                    details={"error": str(e)},
                    level=AuditLevel.ERROR,
                    result="error",
                )
            except:
                return None

    # ------------------------
    # 🔥 FILE APPEND (FAST)
    # ------------------------
    def _append_to_file(self, entry: AuditEntry):
        try:

            # ------------------------
            # 🔥 VALIDATION
            # ------------------------
            if entry is None:
                return

            data = entry.to_dict()

            # ------------------------
            # 🔥 SERIALIZATION SAFETY
            # ------------------------
            try:
                line = json.dumps(data, ensure_ascii=False)
            except Exception:
                line = json.dumps(
                    {
                        "action": "serialization_error",
                        "actor": "system",
                        "error": "failed_to_serialize",
                    }
                )

            # ------------------------
            # 🔥 THREAD-SAFE WRITE
            # ------------------------
            with self.lock:

                # ensure directory exists
                try:
                    os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                except:
                    pass

                # ------------------------
                # 🔥 FILE APPEND
                # ------------------------
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

                # ------------------------
                # 🔥 METRICS
                # ------------------------
                if hasattr(self, "last_flush_time"):
                    from datetime import datetime

                    self.last_flush_time = datetime.utcnow()

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE (NO CRASH)
            # ------------------------
            try:
                print(f"[AuditLogger] Write failed: {e}")
            except:
                pass

    # ------------------------
    # 🔥 FILTER ENTRIES
    # ------------------------
    def get_entries(
        self,
        action_filter: Optional[str] = None,
        actor_filter: Optional[str] = None,
        level_filter: Optional[AuditLevel] = None,
        limit: int = 100,
        sort_desc: bool = True,
        include_metadata: bool = False,
    ) -> List[AuditEntry]:

        try:
            with self.lock:

                # ------------------------
                # 🔥 SAFE COPY (NO MUTATION)
                # ------------------------
                results = list(self.audit_log)

                # ------------------------
                # 🔥 NORMALIZE FILTERS
                # ------------------------
                action_filter = action_filter.lower().strip() if action_filter else None
                actor_filter = actor_filter.lower().strip() if actor_filter else None

                if level_filter and not isinstance(level_filter, AuditLevel):
                    level_filter = AuditLevel.from_string(str(level_filter))

                # ------------------------
                # 🔥 APPLY FILTERS
                # ------------------------
                if action_filter:
                    results = [
                        e for e in results if action_filter in str(e.action).lower()
                    ]

                if actor_filter:
                    results = [
                        e for e in results if actor_filter in str(e.actor).lower()
                    ]

                if level_filter:
                    results = [e for e in results if e.level == level_filter]

                # ------------------------
                # 🔥 SORT BY TIMESTAMP
                # ------------------------
                try:
                    results.sort(
                        key=lambda e: getattr(e, "timestamp", None), reverse=sort_desc
                    )
                except:
                    pass

                # ------------------------
                # 🔥 LIMIT (SAFE)
                # ------------------------
                limit = max(1, min(limit, 1000))
                results = results[:limit]

                # ------------------------
                # 🔥 OPTIONAL METADATA ENRICHMENT
                # ------------------------
                if include_metadata:
                    enriched = []
                    for e in results:
                        try:
                            data = e.to_dict()
                            enriched.append(data)
                        except:
                            enriched.append({"error": "failed_to_serialize"})
                    return enriched

                return results

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            try:
                print(f"[AuditLogger] get_entries error: {e}")
            except:
                pass

            return []

    # ------------------------
    # 🔥 ACTOR HISTORY
    # ------------------------
    def get_actor_history(
        self,
        actor: str,
        limit: int = 50,
        level_filter: Optional[AuditLevel] = None,
        include_metadata: bool = False,
        sort_desc: bool = True,
    ) -> List[Any]:

        try:
            # ------------------------
            # 🔥 INPUT NORMALIZATION
            # ------------------------
            actor = str(actor).strip().lower()

            if not actor:
                return []

            # ------------------------
            # 🔥 LEVEL NORMALIZATION
            # ------------------------
            if level_filter and not isinstance(level_filter, AuditLevel):
                level_filter = AuditLevel.from_string(str(level_filter))

            # ------------------------
            # 🔥 FETCH FROM CORE ENGINE
            # ------------------------
            results = self.get_entries(
                actor_filter=actor,
                level_filter=level_filter,
                limit=limit,
                sort_desc=sort_desc,
                include_metadata=include_metadata,
            )

            return results

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            try:
                print(f"[AuditLogger] get_actor_history error: {e}")
            except:
                pass

            return []

    # ------------------------
    # 🔥 SECURITY EVENTS
    # ------------------------
    def get_security_events(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
        include_metadata: bool = False,
        sort_desc: bool = True,
    ) -> List[Any]:

        try:
            # ------------------------
            # 🔥 BASE FETCH (SECURITY LEVEL)
            # ------------------------
            results = self.get_entries(
                level_filter=AuditLevel.SECURITY,
                limit=limit,
                sort_desc=sort_desc,
                include_metadata=False,  # filter first, convert later
            )

            # ------------------------
            # 🔥 SEVERITY FILTER (OPTIONAL)
            # ------------------------
            if severity:
                severity = str(severity).lower().strip()

                filtered = []
                for e in results:
                    try:
                        details = getattr(e, "details", {})
                        event_severity = str(details.get("severity", "")).lower()

                        if severity == event_severity:
                            filtered.append(e)
                    except:
                        continue

                results = filtered

            # ------------------------
            # 🔥 LIMIT SAFE
            # ------------------------
            limit = max(1, min(limit, 1000))
            results = results[:limit]

            # ------------------------
            # 🔥 OPTIONAL METADATA OUTPUT
            # ------------------------
            if include_metadata:
                enriched = []
                for e in results:
                    try:
                        enriched.append(e.to_dict())
                    except:
                        enriched.append({"error": "serialization_failed"})
                return enriched

            return results

        except Exception as e:
            # ------------------------
            # 🔥 FAILSAFE
            # ------------------------
            try:
                print(f"[AuditLogger] get_security_events error: {e}")
            except:
                pass

            return []

    # ------------------------
    # 🔥 EXPORT FULL LOG
    # ------------------------
    def export_log(
        self,
        filepath: str,
        format: str = "json",
        include_metadata: bool = True,
        compress: bool = False,
    ) -> bool:

        try:
            import json
            import os
            import gzip

            with self.lock:

                # ------------------------
                # 🔥 VALIDATE PATH
                # ------------------------
                filepath = str(filepath).strip()

                if not filepath:
                    return False

                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                # ------------------------
                # 🔥 PREPARE DATA
                # ------------------------
                data = []

                for entry in self.audit_log:
                    try:
                        if include_metadata:
                            data.append(entry.to_dict())
                        else:
                            data.append(
                                {
                                    "action": entry.action,
                                    "actor": entry.actor,
                                    "level": entry.level.value,
                                    "timestamp": entry.timestamp.isoformat(),
                                }
                            )
                    except:
                        continue

                # ------------------------
                # 🔥 EXPORT FORMATS
                # ------------------------
                if format.lower() == "json":

                    serialized = json.dumps(data, indent=2, ensure_ascii=False)

                    # ------------------------
                    # 🔥 OPTIONAL COMPRESSION
                    # ------------------------
                    if compress:
                        with gzip.open(filepath + ".gz", "wt", encoding="utf-8") as f:
                            f.write(serialized)
                    else:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(serialized)

                elif format.lower() == "jsonl":
                    with open(filepath, "w", encoding="utf-8") as f:
                        for entry in data:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                else:
                    return False  # unsupported format

                return True

        except Exception as e:
            try:
                print(f"[AuditLogger] export_log error: {e}")
            except:
                pass

            return False

    # ------------------------
    # 🔥 CLEAR LOG
    # ------------------------
    def clear(
        self, clear_file: bool = True, reset_metrics: bool = True, backup: bool = False
    ) -> bool:

        try:
            import os
            import shutil
            from datetime import datetime

            with self.lock:

                # ------------------------
                # 🔥 BACKUP (OPTIONAL)
                # ------------------------
                if backup and os.path.exists(self.log_file):
                    try:
                        backup_path = (
                            self.log_file + f".bak_{int(datetime.utcnow().timestamp())}"
                        )
                        shutil.copy(self.log_file, backup_path)
                    except:
                        pass

                # ------------------------
                # 🔥 CLEAR MEMORY
                # ------------------------
                cleared_count = len(self.audit_log)
                self.audit_log.clear()

                # ------------------------
                # 🔥 CLEAR FILE
                # ------------------------
                if clear_file:
                    try:
                        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                        with open(self.log_file, "w", encoding="utf-8") as f:
                            f.truncate(0)
                    except:
                        pass

                # ------------------------
                # 🔥 RESET METRICS
                # ------------------------
                if reset_metrics:
                    if hasattr(self, "total_logged"):
                        self.total_logged = 0
                    if hasattr(self, "_pending_writes"):
                        self._pending_writes = 0

                # ------------------------
                # 🔥 LOG ACTION (SELF-AUDIT)
                # ------------------------
                try:
                    self._append_to_file(
                        AuditEntry(
                            action="log_cleared",
                            actor="system",
                            details={"cleared_entries": cleared_count},
                            level=AuditLevel.INFO,
                            result="success",
                        )
                    )
                except:
                    pass

                return True

        except Exception as e:
            try:
                print(f"[AuditLogger] clear error: {e}")
            except:
                pass

            return False

    # ------------------------
    # 🔥 STATS
    # ------------------------
    def get_stats(self) -> Dict[str, Any]:
        try:

            with self.lock:

                # ------------------------
                # 🔥 BASIC COUNTS
                # ------------------------
                total_entries = len(self.audit_log)
                unique_actions = len({e.action for e in self.audit_log})
                unique_actors = len({e.actor for e in self.audit_log})

                # ------------------------
                # 🔥 LEVEL DISTRIBUTION
                # ------------------------
                level_counts = {}
                for e in self.audit_log:
                    lvl = e.level.value
                    level_counts[lvl] = level_counts.get(lvl, 0) + 1

                # ------------------------
                # 🔥 SEVERITY STATS
                # ------------------------
                severe_count = sum(
                    1 for e in self.audit_log if getattr(e, "is_severe", False)
                )
                security_count = sum(
                    1 for e in self.audit_log if getattr(e, "is_security", False)
                )

                # ------------------------
                # 🔥 FILE INFO
                # ------------------------
                file_size = 0
                if os.path.exists(self.log_file):
                    try:
                        file_size = os.path.getsize(self.log_file)
                    except:
                        pass

                # ------------------------
                # 🔥 PERFORMANCE METRICS
                # ------------------------
                total_logged = getattr(self, "total_logged", total_entries)
                pending_writes = getattr(self, "_pending_writes", 0)

                # ------------------------
                # 🔥 LAST ACTIVITY
                # ------------------------
                last_entry_time = None
                if self.audit_log:
                    try:
                        last_entry_time = self.audit_log[-1].timestamp.isoformat()
                    except:
                        pass

                # ------------------------
                # 🔥 HEALTH SCORE
                # ------------------------
                health_score = 1.0

                if total_entries > self.max_entries * 0.9:
                    health_score -= 0.2

                if pending_writes > 50:
                    health_score -= 0.2

                if security_count > 0:
                    health_score -= min(0.3, security_count / max(1, total_entries))

                health_score = max(0.0, round(health_score, 2))

                # ------------------------
                # 🔥 FINAL STATS
                # ------------------------
                return {
                    "total_entries": total_entries,
                    "max_entries": self.max_entries,
                    "unique_actions": unique_actions,
                    "unique_actors": unique_actors,
                    "level_distribution": level_counts,
                    "severe_events": severe_count,
                    "security_events": security_count,
                    "log_file": self.log_file,
                    "file_size_bytes": file_size,
                    "encryption_enabled": self.encrypt_enabled,
                    "total_logged": total_logged,
                    "pending_writes": pending_writes,
                    "last_entry_time": last_entry_time,
                    "health_score": health_score,
                    "status": "healthy" if health_score > 0.7 else "degraded",
                }

        except Exception as e:
            try:
                print(f"[AuditLogger] get_stats error: {e}")
            except:
                pass

            return {"status": "error", "total_entries": 0, "health_score": 0.0}

    def initialize(self, *args, **kwargs) -> bool:
        """
        Initialize AuditLogger runtime safely.
        """
        try:
            if getattr(self, "_initialized_runtime", False):
                return True

            self._initialized_runtime = True
            self._running = False

            # Ensure log directory exists
            try:
                if self.log_file:
                    os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            except Exception:
                pass

            logger.info("📜 AuditLogger initialized")
            return True

        except Exception as e:
            logger.error(f"❌ AuditLogger initialize failed: {e}")
            return False

    def start(self) -> bool:
        """
        Start AuditLogger.
        """
        try:
            if not getattr(self, "_initialized_runtime", False):
                self.initialize()

            if getattr(self, "_running", False):
                return True

            self._running = True

            logger.info("🚀 AuditLogger started")
            return True

        except Exception as e:
            logger.error(f"❌ AuditLogger start failed: {e}")
            self._running = False
            return False

    def stop(self) -> bool:
        """
        Stop AuditLogger safely.
        """
        try:
            if not getattr(self, "_running", False):
                return True

            self._running = False

            # Flush remaining logs if needed
            try:
                if hasattr(self, "_flush_to_file"):
                    self._flush_to_file()
            except Exception:
                pass

            logger.info("🛑 AuditLogger stopped")
            return True

        except Exception as e:
            logger.error(f"❌ AuditLogger stop failed: {e}")
            return False


class AuditFilter:
    def __init__(self):
        # ------------------------
        # 🔥 FILTER CONDITIONS
        # ------------------------
        self.action: Optional[str] = None
        self.actor: Optional[str] = None
        self.level: Optional[AuditLevel] = None

        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        self.severe_only: bool = False
        self.security_only: bool = False

        self.result: Optional[str] = None
        self.trace_id: Optional[str] = None

        self.custom_filters: Dict[str, Any] = {}

    # ------------------------
    # 🔥 BUILDER METHODS
    # ------------------------
    def by_action(self, action: str):
        self.action = action.lower().strip()
        return self

    def by_actor(self, actor: str):
        self.actor = actor.lower().strip()
        return self

    def by_level(self, level):
        if not isinstance(level, AuditLevel):
            level = AuditLevel.from_string(str(level))
        self.level = level
        return self

    def by_time_range(self, start: datetime, end: datetime):
        self.start_time = start
        self.end_time = end
        return self

    def only_severe(self):
        self.severe_only = True
        return self

    def only_security(self):
        self.security_only = True
        return self

    def by_result(self, result: str):
        self.result = result.lower().strip()
        return self

    def by_trace(self, trace_id: str):
        self.trace_id = trace_id
        return self

    def add_custom(self, key: str, value: Any):
        self.custom_filters[key] = value
        return self

    # ------------------------
    # 🔥 APPLY FILTER
    # ------------------------
    def apply(self, entries: List["AuditEntry"]) -> List["AuditEntry"]:
        try:
            results = []

            for e in entries:
                try:
                    # ------------------------
                    # 🔥 ACTION FILTER
                    # ------------------------
                    if self.action and self.action not in str(e.action).lower():
                        continue

                    # ------------------------
                    # 🔥 ACTOR FILTER
                    # ------------------------
                    if self.actor and self.actor not in str(e.actor).lower():
                        continue

                    # ------------------------
                    # 🔥 LEVEL FILTER
                    # ------------------------
                    if self.level and e.level != self.level:
                        continue

                    # ------------------------
                    # 🔥 TIME FILTER
                    # ------------------------
                    if self.start_time and e.timestamp < self.start_time:
                        continue

                    if self.end_time and e.timestamp > self.end_time:
                        continue

                    # ------------------------
                    # 🔥 SEVERITY FILTER
                    # ------------------------
                    if self.severe_only and not getattr(e, "is_severe", False):
                        continue

                    # ------------------------
                    # 🔥 SECURITY FILTER
                    # ------------------------
                    if self.security_only and not getattr(e, "is_security", False):
                        continue

                    # ------------------------
                    # 🔥 RESULT FILTER
                    # ------------------------
                    if self.result and self.result != str(e.result).lower():
                        continue

                    # ------------------------
                    # 🔥 TRACE FILTER
                    # ------------------------
                    if self.trace_id and getattr(e, "trace_id", None) != self.trace_id:
                        continue

                    # ------------------------
                    # 🔥 CUSTOM FILTERS
                    # ------------------------
                    if self.custom_filters:
                        details = getattr(e, "details", {})
                        matched = True

                        for k, v in self.custom_filters.items():
                            if details.get(k) != v:
                                matched = False
                                break

                        if not matched:
                            continue

                    results.append(e)

                except Exception:
                    continue

            return results

        except Exception:
            return []


class AuditAnalyzer:
    def __init__(self):
        self.last_analysis_time = None

    # ------------------------
    # 🔥 BASIC SUMMARY
    # ------------------------
    def summary(self, entries: List["AuditEntry"]) -> Dict[str, Any]:
        try:
            total = len(entries)

            actions = Counter(e.action for e in entries)
            actors = Counter(e.actor for e in entries)
            levels = Counter(e.level.value for e in entries)

            severe = sum(1 for e in entries if getattr(e, "is_severe", False))
            security = sum(1 for e in entries if getattr(e, "is_security", False))

            return {
                "total_entries": total,
                "top_actions": actions.most_common(5),
                "top_actors": actors.most_common(5),
                "level_distribution": dict(levels),
                "severe_count": severe,
                "security_count": security,
            }

        except Exception:
            return {}

    # ------------------------
    # 🔥 ANOMALY DETECTION
    # ------------------------
    def detect_anomalies(self, entries: List["AuditEntry"]) -> List[Dict[str, Any]]:
        anomalies = []

        try:
            now = datetime.utcnow()
            last_5_min = now - timedelta(minutes=5)

            recent = [e for e in entries if e.timestamp >= last_5_min]

            # ------------------------
            # 🔥 HIGH ERROR RATE
            # ------------------------
            errors = [e for e in recent if e.level == AuditLevel.ERROR]
            if len(errors) > 10:
                anomalies.append(
                    {
                        "type": "high_error_rate",
                        "count": len(errors),
                        "message": "Too many errors in last 5 minutes",
                    }
                )

            # ------------------------
            # 🔥 SECURITY SPIKE
            # ------------------------
            security = [e for e in recent if getattr(e, "is_security", False)]
            if len(security) > 5:
                anomalies.append(
                    {
                        "type": "security_spike",
                        "count": len(security),
                        "message": "Spike in security events",
                    }
                )

            # ------------------------
            # 🔥 SINGLE ACTOR SPAM
            # ------------------------
            actor_counts = Counter(e.actor for e in recent)
            for actor, count in actor_counts.items():
                if count > 20:
                    anomalies.append(
                        {
                            "type": "actor_spam",
                            "actor": actor,
                            "count": count,
                            "message": "High activity from single actor",
                        }
                    )

        except Exception:
            pass

        return anomalies

    # ------------------------
    # 🔥 TREND ANALYSIS
    # ------------------------
    def trends(self, entries: List["AuditEntry"]) -> Dict[str, Any]:
        try:
            hourly = {}

            for e in entries:
                hour = e.timestamp.replace(minute=0, second=0, microsecond=0)

                if hour not in hourly:
                    hourly[hour] = 0

                hourly[hour] += 1

            return {"hourly_activity": {str(k): v for k, v in sorted(hourly.items())}}

        except Exception:
            return {}

    # ------------------------
    # 🔥 RISK SCORE
    # ------------------------
    def risk_score(self, entries: List["AuditEntry"]) -> float:
        try:
            if not entries:
                return 0.0

            score = 0

            for e in entries:
                if getattr(e, "is_security", False):
                    score += 3
                elif getattr(e, "is_severe", False):
                    score += 2
                else:
                    score += 0.5

            normalized = score / len(entries)

            return round(min(10.0, normalized), 2)

        except Exception:
            return 0.0

    # ------------------------
    # 🔥 FULL ANALYSIS
    # ------------------------
    def analyze(self, entries: List["AuditEntry"]) -> Dict[str, Any]:
        try:
            self.last_analysis_time = datetime.utcnow()

            return {
                "summary": self.summary(entries),
                "anomalies": self.detect_anomalies(entries),
                "trends": self.trends(entries),
                "risk_score": self.risk_score(entries),
                "analyzed_at": self.last_analysis_time.isoformat(),
            }

        except Exception:
            return {}


__all__ = [
    # ------------------------
    # 🔥 CORE CLASSES
    # ------------------------
    "AuditLogger",
    "AuditEntry",
    "AuditLevel",
    # ------------------------
    # 🔥 INTERNAL HELPERS (OPTIONAL EXPORT)
    # ------------------------
    "EncryptionManager",
    # ------------------------
    # 🔥 FUTURE EXTENSIONS (SAFE EXPORT PLACEHOLDER)
    # ------------------------
    "AuditFilter",
    "AuditAnalyzer",
]
