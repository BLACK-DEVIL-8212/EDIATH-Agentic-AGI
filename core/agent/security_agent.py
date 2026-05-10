"""
Security Agent for EDIATH
Advanced security operations: authentication, authorization, encryption, audit logging, threat detection
"""

import asyncio
import hashlib
import secrets
import base64
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import logging
import hmac

# Cryptography
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# JWT
try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# Password strength
try:
    import zxcvbn

    ZXCVBN_AVAILABLE = True
except ImportError:
    ZXCVBN_AVAILABLE = False


class SecurityLevel(Enum):
    """Security levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(Enum):
    """Encryption algorithms"""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20 = "chacha20"
    RSA_OAEP = "rsa-oaep"


class HashAlgorithm(Enum):
    """Hash algorithms"""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    BCRYPT = "bcrypt"


class Permission(Enum):
    """Permission types"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    AUDIT = "audit"


@dataclass
class User:
    """User information"""

    id: str
    username: str
    email: str
    role: str
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    mfa_enabled: bool = False


@dataclass
class AuditEntry:
    """Audit log entry"""

    id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    severity: SecurityLevel = SecurityLevel.MEDIUM


@dataclass
class SecurityEvent:
    """Security event for monitoring"""

    id: str
    type: str
    severity: SecurityLevel
    timestamp: datetime
    source: str
    description: str
    details: Dict[str, Any]
    resolved: bool = False


class SecurityAgent:
    """
    Advanced security agent capable of:
    - Authentication and authorization
    - Password hashing and verification
    - JWT token management
    - Encryption/decryption (symmetric & asymmetric)
    - Audit logging
    - Security event monitoring
    - Threat detection
    - Rate limiting
    - Input sanitization
    - SQL injection prevention
    - XSS protection
    - CSRF token generation
    - API key management
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Security Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Security settings
        self.secret_key = self.config.get("secret_key", secrets.token_hex(32))
        self.jwt_secret = self.config.get("jwt_secret", secrets.token_hex(32))
        self.jwt_expiry_hours = self.config.get("jwt_expiry_hours", 24)
        self.password_min_length = self.config.get("password_min_length", 8)

        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.rate_limit_config = self.config.get(
            "rate_limits",
            {
                "login": {"max_attempts": 5, "window_seconds": 300},
                "api": {"max_requests": 100, "window_seconds": 60},
                "password_reset": {"max_attempts": 3, "window_seconds": 3600},
            },
        )

        # Encryption keys
        self.encryption_keys: Dict[str, bytes] = {}
        self._init_encryption_keys()

        # User storage (in production, use database)
        self.users: Dict[str, User] = {}
        self.password_hashes: Dict[str, str] = {}

        # Audit and events
        self.audit_logs: List[AuditEntry] = []
        self.security_events: List[SecurityEvent] = []
        self.max_logs = self.config.get("max_logs", 10000)

        # API keys
        self.api_keys: Dict[str, Dict] = {}

        # Statistics
        self.stats = {
            "total_auth_attempts": 0,
            "successful_auth": 0,
            "failed_auth": 0,
            "total_audit_entries": 0,
            "security_events": 0,
            "rate_limit_hits": 0,
        }

        self.logger.info("Security Agent initialized")

    def _init_encryption_keys(self):
        """Initialize encryption keys"""
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Cryptography library not available")
            return

        # Generate or load master key
        key_file = Path(self.config.get("key_file", "./security_keys/master.key"))
        if key_file.exists():
            with open(key_file, "rb") as f:
                self.master_key = f.read()
        else:
            self.master_key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(self.master_key)

        self.cipher = Fernet(self.master_key)

    def _generate_id(self, prefix: str = "sec") -> str:
        """Generate unique ID"""
        import uuid

        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _check_rate_limit(self, key: str, limit_type: str) -> Tuple[bool, int]:
        """Check if rate limit is exceeded"""
        if limit_type not in self.rate_limit_config:
            return True, 0

        config = self.rate_limit_config[limit_type]
        max_attempts = config["max_attempts"]
        window_seconds = config["window_seconds"]

        now = datetime.now()

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old entries
        self.rate_limits[key] = [
            ts
            for ts in self.rate_limits[key]
            if (now - ts).total_seconds() < window_seconds
        ]

        if len(self.rate_limits[key]) >= max_attempts:
            wait_time = (
                window_seconds - (now - self.rate_limits[key][0]).total_seconds()
            )
            return False, int(wait_time)

        self.rate_limits[key].append(now)
        return True, 0

    # ============
    # Password Management
    # ============

    async def hash_password(
        self, password: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256
    ) -> Dict[str, Any]:
        """
        Hash a password securely

        Args:
            password: Plain text password
            algorithm: Hash algorithm to use

        Returns:
            Dictionary with hash and salt
        """
        try:
            salt = secrets.token_hex(16)

            if algorithm == HashAlgorithm.BCRYPT:
                import bcrypt

                password_bytes = password.encode("utf-8")
                salt_bytes = bcrypt.gensalt()
                hash_bytes = bcrypt.hashpw(password_bytes, salt_bytes)
                hash_str = hash_bytes.decode("utf-8")
            else:
                # PBKDF2 for other algorithms
                iterations = 100000
                hash_func = {
                    HashAlgorithm.SHA256: hashlib.sha256,
                    HashAlgorithm.SHA512: hashlib.sha512,
                    HashAlgorithm.SHA1: hashlib.sha1,
                    HashAlgorithm.MD5: hashlib.md5,
                }.get(algorithm, hashlib.sha256)

                password_bytes = password.encode("utf-8")
                salt_bytes = salt.encode("utf-8")

                hash_bytes = hashlib.pbkdf2_hmac(
                    algorithm.value, password_bytes, salt_bytes, iterations
                )
                hash_str = base64.b64encode(hash_bytes).decode("utf-8")

            return {
                "success": True,
                "hash": hash_str,
                "salt": salt if algorithm != HashAlgorithm.BCRYPT else None,
                "algorithm": algorithm.value,
                "message": "Password hashed successfully",
            }

        except Exception as e:
            self.logger.error(f"Password hashing error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def verify_password(
        self,
        password: str,
        hash_str: str,
        salt: Optional[str] = None,
        algorithm: str = "sha256",
    ) -> Dict[str, Any]:
        """
        Verify a password against its hash

        Args:
            password: Plain text password to verify
            hash_str: Stored hash
            salt: Salt (if used)
            algorithm: Hash algorithm used

        Returns:
            Dictionary with verification result
        """
        try:
            if algorithm == "bcrypt":
                import bcrypt

                password_bytes = password.encode("utf-8")
                hash_bytes = hash_str.encode("utf-8")
                is_valid = bcrypt.checkpw(password_bytes, hash_bytes)
            else:
                iterations = 100000
                password_bytes = password.encode("utf-8")
                salt_bytes = salt.encode("utf-8")

                hash_func = {
                    "sha256": hashlib.sha256,
                    "sha512": hashlib.sha512,
                    "sha1": hashlib.sha1,
                    "md5": hashlib.md5,
                }.get(algorithm, hashlib.sha256)

                new_hash_bytes = hashlib.pbkdf2_hmac(
                    algorithm, password_bytes, salt_bytes, iterations
                )
                new_hash_str = base64.b64encode(new_hash_bytes).decode("utf-8")
                is_valid = hmac.compare_digest(new_hash_str, hash_str)

            return {
                "success": True,
                "valid": is_valid,
                "message": "Password verified" if is_valid else "Invalid password",
            }

        except Exception as e:
            self.logger.error(f"Password verification error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def check_password_strength(self, password: str) -> Dict[str, Any]:
        """
        Check password strength

        Args:
            password: Password to check

        Returns:
            Dictionary with strength assessment
        """
        score = 0
        feedback = []

        # Length check
        if len(password) >= self.password_min_length:
            score += 1
        else:
            feedback.append(
                f"Password should be at least {self.password_min_length} characters"
            )

        # Complexity checks
        if re.search(r"[A-Z]", password):
            score += 1
        else:
            feedback.append("Add uppercase letters")

        if re.search(r"[a-z]", password):
            score += 1
        else:
            feedback.append("Add lowercase letters")

        if re.search(r"\d", password):
            score += 1
        else:
            feedback.append("Add numbers")

        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            feedback.append("Add special characters")

        # Use zxcvbn if available
        zxcvbn_score = None
        if ZXCVBN_AVAILABLE:
            result = zxcvbn.password_strength(password)
            zxcvbn_score = result["score"]
            score = max(score, zxcvbn_score)
            if result["feedback"]["warning"]:
                feedback.append(result["feedback"]["warning"])

        # Determine strength level
        if score >= 5:
            strength = "very_strong"
        elif score >= 4:
            strength = "strong"
        elif score >= 3:
            strength = "moderate"
        elif score >= 2:
            strength = "weak"
        else:
            strength = "very_weak"

        return {
            "success": True,
            "score": score,
            "strength": strength,
            "feedback": feedback,
            "zxcvbn_score": zxcvbn_score,
        }

    # ============
    # JWT Token Management
    # ============

    async def generate_jwt(
        self,
        user_id: str,
        claims: Optional[Dict] = None,
        expiry_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JWT token

        Args:
            user_id: User identifier
            claims: Additional claims to include
            expiry_hours: Token expiry in hours

        Returns:
            Dictionary with JWT token
        """
        if not JWT_AVAILABLE:
            return {"success": False, "error": "JWT library not available"}

        try:
            expiry = datetime.utcnow() + timedelta(
                hours=expiry_hours or self.jwt_expiry_hours
            )

            payload = {
                "user_id": user_id,
                "exp": expiry,
                "iat": datetime.utcnow(),
                "jti": self._generate_id("jwt"),
            }

            if claims:
                payload.update(claims)

            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")

            return {
                "success": True,
                "token": token,
                "expires_at": expiry.isoformat(),
                "user_id": user_id,
            }

        except Exception as e:
            self.logger.error(f"JWT generation error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def verify_jwt(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token

        Args:
            token: JWT token to verify

        Returns:
            Dictionary with verification result
        """
        if not JWT_AVAILABLE:
            return {"success": False, "error": "JWT library not available"}

        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

            return {
                "success": True,
                "valid": True,
                "payload": payload,
                "user_id": payload.get("user_id"),
                "expires_at": datetime.fromtimestamp(payload["exp"]).isoformat(),
            }

        except jwt.ExpiredSignatureError:
            return {"success": True, "valid": False, "error": "Token expired"}
        except jwt.InvalidTokenError as e:
            return {"success": True, "valid": False, "error": str(e)}
        except Exception as e:
            self.logger.error(f"JWT verification error: {str(e)}")
            return {"success": False, "error": str(e)}

    # ============
    # Encryption/Decryption
    # ============

    async def encrypt_data(
        self, data: str, key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encrypt sensitive data

        Args:
            data: Data to encrypt
            key_id: Specific key to use (default uses master key)

        Returns:
            Dictionary with encrypted data
        """
        if not CRYPTO_AVAILABLE:
            return {"success": False, "error": "Cryptography library not available"}

        try:
            cipher = self.cipher
            encrypted = cipher.encrypt(data.encode("utf-8"))

            return {
                "success": True,
                "encrypted_data": base64.b64encode(encrypted).decode("utf-8"),
                "key_id": key_id or "master",
                "message": "Data encrypted successfully",
            }

        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def decrypt_data(
        self, encrypted_data: str, key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Decrypt encrypted data

        Args:
            encrypted_data: Base64 encoded encrypted data
            key_id: Key used for encryption

        Returns:
            Dictionary with decrypted data
        """
        if not CRYPTO_AVAILABLE:
            return {"success": False, "error": "Cryptography library not available"}

        try:
            cipher = self.cipher
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = cipher.decrypt(encrypted_bytes)

            return {
                "success": True,
                "decrypted_data": decrypted.decode("utf-8"),
                "message": "Data decrypted successfully",
            }

        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_api_key(
        self, name: str, permissions: List[Permission], expiry_days: int = 365
    ) -> Dict[str, Any]:
        """
        Generate API key

        Args:
            name: Key name/description
            permissions: Permissions for this key
            expiry_days: Days until key expires

        Returns:
            Dictionary with API key
        """
        api_key = secrets.token_urlsafe(32)
        api_secret = secrets.token_urlsafe(32)

        key_id = self._generate_id("apikey")

        self.api_keys[key_id] = {
            "name": name,
            "api_key": api_key,
            "api_secret_hash": hashlib.sha256(api_secret.encode()).hexdigest(),
            "permissions": [p.value for p in permissions],
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=expiry_days),
            "last_used": None,
            "is_active": True,
        }

        return {
            "success": True,
            "key_id": key_id,
            "api_key": api_key,
            "api_secret": api_secret,
            "message": "API key generated. Store the secret securely!",
        }

    async def verify_api_key(self, api_key: str, api_secret: str) -> Dict[str, Any]:
        """
        Verify API key and secret

        Args:
            api_key: API key
            api_secret: API secret

        Returns:
            Dictionary with verification result
        """
        for key_id, key_data in self.api_keys.items():
            if key_data["api_key"] == api_key:
                # Check expiry
                if key_data["expires_at"] < datetime.now():
                    return {"success": True, "valid": False, "error": "API key expired"}

                if not key_data["is_active"]:
                    return {
                        "success": True,
                        "valid": False,
                        "error": "API key inactive",
                    }

                # Verify secret
                secret_hash = hashlib.sha256(api_secret.encode()).hexdigest()
                if secret_hash == key_data["api_secret_hash"]:
                    key_data["last_used"] = datetime.now()
                    return {
                        "success": True,
                        "valid": True,
                        "key_id": key_id,
                        "permissions": key_data["permissions"],
                        "name": key_data["name"],
                    }
                else:
                    return {"success": True, "valid": False, "error": "Invalid secret"}

        return {"success": True, "valid": False, "error": "API key not found"}

    # ============
    # Input Sanitization
    # ============

    async def sanitize_input(
        self,
        input_data: str,
        prevent_sql_injection: bool = True,
        prevent_xss: bool = True,
        max_length: int = 10000,
    ) -> Dict[str, Any]:
        """
        Sanitize user input

        Args:
            input_data: Raw user input
            prevent_sql_injection: Remove SQL patterns
            prevent_xss: Remove XSS patterns
            max_length: Maximum allowed length

        Returns:
            Dictionary with sanitized output
        """
        original_length = len(input_data)

        if len(input_data) > max_length:
            return {
                "success": False,
                "error": f"Input exceeds maximum length of {max_length} characters",
            }

        sanitized = input_data

        if prevent_sql_injection:
            # Remove common SQL injection patterns
            sql_patterns = [
                r"(\bSELECT\b.*\bFROM\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bUPDATE\b.*\bSET\b)",
                r"(\bDELETE\b.*\bFROM\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bOR\b.*=.*\bOR\b)",
                r"(\bAND\b.*=.*\bAND\b)",
                r"(--)",
                r"(;)",
                r"('.*' OR '1'='1')",
            ]

            for pattern in sql_patterns:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

        if prevent_xss:
            # Escape HTML characters
            html_escape_table = {
                "&": "&amp;",
                '"': "&quot;",
                "'": "&apos;",
                ">": "&gt;",
                "<": "&lt;",
            }
            sanitized = "".join(html_escape_table.get(c, c) for c in sanitized)

            # Remove script tags
            sanitized = re.sub(
                r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
                "",
                sanitized,
                flags=re.IGNORECASE,
            )
            sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r"on\w+\s*=", "", sanitized, flags=re.IGNORECASE)

        return {
            "success": True,
            "original_length": original_length,
            "sanitized_length": len(sanitized),
            "sanitized_input": sanitized,
            "sql_injection_prevented": prevent_sql_injection,
            "xss_prevented": prevent_xss,
        }

    async def generate_csrf_token(self, session_id: str) -> Dict[str, Any]:
        """
        Generate CSRF token

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with CSRF token
        """
        token = secrets.token_urlsafe(32)

        # Store token with expiry
        if not hasattr(self, "csrf_tokens"):
            self.csrf_tokens = {}

        self.csrf_tokens[session_id] = {
            "token": token,
            "expires_at": datetime.now() + timedelta(hours=1),
        }

        return {
            "success": True,
            "csrf_token": token,
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
        }

    async def verify_csrf_token(self, session_id: str, token: str) -> Dict[str, Any]:
        """
        Verify CSRF token

        Args:
            session_id: Session identifier
            token: Token to verify

        Returns:
            Dictionary with verification result
        """
        if not hasattr(self, "csrf_tokens"):
            return {"success": True, "valid": False, "error": "No CSRF token found"}

        if session_id not in self.csrf_tokens:
            return {"success": True, "valid": False, "error": "Invalid session"}

        stored = self.csrf_tokens[session_id]

        if stored["expires_at"] < datetime.now():
            del self.csrf_tokens[session_id]
            return {"success": True, "valid": False, "error": "Token expired"}

        if stored["token"] != token:
            return {"success": True, "valid": False, "error": "Invalid token"}

        return {"success": True, "valid": True, "message": "Token verified"}

    # ============
    # Audit Logging
    # ============

    async def audit_log(
        self,
        user_id: str,
        action: str,
        resource: str,
        status: str,
        details: Dict[str, Any],
        severity: SecurityLevel = SecurityLevel.MEDIUM,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create audit log entry

        Args:
            user_id: User performing action
            action: Action performed
            resource: Resource affected
            status: Success/failure status
            details: Additional details
            severity: Event severity
            ip_address: Source IP address

        Returns:
            Dictionary with audit result
        """
        entry = AuditEntry(
            id=self._generate_id("audit"),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            details=details,
            ip_address=ip_address,
            severity=severity,
        )

        self.audit_logs.append(entry)
        self.stats["total_audit_entries"] += 1

        # Trim if needed
        if len(self.audit_logs) > self.max_logs:
            self.audit_logs = self.audit_logs[-self.max_logs :]

        # Log to file
        self.logger.info(
            f"AUDIT: user={user_id} action={action} resource={resource} status={status}"
        )

        return {
            "success": True,
            "audit_id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
        }

    async def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Retrieve audit logs

        Args:
            user_id: Filter by user
            action: Filter by action
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of logs

        Returns:
            Dictionary with audit logs
        """
        logs = self.audit_logs

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if action:
            logs = [l for l in logs if l.action == action]
        if start_date:
            logs = [l for l in logs if l.timestamp >= start_date]
        if end_date:
            logs = [l for l in logs if l.timestamp <= end_date]

        logs = logs[-limit:]

        return {
            "success": True,
            "total": len(logs),
            "logs": [
                {
                    "id": l.id,
                    "timestamp": l.timestamp.isoformat(),
                    "user_id": l.user_id,
                    "action": l.action,
                    "resource": l.resource,
                    "status": l.status,
                    "severity": l.severity.value,
                    "details": l.details,
                }
                for l in logs
            ],
        }

    # ============
    # Security Event Monitoring
    # ============

    async def report_security_event(
        self,
        event_type: str,
        description: str,
        severity: SecurityLevel,
        source: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Report a security event

        Args:
            event_type: Type of security event
            description: Event description
            severity: Event severity
            source: Event source
            details: Event details

        Returns:
            Dictionary with event result
        """
        event = SecurityEvent(
            id=self._generate_id("secevt"),
            type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            source=source,
            description=description,
            details=details,
        )

        self.security_events.append(event)
        self.stats["security_events"] += 1

        # Log based on severity
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"SECURITY: {event_type} - {description}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"SECURITY: {event_type} - {description}")
        else:
            self.logger.warning(f"SECURITY: {event_type} - {description}")

        # Trim if needed
        if len(self.security_events) > self.max_logs:
            self.security_events = self.security_events[-self.max_logs :]

        return {
            "success": True,
            "event_id": event.id,
            "timestamp": event.timestamp.isoformat(),
        }

    async def get_security_events(
        self,
        severity: Optional[SecurityLevel] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Retrieve security events

        Args:
            severity: Filter by severity
            unresolved_only: Only unresolved events
            limit: Maximum number of events

        Returns:
            Dictionary with security events
        """
        events = self.security_events

        if severity:
            events = [e for e in events if e.severity == severity]
        if unresolved_only:
            events = [e for e in events if not e.resolved]

        events = events[-limit:]

        return {
            "success": True,
            "total": len(events),
            "events": [
                {
                    "id": e.id,
                    "type": e.type,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat(),
                    "source": e.source,
                    "description": e.description,
                    "resolved": e.resolved,
                }
                for e in events
            ],
        }

    async def resolve_security_event(self, event_id: str) -> Dict[str, Any]:
        """
        Mark security event as resolved

        Args:
            event_id: Event identifier

        Returns:
            Dictionary with resolution result
        """
        for event in self.security_events:
            if event.id == event_id:
                event.resolved = True
                return {
                    "success": True,
                    "message": f"Event {event_id} marked as resolved",
                }

        return {"success": False, "error": f"Event {event_id} not found"}

    # ============
    # Threat Detection
    # ============

    async def detect_suspicious_activity(
        self, user_id: str, action: str, ip_address: str
    ) -> Dict[str, Any]:
        """
        Detect suspicious activity patterns

        Args:
            user_id: User identifier
            action: Action being performed
            ip_address: Source IP address

        Returns:
            Dictionary with detection result
        """
        suspicious = False
        reasons = []

        # Check for rapid consecutive actions
        recent_actions = [
            l
            for l in self.audit_logs
            if l.user_id == user_id
            and (datetime.now() - l.timestamp).total_seconds() < 60
        ]

        if len(recent_actions) > 10:
            suspicious = True
            reasons.append(f"Rapid actions: {len(recent_actions)} in last minute")

        # Check for failed authentication attempts
        failed_auth = [
            l
            for l in self.audit_logs
            if l.user_id == user_id
            and l.action == "login"
            and l.status == "failed"
            and (datetime.now() - l.timestamp).total_seconds() < 300
        ]

        if len(failed_auth) > 5:
            suspicious = True
            reasons.append(
                f"Multiple failed logins: {len(failed_auth)} in last 5 minutes"
            )

        # Check for unusual access patterns
        if action in ["delete", "admin", "permission_change"]:
            suspicious = True
            reasons.append(f"Sensitive action: {action}")

        if suspicious:
            await self.report_security_event(
                event_type="suspicious_activity",
                description=f"Suspicious activity detected for user {user_id}",
                severity=SecurityLevel.HIGH,
                source="threat_detection",
                details={
                    "user_id": user_id,
                    "action": action,
                    "ip_address": ip_address,
                    "reasons": reasons,
                },
            )

        return {"success": True, "suspicious": suspicious, "reasons": reasons}

    # ============
    # Rate Limiting
    # ============

    async def check_rate_limit(self, key: str, limit_type: str) -> Dict[str, Any]:
        """
        Check if action is rate limited

        Args:
            key: Rate limit key (e.g., user_id, IP address)
            limit_type: Type of rate limit to check

        Returns:
            Dictionary with rate limit status
        """
        allowed, wait_time = self._check_rate_limit(key, limit_type)

        if not allowed:
            self.stats["rate_limit_hits"] += 1
            await self.report_security_event(
                event_type="rate_limit_exceeded",
                description=f"Rate limit exceeded for {key}",
                severity=SecurityLevel.MEDIUM,
                source="rate_limiter",
                details={"key": key, "limit_type": limit_type, "wait_time": wait_time},
            )

        return {
            "success": True,
            "allowed": allowed,
            "wait_time": wait_time if not allowed else 0,
            "limit_type": limit_type,
        }

    # ============
    # User Management
    # ============

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: str = "user",
        permissions: List[Permission] = None,
    ) -> Dict[str, Any]:
        """
        Create a new user

        Args:
            username: Username
            email: Email address
            password: Password
            role: User role
            permissions: User permissions

        Returns:
            Dictionary with user creation result
        """
        # Check password strength
        strength = await self.check_password_strength(password)
        if strength["strength"] in ["very_weak", "weak"]:
            return {
                "success": False,
                "error": "Password too weak",
                "feedback": strength["feedback"],
            }

        # Hash password
        hash_result = await self.hash_password(password, HashAlgorithm.SHA256)
        if not hash_result["success"]:
            return hash_result

        user_id = self._generate_id("user")

        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions or [Permission.READ],
            created_at=datetime.now(),
        )

        self.users[user_id] = user
        self.password_hashes[user_id] = hash_result["hash"]

        await self.audit_log(
            user_id=user_id,
            action="create_user",
            resource=f"user/{user_id}",
            status="success",
            details={"username": username, "role": role},
        )

        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "message": "User created successfully",
        }

    async def authenticate_user(
        self, username: str, password: str, ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Authenticate a user

        Args:
            username: Username
            password: Password
            ip_address: Source IP address

        Returns:
            Dictionary with authentication result
        """
        self.stats["total_auth_attempts"] += 1

        # Check rate limit
        rate_check = await self.check_rate_limit(f"auth_{username}", "login")
        if not rate_check["allowed"]:
            return {
                "success": False,
                "error": f'Too many attempts. Try again in {rate_check["wait_time"]} seconds',
            }

        # Find user
        user = None
        user_id = None
        for uid, u in self.users.items():
            if u.username == username:
                user = u
                user_id = uid
                break

        if not user:
            self.stats["failed_auth"] += 1
            await self.audit_log(
                user_id="unknown",
                action="login",
                resource="auth",
                status="failed",
                details={"username": username, "reason": "user_not_found"},
                ip_address=ip_address,
            )
            return {"success": False, "error": "Invalid credentials"}

        # Verify password
        hash_result = await self.verify_password(
            password, self.password_hashes[user_id], algorithm="sha256"
        )

        if not hash_result["valid"]:
            self.stats["failed_auth"] += 1
            await self.audit_log(
                user_id=user_id,
                action="login",
                resource="auth",
                status="failed",
                details={"reason": "invalid_password"},
                ip_address=ip_address,
            )

            # Check for suspicious activity
            await self.detect_suspicious_activity(
                user_id, "login", ip_address or "unknown"
            )

            return {"success": False, "error": "Invalid credentials"}

        # Success
        self.stats["successful_auth"] += 1
        user.last_login = datetime.now()

        # Generate JWT
        jwt_result = await self.generate_jwt(user_id)

        await self.audit_log(
            user_id=user_id,
            action="login",
            resource="auth",
            status="success",
            details={"username": username},
            ip_address=ip_address,
        )

        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "role": user.role,
            "permissions": [p.value for p in user.permissions],
            "token": jwt_result.get("token"),
            "message": "Authentication successful",
        }

    async def check_permission(
        self, user_id: str, permission: Permission, resource: str
    ) -> Dict[str, Any]:
        """
        Check if user has permission

        Args:
            user_id: User identifier
            permission: Permission to check
            resource: Resource being accessed

        Returns:
            Dictionary with permission check result
        """
        if user_id not in self.users:
            return {"success": True, "authorized": False, "reason": "User not found"}

        user = self.users[user_id]

        # Admin has all permissions
        if user.role == "admin":
            return {"success": True, "authorized": True}

        if permission in user.permissions:
            return {"success": True, "authorized": True}

        await self.audit_log(
            user_id=user_id,
            action="permission_check",
            resource=resource,
            status="denied",
            details={"required_permission": permission.value},
            severity=SecurityLevel.HIGH,
        )

        return {
            "success": True,
            "authorized": False,
            "reason": "Insufficient permissions",
        }

    # ============
    # Utility Methods
    # ============

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            "total_users": len(self.users),
            "active_api_keys": len(
                [k for k in self.api_keys.values() if k["is_active"]]
            ),
            "audit_logs_count": len(self.audit_logs),
            "security_events_count": len(self.security_events),
            "unresolved_events": len(
                [e for e in self.security_events if not e.resolved]
            ),
        }

    async def rotate_keys(self) -> Dict[str, Any]:
        """
        Rotate encryption keys
        """
        if not CRYPTO_AVAILABLE:
            return {"success": False, "error": "Cryptography library not available"}

        old_key = self.master_key
        new_key = Fernet.generate_key()

        # Re-encrypt sensitive data would be needed in production
        self.master_key = new_key
        self.cipher = Fernet(new_key)

        # Save new key
        key_file = Path(self.config.get("key_file", "./security_keys/master.key"))
        with open(key_file, "wb") as f:
            f.write(new_key)

        await self.report_security_event(
            event_type="key_rotation",
            description="Master encryption key rotated",
            severity=SecurityLevel.HIGH,
            source="security_agent",
            details={"timestamp": datetime.now().isoformat()},
        )

        return {"success": True, "message": "Keys rotated successfully"}


# Integration wrapper for EDIATH
class SecurityAgentWrapper:
    """
    Wrapper class to integrate SecurityAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.security_agent = SecurityAgent(config)
        self.agent_type = "security"
        self.capabilities = [
            "authenticate_user",
            "check_permission",
            "encrypt_data",
            "decrypt_data",
            "audit_log",
            "sanitize_input",
            "generate_jwt",
            "verify_jwt",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a security request

        Request format:
        {
            'operation': 'authenticate|encrypt|decrypt|audit|sanitize|check_permission|...',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "authenticate":
            return await self.security_agent.authenticate_user(
                username=request.get("username"),
                password=request.get("password"),
                ip_address=request.get("ip_address"),
            )

        elif operation == "check_permission":
            permission = Permission(request.get("permission"))
            return await self.security_agent.check_permission(
                user_id=request.get("user_id"),
                permission=permission,
                resource=request.get("resource"),
            )

        elif operation == "encrypt":
            return await self.security_agent.encrypt_data(
                data=request.get("data"), key_id=request.get("key_id")
            )

        elif operation == "decrypt":
            return await self.security_agent.decrypt_data(
                encrypted_data=request.get("encrypted_data"),
                key_id=request.get("key_id"),
            )

        elif operation == "audit":
            severity = SecurityLevel(request.get("severity", "medium"))
            return await self.security_agent.audit_log(
                user_id=request.get("user_id"),
                action=request.get("action"),
                resource=request.get("resource"),
                status=request.get("status"),
                details=request.get("details", {}),
                severity=severity,
                ip_address=request.get("ip_address"),
            )

        elif operation == "get_audit_logs":
            start_date = request.get("start_date")
            end_date = request.get("end_date")
            if start_date:
                start_date = datetime.fromisoformat(start_date)
            if end_date:
                end_date = datetime.fromisoformat(end_date)

            return await self.security_agent.get_audit_logs(
                user_id=request.get("user_id"),
                action=request.get("action"),
                start_date=start_date,
                end_date=end_date,
                limit=request.get("limit", 100),
            )

        elif operation == "sanitize":
            return await self.security_agent.sanitize_input(
                input_data=request.get("input"),
                prevent_sql_injection=request.get("prevent_sql", True),
                prevent_xss=request.get("prevent_xss", True),
                max_length=request.get("max_length", 10000),
            )

        elif operation == "generate_jwt":
            return await self.security_agent.generate_jwt(
                user_id=request.get("user_id"),
                claims=request.get("claims"),
                expiry_hours=request.get("expiry_hours"),
            )

        elif operation == "verify_jwt":
            return await self.security_agent.verify_jwt(token=request.get("token"))

        elif operation == "check_password":
            return await self.security_agent.check_password_strength(
                password=request.get("password")
            )

        elif operation == "create_user":
            permissions = request.get("permissions", ["read"])
            permissions = [Permission(p) for p in permissions]
            return await self.security_agent.create_user(
                username=request.get("username"),
                email=request.get("email"),
                password=request.get("password"),
                role=request.get("role", "user"),
                permissions=permissions,
            )

        elif operation == "rate_limit":
            return await self.security_agent.check_rate_limit(
                key=request.get("key"), limit_type=request.get("limit_type")
            )

        elif operation == "csrf_token":
            return await self.security_agent.generate_csrf_token(
                session_id=request.get("session_id")
            )

        elif operation == "verify_csrf":
            return await self.security_agent.verify_csrf_token(
                session_id=request.get("session_id"), token=request.get("token")
            )

        elif operation == "generate_api_key":
            permissions = [Permission(p) for p in request.get("permissions", ["read"])]
            return await self.security_agent.generate_api_key(
                name=request.get("name"),
                permissions=permissions,
                expiry_days=request.get("expiry_days", 365),
            )

        elif operation == "security_events":
            severity = request.get("severity")
            if severity:
                severity = SecurityLevel(severity)
            return await self.security_agent.get_security_events(
                severity=severity,
                unresolved_only=request.get("unresolved_only", False),
                limit=request.get("limit", 100),
            )

        elif operation == "stats":
            return self.security_agent.get_stats()

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "SecurityAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.security_agent.get_stats(),
            "encryption_available": CRYPTO_AVAILABLE,
            "jwt_available": JWT_AVAILABLE,
        }


# Example usage and testing
async def test_security_agent():
    """Test the security agent functionality"""

    # Initialize agent
    agent = SecurityAgent()

    print("=== Security Agent Test ===\n")

    # Test password strength checking
    print("1. Password Strength Check...")
    result = await agent.check_password_strength("weak")
    print(f"   Strength: {result['strength']}")
    print(f"   Feedback: {result['feedback'][:2]}")

    result = await agent.check_password_strength("Str0ng!P@ssw0rd2024")
    print(f"   Strong password strength: {result['strength']}")

    # Test user creation
    print("\n2. Creating User...")
    result = await agent.create_user(
        username="testuser",
        email="test@example.com",
        password="SecurePass123!",
        role="user",
        permissions=["read", "write"],
    )

    if result["success"]:
        user_id = result["user_id"]
        print(f"   User created: {result['username']} (ID: {user_id})")
    else:
        print(f"   Error: {result.get('error')}")
        user_id = None

    # Test authentication
    print("\n3. Authenticating User...")
    result = await agent.authenticate_user("testuser", "SecurePass123!")
    if result["success"]:
        print("   Authentication successful")
        print(f"   Token: {result.get('token', 'N/A')[:50]}...")
    else:
        print(f"   Authentication failed: {result.get('error')}")

    # Test JWT generation and verification
    print("\n4. JWT Token Test...")
    if user_id:
        jwt_result = await agent.generate_jwt(user_id, {"role": "user"})
        if jwt_result["success"]:
            token = jwt_result["token"]
            print(f"   Token generated: {token[:50]}...")

            verify = await agent.verify_jwt(token)
            if verify["valid"]:
                print(f"   Token verified for user: {verify['user_id']}")
            else:
                print(f"   Token invalid: {verify.get('error')}")

    # Test encryption
    print("\n5. Encryption Test...")
    sensitive_data = "This is sensitive information: SSN-123-45-6789"
    encrypted = await agent.encrypt_data(sensitive_data)
    if encrypted["success"]:
        print(f"   Encrypted: {encrypted['encrypted_data'][:50]}...")

        decrypted = await agent.decrypt_data(encrypted["encrypted_data"])
        if decrypted["success"]:
            print(f"   Decrypted: {decrypted['decrypted_data']}")

    # Test input sanitization
    print("\n6. Input Sanitization...")
    malicious_input = "<script>alert('XSS')</script> UNION SELECT * FROM users"
    sanitized = await agent.sanitize_input(malicious_input)
    if sanitized["success"]:
        print(f"   Original: {malicious_input[:50]}...")
        print(f"   Sanitized: {sanitized['sanitized_input'][:50]}...")
        print(
            f"   Removed: {sanitized['original_length'] - sanitized['sanitized_length']} chars"
        )

    # Test audit logging
    print("\n7. Audit Logging...")
    result = await agent.audit_log(
        user_id=user_id or "system",
        action="test_action",
        resource="test_resource",
        status="success",
        details={"test": "data"},
        severity=SecurityLevel.LOW,
    )
    print(f"   Audit log created: {result['audit_id']}")

    # Test rate limiting
    print("\n8. Rate Limiting...")
    for i in range(3):
        result = await agent.check_rate_limit("test_key", "login")
        print(f"   Attempt {i+1}: {'✓' if result['allowed'] else '✗'}")

    # Test API key generation
    print("\n9. API Key Generation...")
    result = await agent.generate_api_key(
        name="Test API Key", permissions=[Permission.READ], expiry_days=30
    )
    if result["success"]:
        print(f"   API Key: {result['api_key']}")
        print(f"   Secret: {result['api_secret']}")

    # Get audit logs
    print("\n10. Retrieving Audit Logs...")
    result = await agent.get_audit_logs(limit=5)
    if result["success"]:
        print(f"   Total logs: {result['total']}")
        for log in result["logs"][:3]:
            print(f"     - {log['action']} by {log['user_id']} ({log['status']})")

    # Get statistics
    print("\n11. Agent Statistics...")
    stats = agent.get_stats()
    print(f"   Total auth attempts: {stats['total_auth_attempts']}")
    print(f"   Successful auth: {stats['successful_auth']}")
    print(f"   Failed auth: {stats['failed_auth']}")
    print(f"   Total users: {stats['total_users']}")
    print(f"   Audit logs: {stats['audit_logs_count']}")
    print(f"   Security events: {stats['security_events_count']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_security_agent())
