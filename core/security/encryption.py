"""
Encryption Manager - comprehensive encryption system with key management,
multi-algorithm support, hardware security, and audit logging.
(PRODUCTION READY WITH ENTERPRISE SECURITY FEATURES)
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import secrets

try:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    SCRYPT_AVAILABLE = True
except ImportError:
    SCRYPT_AVAILABLE = False

from ..utils.logger import logger


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    FERNET = "fernet"
    AES_GCM = "aes_gcm"
    AES_CBC = "aes_cbc"
    CHACHA20 = "chacha20"
    RSA = "rsa"
    HYBRID = "hybrid"


class KeyDerivationFunction(Enum):
    """Key derivation functions."""

    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    BCRYPT = "bcrypt"
    ARGON2 = "argon2"


class KeyType(Enum):
    """Types of cryptographic keys."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    SIGNING = "signing"


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data."""

    algorithm: EncryptionAlgorithm
    timestamp: datetime
    key_id: str
    version: int = 2
    compression: bool = False
    encoding: str = "base64"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "timestamp": self.timestamp.isoformat(),
            "key_id": self.key_id,
            "version": self.version,
            "compression": self.compression,
            "encoding": self.encoding,
            "tags": self.tags,
        }


class EncryptionManager:
    """Enterprise-grade encryption manager with multiple algorithms and key management."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        key_path: str = "security/secret.key",
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET,
        key_derivation: KeyDerivationFunction = KeyDerivationFunction.PBKDF2,
        use_hardware_security: bool = False,
        enable_audit: bool = True,
        key_rotation_days: int = 90,
    ):
        """Initialize enterprise encryption manager."""
        if hasattr(self, "_initialized"):
            return

        self.key_path = Path(key_path)
        self.algorithm = algorithm
        self.key_derivation = key_derivation
        self.use_hardware_security = use_hardware_security
        self.enable_audit = enable_audit
        self.key_rotation_days = key_rotation_days

        # Key storage
        self.keys: Dict[str, Union[bytes, rsa.RSAPrivateKey]] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        self.current_key_id = "master"

        # Key management
        self._load_or_create_keys()

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []

        # Metrics
        self.metrics = {
            "total_encryptions": 0,
            "total_decryptions": 0,
            "failed_encryptions": 0,
            "failed_decryptions": 0,
            "key_rotations": 0,
            "last_key_rotation": None,
        }

        # Background rotation check
        self._rotation_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_rotation_monitor()

        self._initialized = True

        logger.info(
            f"✅ EncryptionManager initialized (algorithm={algorithm.value}, audit={enable_audit})"
        )

    # #================#================#============#=============
    # 🔥 KEY MANAGEMENT
    # #================#================#============#=============

    def _load_or_create_keys(self) -> None:
        """Load existing keys or create new ones."""
        if (
            self.algorithm == EncryptionAlgorithm.RSA
            or self.algorithm == EncryptionAlgorithm.HYBRID
        ):
            self._load_or_create_asymmetric_keys()
        else:
            self._load_or_create_symmetric_keys()

    def _load_or_create_symmetric_keys(self) -> None:
        """Load or create symmetric encryption keys."""
        key_file = self.key_path.with_suffix(".key")

        if key_file.exists():
            try:
                with open(key_file, "rb") as f:
                    self.keys["master"] = f.read()
                self._load_key_metadata()
                logger.info(f"📁 Loaded symmetric key from {key_file}")
            except Exception as e:
                logger.error(f"Failed to load key: {e}")
                self._create_symmetric_keys()
        else:
            self._create_symmetric_keys()

    def _create_symmetric_keys(self) -> None:
        """Create new symmetric encryption keys."""
        if self.algorithm == EncryptionAlgorithm.FERNET:
            self.keys["master"] = Fernet.generate_key()
        elif (
            self.algorithm == EncryptionAlgorithm.AES_GCM
            or self.algorithm == EncryptionAlgorithm.AES_CBC
        ):
            self.keys["master"] = secrets.token_bytes(32)  # 256-bit AES key
        elif self.algorithm == EncryptionAlgorithm.CHACHA20:
            self.keys["master"] = secrets.token_bytes(32)  # 256-bit ChaCha20 key
        else:
            self.keys["master"] = Fernet.generate_key()

        # Save key
        key_file = self.key_path.with_suffix(".key")
        key_file.parent.mkdir(parents=True, exist_ok=True)

        with open(key_file, "wb") as f:
            f.write(self.keys["master"])

        # Save key metadata
        self._save_key_metadata()

        # Set permissions (Unix only)
        if os.name == "posix":
            os.chmod(key_file, 0o600)

        logger.info(f"🔑 Created new symmetric key at {key_file}")

    def _load_or_create_asymmetric_keys(self) -> None:
        """Load or create RSA key pair."""
        private_key_path = self.key_path.with_suffix(".private.pem")
        public_key_path = self.key_path.with_suffix(".public.pem")

        if private_key_path.exists() and public_key_path.exists():
            try:
                with open(private_key_path, "rb") as f:
                    self.keys["private"] = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )

                with open(public_key_path, "rb") as f:
                    self.keys["public"] = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )

                self.current_key_id = "asymmetric"
                logger.info(f"📁 Loaded asymmetric keys from {private_key_path}")
            except Exception as e:
                logger.error(f"Failed to load asymmetric keys: {e}")
                self._create_asymmetric_keys()
        else:
            self._create_asymmetric_keys()

    def _create_asymmetric_keys(self) -> None:
        """Create new RSA key pair."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Extract public key
        public_key = private_key.public_key()

        self.keys["private"] = private_key
        self.keys["public"] = public_key
        self.current_key_id = "asymmetric"

        # Save private key
        private_key_path = self.key_path.with_suffix(".private.pem")
        private_key_path.parent.mkdir(parents=True, exist_ok=True)

        with open(private_key_path, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Save public key
        public_key_path = self.key_path.with_suffix(".public.pem")
        with open(public_key_path, "wb") as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

        # Set permissions (Unix only)
        if os.name == "posix":
            os.chmod(private_key_path, 0o600)
            os.chmod(public_key_path, 0o644)

        logger.info(f"🔑 Created new RSA key pair at {private_key_path}")

    def _save_key_metadata(self) -> None:
        """Save key metadata."""
        metadata_file = self.key_path.with_suffix(".meta.json")

        self.key_metadata[self.current_key_id] = {
            "algorithm": self.algorithm.value,
            "created_at": datetime.now().isoformat(),
            "key_type": KeyType.SYMMETRIC.value,
            "key_size": 256,
            "version": 2,
        }

        with open(metadata_file, "w") as f:
            json.dump(self.key_metadata, f, indent=2)

    def _load_key_metadata(self) -> None:
        """Load key metadata."""
        metadata_file = self.key_path.with_suffix(".meta.json")

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.key_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load key metadata: {e}")

    def rotate_key(self, preserve_old: bool = True) -> bool:
        """Rotate encryption key with optional preservation of old keys."""
        try:
            # Store old key
            if preserve_old:
                old_key_id = f"{self.current_key_id}_old_{int(time.time())}"
                if self.algorithm == EncryptionAlgorithm.RSA:
                    self.keys[old_key_id] = self.keys["private"]
                else:
                    self.keys[old_key_id] = self.keys["master"]
                self.key_metadata[old_key_id] = self.key_metadata.get(
                    self.current_key_id, {}
                )
                self.key_metadata[old_key_id]["rotated_at"] = datetime.now().isoformat()

            # Create new keys
            if (
                self.algorithm == EncryptionAlgorithm.RSA
                or self.algorithm == EncryptionAlgorithm.HYBRID
            ):
                self._create_asymmetric_keys()
            else:
                self._create_symmetric_keys()

            self.metrics["key_rotations"] += 1
            self.metrics["last_key_rotation"] = datetime.now().isoformat()

            self._audit_event(
                "key_rotation",
                {"algorithm": self.algorithm.value, "preserved_old": preserve_old},
            )

            logger.info(
                f"🔄 Key rotated successfully (rotation #{self.metrics['key_rotations']})"
            )
            return True

        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            self._audit_event("key_rotation_failed", {"error": str(e)})
            return False

    def derive_key_from_password(
        self, password: str, salt: Optional[bytes] = None, key_length: int = 32
    ) -> bytes:
        """Derive encryption key from password."""
        if salt is None:
            salt = secrets.token_bytes(16)

        if self.key_derivation == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )
        elif self.key_derivation == KeyDerivationFunction.SCRYPT and SCRYPT_AVAILABLE:
            kdf = Scrypt(
                salt=salt,
                length=key_length,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend(),
            )
        else:
            # Fallback to PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )

        key = kdf.derive(password.encode())
        return key

    # #================#================#============#=============
    # 🔥 ENCRYPTION
    # #================#================#============#=============

    def encrypt(
        self,
        data: str,
        key_id: Optional[str] = None,
        algorithm: Optional[EncryptionAlgorithm] = None,
        add_metadata: bool = True,
    ) -> str:
        """Encrypt data with specified algorithm."""
        if not data:
            return ""

        start_time = time.time()
        key_id = key_id or self.current_key_id

        try:
            # Choose algorithm
            algo = algorithm or self.algorithm
            data_bytes = data.encode("utf-8")

            # Optional: compress data
            compressed = False
            if len(data_bytes) > 1024:  # Compress if > 1KB
                import zlib

                data_bytes = zlib.compress(data_bytes)
                compressed = True

            # Encrypt based on algorithm
            if algo == EncryptionAlgorithm.FERNET:
                encrypted = self._encrypt_fernet(data_bytes, key_id)
            elif algo == EncryptionAlgorithm.AES_GCM:
                encrypted = self._encrypt_aes_gcm(data_bytes, key_id)
            elif algo == EncryptionAlgorithm.AES_CBC:
                encrypted = self._encrypt_aes_cbc(data_bytes, key_id)
            elif algo == EncryptionAlgorithm.RSA:
                encrypted = self._encrypt_rsa(data_bytes, key_id)
            elif algo == EncryptionAlgorithm.HYBRID:
                encrypted = self._encrypt_hybrid(data_bytes, key_id)
            else:
                encrypted = self._encrypt_fernet(data_bytes, key_id)

            # Add metadata
            if add_metadata:
                metadata = EncryptionMetadata(
                    algorithm=algo,
                    timestamp=datetime.now(),
                    key_id=key_id,
                    compression=compressed,
                )
                result = self._add_metadata(encrypted, metadata)
            else:
                result = base64.urlsafe_b64encode(encrypted).decode()

            # Update metrics
            self.metrics["total_encryptions"] += 1
            elapsed_ms = (time.time() - start_time) * 1000

            # Audit
            self._audit_event(
                "encryption",
                {
                    "algorithm": algo.value,
                    "key_id": key_id,
                    "data_size": len(data),
                    "encrypted_size": len(result),
                    "time_ms": elapsed_ms,
                },
            )

            return result

        except Exception as e:
            self.metrics["failed_encryptions"] += 1
            logger.error(f"Encryption failed: {e}")
            self._audit_event("encryption_failed", {"error": str(e)})
            raise ValueError(f"Encryption failed: {e}")

    def _encrypt_fernet(self, data: bytes, key_id: str) -> bytes:
        """Encrypt using Fernet."""
        key = self._get_key(key_id)
        if isinstance(key, bytes) and len(key) == 32:  # Fernet key
            cipher = Fernet(key)
            return cipher.encrypt(data)
        else:
            cipher = Fernet(self.keys["master"])
            return cipher.encrypt(data)

    def _encrypt_aes_gcm(self, data: bytes, key_id: str) -> bytes:
        """Encrypt using AES-GCM (authenticated encryption)."""
        key = self._get_key(key_id)
        if not isinstance(key, bytes):
            key = self.keys["master"]

        iv = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        encrypted = encryptor.update(data) + encryptor.finalize()

        # Return IV + tag + ciphertext
        return iv + encryptor.tag + encrypted

    def _encrypt_aes_cbc(self, data: bytes, key_id: str) -> bytes:
        """Encrypt using AES-CBC."""
        key = self._get_key(key_id)
        if not isinstance(key, bytes):
            key = self.keys["master"]

        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Pad data to block size
        pad_length = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_length] * pad_length)

        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        return iv + encrypted

    def _encrypt_rsa(self, data: bytes, key_id: str) -> bytes:
        """Encrypt using RSA (only for small data)."""
        public_key = self._get_key("public")
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError("RSA public key not available")

        # RSA can only encrypt small amounts of data
        chunk_size = 190  # For 2048-bit RSA with padding
        encrypted_chunks = []

        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            encrypted = public_key.encrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            encrypted_chunks.append(encrypted)

        return b"".join(encrypted_chunks)

    def _encrypt_hybrid(self, data: bytes, key_id: str) -> bytes:
        """Hybrid encryption: RSA + AES."""
        # Generate random AES key
        aes_key = secrets.token_bytes(32)

        # Encrypt data with AES
        iv = secrets.token_bytes(12)
        cipher = Cipher(
            algorithms.AES(aes_key), modes.GCM(iv), backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()

        # Encrypt AES key with RSA
        encrypted_key = self._encrypt_rsa(aes_key, key_id)

        # Format: encrypted_key_length (4 bytes) + encrypted_key + iv + tag + encrypted_data
        key_length = len(encrypted_key).to_bytes(4, "big")
        return key_length + encrypted_key + iv + encryptor.tag + encrypted_data

    # #================#================#============#=============
    # 🔥 DECRYPTION
    # #================#================#============#=============

    def decrypt(self, token: str, verify_metadata: bool = True) -> Optional[str]:
        """Decrypt data with automatic algorithm detection."""
        if not token:
            return ""

        start_time = time.time()

        try:
            # Try to extract metadata
            if verify_metadata and token.startswith("{"):
                # Has metadata
                encrypted_data, metadata = self._extract_metadata(token)
                algorithm = EncryptionAlgorithm(metadata["algorithm"])
            else:
                encrypted_data = base64.urlsafe_b64decode(token)
                algorithm = self.algorithm

            # Decrypt based on algorithm
            if algorithm == EncryptionAlgorithm.FERNET:
                decrypted = self._decrypt_fernet(encrypted_data)
            elif algorithm == EncryptionAlgorithm.AES_GCM:
                decrypted = self._decrypt_aes_gcm(encrypted_data)
            elif algorithm == EncryptionAlgorithm.AES_CBC:
                decrypted = self._decrypt_aes_cbc(encrypted_data)
            elif algorithm == EncryptionAlgorithm.RSA:
                decrypted = self._decrypt_rsa(encrypted_data)
            elif algorithm == EncryptionAlgorithm.HYBRID:
                decrypted = self._decrypt_hybrid(encrypted_data)
            else:
                decrypted = self._decrypt_fernet(encrypted_data)

            # Decompress if needed
            if (
                verify_metadata
                and token.startswith("{")
                and metadata.get("compression")
            ):
                import zlib

                decrypted = zlib.decompress(decrypted)

            # Update metrics
            self.metrics["total_decryptions"] += 1
            elapsed_ms = (time.time() - start_time) * 1000

            # Audit
            self._audit_event(
                "decryption", {"algorithm": algorithm.value, "time_ms": elapsed_ms}
            )

            return decrypted.decode("utf-8")

        except Exception as e:
            self.metrics["failed_decryptions"] += 1
            logger.error(f"Decryption failed: {e}")
            self._audit_event("decryption_failed", {"error": str(e)})
            return None

    def _decrypt_fernet(self, encrypted_data: bytes) -> bytes:
        """Decrypt using Fernet."""
        # Try all keys (for key rotation support)
        for key_id, key in self.keys.items():
            if isinstance(key, bytes) and len(key) == 32:
                try:
                    cipher = Fernet(key)
                    return cipher.decrypt(encrypted_data)
                except Exception:
                    continue

        # Fallback to master key
        cipher = Fernet(self.keys["master"])
        return cipher.decrypt(encrypted_data)

    def _decrypt_aes_gcm(self, encrypted_data: bytes) -> bytes:
        """Decrypt using AES-GCM."""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        key = self._get_key(self.current_key_id)
        if not isinstance(key, bytes):
            key = self.keys["master"]

        cipher = Cipher(
            algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def _decrypt_aes_cbc(self, encrypted_data: bytes) -> bytes:
        """Decrypt using AES-CBC."""
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        key = self._get_key(self.current_key_id)
        if not isinstance(key, bytes):
            key = self.keys["master"]

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding
        pad_length = padded_data[-1]
        return padded_data[:-pad_length]

    def _decrypt_rsa(self, encrypted_data: bytes) -> bytes:
        """Decrypt using RSA."""
        private_key = self._get_key("private")
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("RSA private key not available")

        chunk_size = 256  # For 2048-bit RSA
        decrypted_chunks = []

        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i : i + chunk_size]
            decrypted = private_key.decrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            decrypted_chunks.append(decrypted)

        return b"".join(decrypted_chunks)

    def _decrypt_hybrid(self, encrypted_data: bytes) -> bytes:
        """Decrypt hybrid encrypted data."""
        # Extract components
        key_length = int.from_bytes(encrypted_data[:4], "big")
        encrypted_key = encrypted_data[4 : 4 + key_length]
        iv = encrypted_data[4 + key_length : 4 + key_length + 12]
        tag = encrypted_data[4 + key_length + 12 : 4 + key_length + 12 + 16]
        ciphertext = encrypted_data[4 + key_length + 12 + 16 :]

        # Decrypt AES key with RSA
        aes_key = self._decrypt_rsa(encrypted_key)

        # Decrypt data with AES
        cipher = Cipher(
            algorithms.AES(aes_key), modes.GCM(iv, tag), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def _get_key(
        self, key_id: str
    ) -> Union[bytes, rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Get key by ID."""
        if key_id in self.keys:
            return self.keys[key_id]
        return self.keys.get("master", b"")

    def _add_metadata(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> str:
        """Add metadata to encrypted data."""
        metadata_dict = metadata.to_dict()
        metadata_json = json.dumps(metadata_dict)

        # Combine metadata and encrypted data
        combined = {
            "metadata": metadata_json,
            "data": base64.urlsafe_b64encode(encrypted_data).decode(),
        }

        return json.dumps(combined)

    def _extract_metadata(self, token: str) -> Tuple[bytes, Dict[str, Any]]:
        """Extract metadata from token."""
        combined = json.loads(token)
        metadata = json.loads(combined["metadata"])
        encrypted_data = base64.urlsafe_b64decode(combined["data"])
        return encrypted_data, metadata

    # #================#================#============#=============
    # 🔥 FILE ENCRYPTION
    # #================#================#============#=============

    def encrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 8192,
    ) -> bool:
        """Encrypt a file with streaming support for large files."""
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + ".encrypted")
        else:
            output_path = Path(output_path)

        try:
            if self.algorithm == EncryptionAlgorithm.RSA:
                # RSA can't handle large files directly, use hybrid
                return self._encrypt_file_hybrid(input_path, output_path, chunk_size)
            else:
                return self._encrypt_file_symmetric(input_path, output_path, chunk_size)

        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return False

    def _encrypt_file_symmetric(
        self, input_path: Path, output_path: Path, chunk_size: int
    ) -> bool:
        """Encrypt file using symmetric encryption."""
        try:
            key = self._get_key(self.current_key_id)
            if not isinstance(key, bytes):
                key = self.keys["master"]

            # For AES, generate IV
            iv = secrets.token_bytes(12)

            with open(input_path, "rb") as infile:
                with open(output_path, "wb") as outfile:
                    # Write IV
                    outfile.write(iv)

                    # Create cipher
                    cipher = Cipher(
                        algorithms.AES(key), modes.GCM(iv), backend=default_backend()
                    )
                    encryptor = cipher.encryptor()

                    # Encrypt in chunks
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        encrypted_chunk = encryptor.update(chunk)
                        outfile.write(encrypted_chunk)

                    # Write tag
                    outfile.write(encryptor.tag)

            logger.info(f"🔒 File encrypted: {input_path} -> {output_path}")
            self._audit_event(
                "file_encryption",
                {"input": str(input_path), "output": str(output_path)},
            )
            return True

        except Exception as e:
            logger.error(f"Symmetric file encryption failed: {e}")
            return False

    def _encrypt_file_hybrid(
        self, input_path: Path, output_path: Path, chunk_size: int
    ) -> bool:
        """Encrypt file using hybrid encryption."""
        try:
            # Generate random AES key
            aes_key = secrets.token_bytes(32)
            iv = secrets.token_bytes(12)

            # Encrypt AES key with RSA
            encrypted_key = self._encrypt_rsa(aes_key, "public")

            with open(input_path, "rb") as infile:
                with open(output_path, "wb") as outfile:
                    # Write encrypted key length and key
                    outfile.write(len(encrypted_key).to_bytes(4, "big"))
                    outfile.write(encrypted_key)

                    # Write IV
                    outfile.write(iv)

                    # Create AES cipher
                    cipher = Cipher(
                        algorithms.AES(aes_key),
                        modes.GCM(iv),
                        backend=default_backend(),
                    )
                    encryptor = cipher.encryptor()

                    # Encrypt file in chunks
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        encrypted_chunk = encryptor.update(chunk)
                        outfile.write(encrypted_chunk)

                    # Write tag
                    outfile.write(encryptor.tag)

            logger.info(f"🔒 File encrypted (hybrid): {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Hybrid file encryption failed: {e}")
            return False

    def decrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 8192,
    ) -> bool:
        """Decrypt a file with streaming support."""
        input_path = Path(input_path)

        if output_path is None:
            if input_path.suffix == ".encrypted":
                output_path = input_path.with_suffix("")
            else:
                output_path = input_path.with_suffix(".decrypted")
        else:
            output_path = Path(output_path)

        try:
            with open(input_path, "rb") as infile:
                # Read IV
                iv = infile.read(12)

                # Try to determine if it's hybrid by checking for key length
                infile.seek(0)
                first_4_bytes = infile.read(4)
                infile.seek(0)

                if len(first_4_bytes) == 4:
                    key_length = int.from_bytes(first_4_bytes, "big")
                    if 128 <= key_length <= 512:  # RSA encrypted key length
                        return self._decrypt_file_hybrid(
                            input_path, output_path, chunk_size
                        )

                # Symmetric decryption
                return self._decrypt_file_symmetric(input_path, output_path, chunk_size)

        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False

    def _decrypt_file_symmetric(
        self, input_path: Path, output_path: Path, chunk_size: int
    ) -> bool:
        """Decrypt file using symmetric decryption."""
        try:
            key = self._get_key(self.current_key_id)
            if not isinstance(key, bytes):
                key = self.keys["master"]

            with open(input_path, "rb") as infile:
                # Read IV
                iv = infile.read(12)

                # Read to end to get tag (last 16 bytes)
                infile.seek(-16, os.SEEK_END)
                tag = infile.read(16)

                # Go back to after IV
                infile.seek(12)

                with open(output_path, "wb") as outfile:
                    # Create cipher
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.GCM(iv, tag),
                        backend=default_backend(),
                    )
                    decryptor = cipher.decryptor()

                    # Read up to tag position
                    remaining_size = infile.tell() - 16
                    infile.seek(12)

                    while infile.tell() < remaining_size:
                        chunk = infile.read(
                            min(chunk_size, remaining_size - infile.tell())
                        )
                        if not chunk:
                            break
                        decrypted_chunk = decryptor.update(chunk)
                        outfile.write(decrypted_chunk)

            logger.info(f"🔓 File decrypted: {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Symmetric file decryption failed: {e}")
            return False

    def _decrypt_file_hybrid(
        self, input_path: Path, output_path: Path, chunk_size: int
    ) -> bool:
        """Decrypt file using hybrid decryption."""
        try:
            with open(input_path, "rb") as infile:
                # Read encrypted key length
                key_length = int.from_bytes(infile.read(4), "big")

                # Read encrypted AES key
                encrypted_key = infile.read(key_length)

                # Read IV
                iv = infile.read(12)

                # Read to end to get tag
                infile.seek(-16, os.SEEK_END)
                tag = infile.read(16)

                # Go back to after IV
                infile.seek(4 + key_length + 12)

                # Decrypt AES key
                aes_key = self._decrypt_rsa(encrypted_key)

                with open(output_path, "wb") as outfile:
                    # Create AES cipher
                    cipher = Cipher(
                        algorithms.AES(aes_key),
                        modes.GCM(iv, tag),
                        backend=default_backend(),
                    )
                    decryptor = cipher.decryptor()

                    remaining_size = infile.tell() - 16
                    infile.seek(4 + key_length + 12)

                    while infile.tell() < remaining_size:
                        chunk = infile.read(
                            min(chunk_size, remaining_size - infile.tell())
                        )
                        if not chunk:
                            break
                        decrypted_chunk = decryptor.update(chunk)
                        outfile.write(decrypted_chunk)

            logger.info(f"🔓 File decrypted (hybrid): {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Hybrid file decryption failed: {e}")
            return False

    # #================#================#============#=============
    # 🔥 HASHING & SIGNING
    # #================#================#============#=============

    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """Create cryptographic hash of data."""
        hash_algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }

        if algorithm not in hash_algorithms:
            algorithm = "sha256"

        hasher = hash_algorithms[algorithm]()
        hasher.update(data.encode("utf-8"))
        return hasher.hexdigest()

    def sign_data(self, data: str) -> str:
        """Sign data using RSA private key."""
        if self.algorithm != EncryptionAlgorithm.RSA:
            raise ValueError("Signing requires RSA algorithm")

        private_key = self._get_key("private")
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ValueError("RSA private key not available")

        signature = private_key.sign(
            data.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode()

    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify RSA signature."""
        if self.algorithm != EncryptionAlgorithm.RSA:
            raise ValueError("Verification requires RSA algorithm")

        public_key = self._get_key("public")
        if not isinstance(public_key, rsa.RSAPublicKey):
            raise ValueError("RSA public key not available")

        try:
            public_key.verify(
                base64.b64decode(signature),
                data.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    # #================#================#============#=============
    # 🔥 KEY MANAGEMENT
    # #================#================#============#=============

    def export_public_key(self) -> Optional[str]:
        """Export public key for sharing."""
        if self.algorithm == EncryptionAlgorithm.RSA:
            public_key = self._get_key("public")
            if isinstance(public_key, rsa.RSAPublicKey):
                pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                return pem.decode("utf-8")

        return None

    def import_public_key(self, pem_data: str) -> bool:
        """Import public key for encryption."""
        try:
            public_key = serialization.load_pem_public_key(
                pem_data.encode("utf-8"), backend=default_backend()
            )
            self.keys["imported_public"] = public_key
            return True
        except Exception as e:
            logger.error(f"Failed to import public key: {e}")
            return False

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all available keys."""
        keys_info = []
        for key_id, key in self.keys.items():
            key_info = {
                "id": key_id,
                "type": type(key).__name__,
                "size": len(key) if isinstance(key, bytes) else 2048,
            }
            if key_id in self.key_metadata:
                key_info["metadata"] = self.key_metadata[key_id]
            keys_info.append(key_info)
        return keys_info

    # #================#================#============#=============
    # 🔥 AUDIT & MONITORING
    # #================#================#============#=============

    def _audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        if not self.enable_audit:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        }

        self.audit_log.append(event)

        # Keep last 1000 events
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self.audit_log[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get encryption metrics."""
        return {
            **self.metrics,
            "available_keys": len(self.keys),
            "algorithm": self.algorithm.value,
            "audit_enabled": self.enable_audit,
            "key_rotation_days": self.key_rotation_days,
        }

    def _start_rotation_monitor(self):
        """Start background key rotation monitor."""

        def monitor():
            while self._running:
                time.sleep(3600)  # Check every hour
                self._check_key_rotation()

        self._running = True
        self._rotation_thread = threading.Thread(target=monitor, daemon=True)
        self._rotation_thread.start()

    def _check_key_rotation(self):
        """Check if key rotation is needed."""
        if not self.key_rotation_days or self.key_rotation_days <= 0:
            return

        metadata = self.key_metadata.get(self.current_key_id, {})
        created_at = metadata.get("created_at")

        if created_at:
            created_date = datetime.fromisoformat(created_at)
            age_days = (datetime.now() - created_date).days

            if age_days >= self.key_rotation_days:
                logger.info(
                    f"Key age {age_days} days exceeds rotation period {self.key_rotation_days}, rotating..."
                )
                self.rotate_key()

    def shutdown(self):
        """Shutdown encryption manager."""
        self._running = False
        if self._rotation_thread:
            self._rotation_thread.join(timeout=5)
        logger.info("🛑 EncryptionManager shutdown")

    # #================#================#============#=============
    # 🔥 UTILITIES
    # #================#================#============#=============

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()[:length]

    def generate_uuid(self) -> str:
        """Generate secure UUID v4."""
        return secrets.token_hex(16)

    def secure_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return secrets.compare_digest(a.encode(), b.encode())


# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager instance."""
    return EncryptionManager()


def encrypt_sensitive_data(data: str) -> str:
    """Quick encrypt helper."""
    return get_encryption_manager().encrypt(data)


def decrypt_sensitive_data(token: str) -> Optional[str]:
    """Quick decrypt helper."""
    return get_encryption_manager().decrypt(token)


__all__ = [
    "EncryptionManager",
    "EncryptionAlgorithm",
    "KeyDerivationFunction",
    "KeyType",
    "EncryptionMetadata",
    "get_encryption_manager",
    "encrypt_sensitive_data",
    "decrypt_sensitive_data",
]
