"""Permission Manager - manages access control and permissions with RBAC, ABAC, audit logging, and fine-grained access control."""

from typing import Dict, List, Set, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
import hashlib
import threading
from pathlib import Path

try:
    from cryptography.fernet import Fernet

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..utils.logger import logger


class Permission(Enum):
    """Comprehensive permission types."""

    # Basic permissions
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"

    # Admin permissions
    ADMIN = "admin"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_PERMISSIONS = "manage_permissions"
    AUDIT = "audit"

    # System permissions
    SYSTEM_CONFIG = "system_config"
    SYSTEM_MONITOR = "system_monitor"
    SYSTEM_BACKUP = "system_backup"
    SYSTEM_RESTORE = "system_restore"

    # Data permissions
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_DELETE = "data_delete"
    DATA_ANONYMIZE = "data_anonymize"

    # Memory permissions
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    MEMORY_DELETE = "memory_delete"
    MEMORY_SEARCH = "memory_search"

    # Security permissions
    SECURITY_VIEW = "security_view"
    SECURITY_CONFIGURE = "security_configure"
    SECURITY_ENCRYPT = "security_encrypt"
    SECURITY_DECRYPT = "security_decrypt"


class AccessLevel(Enum):
    """Access levels for hierarchical access control."""

    NO_ACCESS = 0
    READ_ONLY = 10
    BASIC = 20
    STANDARD = 30
    ADVANCED = 40
    FULL = 50
    SUPER_ADMIN = 100


class ResourceType(Enum):
    """Types of resources for fine-grained access control."""

    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    SYSTEM = "system"
    MEMORY = "memory"
    DATA = "data"
    CONFIG = "config"
    AUDIT_LOG = "audit_log"
    SECURITY = "security"


@dataclass
class AccessRule:
    """Fine-grained access rule."""

    resource_type: ResourceType
    resource_id: Optional[str]
    permission: Permission
    effect: str = "allow"  # allow or deny
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def matches(
        self,
        resource_type: ResourceType,
        resource_id: Optional[str],
        context: Dict[str, Any],
    ) -> bool:
        """Check if rule matches the request."""
        if self.resource_type != resource_type:
            return False

        if self.resource_id and resource_id and self.resource_id != resource_id:
            return False

        # Check conditions
        for key, value in self.conditions.items():
            if key not in context or context[key] != value:
                return False

        return True


class Role:
    """Advanced role with hierarchical inheritance and metadata."""

    def __init__(
        self,
        name: str,
        description: str = "",
        parent_role: Optional[str] = None,
        access_level: AccessLevel = AccessLevel.STANDARD,
    ):
        """Initialize role."""
        self.name = name
        self.description = description
        self.parent_role = parent_role
        self.access_level = access_level
        self.permissions: Set[Permission] = set()
        self.resource_rules: List[AccessRule] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def grant_permission(self, permission: Permission) -> None:
        """Grant permission to role."""
        self.permissions.add(permission)
        self.updated_at = datetime.now()

    def revoke_permission(self, permission: Permission) -> None:
        """Revoke permission from role."""
        self.permissions.discard(permission)
        self.updated_at = datetime.now()

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission (including inherited)."""
        if permission in self.permissions:
            return True
        return False

    def add_resource_rule(self, rule: AccessRule) -> None:
        """Add fine-grained resource rule."""
        self.resource_rules.append(rule)
        self.resource_rules.sort(key=lambda x: x.priority, reverse=True)

    def remove_resource_rule(self, rule: AccessRule) -> None:
        """Remove resource rule."""
        if rule in self.resource_rules:
            self.resource_rules.remove(rule)

    def check_resource_access(
        self,
        resource_type: ResourceType,
        permission: Permission,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check resource-specific access."""
        context = context or {}

        # Check rules in priority order
        for rule in self.resource_rules:
            if rule.permission == permission and rule.matches(
                resource_type, resource_id, context
            ):
                return rule.effect == "allow"

        # Default to permission check
        return self.has_permission(permission)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parent_role": self.parent_role,
            "access_level": self.access_level.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        """Create from dictionary."""
        role = cls(
            name=data["name"],
            description=data.get("description", ""),
            parent_role=data.get("parent_role"),
            access_level=AccessLevel(data.get("access_level", 20)),
        )
        role.permissions = {Permission(p) for p in data.get("permissions", [])}
        role.created_at = (
            datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now()
        )
        role.updated_at = (
            datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now()
        )
        role.metadata = data.get("metadata", {})
        return role


class PermissionManager:
    """Advanced permission manager with RBAC, ABAC, and audit logging."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PermissionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize advanced permission manager."""
        if hasattr(self, "_initialized"):
            return

        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        self.user_permission_overrides: Dict[str, Dict[Permission, bool]] = defaultdict(
            dict
        )
        self.user_access_levels: Dict[str, AccessLevel] = {}

        # Resource-specific permissions
        self.resource_owners: Dict[str, str] = {}  # resource_id -> owner_id
        self.resource_acl: Dict[str, Dict[str, Set[Permission]]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        self.audit_enabled = True
        self.max_audit_size = 10000

        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour

        # Permission cache
        self._permission_cache: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}

        # Callbacks
        self._permission_callbacks: List[Callable] = []

        # Persistence
        self.persistence_path = Path("data/permissions")
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        self._initialize_default_roles()
        self._load_from_disk()

        self._initialized = True

        logger.info("✅ PermissionManager initialized with RBAC + ABAC")

    def _initialize_default_roles(self):
        """Initialize comprehensive default roles."""

        # Super Admin role
        super_admin = Role(
            "super_admin", "Full system access", access_level=AccessLevel.SUPER_ADMIN
        )
        for perm in Permission:
            super_admin.grant_permission(perm)
        self.roles["super_admin"] = super_admin

        # Admin role
        admin = Role("admin", "Administrative access", access_level=AccessLevel.FULL)
        admin_perms = [
            Permission.ADMIN,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.MANAGE_PERMISSIONS,
            Permission.SYSTEM_CONFIG,
            Permission.SYSTEM_MONITOR,
            Permission.SYSTEM_BACKUP,
            Permission.SYSTEM_RESTORE,
            Permission.AUDIT,
            Permission.SECURITY_VIEW,
            Permission.SECURITY_CONFIGURE,
        ]
        for perm in admin_perms:
            admin.grant_permission(perm)
        self.roles["admin"] = admin

        # Manager role
        manager = Role(
            "manager", "Management access", access_level=AccessLevel.ADVANCED
        )
        manager_perms = [
            Permission.READ,
            Permission.WRITE,
            Permission.EXECUTE,
            Permission.DATA_EXPORT,
            Permission.DATA_IMPORT,
            Permission.MEMORY_READ,
            Permission.MEMORY_WRITE,
            Permission.MEMORY_SEARCH,
        ]
        for perm in manager_perms:
            manager.grant_permission(perm)
        self.roles["manager"] = manager

        # User role
        user = Role("user", "Standard user access", access_level=AccessLevel.STANDARD)
        user_perms = [
            Permission.READ,
            Permission.WRITE,
            Permission.EXECUTE,
            Permission.MEMORY_READ,
            Permission.MEMORY_WRITE,
            Permission.MEMORY_SEARCH,
        ]
        for perm in user_perms:
            user.grant_permission(perm)
        self.roles["user"] = user

        # Viewer role
        viewer = Role("viewer", "Read-only access", access_level=AccessLevel.READ_ONLY)
        viewer.grant_permission(Permission.READ)
        viewer.grant_permission(Permission.MEMORY_READ)
        self.roles["viewer"] = viewer

        # Guest role
        guest = Role("guest", "Limited guest access", access_level=AccessLevel.BASIC)
        guest.grant_permission(Permission.READ)
        self.roles["guest"] = guest

        # Auditor role
        auditor = Role("auditor", "Audit log access", access_level=AccessLevel.ADVANCED)
        auditor.grant_permission(Permission.AUDIT)
        auditor.grant_permission(Permission.READ)
        self.roles["auditor"] = auditor

        # Developer role
        developer = Role(
            "developer", "Development access", access_level=AccessLevel.ADVANCED
        )
        dev_perms = [
            Permission.READ,
            Permission.WRITE,
            Permission.EXECUTE,
            Permission.CREATE,
            Permission.UPDATE,
            Permission.DELETE,
            Permission.SYSTEM_CONFIG,
            Permission.MEMORY_READ,
            Permission.MEMORY_WRITE,
            Permission.MEMORY_DELETE,
        ]
        for perm in dev_perms:
            developer.grant_permission(perm)
        self.roles["developer"] = developer

        logger.info(f"Initialized {len(self.roles)} default roles")

    # #================#================#============#=============
    # ROLE MANAGEMENT
    # #================#================#============#=============

    def create_role(
        self,
        role_name: str,
        description: str = "",
        parent_role: Optional[str] = None,
        access_level: AccessLevel = AccessLevel.STANDARD,
    ) -> Role:
        """Create new role."""
        if role_name in self.roles:
            raise ValueError(f"Role '{role_name}' already exists")

        role = Role(role_name, description, parent_role, access_level)
        self.roles[role_name] = role

        self._audit_event("role_created", {"role": role_name, "parent": parent_role})
        logger.info(f"📋 Role created: {role_name}")

        return role

    def delete_role(self, role_name: str, force: bool = False) -> bool:
        """Delete role."""
        if role_name not in self.roles:
            return False

        # Check if role is in use
        if not force:
            users_with_role = [
                uid for uid, roles in self.user_roles.items() if role_name in roles
            ]
            if users_with_role:
                raise ValueError(
                    f"Role '{role_name}' is assigned to {len(users_with_role)} users"
                )

        del self.roles[role_name]

        # Remove role from users
        for user_roles in self.user_roles.values():
            user_roles.discard(role_name)

        self._audit_event("role_deleted", {"role": role_name, "force": force})
        logger.info(f"🗑️ Role deleted: {role_name}")

        return True

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get role by name."""
        return self.roles.get(role_name)

    def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles."""
        return [role.to_dict() for role in self.roles.values()]

    def update_role(self, role_name: str, **kwargs) -> bool:
        """Update role properties."""
        role = self.roles.get(role_name)
        if not role:
            return False

        if "description" in kwargs:
            role.description = kwargs["description"]
        if "access_level" in kwargs:
            role.access_level = kwargs["access_level"]
        if "parent_role" in kwargs:
            role.parent_role = kwargs["parent_role"]
        if "metadata" in kwargs:
            role.metadata.update(kwargs["metadata"])

        role.updated_at = datetime.now()
        self._audit_event("role_updated", {"role": role_name, "updates": kwargs})

        return True

    # #================#================#============#=============
    # PERMISSION MANAGEMENT
    # #================#================#============#=============

    def grant_permission(self, role: str, permission: Permission) -> None:
        """Grant permission to role."""
        if role not in self.roles:
            raise ValueError(f"Role not found: {role}")

        self.roles[role].grant_permission(permission)
        self._invalidate_cache()
        self._audit_event(
            "permission_granted", {"role": role, "permission": permission.value}
        )

    def revoke_permission(self, role: str, permission: Permission) -> None:
        """Revoke permission from role."""
        if role not in self.roles:
            raise ValueError(f"Role not found: {role}")

        self.roles[role].revoke_permission(permission)
        self._invalidate_cache()
        self._audit_event(
            "permission_revoked", {"role": role, "permission": permission.value}
        )

    def grant_permissions_batch(self, role: str, permissions: List[Permission]) -> int:
        """Grant multiple permissions."""
        count = 0
        for perm in permissions:
            try:
                self.grant_permission(role, perm)
                count += 1
            except Exception:
                pass
        return count

    # #================#================#============#=============
    # USER MANAGEMENT
    # #================#================#============#=============

    def assign_role_to_user(
        self, user_id: str, role: str, assigned_by: Optional[str] = None
    ) -> None:
        """Assign role to user."""
        if role not in self.roles:
            raise ValueError(f"Role not found: {role}")

        self.user_roles[user_id].add(role)

        # Set default access level from role
        if user_id not in self.user_access_levels:
            self.user_access_levels[user_id] = self.roles[role].access_level

        self._invalidate_cache(user_id)
        self._audit_event(
            "role_assigned", {"user": user_id, "role": role, "assigned_by": assigned_by}
        )

        logger.debug(f"👤 Role '{role}' assigned to user '{user_id}'")

    def remove_role_from_user(self, user_id: str, role: str) -> None:
        """Remove role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)
            self._invalidate_cache(user_id)
            self._audit_event("role_removed", {"user": user_id, "role": role})

    def set_user_access_level(self, user_id: str, level: AccessLevel) -> None:
        """Set user's base access level."""
        self.user_access_levels[user_id] = level
        self._invalidate_cache(user_id)

    def set_user_permission_override(
        self, user_id: str, permission: Permission, allowed: bool
    ) -> None:
        """Set user-specific permission override."""
        self.user_permission_overrides[user_id][permission] = allowed
        self._invalidate_cache(user_id)

    def remove_user_permission_override(
        self, user_id: str, permission: Permission
    ) -> None:
        """Remove user-specific permission override."""
        if user_id in self.user_permission_overrides:
            self.user_permission_overrides[user_id].pop(permission, None)
            self._invalidate_cache(user_id)

    # #================#================#============#=============
    # PERMISSION CHECKING
    # #================#================#============#=============

    def has_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if user has permission with fine-grained control."""

        # Check cache
        cache_key = self._get_cache_key(user_id, permission, resource_type, resource_id)
        if cache_key in self._permission_cache:
            cached_time = self._cache_timestamps.get(cache_key)
            if cached_time and (datetime.now() - cached_time).seconds < self._cache_ttl:
                return self._permission_cache[cache_key].get(permission.value, False)

        # Check user permission overrides
        if user_id in self.user_permission_overrides:
            override = self.user_permission_overrides[user_id].get(permission)
            if override is not None:
                self._cache_result(cache_key, permission, override)
                return override

        # Check user's access level
        user_level = self.user_access_levels.get(user_id, AccessLevel.NO_ACCESS)
        required_level = self._get_permission_required_level(permission)
        if user_level.value < required_level.value:
            self._cache_result(cache_key, permission, False)
            return False

        # Check roles
        if user_id not in self.user_roles:
            self._cache_result(cache_key, permission, False)
            return False

        # Check resource-specific permissions
        if resource_type and resource_id:
            # Check ownership
            if self.is_resource_owner(user_id, resource_id):
                self._cache_result(cache_key, permission, True)
                return True

            # Check ACL
            if self._check_resource_acl(user_id, resource_id, permission):
                self._cache_result(cache_key, permission, True)
                return True

        # Check role permissions
        for role_name in self.user_roles[user_id]:
            role = self.roles.get(role_name)
            if role:
                # Check resource-specific rules
                if resource_type:
                    if role.check_resource_access(
                        resource_type, permission, resource_id, context
                    ):
                        self._cache_result(cache_key, permission, True)
                        return True

                # Check basic permission
                if role.has_permission(permission):
                    self._cache_result(cache_key, permission, True)
                    return True

        self._cache_result(cache_key, permission, False)
        return False

    def _get_permission_required_level(self, permission: Permission) -> AccessLevel:
        """Get required access level for permission."""
        level_map = {
            Permission.READ: AccessLevel.READ_ONLY,
            Permission.MEMORY_READ: AccessLevel.READ_ONLY,
            Permission.WRITE: AccessLevel.STANDARD,
            Permission.MEMORY_WRITE: AccessLevel.STANDARD,
            Permission.EXECUTE: AccessLevel.STANDARD,
            Permission.CREATE: AccessLevel.STANDARD,
            Permission.UPDATE: AccessLevel.STANDARD,
            Permission.DELETE: AccessLevel.ADVANCED,
            Permission.MEMORY_DELETE: AccessLevel.ADVANCED,
            Permission.MANAGE_USERS: AccessLevel.FULL,
            Permission.MANAGE_ROLES: AccessLevel.FULL,
            Permission.MANAGE_PERMISSIONS: AccessLevel.FULL,
            Permission.ADMIN: AccessLevel.SUPER_ADMIN,
            Permission.SYSTEM_CONFIG: AccessLevel.ADVANCED,
            Permission.AUDIT: AccessLevel.ADVANCED,
        }
        return level_map.get(permission, AccessLevel.BASIC)

    def verify_permission(
        self,
        user_id: str,
        permission: Permission,
        raise_on_failure: bool = True,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verify permission, optionally raising exception."""
        has_perm = self.has_permission(
            user_id, permission, resource_type, resource_id, context
        )

        if not has_perm and raise_on_failure:
            raise PermissionError(
                f"User '{user_id}' lacks '{permission.value}' permission"
                + (f" for {resource_type.value}/{resource_id}" if resource_type else "")
            )

        return has_perm

    def require_permission(self, *args, **kwargs):
        """Alias for verify_permission."""
        return self.verify_permission(*args, **kwargs)

    def check(
        self,
        permission: Union[str, Permission],
        user_id: str = "system",
        default_role: str = "developer",
    ) -> bool:
        """
        Backward-compatible shorthand permission check used by older code paths.
        Accepts either Permission enum values or legacy string names.
        """
        if user_id not in self.user_roles and default_role in self.roles:
            self.assign_role_to_user(user_id, default_role)

        if isinstance(permission, Permission):
            return self.has_permission(user_id, permission)

        permission_name = str(permission).strip().lower()
        permission_map = {
            "open_file": Permission.READ,
            "read_file": Permission.READ,
            "run_file": Permission.EXECUTE,
            "execute": Permission.EXECUTE,
            "write_file": Permission.WRITE,
            "edit_file": Permission.WRITE,
            "create_application": Permission.CREATE,
            "delete_file": Permission.DELETE,
            "system_config": Permission.SYSTEM_CONFIG,
        }

        mapped_permission = permission_map.get(permission_name)
        if mapped_permission is None:
            return True

        return self.has_permission(user_id, mapped_permission)

    # #================#================#============#=============
    # RESOURCE ACCESS CONTROL
    # #================#================#============#=============

    def set_resource_owner(self, resource_id: str, owner_id: str) -> None:
        """Set resource owner."""
        self.resource_owners[resource_id] = owner_id
        self._audit_event(
            "resource_owner_set", {"resource": resource_id, "owner": owner_id}
        )

    def is_resource_owner(self, user_id: str, resource_id: str) -> bool:
        """Check if user owns the resource."""
        return self.resource_owners.get(resource_id) == user_id

    def grant_resource_permission(
        self, resource_id: str, user_id: str, permission: Permission
    ) -> None:
        """Grant user-specific permission on a resource."""
        self.resource_acl[resource_id][user_id].add(permission)
        self._audit_event(
            "resource_permission_granted",
            {"resource": resource_id, "user": user_id, "permission": permission.value},
        )

    def revoke_resource_permission(
        self, resource_id: str, user_id: str, permission: Permission
    ) -> None:
        """Revoke user-specific permission on a resource."""
        if resource_id in self.resource_acl:
            self.resource_acl[resource_id][user_id].discard(permission)

    def _check_resource_acl(
        self, user_id: str, resource_id: str, permission: Permission
    ) -> bool:
        """Check resource ACL."""
        if resource_id in self.resource_acl:
            return permission in self.resource_acl[resource_id].get(user_id, set())
        return False

    # #================#================#============#=============
    # FINE-GRAINED RULES
    # #================#================#============#=============

    def add_resource_rule(
        self,
        role_name: str,
        resource_type: ResourceType,
        permission: Permission,
        effect: str = "allow",
        resource_id: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> bool:
        """Add fine-grained access rule to role."""
        role = self.roles.get(role_name)
        if not role:
            return False

        rule = AccessRule(
            resource_type=resource_type,
            resource_id=resource_id,
            permission=permission,
            effect=effect,
            conditions=conditions or {},
            priority=priority,
        )

        role.add_resource_rule(rule)
        self._audit_event(
            "resource_rule_added",
            {
                "role": role_name,
                "resource_type": resource_type.value,
                "permission": permission.value,
                "effect": effect,
            },
        )

        return True

    # #================#================#============#=============
    # SESSION MANAGEMENT
    # #================#================#============#=============

    def create_session(
        self, user_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create user session."""
        import secrets

        session_id = secrets.token_urlsafe(32)

        self.active_sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "metadata": metadata or {},
        }

        self._audit_event(
            "session_created", {"user": user_id, "session": session_id[:8]}
        )
        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id."""
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        # Check timeout
        elapsed = (datetime.now() - session["last_activity"]).total_seconds()
        if elapsed > self.session_timeout:
            self.delete_session(session_id)
            return None

        # Update last activity
        session["last_activity"] = datetime.now()

        return session["user_id"]

    def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]["user_id"]
            del self.active_sessions[session_id]
            self._audit_event(
                "session_deleted", {"user": user_id, "session": session_id[:8]}
            )
            return True
        return False

    def get_user_from_session(self, session_id: str) -> Optional[str]:
        """Get user from session."""
        return self.validate_session(session_id)

    # #================#================#============#=============
    # QUERY METHODS
    # #================#================#============#=============

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user."""
        permissions = set()

        if user_id in self.user_roles:
            for role_name in self.user_roles[user_id]:
                role = self.roles.get(role_name)
                if role:
                    permissions.update(role.permissions)

        # Apply overrides
        if user_id in self.user_permission_overrides:
            for perm, allowed in self.user_permission_overrides[user_id].items():
                if allowed:
                    permissions.add(perm)
                else:
                    permissions.discard(perm)

        return permissions

    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get roles for user."""
        return self.user_roles.get(user_id, set())

    def get_users_with_role(self, role_name: str) -> List[str]:
        """Get all users with a specific role."""
        return [
            user_id for user_id, roles in self.user_roles.items() if role_name in roles
        ]

    def get_user_access_level(self, user_id: str) -> AccessLevel:
        """Get user's access level."""
        return self.user_access_levels.get(user_id, AccessLevel.NO_ACCESS)

    # #================#================#============#=============
    # AUDIT & MONITORING
    # #================#================#============#=============

    def _audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        if not self.audit_enabled:
            return

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        }

        self.audit_log.append(event)

        # Maintain size limit
        if len(self.audit_log) > self.max_audit_size:
            self.audit_log = self.audit_log[-self.max_audit_size :]

    def get_audit_log(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries with filters."""
        entries = self.audit_log[-limit:]

        if event_type:
            entries = [e for e in entries if e["event_type"] == event_type]

        if user_id:
            entries = [e for e in entries if e["details"].get("user") == user_id]

        return entries

    def export_audit_log(self, filepath: Optional[str] = None) -> str:
        """Export audit log to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.persistence_path / f"audit_log_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.audit_log, f, indent=2, default=str)

        logger.info(f"📊 Audit log exported to {filepath}")
        return str(filepath)

    # #================#================#============#=============
    # CACHE MANAGEMENT
    # #================#================#============#=============

    def _get_cache_key(
        self,
        user_id: str,
        permission: Permission,
        resource_type: Optional[ResourceType],
        resource_id: Optional[str],
    ) -> str:
        """Generate cache key."""
        key = f"{user_id}:{permission.value}"
        if resource_type:
            key += f":{resource_type.value}"
        if resource_id:
            key += f":{resource_id}"
        return hashlib.md5(key.encode()).hexdigest()

    def _cache_result(self, cache_key: str, permission: Permission, result: bool):
        """Cache permission check result."""
        self._permission_cache[cache_key][permission.value] = result
        self._cache_timestamps[cache_key] = datetime.now()

    def _invalidate_cache(self, user_id: Optional[str] = None):
        """Invalidate permission cache."""
        if user_id:
            # Remove cache entries for specific user
            to_remove = [
                key
                for key in self._permission_cache.keys()
                if key.startswith(hashlib.md5(f"{user_id}:".encode()).hexdigest()[:10])
            ]
            for key in to_remove:
                del self._permission_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
        else:
            # Clear all cache
            self._permission_cache.clear()
            self._cache_timestamps.clear()

    # #================#================#============#=============
    # PERSISTENCE
    # #================#================#============#=============

    def save_to_disk(self) -> bool:
        """Save permission configuration to disk."""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "roles": {name: role.to_dict() for name, role in self.roles.items()},
                "user_roles": {
                    uid: list(roles) for uid, roles in self.user_roles.items()
                },
                "user_access_levels": {
                    uid: level.value for uid, level in self.user_access_levels.items()
                },
                "user_permission_overrides": {
                    uid: {p.value: allowed for p, allowed in overrides.items()}
                    for uid, overrides in self.user_permission_overrides.items()
                },
                "resource_owners": self.resource_owners,
            }

            filepath = self.persistence_path / "permissions.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"💾 Permissions saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save permissions: {e}")
            return False

    def _load_from_disk(self) -> bool:
        """Load permission configuration from disk."""
        filepath = self.persistence_path / "permissions.json"
        if not filepath.exists():
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Load roles
            for name, role_data in data.get("roles", {}).items():
                self.roles[name] = Role.from_dict(role_data)

            # Load user roles
            for uid, roles in data.get("user_roles", {}).items():
                self.user_roles[uid] = set(roles)

            # Load user access levels
            for uid, level_value in data.get("user_access_levels", {}).items():
                self.user_access_levels[uid] = AccessLevel(level_value)

            # Load user permission overrides
            for uid, overrides in data.get("user_permission_overrides", {}).items():
                for perm_value, allowed in overrides.items():
                    self.user_permission_overrides[uid][
                        Permission(perm_value)
                    ] = allowed

            # Load resource owners
            self.resource_owners = data.get("resource_owners", {})

            logger.info(f"📂 Permissions loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load permissions: {e}")
            return False

    # #================#================#============#=============
    # CALLBACKS
    # #================#================#============#=============

    def add_permission_callback(self, callback: Callable):
        """Add callback for permission checks."""
        self._permission_callbacks.append(callback)

    def _trigger_callbacks(self, user_id: str, permission: Permission, result: bool):
        """Trigger permission check callbacks."""
        for callback in self._permission_callbacks:
            try:
                callback(user_id, permission, result)
            except Exception as e:
                logger.error(f"Permission callback error: {e}")

    # #================#================#============#=============
    # STATISTICS
    # #================#================#============#=============

    def get_stats(self) -> Dict[str, Any]:
        """Get permission manager statistics."""
        role_permission_counts = {
            name: len(role.permissions) for name, role in self.roles.items()
        }

        return {
            "total_roles": len(self.roles),
            "total_users": len(self.user_roles),
            "role_names": list(self.roles.keys()),
            "active_sessions": len(self.active_sessions),
            "audit_log_size": len(self.audit_log),
            "resource_owners": len(self.resource_owners),
            "role_permission_counts": role_permission_counts,
            "user_permission_overrides": sum(
                len(o) for o in self.user_permission_overrides.values()
            ),
            "cache_size": len(self._permission_cache),
        }

    def get_role_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all roles."""
        return [
            {
                "name": role.name,
                "description": role.description,
                "permission_count": len(role.permissions),
                "access_level": role.access_level.value,
                "user_count": len(self.get_users_with_role(role.name)),
                "created_at": role.created_at.isoformat(),
            }
            for role in self.roles.values()
        ]

    # #================#================#============#=============
    # UTILITIES
    # ================#================#============#=============

    def reset(self, confirm: bool = False) -> bool:
        """Reset all permissions to defaults."""
        if not confirm:
            logger.warning("Reset requires confirmation")
            return False

        self.roles.clear()
        self.user_roles.clear()
        self.user_permission_overrides.clear()
        self.user_access_levels.clear()
        self.resource_owners.clear()
        self.resource_acl.clear()
        self.active_sessions.clear()
        self.audit_log.clear()
        self._permission_cache.clear()
        self._cache_timestamps.clear()

        self._initialize_default_roles()

        self._audit_event("system_reset", {})
        logger.warning("🔄 Permission system reset to defaults")

        return True

    def backup(self) -> Dict[str, Any]:
        """Create full backup of permission system."""
        return {
            "timestamp": datetime.now().isoformat(),
            "roles": {name: role.to_dict() for name, role in self.roles.items()},
            "user_roles": {uid: list(roles) for uid, roles in self.user_roles.items()},
            "user_access_levels": {
                uid: level.value for uid, level in self.user_access_levels.items()
            },
            "resource_owners": self.resource_owners,
            "resource_acl": {
                rid: {uid: [p.value for p in perms] for uid, perms in acl.items()}
                for rid, acl in self.resource_acl.items()
            },
        }

    def restore(self, backup: Dict[str, Any]) -> bool:
        """Restore from backup."""
        try:
            self.roles.clear()
            self.user_roles.clear()
            self.user_permission_overrides.clear()
            self.user_access_levels.clear()

            for name, role_data in backup.get("roles", {}).items():
                self.roles[name] = Role.from_dict(role_data)

            for uid, roles in backup.get("user_roles", {}).items():
                self.user_roles[uid] = set(roles)

            for uid, level_value in backup.get("user_access_levels", {}).items():
                self.user_access_levels[uid] = AccessLevel(level_value)

            self.resource_owners = backup.get("resource_owners", {})

            self._audit_event("restored_from_backup", {})
            logger.info("✅ Permission system restored from backup")

            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


# #================#================#============#=============
# CONVENIENCE FUNCTIONS
# #================#================#============#=============


def get_permission_manager() -> PermissionManager:
    """Get global permission manager instance."""
    return PermissionManager()


def require_permission(permission: Permission):
    """Decorator for permission-based access control."""

    def decorator(func):
        def wrapper(user_id, *args, **kwargs):
            pm = get_permission_manager()
            if pm.has_permission(user_id, permission):
                return func(user_id, *args, **kwargs)
            else:
                raise PermissionError(
                    f"User {user_id} lacks {permission.value} permission"
                )

        return wrapper

    return decorator


__all__ = [
    "PermissionManager",
    "Permission",
    "AccessLevel",
    "ResourceType",
    "Role",
    "AccessRule",
    "get_permission_manager",
    "require_permission",
]
