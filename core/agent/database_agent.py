"""
Database Agent for EDIATH
Advanced database operations: SQL and NoSQL queries, connection management, data migration, schema management
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import hashlib

# SQL Libraries
try:
    import asyncpg

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import aiomysql

    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import aiosqlite

    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import pymongo
    from motor.motor_asyncio import AsyncIOMotorClient

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncSession,
        async_sessionmaker,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy import (
        text,
        MetaData,
        Table,
        Column,
        String,
        Integer,
        Float,
        DateTime,
        Boolean,
        JSON,
    )

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DatabaseType(Enum):
    """Supported database types"""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


class QueryType(Enum):
    """Query operation types"""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    RAW = "raw"


@dataclass
class DatabaseConnection:
    """Database connection information"""

    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    timeout: int = 30


@dataclass
class QueryResult:
    """Query result container"""

    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseAgent:
    """
    Advanced database agent capable of:
    - SQL database operations (PostgreSQL, MySQL, SQLite)
    - NoSQL database operations (MongoDB, Redis)
    - Connection pooling and management
    - Query optimization and analysis
    - Schema management and migration
    - Data import/export
    - Query building and sanitization
    - Transaction management
    - Backup and restore
    - Multiple database connections
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Database Agent

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Connection management
        self.connections: Dict[str, Any] = {}
        self.connection_configs: Dict[str, DatabaseConnection] = {}
        self.active_connections: Dict[str, bool] = {}

        # Connection pools
        self.pools: Dict[str, Any] = {}

        # Query cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        self.query_cache: Dict[str, Tuple[datetime, QueryResult]] = {}

        # SQLAlchemy setup
        self.engines: Dict[str, Any] = {}
        self.session_makers: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "by_db_type": {},
        }

        # Query history
        self.query_history: List[QueryResult] = []
        self.max_history = self.config.get("max_history", 1000)

        # Allowed databases (security)
        self.allowed_databases = set(self.config.get("allowed_databases", []))
        self.blocked_keywords = set(
            self.config.get(
                "blocked_keywords",
                [
                    "DROP",
                    "TRUNCATE",
                    "ALTER",
                    "CREATE",
                    "DELETE",
                    "UPDATE",
                    "INSERT",
                    "REPLACE",
                    "GRANT",
                    "REVOKE",
                ],
            )
        )

        self.logger.info("Database Agent initialized")

    async def connect(
        self, connection_name: str, connection_config: DatabaseConnection
    ) -> Dict[str, Any]:
        """
        Create a database connection

        Args:
            connection_name: Unique name for this connection
            connection_config: Database connection configuration

        Returns:
            Dictionary with connection result
        """
        if connection_name in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} already exists",
            }

        try:
            if connection_config.db_type == DatabaseType.POSTGRESQL:
                conn = await self._connect_postgresql(connection_config)
            elif connection_config.db_type == DatabaseType.MYSQL:
                conn = await self._connect_mysql(connection_config)
            elif connection_config.db_type == DatabaseType.SQLITE:
                conn = await self._connect_sqlite(connection_config)
            elif connection_config.db_type == DatabaseType.MONGODB:
                conn = await self._connect_mongodb(connection_config)
            elif connection_config.db_type == DatabaseType.REDIS:
                conn = await self._connect_redis(connection_config)
            else:
                raise ValueError(
                    f"Unsupported database type: {connection_config.db_type}"
                )

            self.connections[connection_name] = conn
            self.connection_configs[connection_name] = connection_config
            self.active_connections[connection_name] = True

            # Update statistics
            db_type = connection_config.db_type.value
            if db_type not in self.stats["by_db_type"]:
                self.stats["by_db_type"][db_type] = 0
            self.stats["by_db_type"][db_type] += 1

            self.logger.info(
                f"Connected to {connection_name} ({connection_config.db_type.value})"
            )

            return {
                "success": True,
                "connection_name": connection_name,
                "db_type": connection_config.db_type.value,
                "database": connection_config.database,
                "message": f"Connected to {connection_name}",
            }

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "connection_name": connection_name,
            }

    async def _connect_postgresql(self, config: DatabaseConnection) -> Any:
        """Connect to PostgreSQL database"""
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "asyncpg not available. Install with: pip install asyncpg"
            )

        # Create connection pool
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            database=config.database,
            min_size=1,
            max_size=config.pool_size,
            timeout=config.timeout,
            ssl=config.ssl,
        )

        self.pools[config.database] = pool
        return pool

    async def _connect_mysql(self, config: DatabaseConnection) -> Any:
        """Connect to MySQL database"""
        if not MYSQL_AVAILABLE:
            raise ImportError(
                "aiomysql not available. Install with: pip install aiomysql"
            )

        # Create connection pool
        pool = await aiomysql.create_pool(
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            db=config.database,
            minsize=1,
            maxsize=config.pool_size,
            autocommit=True,
        )

        self.pools[config.database] = pool
        return pool

    async def _connect_sqlite(self, config: DatabaseConnection) -> Any:
        """Connect to SQLite database"""
        if not SQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite not available. Install with: pip install aiosqlite"
            )

        # SQLite uses file path instead of host/port
        db_path = config.database if config.database else ":memory:"

        # Create connection
        conn = await aiosqlite.connect(db_path)
        self.pools[config.database] = conn
        return conn

    async def _connect_mongodb(self, config: DatabaseConnection) -> Any:
        """Connect to MongoDB database"""
        if not MONGODB_AVAILABLE:
            raise ImportError("motor not available. Install with: pip install motor")

        # Build connection URI
        if config.username and config.password:
            uri = f"mongodb://{config.username}:{config.password}@{config.host}:{config.port}/"
        else:
            uri = f"mongodb://{config.host}:{config.port}/"

        client = AsyncIOMotorClient(uri)
        db = client[config.database]

        return db

    async def _connect_redis(self, config: DatabaseConnection) -> Any:
        """Connect to Redis database"""
        if not REDIS_AVAILABLE:
            raise ImportError("redis not available. Install with: pip install redis")

        # Create Redis client
        client = redis.Redis(
            host=config.host,
            port=config.port,
            password=config.password,
            db=int(config.database) if config.database else 0,
            decode_responses=True,
        )

        # Test connection
        await client.ping()

        return client

    async def disconnect(self, connection_name: str) -> Dict[str, Any]:
        """
        Disconnect from database

        Args:
            connection_name: Connection name to disconnect

        Returns:
            Dictionary with disconnection result
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        try:
            conn = self.connections[connection_name]
            config = self.connection_configs[connection_name]

            if config.db_type == DatabaseType.POSTGRESQL:
                await conn.close()
            elif config.db_type == DatabaseType.MYSQL:
                conn.close()
                await conn.wait_closed()
            elif config.db_type == DatabaseType.SQLITE:
                await conn.close()
            elif config.db_type == DatabaseType.MONGODB:
                # MongoDB client doesn't need explicit close
                pass
            elif config.db_type == DatabaseType.REDIS:
                await conn.close()

            # Clean up
            del self.connections[connection_name]
            del self.connection_configs[connection_name]
            self.active_connections[connection_name] = False

            return {
                "success": True,
                "connection_name": connection_name,
                "message": f"Disconnected from {connection_name}",
            }

        except Exception as e:
            self.logger.error(f"Disconnection error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "connection_name": connection_name,
            }

    async def execute_query(
        self,
        connection_name: str,
        query: str,
        params: Optional[Dict] = None,
        query_type: QueryType = QueryType.SELECT,
        use_cache: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a database query

        Args:
            connection_name: Name of the connection to use
            query: SQL or NoSQL query
            params: Query parameters
            query_type: Type of query
            use_cache: Use cached results
            timeout: Query timeout in seconds

        Returns:
            Dictionary with query results
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        # Security check
        if not self._is_query_safe(query, query_type):
            return {
                "success": False,
                "error": "Query contains blocked keywords or operations",
                "blocked": True,
            }

        # Generate cache key
        cache_key = self._get_cache_key(connection_name, query, params)

        # Check cache
        if use_cache and self.cache_enabled and query_type == QueryType.SELECT:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return self._format_result(cached_result)

        start_time = datetime.now()

        try:
            config = self.connection_configs[connection_name]
            conn = self.connections[connection_name]

            # Execute based on database type
            if config.db_type in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
                result = await self._execute_sql_query(
                    conn, config, query, params, query_type, timeout
                )
            elif config.db_type == DatabaseType.SQLITE:
                result = await self._execute_sqlite_query(
                    conn, query, params, query_type
                )
            elif config.db_type == DatabaseType.MONGODB:
                result = await self._execute_mongodb_query(
                    conn, query, params, query_type
                )
            elif config.db_type == DatabaseType.REDIS:
                result = await self._execute_redis_query(
                    conn, query, params, query_type
                )
            else:
                raise ValueError(f"Unsupported database type: {config.db_type}")

            execution_time = (datetime.now() - start_time).total_seconds()

            # Create query result
            query_result = QueryResult(
                success=result["success"],
                data=result.get("data", []),
                row_count=result.get("row_count", 0),
                execution_time=execution_time,
                query=query,
                error=result.get("error"),
                metadata=result.get("metadata", {}),
            )

            # Update statistics
            self.stats["total_queries"] += 1
            if query_result.success:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1

            self.stats["total_execution_time"] += execution_time

            # Add to history
            self._add_to_history(query_result)

            # Cache result for SELECT queries
            if query_result.success and query_type == QueryType.SELECT:
                self._add_to_cache(cache_key, query_result)

            return self._format_result(query_result)

        except Exception as e:
            self.logger.error(f"Query execution error: {str(e)}")
            self.stats["failed_queries"] += 1

            return {
                "success": False,
                "error": str(e),
                "query": query,
                "connection": connection_name,
            }

    async def _execute_sql_query(
        self,
        conn,
        config: DatabaseConnection,
        query: str,
        params: Optional[Dict],
        query_type: QueryType,
        timeout: Optional[int],
    ) -> Dict:
        """Execute SQL query (PostgreSQL/MySQL)"""
        if config.db_type == DatabaseType.POSTGRESQL:
            # Use connection pool
            async with conn.acquire() as connection:
                if query_type == QueryType.SELECT:
                    if params:
                        rows = await connection.fetch(query, *params.values())
                    else:
                        rows = await connection.fetch(query)

                    data = [dict(row) for row in rows]
                    return {"success": True, "data": data, "row_count": len(data)}
                else:
                    if params:
                        result = await connection.execute(query, *params.values())
                    else:
                        result = await connection.execute(query)

                    return {
                        "success": True,
                        "row_count": result if isinstance(result, int) else 0,
                        "data": [],
                    }

        elif config.db_type == DatabaseType.MYSQL:
            async with conn.acquire() as connection:
                async with connection.cursor() as cursor:
                    if params:
                        await cursor.execute(query, list(params.values()))
                    else:
                        await cursor.execute(query)

                    if query_type == QueryType.SELECT:
                        rows = await cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                        data = [dict(zip(columns, row)) for row in rows]
                        return {"success": True, "data": data, "row_count": len(data)}
                    else:
                        return {
                            "success": True,
                            "row_count": cursor.rowcount,
                            "data": [],
                        }

    async def _execute_sqlite_query(
        self, conn, query: str, params: Optional[Dict], query_type: QueryType
    ) -> Dict:
        """Execute SQLite query"""
        async with conn.execute(query, params or {}) as cursor:
            if query_type == QueryType.SELECT:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                data = [dict(zip(columns, row)) for row in rows]
                return {"success": True, "data": data, "row_count": len(data)}
            else:
                await conn.commit()
                return {"success": True, "row_count": cursor.rowcount, "data": []}

    async def _execute_mongodb_query(
        self, db, query: str, params: Optional[Dict], query_type: QueryType
    ) -> Dict:
        """Execute MongoDB query"""
        # Parse query as JSON
        if isinstance(query, str):
            query_dict = json.loads(query)
        else:
            query_dict = query

        collection = query_dict.get("collection")
        operation = query_dict.get("operation", "find")

        if not collection:
            raise ValueError("Collection name required for MongoDB query")

        coll = db[collection]

        if operation == "find":
            filter_query = query_dict.get("filter", {})
            projection = query_dict.get("projection", None)
            limit = query_dict.get("limit", 100)

            cursor = coll.find(filter_query, projection).limit(limit)
            data = await cursor.to_list(length=limit)

            return {"success": True, "data": data, "row_count": len(data)}

        elif operation == "insert_one":
            document = query_dict.get("document")
            result = await coll.insert_one(document)
            return {
                "success": True,
                "data": [{"inserted_id": str(result.inserted_id)}],
                "row_count": 1,
            }

        elif operation == "insert_many":
            documents = query_dict.get("documents", [])
            result = await coll.insert_many(documents)
            return {
                "success": True,
                "data": [{"inserted_ids": [str(id) for id in result.inserted_ids]}],
                "row_count": len(result.inserted_ids),
            }

        elif operation == "update":
            filter_query = query_dict.get("filter", {})
            update_data = query_dict.get("update", {})
            result = await coll.update_many(filter_query, {"$set": update_data})
            return {
                "success": True,
                "data": [{"modified_count": result.modified_count}],
                "row_count": result.modified_count,
            }

        elif operation == "delete":
            filter_query = query_dict.get("filter", {})
            result = await coll.delete_many(filter_query)
            return {
                "success": True,
                "data": [{"deleted_count": result.deleted_count}],
                "row_count": result.deleted_count,
            }

        elif operation == "aggregate":
            pipeline = query_dict.get("pipeline", [])
            cursor = coll.aggregate(pipeline)
            data = await cursor.to_list(length=1000)
            return {"success": True, "data": data, "row_count": len(data)}

        else:
            raise ValueError(f"Unsupported MongoDB operation: {operation}")

    async def _execute_redis_query(
        self, client, query: str, params: Optional[Dict], query_type: QueryType
    ) -> Dict:
        """Execute Redis command"""
        # Parse command
        parts = query.strip().split()
        command = parts[0].upper()
        args = parts[1:] if len(parts) > 1 else []

        # Execute Redis command
        method = getattr(client, command.lower(), None)
        if not method:
            raise ValueError(f"Unsupported Redis command: {command}")

        result = await method(*args)

        # Format result
        if isinstance(result, list):
            data = [{"value": v} for v in result]
        elif isinstance(result, dict):
            data = [result]
        else:
            data = [{"result": result}]

        return {
            "success": True,
            "data": data,
            "row_count": len(data),
            "metadata": {"command": command},
        }

    async def execute_many(
        self,
        connection_name: str,
        queries: List[str],
        params_list: Optional[List[Dict]] = None,
        use_transaction: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute multiple queries in batch

        Args:
            connection_name: Connection name
            queries: List of queries to execute
            params_list: List of parameters for each query
            use_transaction: Use transaction for all queries

        Returns:
            Dictionary with batch results
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        results = []
        failed = []

        config = self.connection_configs[connection_name]

        try:
            if use_transaction and config.db_type in [
                DatabaseType.POSTGRESQL,
                DatabaseType.MYSQL,
            ]:
                # Start transaction
                conn = self.connections[connection_name]

                if config.db_type == DatabaseType.POSTGRESQL:
                    async with conn.acquire() as connection:
                        async with connection.transaction():
                            for i, query in enumerate(queries):
                                params = (
                                    params_list[i]
                                    if params_list and i < len(params_list)
                                    else None
                                )
                                result = await self.execute_query(
                                    connection_name, query, params, use_cache=False
                                )
                                results.append(result)
                                if not result["success"]:
                                    failed.append(
                                        {"index": i, "error": result.get("error")}
                                    )
                                    raise Exception(
                                        f"Query {i} failed: {result.get('error')}"
                                    )
            else:
                # Execute without transaction
                for i, query in enumerate(queries):
                    params = (
                        params_list[i] if params_list and i < len(params_list) else None
                    )
                    result = await self.execute_query(
                        connection_name, query, params, use_cache=False
                    )
                    results.append(result)
                    if not result["success"]:
                        failed.append({"index": i, "error": result.get("error")})

            return {
                "success": len(failed) == 0,
                "total_queries": len(queries),
                "successful_queries": len([r for r in results if r["success"]]),
                "failed_queries": len(failed),
                "results": results,
                "failed": failed if failed else None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_queries": len(queries),
                "successful_queries": len([r for r in results if r["success"]]),
                "failed_queries": len(queries)
                - len([r for r in results if r["success"]]),
            }

    async def get_tables(self, connection_name: str) -> Dict[str, Any]:
        """
        Get list of tables in database

        Args:
            connection_name: Connection name

        Returns:
            Dictionary with tables list
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        config = self.connection_configs[connection_name]

        try:
            if config.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    tables = [row["table_name"] for row in result["data"]]
                    return {"success": True, "tables": tables, "count": len(tables)}

            elif config.db_type == DatabaseType.MYSQL:
                query = "SHOW TABLES"
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    tables = [list(row.values())[0] for row in result["data"]]
                    return {"success": True, "tables": tables, "count": len(tables)}

            elif config.db_type == DatabaseType.SQLITE:
                query = "SELECT name FROM sqlite_master WHERE type='table'"
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    tables = [row["name"] for row in result["data"]]
                    return {"success": True, "tables": tables, "count": len(tables)}

            elif config.db_type == DatabaseType.MONGODB:
                db = self.connections[connection_name]
                collections = await db.list_collection_names()
                return {
                    "success": True,
                    "tables": collections,
                    "count": len(collections),
                }

            elif config.db_type == DatabaseType.REDIS:
                # Redis doesn't have tables, return keyspace info
                client = self.connections[connection_name]
                info = await client.info("keyspace")
                return {
                    "success": True,
                    "tables": list(info.keys()),
                    "count": len(info),
                }

            return {
                "success": False,
                "error": f"Unsupported database type: {config.db_type}",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_schema(self, connection_name: str, table_name: str) -> Dict[str, Any]:
        """
        Get schema information for a table

        Args:
            connection_name: Connection name
            table_name: Name of the table

        Returns:
            Dictionary with schema information
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        config = self.connection_configs[connection_name]

        try:
            if config.db_type == DatabaseType.POSTGRESQL:
                query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = $1
                    ORDER BY ordinal_position
                """
                result = await self.execute_query(
                    connection_name, query, {"table_name": table_name}
                )

                if result["success"]:
                    return {
                        "success": True,
                        "table": table_name,
                        "columns": result["data"],
                        "count": len(result["data"]),
                    }

            elif config.db_type == DatabaseType.MYSQL:
                query = f"DESCRIBE {table_name}"
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    columns = []
                    for row in result["data"]:
                        columns.append(
                            {
                                "column_name": row.get("Field"),
                                "data_type": row.get("Type"),
                                "is_nullable": row.get("Null"),
                                "column_default": row.get("Default"),
                            }
                        )
                    return {
                        "success": True,
                        "table": table_name,
                        "columns": columns,
                        "count": len(columns),
                    }

            elif config.db_type == DatabaseType.SQLITE:
                query = f"PRAGMA table_info({table_name})"
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    columns = []
                    for row in result["data"]:
                        columns.append(
                            {
                                "column_name": row.get("name"),
                                "data_type": row.get("type"),
                                "is_nullable": (
                                    "YES" if row.get("notnull") == 0 else "NO"
                                ),
                                "column_default": row.get("dflt_value"),
                            }
                        )
                    return {
                        "success": True,
                        "table": table_name,
                        "columns": columns,
                        "count": len(columns),
                    }

            elif config.db_type == DatabaseType.MONGODB:
                # Get a sample document to infer schema
                db = self.connections[connection_name]
                coll = db[table_name]
                sample = await coll.find_one()

                if sample:
                    schema = []
                    for key, value in sample.items():
                        schema.append(
                            {
                                "column_name": key,
                                "data_type": type(value).__name__,
                                "sample_value": str(value)[:100],
                            }
                        )

                    return {
                        "success": True,
                        "table": table_name,
                        "columns": schema,
                        "count": len(schema),
                        "sample_document": sample,
                    }

            return {"success": False, "error": f"Cannot get schema for {table_name}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def backup_database(
        self, connection_name: str, backup_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backup database to file

        Args:
            connection_name: Connection name
            backup_path: Path to save backup

        Returns:
            Dictionary with backup result
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_{connection_name}_{timestamp}.json"

        try:
            config = self.connection_configs[connection_name]

            # Get all tables
            tables_result = await self.get_tables(connection_name)
            if not tables_result["success"]:
                return tables_result

            backup_data = {
                "connection_name": connection_name,
                "db_type": config.db_type.value,
                "database": config.database,
                "backup_time": datetime.now().isoformat(),
                "tables": {},
            }

            # Backup each table
            for table in tables_result["tables"]:
                query = f"SELECT * FROM {table}"
                result = await self.execute_query(connection_name, query)

                if result["success"]:
                    backup_data["tables"][table] = {
                        "row_count": len(result["data"]),
                        "data": result["data"],
                    }

            # Save to file
            backup_path_obj = Path(backup_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(backup_path_obj, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, default=str)

            return {
                "success": True,
                "backup_path": str(backup_path_obj),
                "size": backup_path_obj.stat().st_size,
                "tables": len(backup_data["tables"]),
                "message": f"Backup saved to {backup_path}",
            }

        except Exception as e:
            self.logger.error(f"Backup error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def restore_backup(
        self, connection_name: str, backup_path: str
    ) -> Dict[str, Any]:
        """
        Restore database from backup file

        Args:
            connection_name: Connection name
            backup_path: Path to backup file

        Returns:
            Dictionary with restore result
        """
        if connection_name not in self.connections:
            return {
                "success": False,
                "error": f"Connection {connection_name} not found",
            }

        try:
            # Load backup data
            with open(backup_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

            restored_tables = []
            failed_tables = []

            # Restore each table
            for table_name, table_data in backup_data["tables"].items():
                # Clear existing data
                clear_query = f"DELETE FROM {table_name}"
                await self.execute_query(
                    connection_name, clear_query, query_type=QueryType.DELETE
                )

                # Insert data
                for row in table_data["data"]:
                    columns = ", ".join(row.keys())
                    placeholders = ", ".join(["?" for _ in row])
                    insert_query = (
                        f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    )

                    result = await self.execute_query(
                        connection_name,
                        insert_query,
                        params=list(row.values()),
                        query_type=QueryType.INSERT,
                    )

                    if result["success"]:
                        restored_tables.append(table_name)
                    else:
                        failed_tables.append(table_name)

            return {
                "success": len(failed_tables) == 0,
                "restored_tables": restored_tables,
                "failed_tables": failed_tables,
                "total_tables": len(backup_data["tables"]),
                "message": f"Restored {len(restored_tables)} tables",
            }

        except Exception as e:
            self.logger.error(f"Restore error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _is_query_safe(self, query: str, query_type: QueryType) -> bool:
        """Check if query is safe to execute"""
        query_upper = query.upper()

        # Block dangerous keywords for non-SELECT queries
        if query_type != QueryType.SELECT:
            for keyword in self.blocked_keywords:
                if keyword in query_upper:
                    return False

        # Check for SQL injection patterns
        dangerous_patterns = [
            r";\s*DROP",
            r";\s*DELETE",
            r";\s*UPDATE",
            r";\s*INSERT",
            r"--\s*DROP",
            r"\/\*\s*DROP",
            r"UNION\s+SELECT",
            r"OR\s+1\s*=\s*1",
            r"OR\s+\'1\'=\'1",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False

        return True

    def _get_cache_key(
        self, connection_name: str, query: str, params: Optional[Dict]
    ) -> str:
        """Generate cache key for query"""
        key_data = f"{connection_name}:{query}:{json.dumps(params or {})}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[QueryResult]:
        """Get query result from cache"""
        if cache_key in self.query_cache:
            timestamp, result = self.query_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return result
            else:
                del self.query_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: QueryResult):
        """Add query result to cache"""
        if self.cache_enabled:
            # Manage cache size
            if len(self.query_cache) > 1000:
                # Remove oldest 10%
                items = sorted(self.query_cache.items(), key=lambda x: x[1][0])
                for key, _ in items[:100]:
                    del self.query_cache[key]

            self.query_cache[cache_key] = (datetime.now(), result)
            self.stats["cache_misses"] += 1

    def _add_to_history(self, result: QueryResult):
        """Add query to history"""
        self.query_history.append(result)
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history :]

    def _format_result(self, result: QueryResult) -> Dict[str, Any]:
        """Format query result for output"""
        return {
            "success": result.success,
            "data": result.data,
            "row_count": result.row_count,
            "execution_time": result.execution_time,
            "error": result.error,
            "metadata": result.metadata,
        }

    def get_query_history(
        self, limit: int = None, success_only: bool = False
    ) -> List[Dict]:
        """Get query history"""
        history = self.query_history

        if success_only:
            history = [h for h in history if h.success]

        if limit:
            history = history[-limit:]

        return [
            {
                "success": h.success,
                "row_count": h.row_count,
                "execution_time": h.execution_time,
                "query_preview": h.query[:100],
                "error": h.error,
            }
            for h in history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        avg_time = (
            (self.stats["total_execution_time"] / self.stats["total_queries"])
            if self.stats["total_queries"] > 0
            else 0
        )

        return {
            **self.stats,
            "average_execution_time": avg_time,
            "success_rate": (
                (self.stats["successful_queries"] / self.stats["total_queries"] * 100)
                if self.stats["total_queries"] > 0
                else 0
            ),
            "cache_hit_rate": (
                (
                    self.stats["cache_hits"]
                    / (self.stats["cache_hits"] + self.stats["cache_misses"])
                    * 100
                )
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                else 0
            ),
            "active_connections": len(
                [c for c in self.active_connections.values() if c]
            ),
            "history_size": len(self.query_history),
            "cache_size": len(self.query_cache),
        }

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")

    def clear_history(self):
        """Clear query history"""
        self.query_history.clear()
        self.logger.info("Query history cleared")

    async def close_all_connections(self):
        """Close all active database connections"""
        for conn_name in list(self.connections.keys()):
            await self.disconnect(conn_name)

        self.logger.info("All database connections closed")


# Integration wrapper for EDIATH
class DatabaseAgentWrapper:
    """
    Wrapper class to integrate DatabaseAgent with EDIATH's agent architecture
    """

    def __init__(self, config: Optional[Dict] = None):
        self.db_agent = DatabaseAgent(config)
        self.agent_type = "database"
        self.capabilities = [
            "connect",
            "execute_query",
            "batch_execution",
            "get_tables",
            "get_schema",
            "backup_restore",
            "transaction_management",
        ]

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a database request

        Request format:
        {
            'operation': 'connect|query|batch|tables|schema|backup|restore|history',
            ... operation specific parameters ...
        }
        """
        operation = request.get("operation")

        if operation == "connect":
            db_type = DatabaseType(request.get("db_type"))
            connection_config = DatabaseConnection(
                db_type=db_type,
                host=request.get("host", "localhost"),
                port=request.get("port", self._get_default_port(db_type)),
                database=request.get("database"),
                username=request.get("username"),
                password=request.get("password"),
                ssl=request.get("ssl", False),
                pool_size=request.get("pool_size", 10),
                timeout=request.get("timeout", 30),
            )

            return await self.db_agent.connect(
                connection_name=request.get("connection_name"),
                connection_config=connection_config,
            )

        elif operation == "disconnect":
            return await self.db_agent.disconnect(
                connection_name=request.get("connection_name")
            )

        elif operation == "query":
            query_type = request.get("query_type", "select")

            return await self.db_agent.execute_query(
                connection_name=request.get("connection_name"),
                query=request.get("query"),
                params=request.get("params"),
                query_type=QueryType(query_type),
                use_cache=request.get("use_cache", True),
                timeout=request.get("timeout"),
            )

        elif operation == "batch":
            query_type = request.get("query_type", "select")
            queries = request.get("queries", [])
            params_list = request.get("params_list")

            results = []
            for query in queries:
                result = await self.db_agent.execute_query(
                    connection_name=request.get("connection_name"),
                    query=query,
                    params=params_list,
                    query_type=QueryType(query_type),
                )
                results.append(result)

            return {
                "success": all(r["success"] for r in results),
                "results": results,
                "total": len(results),
            }

        elif operation == "tables":
            return await self.db_agent.get_tables(
                connection_name=request.get("connection_name")
            )

        elif operation == "schema":
            return await self.db_agent.get_schema(
                connection_name=request.get("connection_name"),
                table_name=request.get("table_name"),
            )

        elif operation == "backup":
            return await self.db_agent.backup_database(
                connection_name=request.get("connection_name"),
                backup_path=request.get("backup_path"),
            )

        elif operation == "restore":
            return await self.db_agent.restore_backup(
                connection_name=request.get("connection_name"),
                backup_path=request.get("backup_path"),
            )

        elif operation == "history":
            return {
                "success": True,
                "history": self.db_agent.get_query_history(
                    limit=request.get("limit"),
                    success_only=request.get("success_only", False),
                ),
            }

        elif operation == "stats":
            return self.db_agent.get_stats()

        elif operation == "clear_cache":
            self.db_agent.clear_cache()
            return {"success": True, "message": "Cache cleared"}

        elif operation == "clear_history":
            self.db_agent.clear_history()
            return {"success": True, "message": "History cleared"}

        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}

    def _get_default_port(self, db_type: DatabaseType) -> int:
        """Get default port for database type"""
        ports = {
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.MYSQL: 3306,
            DatabaseType.MONGODB: 27017,
            DatabaseType.REDIS: 6379,
            DatabaseType.ELASTICSEARCH: 9200,
        }
        return ports.get(db_type, 0)

    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": "DatabaseAgent",
            "type": self.agent_type,
            "capabilities": self.capabilities,
            "stats": self.db_agent.get_stats(),
            "supported_databases": [db.value for db in DatabaseType],
            "active_connections": len(
                [c for c in self.db_agent.active_connections.values() if c]
            ),
        }

    async def close(self):
        """Clean up resources"""
        await self.db_agent.close_all_connections()


# Example usage and testing
async def test_database_agent():
    """Test the database agent functionality"""

    # Initialize agent
    agent = DatabaseAgent()

    print("=== Database Agent Test ===\n")

    # Test SQLite connection (in-memory)
    print("1. Connecting to SQLite...")
    sqlite_config = DatabaseConnection(
        db_type=DatabaseType.SQLITE,
        host="localhost",
        port=0,
        database=":memory:",
        username=None,
        password=None,
    )

    result = await agent.connect("test_sqlite", sqlite_config)
    print(f"   Connection: {result['success']}")

    # Test creating a table
    print("\n2. Creating table...")
    create_query = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    result = await agent.execute_query(
        "test_sqlite", create_query, query_type=QueryType.RAW
    )
    print(f"   Table created: {result['success']}")

    # Test inserting data
    print("\n3. Inserting data...")
    insert_query = "INSERT INTO users (name, email, age) VALUES (?, ?, ?)"
    params_list = [
        ("Alice", "alice@example.com", 30),
        ("Bob", "bob@example.com", 25),
        ("Charlie", "charlie@example.com", 35),
    ]

    for params in params_list:
        result = await agent.execute_query(
            "test_sqlite", insert_query, params=params, query_type=QueryType.INSERT
        )
    print(f"   Inserted {len(params_list)} records")

    # Test selecting data
    print("\n4. Selecting data...")
    select_query = "SELECT * FROM users WHERE age > ?"
    result = await agent.execute_query(
        "test_sqlite", select_query, params=(25,), query_type=QueryType.SELECT
    )

    if result["success"]:
        print(f"   Found {result['row_count']} users:")
        for row in result["data"]:
            print(f"     - {row['name']} ({row['age']}) - {row['email']}")

    # Test getting tables
    print("\n5. Getting tables...")
    tables = await agent.get_tables("test_sqlite")
    if tables["success"]:
        print(f"   Tables: {', '.join(tables['tables'])}")

    # Test getting schema
    print("\n6. Getting schema...")
    schema = await agent.get_schema("test_sqlite", "users")
    if schema["success"]:
        print("   Columns in 'users':")
        for col in schema["columns"]:
            print(f"     - {col['column_name']} ({col['data_type']})")

    # Test update operation
    print("\n7. Updating data...")
    update_query = "UPDATE users SET age = ? WHERE name = ?"
    result = await agent.execute_query(
        "test_sqlite", update_query, params=(31, "Alice"), query_type=QueryType.UPDATE
    )
    print(f"   Update: {result['success']}, affected: {result['row_count']}")

    # Test batch operations
    print("\n8. Batch operations...")
    batch_queries = [
        "SELECT COUNT(*) as count FROM users",
        "SELECT AVG(age) as avg_age FROM users",
        "SELECT * FROM users ORDER BY age",
    ]

    for query in batch_queries:
        result = await agent.execute_query("test_sqlite", query)
        if result["success"]:
            print(f"   {query[:30]}: {result['data']}")

    # Test backup
    print("\n9. Creating backup...")
    backup = await agent.backup_database("test_sqlite", "test_backup.json")
    if backup["success"]:
        print(f"   Backup saved: {backup['backup_path']}")
        print(f"   Size: {backup['size']} bytes")

    # Get statistics
    print("\n10. Agent Statistics...")
    stats = agent.get_stats()
    print(f"    Total queries: {stats['total_queries']}")
    print(f"    Successful: {stats['successful_queries']}")
    print(f"    Success rate: {stats['success_rate']:.1f}%")
    print(f"    Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"    Avg execution time: {stats['average_execution_time']:.3f}s")

    # Disconnect
    print("\n11. Disconnecting...")
    result = await agent.disconnect("test_sqlite")
    print(f"    Disconnected: {result['success']}")

    print("\n=== Test Complete ===")


# Run test
if __name__ == "__main__":
    asyncio.run(test_database_agent())
