from abc import ABC, abstractmethod
from typing import Any


class ConnectorPort(ABC):
    """Base class for database connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Establishes the connection."""

    @abstractmethod
    def execute_query(self, sql: str) -> Any:
        """Executes a SQL query and returns raw data."""


class SqlServerAdapter(ConnectorPort):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.connection = None

    def connect(self) -> None:
        # Example placeholder; in real implementation use pyodbc or similar
        self.connection = f"Connected to SQL Server using {self.conn_str}"

    def execute_query(self, sql: str) -> Any:
        if not self.connection:
            raise ConnectionError("Not connected to SQL Server")
        return f"Results of '{sql}'"


class PostgresAdapter(ConnectorPort):
    def __init__(self, conn_str: str) -> None:
        self.conn_str = conn_str
        self.connection = None

    def connect(self) -> None:
        self.connection = f"Connected to Postgres using {self.conn_str}"

    def execute_query(self, sql: str) -> Any:
        if not self.connection:
            raise ConnectionError("Not connected to Postgres")
        return f"Results of '{sql}'"
