import sys
import types

# Provide dummy pyodbc module for tests
if 'pyodbc' not in sys.modules:
    sys.modules['pyodbc'] = types.ModuleType('pyodbc')
if 'psycopg2' not in sys.modules:
    sys.modules['psycopg2'] = types.ModuleType('psycopg2')
if 'databricks' not in sys.modules:
    databricks = types.ModuleType('databricks')
    databricks.sql = types.ModuleType('databricks.sql')
    sys.modules['databricks'] = databricks
    sys.modules['databricks.sql'] = databricks.sql
