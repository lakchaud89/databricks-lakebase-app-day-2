"""
One-time setup script: creates the Databricks secret scope for this
homework assignment and stores its Lakebase connection URL. Run this once
locally (with the Databricks CLI configured) or from a notebook - never
commit the resulting secret value anywhere.

Uses scope "database_weather" -- a dedicated scope for this assignment,
kept separate from the class reference app's "database" scope so the two
apps' Lakebase connections can never be confused with each other.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# w.secrets.create_scope(scope="database_weather")
w.secrets.put_secret(
    scope="database_weather",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: "),
)

w.secrets.put_acl(
    scope="database_weather",
    principal="users",
    permission=workspace.AclPermission.READ,
)
