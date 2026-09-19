from pathlib import Path
import sqlite3
from flask import current_app, g

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL, business_name TEXT NOT NULL,
 email TEXT NOT NULL UNIQUE, phone TEXT NOT NULL, address TEXT NOT NULL, city TEXT NOT NULL,
 state TEXT NOT NULL, region TEXT NOT NULL, password_hash TEXT NOT NULL,
 security_question TEXT, security_answer_hash TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recovery_tokens (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, token_hash TEXT NOT NULL UNIQUE,
 expires_at TEXT NOT NULL, used_at TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS recovery_attempts (
 attempt_key TEXT PRIMARY KEY, failure_count INTEGER NOT NULL DEFAULT 0,
 window_started_at TEXT NOT NULL, blocked_until TEXT
);
CREATE TABLE IF NOT EXISTS products (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, name TEXT NOT NULL, category TEXT NOT NULL,
 unit TEXT NOT NULL, region TEXT NOT NULL, current_stock REAL NOT NULL DEFAULT 0, safety_stock REAL NOT NULL DEFAULT 0,
 lead_time_days INTEGER NOT NULL DEFAULT 7, description TEXT, is_active INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS sales (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, product_id INTEGER NOT NULL, sale_date TEXT NOT NULL,
 quantity REAL NOT NULL, unit TEXT NOT NULL, selling_price REAL, buyer TEXT, notes TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS inventory_transactions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, product_id INTEGER NOT NULL, transaction_type TEXT NOT NULL,
 quantity REAL NOT NULL, reference_id INTEGER, notes TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS forecasts (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, product_id INTEGER NOT NULL, forecast_date TEXT NOT NULL,
 forecast_value REAL NOT NULL, model_name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS alerts (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, product_id INTEGER NOT NULL, alert_type TEXT NOT NULL,
 message TEXT NOT NULL, severity TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'ACTIVE',
 previous_stock REAL NOT NULL DEFAULT 0, resolved_at TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS reorder_requests (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, product_id INTEGER NOT NULL, product_name TEXT NOT NULL,
 recommended_quantity REAL NOT NULL, confirmed_quantity REAL NOT NULL, current_stock REAL NOT NULL, safety_stock REAL NOT NULL,
 reorder_point REAL NOT NULL, target_stock REAL NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS user_settings (
 id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL UNIQUE, default_forecast_horizon INTEGER NOT NULL DEFAULT 7,
 default_region TEXT, default_unit TEXT, alert_enabled INTEGER NOT NULL DEFAULT 1,
 FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_products_user ON products(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_sales_user_date ON sales(user_id, sale_date);
CREATE INDEX IF NOT EXISTS idx_alerts_user_read ON alerts(user_id, is_read);
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_reorder_request ON reorder_requests(user_id, product_id, status) WHERE status='PENDING';
"""


def db_path():
    return Path(current_app.instance_path) / "wholesale.db"


def get_db():
    if "db" not in g:
        connection = sqlite3.connect(db_path())
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


def close_db(_error=None):
    connection = g.pop("db", None)
    if connection:
        connection.close()


def init_db():
    Path(current_app.instance_path).mkdir(parents=True, exist_ok=True)
    connection = get_db()
    connection.executescript(SCHEMA)
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(sales)").fetchall()}
    if "entry_method" not in columns:
        connection.execute("ALTER TABLE sales ADD COLUMN entry_method TEXT NOT NULL DEFAULT 'Manual'")
    user_columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()}
    for column_name in ("security_question", "security_answer_hash"):
        if column_name not in user_columns:
            connection.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT")
    alert_columns = {row["name"] for row in connection.execute("PRAGMA table_info(alerts)").fetchall()}
    for column_name, definition in {
        "status": "TEXT NOT NULL DEFAULT 'ACTIVE'",
        "previous_stock": "REAL NOT NULL DEFAULT 0",
        "resolved_at": "TEXT",
    }.items():
        if column_name not in alert_columns:
            connection.execute(f"ALTER TABLE alerts ADD COLUMN {column_name} {definition}")
    reorder_columns = {row["name"] for row in connection.execute("PRAGMA table_info(reorder_requests)").fetchall()}
    for column_name, definition in {
        "product_name": "TEXT NOT NULL DEFAULT ''",
        "recommended_quantity": "REAL NOT NULL DEFAULT 0",
        "confirmed_quantity": "REAL NOT NULL DEFAULT 0",
        "current_stock": "REAL NOT NULL DEFAULT 0",
        "safety_stock": "REAL NOT NULL DEFAULT 0",
        "reorder_point": "REAL NOT NULL DEFAULT 0",
        "target_stock": "REAL NOT NULL DEFAULT 0",
        "status": "TEXT NOT NULL DEFAULT 'PENDING'",
    }.items():
        if column_name not in reorder_columns:
            connection.execute(f"ALTER TABLE reorder_requests ADD COLUMN {column_name} {definition}")
    connection.execute("UPDATE alerts SET is_read=1 WHERE is_read=0 AND id NOT IN (SELECT MAX(id) FROM alerts WHERE is_read=0 GROUP BY user_id, product_id, alert_type)")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_active_alert ON alerts(user_id, product_id, alert_type) WHERE is_read = 0")
    connection.commit()


def query(sql, parameters=(), one=False):
    cursor = get_db().execute(sql, parameters)
    rows = cursor.fetchall()
    cursor.close()
    return (rows[0] if rows else None) if one else rows


def execute(sql, parameters=()):
    connection = get_db()
    cursor = connection.execute(sql, parameters)
    connection.commit()
    return cursor
