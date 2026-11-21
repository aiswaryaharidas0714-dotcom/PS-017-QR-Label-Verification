import os
from pathlib import Path
from datetime import datetime, date
import mysql.connector
from mysql.connector import Error as MySQLError

# Try importing streamlit safely
try:
    import streamlit as st
except ImportError:
    st = None

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def get_db_credentials():
    """
    Safely retrieves MySQL credentials.
    Priority:
    1. Streamlit secrets (st.secrets["mysql"])
    2. Local .streamlit/secrets.toml using tomllib
    3. Environment variables (MYSQL_HOST, MYSQL_PORT, etc.)
    """
    # 1. Try Streamlit secrets
    if st is not None:
        try:
            if hasattr(st, "secrets") and "mysql" in st.secrets:
                cfg = st.secrets["mysql"]
                return {
                    "host": cfg["host"],
                    "port": int(cfg.get("port", 3306)),
                    "user": cfg["user"],
                    "password": cfg["password"],
                    "database": cfg["database"],
                }
        except Exception:
            pass

    # 2. Try local .streamlit/secrets.toml
    possible_paths = [
        Path(".streamlit/secrets.toml"),
        Path(__file__).parent / ".streamlit" / "secrets.toml",
    ]
    if tomllib is not None:
        for p in possible_paths:
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        data = tomllib.load(f)
                        if "mysql" in data:
                            cfg = data["mysql"]
                            return {
                                "host": cfg["host"],
                                "port": int(cfg.get("port", 3306)),
                                "user": cfg["user"],
                                "password": cfg["password"],
                                "database": cfg["database"],
                            }
                except Exception:
                    continue

    # 3. Try Environment Variables
    if os.getenv("MYSQL_HOST") and os.getenv("MYSQL_USER"):
        return {
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "defaultdb"),
        }

    return None


def connect_database():
    """
    Establishes connection to MySQL database using credentials.
    Returns connection object or None on failure.
    """
    creds = get_db_credentials()
    if not creds:
        if st is not None:
            try:
                st.error("Database configuration not found in Streamlit secrets or secrets.toml.")
            except Exception:
                pass
        return None

    try:
        connection = mysql.connector.connect(
            host=creds["host"],
            port=creds["port"],
            user=creds["user"],
            password=creds["password"],
            database=creds["database"],
            ssl_disabled=False,
            connection_timeout=10,
            autocommit=True
        )
        return connection
    except Exception as error:
        if st is not None:
            try:
                st.error(f"Database connection error: {error}")
            except Exception:
                pass
        return None


def check_connection():
    """Checks if the MySQL database is reachable and returns (is_connected, latency_ms or error_msg)."""
    start_time = datetime.now()
    conn = connect_database()
    if conn is None:
        return False, "Could not establish database connection."
    try:
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return True, f"{latency:.1f} ms"
        return False, "Connection failed is_connected check."
    except Exception as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def get_product(product_id):
    """
    Queries product record by product_id using parameterized SQL.
    Returns dictionary with product details or None.
    """
    conn = connect_database()
    if conn is None:
        return None
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT product_id, product_name, batch_number,
                   manufacturing_date, expiry_date
            FROM products
            WHERE product_id = %s
            """,
            (product_id,)
        )
        return cursor.fetchone()
    except Exception as error:
        if st is not None:
            try:
                st.error(f"Failed to query product: {error}")
            except Exception:
                pass
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_all_products():
    """
    Fetches all registered products from MySQL.
    """
    conn = connect_database()
    if conn is None:
        return []
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT product_id, product_name, batch_number,
                   manufacturing_date, expiry_date
            FROM products
            ORDER BY product_id ASC
            """
        )
        return cursor.fetchall()
    except Exception as error:
        if st is not None:
            try:
                st.error(f"Failed to query products: {error}")
            except Exception:
                pass
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def add_product(product_id, product_name, batch_number, mfg_date, exp_date):
    """
    Inserts a new product into MySQL using parameterized SQL.
    Returns (success: bool, message: str).
    """
    if not product_id or not product_name or not batch_number:
        return False, "Product ID, Name, and Batch Number are required."

    if isinstance(mfg_date, str):
        mfg_date = datetime.strptime(mfg_date, "%Y-%m-%d").date()
    if isinstance(exp_date, str):
        exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()

    if mfg_date > exp_date:
        return False, "Manufacturing date cannot be after Expiry date."

    conn = connect_database()
    if conn is None:
        return False, "Database connection unavailable."

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO products (product_id, product_name, batch_number, manufacturing_date, expiry_date)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (product_id.strip(), product_name.strip(), batch_number.strip(), mfg_date, exp_date)
        )
        conn.commit()
        return True, f"Product '{product_id}' registered successfully."
    except MySQLError as err:
        if err.errno == 1062:  # Duplicate entry
            return False, f"Product ID '{product_id}' already exists."
        return False, f"Database error: {err}"
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_product(product_id, product_name, batch_number, mfg_date, exp_date):
    """
    Updates an existing product in MySQL using parameterized SQL.
    Returns (success: bool, message: str).
    """
    if not product_id or not product_name or not batch_number:
        return False, "Product ID, Name, and Batch Number are required."

    if isinstance(mfg_date, str):
        mfg_date = datetime.strptime(mfg_date, "%Y-%m-%d").date()
    if isinstance(exp_date, str):
        exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()

    if mfg_date > exp_date:
        return False, "Manufacturing date cannot be after Expiry date."

    conn = connect_database()
    if conn is None:
        return False, "Database connection unavailable."

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE products
            SET product_name = %s, batch_number = %s, manufacturing_date = %s, expiry_date = %s
            WHERE product_id = %s
            """,
            (product_name.strip(), batch_number.strip(), mfg_date, exp_date, product_id.strip())
        )
        conn.commit()
        if cursor.rowcount == 0:
            return False, f"Product ID '{product_id}' not found."
        return True, f"Product '{product_id}' updated successfully."
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_inspection(product_id, qr_status, text_status, appearance_status, final_result, remarks):
    """
    Inserts an inspection record into inspection_history using parameterized SQL.
    Returns (success: bool, inspection_id or error message).
    """
    conn = connect_database()
    if conn is None:
        return False, "Database connection failed."

    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO inspection_history
            (product_id, qr_status, text_status, appearance_status, final_result, remarks)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(product_id)[:20] if product_id else "UNKNOWN",
                str(qr_status)[:20],
                str(text_status)[:20],
                str(appearance_status)[:20],
                str(final_result)[:20],
                str(remarks)[:255] if remarks else ""
            )
        )
        conn.commit()
        new_id = cursor.lastrowid
        return True, new_id
    except Exception as e:
        return False, str(e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_inspection_history(limit=100, status_filter=None, search_query=None):
    """
    Queries inspection history with optional filtering and search.
    Never deletes or alters existing records.
    """
    conn = connect_database()
    if conn is None:
        return []

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT inspection_id, product_id, inspection_date,
                   qr_status, text_status, appearance_status, final_result, remarks
            FROM inspection_history
            WHERE 1=1
        """
        params = []

        if status_filter and status_filter.upper() != "ALL":
            query += " AND final_result = %s"
            params.append(status_filter.upper())

        if search_query:
            query += " AND (product_id LIKE %s OR remarks LIKE %s)"
            search_param = f"%{search_query}%"
            params.extend([search_param, search_param])

        query += " ORDER BY inspection_id DESC"

        if limit and isinstance(limit, int):
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, tuple(params))
        return cursor.fetchall()
    except Exception as error:
        if st is not None:
            try:
                st.error(f"Failed to fetch history: {error}")
            except Exception:
                pass
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_dashboard_metrics():
    """
    Calculates summary KPIs from MySQL inspection_history records:
    - total_inspections
    - pass_count, review_count, fail_count
    - pass_percentage
    - recent_inspections (last 10)
    All values come from real MySQL records, never hardcoded.
    """
    conn = connect_database()
    default_metrics = {
        "total": 0,
        "pass": 0,
        "review": 0,
        "fail": 0,
        "pass_rate": 0.0,
        "status_counts": {"PASS": 0, "REVIEW": 0, "FAIL": 0},
        "daily_trends": [],
        "recent_inspections": []
    }

    if conn is None:
        return default_metrics

    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)

        # 1. Status aggregates
        cursor.execute("""
            SELECT final_result, COUNT(*) AS count
            FROM inspection_history
            GROUP BY final_result
        """)
        counts_raw = cursor.fetchall()
        status_map = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        total = 0
        for row in counts_raw:
            st_name = (row["final_result"] or "").upper()
            cnt = row["count"]
            if st_name in status_map:
                status_map[st_name] = cnt
            total += cnt

        pass_rate = (status_map["PASS"] / total * 100) if total > 0 else 0.0

        # 2. Daily trend (past 7 days)
        cursor.execute("""
            SELECT DATE(inspection_date) as date,
                   SUM(CASE WHEN final_result = 'PASS' THEN 1 ELSE 0 END) as pass_count,
                   SUM(CASE WHEN final_result = 'REVIEW' THEN 1 ELSE 0 END) as review_count,
                   SUM(CASE WHEN final_result = 'FAIL' THEN 1 ELSE 0 END) as fail_count,
                   COUNT(*) as total
            FROM inspection_history
            GROUP BY DATE(inspection_date)
            ORDER BY date DESC
            LIMIT 7
        """)
        daily_trends = cursor.fetchall()
        daily_trends.reverse()

        # 3. Recent inspections
        cursor.execute("""
            SELECT inspection_id, product_id, inspection_date,
                   qr_status, text_status, appearance_status, final_result, remarks
            FROM inspection_history
            ORDER BY inspection_id DESC
            LIMIT 10
        """)
        recent_inspections = cursor.fetchall()

        return {
            "total": total,
            "pass": status_map["PASS"],
            "review": status_map["REVIEW"],
            "fail": status_map["FAIL"],
            "pass_rate": round(pass_rate, 1),
            "status_counts": status_map,
            "daily_trends": daily_trends,
            "recent_inspections": recent_inspections
        }
    except Exception as error:
        if st is not None:
            try:
                st.error(f"Failed to fetch metrics: {error}")
            except Exception:
                pass
        return default_metrics
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
