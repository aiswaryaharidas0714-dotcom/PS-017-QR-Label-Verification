import os
import mysql.connector

def connect_database():
    try:
        try:
            import streamlit as st
            if "mysql" in st.secrets:
                config = st.secrets["mysql"]
                return mysql.connector.connect(
                    host=config["host"],
                    user=config["user"],
                    password=config["password"],
                    database=config["database"],
                    port=int(config.get("port", 3306))
                )
        except Exception:
            pass

        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME", "ps017_inspection"),
            port=int(os.getenv("DB_PORT", "3306"))
        )

    except Exception as e:
        print("Database connection failed:", e)
        return None