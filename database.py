
import mysql.connector
import streamlit as st

def connect_database():
    try:
        config = st.secrets["mysql"]

        return mysql.connector.connect(
            host=config["host"],
            port=int(config["port"]),
            user=config["user"],
            password=config["password"],
            database=config["database"],
            ssl_disabled=False,
            connection_timeout=15
        )

    except Exception as error:
        st.error("Database error: " + str(error))
        return None
