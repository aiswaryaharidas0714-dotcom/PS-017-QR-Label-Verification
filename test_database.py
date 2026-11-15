from database import connect_database

connection = connect_database()

if connection.is_connected():
    print("MySQL Database Connected Successfully!")

connection.close()