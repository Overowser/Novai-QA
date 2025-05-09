import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_DB = os.getenv("PG_DB")

conn = psycopg2.connect(
    host=PG_HOST,
    dbname=PG_DB,
    user=PG_USER,
    password=PG_PASSWORD,
    port="5432"
)

cursor = conn.cursor()




cursor.close()
conn.close()