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
cursor.execute('''
CREATE TABLE IF NOT EXISTS novels (
    id SERIAL PRIMARY KEY,
    novel_title TEXT NOT NULL UNIQUE,
    novel_image TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS chapters (
    id SERIAL PRIMARY KEY,
    novel_id INTEGER REFERENCES novels(id) ON DELETE CASCADE,
    chapter_number INT NOT NULL,
    chapter_title TEXT,
    chapter_url TEXT,
    chapter_content TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    novel_id INTEGER REFERENCES novels(id) ON DELETE CASCADE,
    chunk_number INT NOT NULL,
    chunk_content TEXT
);
''')

conn.commit()

cursor.close()
conn.close()
