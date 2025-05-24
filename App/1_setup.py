from utils import get_db_connection
from logger_config import setup_logger
import nltk

logger = setup_logger("database")


conn = get_db_connection()

cursor = conn.cursor()
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS novels (
    id SERIAL PRIMARY KEY,
    novel_title TEXT NOT NULL UNIQUE,
    novel_image TEXT
);
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS chapters (
    id SERIAL PRIMARY KEY,
    novel_id INTEGER REFERENCES novels(id) ON DELETE CASCADE,
    chapter_number INT NOT NULL,
    chapter_title TEXT,
    chapter_url TEXT,
    chapter_content TEXT
);
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    chapter_id INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    novel_id INTEGER REFERENCES novels(id) ON DELETE CASCADE,
    chunk_number INT NOT NULL,
    chunk_content TEXT,
    preprocessed_chunk_content TEXT[]
);
"""
)

conn.commit()

cursor.close()
conn.close()

logger.info("Database setup completed.")

# Download the necessary NLTK resources
logger.info("Downloading NLTK resources...")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
logger.info("NLTK resources downloaded.")
