import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from logger_config import setup_logger
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger("utils")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_db_connection():
    PG_PASSWORD = os.getenv("PG_PASSWORD")
    PG_HOST = os.getenv("PG_HOST")
    PG_USER = os.getenv("PG_USER")
    PG_DB = os.getenv("PG_DB")
    return psycopg2.connect(
        host=PG_HOST, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD, port="5432"
    )

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_with_pos(tokens):
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

def preprocess(text, do_lemmatize=True):
    logger.debug("Starting preprocessing of text.")
    # Lowercase
    text = text.lower()
    # Remove non-alphabetical characters (optional)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_words]
    # Lemmatize or stem
    if do_lemmatize:
        tokens = lemmatize_with_pos(tokens)
    logger.debug("Finished preprocessing of text.")
    return tokens

def get_novel_id(novel_title, cursor):
    """
    Fetch the novel ID from the database using the novel title.
    """
    logger.debug("Fetching novel ID for title: %s", novel_title)
    cursor.execute("SELECT id FROM novels WHERE novel_title = %s", (novel_title,))
    result = cursor.fetchone()
    if result:
        novel_id = result[0]
        logger.info("Found novel ID: %s for title: %s", novel_id, novel_title)
        return novel_id
    else:
        logger.warning("Novel '%s' not found in the database.", novel_title)
        return None
