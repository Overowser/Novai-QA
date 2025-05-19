import psycopg2
import os
from dotenv import load_dotenv

from time import time

from utils import preprocess


import re
import nltk
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_DB = os.getenv("PG_DB")

# Download required NLTK resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize once
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# def preprocess(text, do_lemmatize=True):
#     # Lowercase
#     text = text.lower()
#     # Remove non-alphabetical characters (optional)
#     text = re.sub(r'[^a-z\s]', '', text)
#     # Tokenize
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     tokens = [w for w in tokens if w not in stop_words]
#     # Lemmatize or stem
#     if do_lemmatize:
#         tokens = [lemmatizer.lemmatize(w) for w in tokens]
#     return tokens



# could be put in utils.py (can be used in retrieval)
def get_novel_id(novel_title, cursor, conn):
    """
    Fetch the novel ID from the database using the novel title.
    """
    cursor.execute("SELECT id FROM novels WHERE novel_title = %s", (novel_title,))

    result = cursor.fetchone()
    if result:
        novel_id = result[0]
        return novel_id
    else:
        print(f"Novel '{novel_title}' not found in the database.")
        return None

# could be put in utils.py (can be used in retrieval)
def collection_name_from_title(novel_title):
    """
    Generate a collection name based on the novel title.
    """
    # Remove spaces and special characters from the novel title
    collection_name = "".join(e.lower() for e in novel_title if e.isalnum())
    return collection_name


def indexing_novel_chunks(novel_title):

    # connect to the database and fetch the chunks
    conn = psycopg2.connect(
        host=PG_HOST,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
        port="5432"
    )
    cursor = conn.cursor()

    novel_id = get_novel_id(novel_title, cursor, conn)
    print(f"Novel ID for {novel_title}: {novel_id}")

    # Fetch the chunks from the database
    cursor.execute("SELECT id, chunk_content FROM chunks WHERE novel_id = %s", (novel_id,))

    chunks = cursor.fetchall()
    print(f"Number of chunks for novel {novel_title}: {len(chunks)}")

    if len(chunks) == 0:
        print(f"No chunks found for novel {novel_title}.")
        return

    
    ids, documents = map(list, zip(*chunks))


    print("Tokenizing documents...")
    tokenized_docs = [preprocess(doc) for doc in documents]
    print("Done tokenizing documents.")
    
    # store the tokenized documents in the database
    # timing this
    time_start = time()
    for doc_id, tokens in zip(ids, tokenized_docs):
        cursor.execute(
            "UPDATE chunks SET preprocessed_chunk_content = %s WHERE id = %s",
            (tokens, doc_id)
        )
    time_end = time()
    print(f"Time taken to store tokenized documents: {time_end - time_start} seconds")
    conn.commit()
    print("Done storing tokenized documents.")

    cursor.close()
    conn.close()

    return


# test the function

if __name__ == "__main__":
    # test the function
    novel_title = "Infinite Mana In The Apocalypse"
    indexing_novel_chunks(novel_title)


# TODO:
# - store the preprocessed chunks in the database (done)
# - add a check to see if the chunks are already preprocessed
# - remove the pickle dump (done)
# - at retrieval time, retrieve the preprocessed chunks from the database and use bm25 to rank them 
# - add multiprossing to the preprocessing function
# - create a temporary table to store the preprocessed chunks to fasten the storing process