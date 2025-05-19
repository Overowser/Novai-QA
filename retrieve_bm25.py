import psycopg2
from dotenv import load_dotenv
import os
from utils import *
from rank_bm25 import BM25Okapi
import numpy as np

load_dotenv()

def get_chunk_from_id(chunk_id_list):
    """
    Fetch the chunk content from the database using the chunk ID.
    """

    # connect to the database
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

    ids_tuple = tuple([int(id) for id in chunk_id_list])

    cursor.execute("SELECT chunk_content FROM chunks WHERE id in %s", (ids_tuple,))

    results = cursor.fetchall()
    cursor.close()
    conn.close()
    if len(results) == len(chunk_id_list):
        chunks_content = [row[0] for row in results]
        return chunks_content
    else:
        print(f"Some or all chunk IDs not found in the database.")
        return None


def retrieve_context(query, novel_name, spoiler_threshold=None, k=5):
    """
    Retrieve the top k most similar chunks from the index based on the query.
    """
    # connect to the database
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

    # Get the novel ID
    novel_id = get_novel_id(novel_name, cursor, conn)
    print(f"Novel ID for {novel_name}: {novel_id}")

    print(f"Retrieving chunks for {novel_name} ...")
    if spoiler_threshold:
        cursor.execute("SELECT chunks.id, chunks.preprocessed_chunk_content FROM chunks JOIN chapters ON chunks.chapter_id = chapters.id WHERE chapters.novel_id = %s AND chapters.chapter_number <= %s;",
         (novel_id,spoiler_threshold,))
    else:
        cursor.execute("SELECT id, preprocessed_chunk_content FROM chunks WHERE novel_id = %s", (novel_id,))

    chunks = cursor.fetchall()

    ids, tokenized_docs = map(list, zip(*chunks))
    print(f"Number of chunks for novel {novel_name}: {len(tokenized_docs)}")

    if len(tokenized_docs) == 0:
        print(f"No chunks found for novel {novel_name}.")
        return

    print("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    print("Done creating BM25 index.")

    # Tokenize the query
    query_tokens = preprocess(query)    

    # Search for the top k nearest neighbors
    print(f"Searching for the top {k} nearest neighbors...")
    # results = bm25.get_top_n(query_tokens, tokenized_docs, n=k)
    # print(query_vector.tolist())
    scores = bm25.get_scores(query_tokens)
    top_n_indices = np.argsort(scores)[::-1][:k]
    

    top_ids = [ids[i] for i in top_n_indices]

    chunks = get_chunk_from_id(top_ids)

    return chunks


# example usage
if __name__ == "__main__":

    print(retrieve_context("Who is the main character of the story?","Infinite Mana In The Apocalypse", spoiler_threshold=10, k=3))


# TODO:
# 1. Make the search function able to make hybrid searches
# 2. Make the search function able to make metadata filtering
# 3. Async when fetching the chunks
