import psycopg2
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

import chromadb
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

# could be put in utils.py (can be used in retrieval)
def collection_name_from_title(novel_title):
    """
    Generate a collection name based on the novel title.
    """
    # Remove spaces and special characters from the novel title
    collection_name = "".join(e.lower() for e in novel_title if e.isalnum())
    return collection_name


def retrieve_context(query, novel_name, spoiler_threshold=None, k=5, embedding_model="mixedbread-ai/mxbai-embed-large-v1"):
    """
    Retrieve the top k most similar chunks from the index based on the query.
    """

    query_prompt = 'Represent this sentence for searching relevant passages: '

    # load the model
    model = SentenceTransformer(embedding_model, device='cuda')

    # Encode the query
    query_vector = model.encode(query_prompt + query)

    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    collection = chromadb.PersistentClient().get_or_create_collection(name=collection_name_from_title(novel_name))

    # Search for the top k nearest neighbors
    if spoiler_threshold:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where={
                "chapter_id":{
                    "$lte": spoiler_threshold
                }
            }
        )
    else:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k
        )

    # print(query_vector.tolist())

    print(results)
    

    ids = results['ids'][0]
    print(ids)

    chunks = get_chunk_from_id(ids)

    return chunks


# example usage
if __name__ == "__main__":

    print(retrieve_context("Who is the main character of the story?","Infinite Mana In The Apocalypse", spoiler_threshold=10, k=3))


# TODO:
# 1. Make the search function able to make hybrid searches
# 2. Make the search function able to make metadata filtering
# 3. Async when fetching the chunks
