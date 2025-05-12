import psycopg2
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

import faiss
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

    cursor.execute("SELECT chunk_content FROM chunks WHERE id in %s", (tuple(chunk_id_list),))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    if len(results) == len(chunk_id_list):
        chunks_content = [row[0] for row in results]
        return chunks_content
    else:
        print(f"Some or all chunk IDs not found in the database.")
        return None


def retrieve_context(query, index, k=5, embedding_model="mixedbread-ai/mxbai-embed-large-v1"):
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

    # Convert the query vector to a 2D array
    query_vector = query_vector.reshape(1, -1)

    # Search for the top k nearest neighbors
    results = index.search(query_vector, k)
    
    similarity_scores, indices = index.search(query_vector, k)

    indices = [id.item() for id in indices[0]]

    chunks = get_chunk_from_id(indices)

    return similarity_scores[0], indices, chunks


# example usage
if __name__ == "__main__":

    index = faiss.read_index("faiss_database.index")

    print(retrieve_context("Who is the main character of the story?", index, k=5))



# TODO:
# 1. Make the search function able to make hybrid searches
# 2. Make the search function able to make metadata filtering
# 3. Async when fetching the chunks
