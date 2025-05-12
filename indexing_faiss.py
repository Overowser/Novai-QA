import psycopg2
import os
from dotenv import load_dotenv
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_DB = os.getenv("PG_DB")



# could be put in utils.py
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


def indexing_novel_chunks(novel_title, embedding_model="mixedbread-ai/mxbai-embed-large-v1"):

    # mixedbread-ai/mxbai-embed-large-v1 is hardcoded could be passed as an argument from .env file
    model = SentenceTransformer(embedding_model, device='cuda')


    # dimension could be hardcoded as 1024 for this model.
    # can use IndexFlatL2 (for euclidian distance) or IndexFlatIP for cosine similarity with normalized vectors
    dimension = len(model.encode("hello world"))

    # create the index with ids to map the chunks to their ids
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))


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

    ids = [id for id, _ in chunks]
    documents = [chunk for _, chunk in chunks]

    print("Encoding the chunks...")
    # create normalized embeddings
    embeddings = model.encode(documents, convert_to_numpy=True,batch_size=32,show_progress_bar=True)
    print("Done encoding.")
    print("Normalizing the embeddings...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print("Done normalizing.")

    index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

    print("Writing the index to file...")
    faiss.write_index(index, f"{novel_title}_index.index")
    print("Done writing index.")

    cursor.close()
    conn.close()

    return


# test the function

if __name__ == "__main__":
    # test the function
    novel_title = "Infinite Mana In The Apocalypse"
    embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
    indexing_novel_chunks(novel_title)