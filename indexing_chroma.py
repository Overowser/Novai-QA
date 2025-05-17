import psycopg2
import os
from dotenv import load_dotenv
import chromadb
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

load_dotenv()

PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_USER = os.getenv("PG_USER")
PG_DB = os.getenv("PG_DB")


def get_ids_from_chroma(collection):
    """
    Fetch the chunk IDs already indexed in the vector database.
    """
    results = collection.get()

    return results["ids"]


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


def indexing_novel_chunks(novel_title, embedding_model="mixedbread-ai/mxbai-embed-large-v1"):

    # mixedbread-ai/mxbai-embed-large-v1 is hardcoded could be passed as an argument from .env file
    model = SentenceTransformer(embedding_model, device='cuda')

    # create a new chroma collection
    collection_name = collection_name_from_title(novel_title)

    chroma_client = chromadb.PersistentClient()

    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids_in_chroma = get_ids_from_chroma(collection)
    print(f"Number of IDs already in Chroma: {len(ids_in_chroma)}")


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
    cursor.execute("SELECT id, chapter_id, chunk_content FROM chunks WHERE novel_id = %s", (novel_id,))

    chunks = cursor.fetchall()
    print(f"Number of chunks for novel {novel_title}: {len(chunks)}")

    chunks = [chunk for chunk in chunks if str(chunk[0]) not in ids_in_chroma]

    if len(chunks) == 0:
        print("No new chunks to add to the collection.")
        return

    print(f"Number of chunks to be added to Chroma: {len(chunks)}")
    
    ids, chapter_ids, documents = map(list, zip(*chunks))
    ids = [str(id) for id in ids]

    # ids = [id for id, _, _  in chunks]
    # chapter_ids = [chapter_id for _, chapter_id, _ in chunks]
    # documents = [chunk for _, chunk in chunks]
    
    # for testing:
    # ids = [str(id) for id in ids[:10]]
    # chapter_ids = chapter_ids[:10]
    # documents = documents[:10]

    # create normalized embeddings
    chroma_batch_size = 1024

    ids_batches = [ids[i:i + chroma_batch_size] for i in range(0, len(ids), chroma_batch_size)]
    chapter_ids_batches = [chapter_ids[i:i + chroma_batch_size] for i in range(0, len(chapter_ids), chroma_batch_size)]
    documents_batches = [documents[i:i + chroma_batch_size] for i in range(0, len(documents), chroma_batch_size)]

    for ids_batch, chapter_ids_batch, documents_batch in tqdm(zip(ids_batches, chapter_ids_batches, documents_batches), total=len(ids_batches), desc="Encoding chunks"):
        embeddings = model.encode(documents_batch, convert_to_numpy=True,batch_size=32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        collection.add(
        embeddings=embeddings.tolist(),
        ids=ids_batch,
        metadatas=[{"chapter_id": chapter_id} for chapter_id in chapter_ids_batch]
        )

    print("Done adding chunks to the collection.")

    cursor.close()
    conn.close()

    return


# test the function

if __name__ == "__main__":
    # test the function
    novel_title = "Infinite Mana In The Apocalypse"
    embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
    indexing_novel_chunks(novel_title)


# TODO:
# - add a check to see if an id already exists in chroma before embedding it (done)