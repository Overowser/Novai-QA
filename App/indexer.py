import chromadb
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from time import time
from utils import preprocess, get_novel_id, get_db_connection
from logger_config import setup_logger

logger = setup_logger("indexer")


def collection_name_from_title(novel_title):
    # Remove spaces and special characters from the novel title
    collection_name = "".join(e.lower() for e in novel_title if e.isalnum())
    return collection_name


def indexing_novel_chunks_chroma(
    novel_title, embedding_model="mixedbread-ai/mxbai-embed-large-v1"
):

    # mixedbread-ai/mxbai-embed-large-v1 is hardcoded could be passed as an argument from .env file
    model = SentenceTransformer(embedding_model, device="cuda")

    # create a new chroma collection
    collection_name = collection_name_from_title(novel_title)

    chroma_client = chromadb.PersistentClient()

    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids_in_chroma = collection.get()["ids"]
    logger.info(f"Number of IDs already in Chroma: {len(ids_in_chroma)}")

    # connect to the database and fetch the chunks
    conn = get_db_connection()
    cursor = conn.cursor()

    novel_id = get_novel_id(novel_title, cursor)
    logger.info(f"Novel ID for {novel_title}: {novel_id}")

    # Fetch the chunks from the database
    cursor.execute(
        "SELECT id, chapter_id, chunk_content FROM chunks WHERE novel_id = %s",
        (novel_id,),
    )

    chunks = cursor.fetchall()
    logger.info(f"Number of chunks for novel {novel_title}: {len(chunks)}")

    chunks = [chunk for chunk in chunks if str(chunk[0]) not in ids_in_chroma]

    if len(chunks) == 0:
        logger.info("No new chunks to add to the collection.")
        return

    logger.info(f"Number of chunks to be added to Chroma: {len(chunks)}")

    ids, chapter_ids, documents = map(list, zip(*chunks))
    ids = [str(id) for id in ids]

    # create normalized embeddings
    chroma_batch_size = 1024

    ids_batches = [
        ids[i : i + chroma_batch_size] for i in range(0, len(ids), chroma_batch_size)
    ]
    chapter_ids_batches = [
        chapter_ids[i : i + chroma_batch_size]
        for i in range(0, len(chapter_ids), chroma_batch_size)
    ]
    documents_batches = [
        documents[i : i + chroma_batch_size]
        for i in range(0, len(documents), chroma_batch_size)
    ]

    for ids_batch, chapter_ids_batch, documents_batch in tqdm(
        zip(ids_batches, chapter_ids_batches, documents_batches),
        total=len(ids_batches),
        desc="Encoding chunks",
    ):
        embeddings = model.encode(documents_batch, convert_to_numpy=True, batch_size=32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        collection.add(
            embeddings=embeddings.tolist(),
            ids=ids_batch,
            metadatas=[{"chapter_id": chapter_id} for chapter_id in chapter_ids_batch],
        )

    logger.info("Done adding chunks to the collection.")

    cursor.close()
    conn.close()

    return


def indexing_novel_chunks_bm25(novel_title):

    # connect to the database and fetch the chunks
    conn = get_db_connection()
    cursor = conn.cursor()

    novel_id = get_novel_id(novel_title, cursor)
    logger.info(f"Novel ID for {novel_title}: {novel_id}")

    # Fetch the chunks from the database
    cursor.execute(
        "SELECT id, chunk_content FROM chunks WHERE novel_id = %s", (novel_id,)
    )

    chunks = cursor.fetchall()
    logger.info(f"Number of chunks for novel {novel_title}: {len(chunks)}")

    if len(chunks) == 0:
        logger.warning(f"No chunks found for novel {novel_title}.")
        return

    ids, documents = map(list, zip(*chunks))

    logger.info("Tokenizing documents...")
    tokenized_docs = [preprocess(doc) for doc in documents]
    logger.info("Done tokenizing documents.")

    # store the tokenized documents in the database
    # timing this
    time_start = time()
    for doc_id, tokens in zip(ids, tokenized_docs):
        cursor.execute(
            "UPDATE chunks SET preprocessed_chunk_content = %s WHERE id = %s",
            (tokens, doc_id),
        )
    time_end = time()
    logger.info(
        f"Time taken to store tokenized documents: {time_end - time_start} seconds"
    )
    conn.commit()
    logger.info("Done storing tokenized documents.")

    cursor.close()
    conn.close()

    return

