import chromadb
import numpy as np
from utils import get_db_connection, preprocess, get_novel_id
from rank_bm25 import BM25Okapi
from logger_config import setup_logger

logger = setup_logger("retriever")


# ----------------------------------------
# RETRIEVAL - CHROMADB
# ----------------------------------------


def get_chunk_from_id(chunk_id_list):
    """
    Fetch the chunk content from the database using the chunk ID.
    """
    if not chunk_id_list:
        logger.warning("No chunk IDs provided.")
        return []

    conn = get_db_connection()
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
        logger.warning("Some or all chunk IDs not found in the database.")
        return []


def collection_name_from_title(novel_title):
    """
    Generate a collection name based on the novel title.
    """
    # Remove spaces and special characters from the novel title
    collection_name = "".join(e.lower() for e in novel_title if e.isalnum())
    return collection_name


def retrieve_context_chroma(query, novel_name, model, spoiler_threshold=None, k=5):
    """
    Retrieve the top k most similar chunks from the index based on the query.
    """

    query_prompt = "Represent this sentence for searching relevant passages: "

    # Encode the query
    query_vector = model.encode(query_prompt + query)

    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    collection = chromadb.PersistentClient().get_or_create_collection(
        name=collection_name_from_title(novel_name)
    )

    # Search for the top k nearest neighbors
    if spoiler_threshold:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where={"chapter_id": {"$lte": spoiler_threshold}},
        )
    else:
        results = collection.query(
            query_embeddings=[query_vector.tolist()], n_results=k
        )

    # logger.debug("Query vector: %s", query_vector.tolist())

    logger.debug("Results: %s", results)

    ids = results["ids"][0]
    logger.debug("Chunk IDs: %s", ids)

    chunks = get_chunk_from_id(ids)

    return chunks


# ----------------------------------------
# RETRIEVAL - BM25
# ----------------------------------------


def retrieve_context_bm25(query, novel_name, spoiler_threshold=None, k=5):
    """
    Retrieve the top k most similar chunks from the index based on the query.
    """
    # connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get the novel ID
    novel_id = get_novel_id(novel_name, cursor)
    logger.info("Novel ID for %s: %s", novel_name, novel_id)

    logger.info("Retrieving chunks for %s ...", novel_name)
    if spoiler_threshold:
        cursor.execute(
            "SELECT chunks.id, chunks.preprocessed_chunk_content FROM chunks JOIN chapters ON chunks.chapter_id = chapters.id WHERE chapters.novel_id = %s AND chapters.chapter_number <= %s;",
            (
                novel_id,
                spoiler_threshold,
            ),
        )
    else:
        cursor.execute(
            "SELECT id, preprocessed_chunk_content FROM chunks WHERE novel_id = %s",
            (novel_id,),
        )

    chunks = cursor.fetchall()

    ids, tokenized_docs = map(list, zip(*chunks))
    logger.info("Number of chunks for novel %s: %s", novel_name, len(tokenized_docs))

    if len(tokenized_docs) == 0:
        logger.warning("No chunks found for novel %s.", novel_name)
        return

    logger.info("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    logger.info("Done creating BM25 index.")

    # Tokenize the query
    query_tokens = preprocess(query)

    # Search for the top k nearest neighbors
    logger.info("Searching for the top %s nearest neighbors...", k)
    scores = bm25.get_scores(query_tokens)
    top_n_indices = np.argsort(scores)[::-1][:k]

    top_ids = [ids[i] for i in top_n_indices]

    chunks = get_chunk_from_id(top_ids)

    return chunks


def retrieve_context(query, novel_name, model, spoiler_threshold=None, k=10):
    """
    Retrieve the top k most similar chunks from the index based on the query.
    """

    # Use BM25 for retrieval
    chunks_bm25 = retrieve_context_bm25(
        query, novel_name, spoiler_threshold=spoiler_threshold, k=k
    )

    # Use ChromaDB for retrieval
    chunks_chroma = retrieve_context_chroma(
        query, novel_name, model, spoiler_threshold=spoiler_threshold, k=k
    )

    # Combine the results
    combined_chunks = list(set(chunks_bm25 + chunks_chroma))

    logger.info("Number of combined chunks: %s", len(combined_chunks))

    return combined_chunks


# ----------------------------------------
# RERANKING
# ----------------------------------------


def rerank_chunks(query, chunks):
    logger.info("Reranking chunks...")
    return chunks

