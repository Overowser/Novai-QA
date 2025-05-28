from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from logger_config import setup_logger
from utils import get_novel_id, get_db_connection

logger = setup_logger("chunker")


def segment_text(text, max_chunk_size, overlap, tokenizer=None):
    logger.info(
        f"Segmenting text into paragraphs with max size {max_chunk_size} and overlap {overlap}"
    )
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs = [
        (p, len(tokenizer.encode(p, add_special_tokens=False))) for p in paragraphs
    ]

    # adding a check for very large paragraphs
    if any(size + overlap > max_chunk_size for _, size in paragraphs):

        # we go to sentence level tokenization
        logger.info(
            f"Some paragraphs are too large, going to sentence level tokenization"
        )
        sentences = sent_tokenize(text)
        # the rest of the code remains the same so we use the same name for the variable
        paragraphs = [
            (s, len(tokenizer.encode(s, add_special_tokens=False))) for s in sentences
        ]

        # adding a check for very large sentences
        if any(size > max_chunk_size for _, size in paragraphs):
            logger.info(f"Some sentences are too large, breaking them down further")
            pars = []
            for s, size in paragraphs:
                if size <= max_chunk_size:
                    pars.append((s, size))
                else:
                    sentences = [
                        (
                            sentence.strip(),
                            len(
                                tokenizer.encode(
                                    sentence.strip(), add_special_tokens=False
                                )
                            ),
                        )
                        for sentence in s.split("\n")
                        if sentence.strip()
                    ]
                    pars += sentences

            paragraphs = pars

    logger.info(f"Segmented text into {len(paragraphs)} paragraphs")
    return paragraphs


def chunk_text(text, max_chunk_size=512, overlap=200, tokenizer=None):
    """
    This function chunks the text into smaller pieces based on the max_chunk_size and overlap.
    It uses the tokenizer to calculate the size of each chunk.
    If the first paragraph is too large, it will reduce the overlap down to a minimum of 0.
    It will still try to keep the overlap as large as possible, but it will not exceed the overlap size.
    When the overlap is done, it will add the next paragraph to the chunk until the max_chunk_size is reached.
    It will then create a new chunk and repeat the process until all paragraphs are processed.
    The first paragraph that is added first to the chunk should have a size less than the max_chunk_size.
    Really hopes this doesn't break, the logic of it all melted my brain.
    """

    paragraphs = segment_text(text, max_chunk_size, overlap, tokenizer=tokenizer)

    chunks = []
    current_chunk = []
    current_chunk_size = 0

    k = 0
    finished = False

    logger.info(
        f"Starting chunking with max size {max_chunk_size} and overlap {overlap}"
    )

    for i, (paragraph, size) in enumerate(paragraphs):
        if i != k:
            continue
        if not current_chunk:
            current_chunk.append(paragraph)
            current_chunk_size += size
            j = i - 1
            while current_chunk_size <= size + overlap:
                if j >= 0:
                    if current_chunk_size <= max_chunk_size - paragraphs[j][1]:
                        current_chunk.insert(0, paragraphs[j][0])
                        current_chunk_size += paragraphs[j][1]
                        j -= 1
                    else:
                        break
                else:
                    break

            while current_chunk_size <= max_chunk_size:
                k += 1
                if (
                    k < len(paragraphs)
                    and current_chunk_size + paragraphs[k][1] <= max_chunk_size
                ):
                    current_chunk.append(paragraphs[k][0])
                    current_chunk_size += paragraphs[k][1]
                elif k >= len(paragraphs):
                    finished = True
                    chunks.append(" ".join(current_chunk))
                    break
                elif current_chunk_size + paragraphs[k][1] > max_chunk_size:
                    chunks.append(" ".join(current_chunk))
                    break

            current_chunk = []
            current_chunk_size = 0

        if finished:
            break

    logger.info(f"Chunked text into {len(chunks)} chunks.")
    return chunks


def chunking_novel(
    novel_title,
    max_chunk_size=512,
    overlap=200,
    embedding_model="mixedbread-ai/mxbai-embed-large-v1",
):

    conn = get_db_connection()

    cursor = conn.cursor()

    novel_id = get_novel_id(novel_title, cursor)
    if not novel_id:
        logger.info(f"Novel '{novel_title}' not found in the database.")
        return

    cursor.execute("SELECT * FROM chunks WHERE novel_id = %s", (novel_id,))
    existing_chunks = cursor.fetchall()
    if existing_chunks:
        logger.info(f"Chunks for novel '{novel_title}' already exist in the database.")
        response = input("Do you want to delete them and rechunk? (y/n): ")
        if response.lower() == "y":
            cursor.execute("DELETE FROM chunks WHERE novel_id = %s", (novel_id,))
            conn.commit()
            logger.info(f"Deleted existing chunks for novel '{novel_title}'.")
        else:
            logger.info("Exiting without chunking.")
            return

    logger.info(f"Chunking novel '{novel_title}'...")
    logger.info(f"Using embedding model: {embedding_model}")
    logger.info(f"Using max_chunk_size: {max_chunk_size}")
    logger.info(f"Using overlap: {overlap}")

    model = SentenceTransformer(embedding_model)
    tokenizer = model.tokenizer

    cursor.execute(
        "SELECT id, chapter_content FROM chapters WHERE novel_id = %s", (novel_id,)
    )

    chunky = []

    chapters = cursor.fetchall()
    for chapter_id, chapter_content in chapters:
        logger.info(f"Chunking chapter ID {chapter_id}...")
        chunks = chunk_text(
            chapter_content,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
        logger.info(f"Chapter ID {chapter_id} has {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            # Insert into chunks table
            cursor.execute(
                "INSERT INTO chunks (chapter_id, novel_id, chunk_number, chunk_content) VALUES (%s, %s, %s, %s)",
                (chapter_id, novel_id, i + 1, chunk),
            )

            conn.commit()
        logger.info(f"Inserted {len(chunks)} chunks for chapter ID {chapter_id}")
        if len(chunks) > 5:
            chunky.append((chapter_id, len(chunks)))

    cursor.close()
    conn.close()
    logger.info("Chunking completed")
    logger.info(f"Here are the chunky chapters:")
    for chapter_id, num_chunks in chunky:
        logger.info(f"Chapter ID {chapter_id} has {num_chunks} chunks.")
    return


if __name__ == "__main__":
    chunking_novel("Supreme Magus")
    # "Infinite Mana In The Apocalypse"
