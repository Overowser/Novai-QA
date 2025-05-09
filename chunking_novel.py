import psycopg2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

load_dotenv()


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


def segment_text(text, tokenizer=None):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    paragraphs = [(p, len(tokenizer.encode(p, add_special_tokens=False))) for p in paragraphs]

    # adding a check for very large paragraphs
    if any(size + overlap > max_chunk_size for _, size in paragraphs):
        
        # we go to sentence level tokenization
        sentences = sent_tokenize(text)
        # the rest of the code remains the same so we use the same name for the variable
        paragraphs = [(s, len(tokenizer.encode(s, add_special_tokens=False))) for s in sentences]

        # adding a check for very large sentences
        if any(size > max_chunk_size for _, size in paragraphs):
            pars = []
            for s, size in paragraphs:
                if size <= max_chunk_size:
                    pars.append((s, size))
                else:
                    sentences = [(sentence.strip(), len(tokenizer.encode(sentence.strip(), add_special_tokens=False)))  for sentence in s.split("\n") if sentence.strip()]
                    pars+= sentences

            paragraphs = pars


def chunk_text(text, max_chunk_size=512, overlap=200, tokenizer=None):
    
    paragraphs = segment_text(text, tokenizer=tokenizer)


    chunks = []
    current_chunk = []
    current_chunk_size = 0
    overlap_buffer = []
    new_overlap = overlap

    for i, (paragraph, size) in enumerate(paragraphs):
        if current_chunk_size + size <= max_chunk_size:
            current_chunk.append(paragraph)
            current_chunk_size += size
        else:
            # Finalize current chunk
            chunks.append("\n".join(current_chunk))

            # Prepare overlap buffer (from the end of current chunk)
            overlap_buffer = []
            temp_size = 0
            new_overlap

            for p, s in reversed([(p, len(tokenizer.encode(p, add_special_tokens=False))) for p in current_chunk]):
                if temp_size + s <= overlap:
                    overlap_buffer.insert(0, p)
                    temp_size += s
                else:
                    break 

            # Start new chunk with overlap buffer and current paragraph
            current_chunk = overlap_buffer + [paragraph]
            current_chunk_size = temp_size + size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def chunking_novel(novel_title, max_chunk_size=512, overlap=200, embedding_model="mixedbread-ai/mxbai-embed-large-v1"):

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

    novel_id = get_novel_id(novel_title, cursor, conn)
    if not novel_id:
        print(f"Novel '{novel_title}' not found in the database.")
        return

    cursor.execute("SELECT * FROM chunks WHERE novel_id = %s", (novel_id,))
    existing_chunks = cursor.fetchall()
    if existing_chunks:
        print(f"Chunks for novel '{novel_title}' already exist in the database.")
        response = input("Do you want to delete them and rechunk? (y/n): ")
        if response.lower() == 'y':
            cursor.execute("DELETE FROM chunks WHERE novel_id = %s", (novel_id,))
            conn.commit()
            print(f"Deleted existing chunks for novel '{novel_title}'.")
        else:
            print("Exiting without chunking.")
            return

    print(f"Chunking novel '{novel_title}'...")
    print(f"Using embedding model: {embedding_model}")
    print(f"Using max_chunk_size: {max_chunk_size}")
    print(f"Using overlap: {overlap}")

    model = SentenceTransformer(embedding_model)
    tokenizer = model.tokenizer

    cursor.execute("SELECT id, chapter_content FROM chapters WHERE novel_id = %s", (novel_id,))

    chunky = []

    chapters = cursor.fetchall()
    for chapter_id, chapter_content in chapters:
        print(f"Chunking chapter ID {chapter_id}...")
        chunks = chunk_text(chapter_content, max_chunk_size=max_chunk_size, overlap=overlap, tokenizer=tokenizer)
        print(f"Chapter ID {chapter_id} has {len(chunks)} chunks.")
        for i, chunk in enumerate(chunks):
            # Insert into chunks table
            cursor.execute("INSERT INTO chunks (chapter_id, novel_id, chunk_number, chunk_content) VALUES (%s, %s, %s, %s)",
                           (chapter_id, novel_id, i + 1, chunk))
    
            conn.commit()
        print(f"Inserted {len(chunks)} chunks for chapter ID {chapter_id}")
        if len(chunks) > 5:
            chunky.append((chapter_id, len(chunks)))
    
    cursor.close()
    conn.close()
    print("Chunking completed")
    print(f"Here are the chunky chapters:")
    for chapter_id, num_chunks in chunky:
        print(f"Chapter ID {chapter_id} has {num_chunks} chunks.")
    return

# test
chunking_novel("Infinite Mana In The Apocalypse")
# "Infinite Mana In The Apocalypse"


# TODO:
# 1. Make it async
# 2. Make it check if the chunk already exists in the database before inserting it
