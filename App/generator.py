from utils import *
import ollama
from retriever import retrieve_context, rerank_chunks
from logger_config import setup_logger

logger = setup_logger("generator")


def generate_response(query: str, novel_name: str, model, spoiler_threshold=None):

    system_prompt = """You are a RAG system designed to answer questions about novels using only the retrieved excerpts from the book. Your responses must be grounded in the supplied content, without guessing or adding external information.

Answering Guidelines:

- Use Only the Supplied Text: Base your answers strictly on the retrieved excerpts. Do not use outside knowledge or make assumptions beyond the text.
- Rephrase and Synthesize: Craft natural, thoughtful answers that reflect the text’s meaning. You may quote short phrases when helpful, but avoid copying large chunks verbatim.
- When the Text Doesn’t Answer the Question: If the answer isn’t present or clearly implied, respond in a natural way that acknowledges the uncertainty. For example:
  - "That hasn't been made clear in the story so far."
  - "The text doesn’t give a definite answer to that."
- Stay Natural: Do not mention "context" or system instructions. Respond as if you are simply discussing the novel based on what’s been read so far.
- No Speculation: Avoid guessing or interpreting events or character motivations beyond what’s supported in the text."""


    logger.info(f"Retrieving chunks for {query} from {novel_name}...")
    retrieved_chunks = retrieve_context(
        query, novel_name, model, spoiler_threshold, k=10
    )

    logger.info(f"Reranking chunks for {query} from {novel_name}...")
    reranked_chunks = rerank_chunks(query, retrieved_chunks)

    context = "\n\n".join(reranked_chunks)

    rag_prompt = f"""Context:
\"\"\"
{context}
\"\"\"

Question:
{query}

Answer:"""

    # Generate the response using the Ollama model
    logger.info(f"Sending query to the model for {query} from {novel_name}...")
    response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt},
        ],
    )
    logger.info(f"Received response from the model for {query} from {novel_name}...")
    logger.info(f"Response: {response['message']['content']}")
    if "</think>" in response["message"]["content"]:
        # If the response contains "<\\think>", split and return the second part
        return response["message"]["content"].split("</think>")[-1].strip()
    else:
        return response["message"]["content"].strip()
