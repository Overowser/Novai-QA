import ollama
from retrieval import retrieve_context
import faiss

def generate_response(query: str) -> str:

    system_prompt = """ You are a specialized RAG system for answering questions about novels. Your primary directive is to derive answers exclusively from the contextual information provided.

**When answering:**
* Base your entire answer on the provided text snippets.
* If possible and relevant, you can quote short phrases from the context to support your answer, but avoid simply copying large chunks of text. Rephrase and synthesize the information.
* If the context does not contain the information needed to answer the question, clearly state: 'The provided context does not contain information to answer this question.'
* Do not infer, speculate, or use any external knowledge beyond the supplied text."""


    index = faiss.read_index("faiss_database.index")

    _, _, context = retrieve_context(query, index, k=5)

    context = "\n\n".join(context)

    rag_prompt = f"""
**Context:**
{context}

**Question:**
{query}"""

    # Generate the response using the Ollama model
    response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rag_prompt}
        ],
        stream=True,
    )

    return response


# example usage
if __name__ == "__main__":
    query = input("Enter your question: ")
    stream = generate_response(query)
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
