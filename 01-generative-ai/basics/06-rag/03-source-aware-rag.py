import os

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents():
    with open(
        "01-generative-ai/basics/06-rag/data/devops-knowledge.txt",
        "r",
        encoding="utf-8",
    ) as file:
        text = file.read()

    return [
        {
            "id": index + 1,
            "content": chunk.strip(),
        }
        for index, chunk in enumerate(text.split("\n\n"))
        if chunk.strip()
    ]


def retrieve_top_documents(query, documents, model, top_k=3):
    document_texts = [document["content"] for document in documents]

    document_embeddings = model.encode(document_texts)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append(
            {
                "id": documents[index]["id"],
                "content": documents[index]["content"],
                "score": similarities[index],
            }
        )

    return results


def main():
    documents = load_documents()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    query = "How can I troubleshoot a Kubernetes pod that keeps restarting?"

    results = retrieve_top_documents(
        query,
        documents,
        embedding_model,
        top_k=3,
    )

    context = "\n\n".join(
        f"[Source {result['id']}]\n{result['content']}"
        for result in results
    )

    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = f"""
You are a DevOps/SRE assistant.

Answer the user's question using the provided context.

For every important recommendation, reference the source number
such as [Source 1] or [Source 2].

Context:

{context}

Question:

{query}

Provide a concise troubleshooting response.
Do not invent information that is not supported by the context.
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    print("Question:")
    print(query)

    print("\nRetrieved Sources:")

    for result in results:
        print(f"\n[Source {result['id']}]")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Document: {result['content']}")

    print("\nLLM Answer:")
    print(response.choices[0].message.content)

    print("\nSources Used:")
    for result in results:
        print(
            f"[Source {result['id']}] "
            f"Similarity: {result['score']:.4f}"
        )


if __name__ == "__main__":
    main()
