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

    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def retrieve_top_documents(query, documents, model, top_k=3):
    document_embeddings = model.encode(documents)
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
                "document": documents[index],
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
        result["document"]
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

Context:
{context}

Question:
{query}

Provide a concise troubleshooting response.
Mention the relevant Kubernetes commands where appropriate.
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

    print("\nRetrieved Documents:")

    for number, result in enumerate(results, start=1):
        print(f"\n--- Result {number} ---")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")

    print("\nLLM Answer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
