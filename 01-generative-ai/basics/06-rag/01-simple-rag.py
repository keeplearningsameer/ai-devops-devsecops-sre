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


def retrieve_context(query, documents, model):
    document_embeddings = model.encode(documents)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    best_index = similarities.argmax()

    return documents[best_index], similarities[best_index]


def main():
    documents = load_documents()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    query = "Why is my Kubernetes pod restarting repeatedly?"

    context, score = retrieve_context(
        query,
        documents,
        embedding_model,
    )

    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    prompt = f"""
You are a DevOps/SRE assistant.

Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{query}

If the context does not contain enough information, say:
"I don't have enough information in the provided context."

Provide a concise operational explanation.
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

    print("\nRetrieved Context:")
    print(context)

    print(f"\nSimilarity Score: {score:.4f}")

    print("\nLLM Answer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
