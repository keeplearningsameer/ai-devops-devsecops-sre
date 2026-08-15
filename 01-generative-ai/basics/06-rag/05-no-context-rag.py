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


def retrieve_documents(
    query,
    documents,
    model,
    top_k=3,
    threshold=0.70,
):
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
        score = similarities[index]

        if score >= threshold:
            results.append(
                {
                    "id": documents[index]["id"],
                    "content": documents[index]["content"],
                    "score": score,
                }
            )

    return results


def main():
    documents = load_documents()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Deliberately asking something our knowledge base does not cover.
    query = "How do I configure PostgreSQL replication for disaster recovery?"

    threshold = 0.70

    results = retrieve_documents(
        query,
        documents,
        embedding_model,
        top_k=3,
        threshold=threshold,
    )

    print("Question:")
    print(query)

    print(f"\nRelevance Threshold: {threshold}")

    print("\nRetrieved Relevant Sources:")

    for result in results:
        print(f"\n[Source {result['id']}]")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Document: {result['content']}")

    if not results:
        print("\nNo sufficiently relevant information was found.")
        print(
            "I don't have enough information in the knowledge base "
            "to answer this question."
        )
        return

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

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer the
question, say:

"I don't have enough information in the provided context."

Do not use outside knowledge.
Do not guess.
Do not invent commands or configuration.

Context:

{context}

Question:

{query}
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

    print("\nLLM Answer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
