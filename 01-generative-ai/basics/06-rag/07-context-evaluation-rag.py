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


def retrieve_documents(query, documents, model, top_k=3):
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


def evaluate_context(results, minimum_score=0.55):
    if not results:
        return False

    best_score = results[0]["score"]

    return best_score >= minimum_score


def main():
    documents = load_documents()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    query = "Why is my Kubernetes pod restarting repeatedly?"

    results = retrieve_documents(
        query,
        documents,
        embedding_model,
        top_k=3,
    )

    print("Question:")
    print(query)

    print("\nRetrieved Sources:")

    for result in results:
        print(f"\n[Source {result['id']}]")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Document: {result['content']}")

    context_is_relevant = evaluate_context(
        results,
        minimum_score=0.55,
    )

    print("\nContext Evaluation:")

    if context_is_relevant:
        print("Relevant context found.")
    else:
        print("Context is not sufficiently relevant.")
        print("The LLM will not be called.")
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

First, consider whether the retrieved context actually supports
the user's question.

Only answer using information supported by the context.

If the context is insufficient, say:
"I don't have enough information in the provided context."

Context:

{context}

Question:

{query}

Give a concise SRE troubleshooting answer.
Reference the relevant source numbers.
Do not invent information.
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
