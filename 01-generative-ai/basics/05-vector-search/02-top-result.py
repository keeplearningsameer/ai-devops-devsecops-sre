from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    documents = [
        "Kubernetes manages containerized applications across a cluster.",
        "Prometheus collects metrics and provides monitoring and alerting.",
        "Terraform is an infrastructure as code tool used to provision cloud resources.",
        "Docker packages applications and their dependencies into containers.",
    ]

    query = "How can I monitor my Kubernetes environment?"

    document_embeddings = model.encode(documents)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    best_index = similarities.argmax()

    print("Query:")
    print(query)
    print()
    print("Most relevant document:")
    print(documents[best_index])
    print()
    print("Similarity score:")
    print(f"{similarities[best_index]:.4f}")


if __name__ == "__main__":
    main()
