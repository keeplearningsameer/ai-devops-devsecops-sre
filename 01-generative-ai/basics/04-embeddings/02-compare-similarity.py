from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    text_a = "Kubernetes manages containerized applications."
    text_b = "Kubernetes orchestrates containers across a cluster."
    text_c = "The weather is sunny today."

    embeddings = model.encode([text_a, text_b, text_c])

    similarity_ab = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]

    similarity_ac = cosine_similarity(
        [embeddings[0]], [embeddings[2]]
    )[0][0]

    print("Embedding similarity experiment")
    print()
    print("Text A:", text_a)
    print("Text B:", text_b)
    print("Text C:", text_c)
    print()
    print("Similarity A ↔ B:", round(similarity_ab, 4))
    print("Similarity A ↔ C:", round(similarity_ac, 4))


if __name__ == "__main__":
    main()
