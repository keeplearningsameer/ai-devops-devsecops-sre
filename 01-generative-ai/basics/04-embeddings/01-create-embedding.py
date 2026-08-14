from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    text = """
    Kubernetes is a container orchestration platform.
    It manages containerized applications across a cluster.
    """

    embedding = model.encode(text)

    print("Embedding created successfully!")
    print()
    print("Input text:")
    print(text.strip())
    print()
    print("Embedding dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])


if __name__ == "__main__":
    main()
