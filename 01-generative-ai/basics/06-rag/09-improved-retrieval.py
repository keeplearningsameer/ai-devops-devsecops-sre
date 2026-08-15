from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents():
    return [
        {
            "id": 1,
            "topic": "Kubernetes",
            "category": "orchestration",
            "content": (
                "Kubernetes is a container orchestration platform used "
                "to deploy, manage, and scale containerized applications. "
                "Kubernetes uses Pods as the smallest deployable units."
            ),
        },
        {
            "id": 2,
            "topic": "Docker",
            "category": "containers",
            "content": (
                "Docker packages applications and their dependencies "
                "into containers. Containers provide a consistent "
                "runtime environment across different systems."
            ),
        },
        {
            "id": 3,
            "topic": "Prometheus",
            "category": "monitoring",
            "content": (
                "Prometheus collects metrics and provides monitoring "
                "and alerting for infrastructure and applications."
            ),
        },
        {
            "id": 4,
            "topic": "Terraform",
            "category": "infrastructure-as-code",
            "content": (
                "Terraform is an infrastructure as code tool. "
                "It allows infrastructure resources to be defined "
                "in configuration files and managed consistently "
                "through automation."
            ),
        },
    ]


def retrieve_top_document(query, documents, model):
    searchable_text = [
        f"{document['topic']} "
        f"{document['category']} "
        f"{document['content']}"
        for document in documents
    ]

    document_embeddings = model.encode(searchable_text)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    best_index = similarities.argmax()

    return {
        "document": documents[best_index],
        "score": similarities[best_index],
    }


def main():
    documents = load_documents()

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    test_cases = [
        {
            "question": "Why is my Kubernetes pod restarting repeatedly?",
            "expected_topic": "Kubernetes",
        },
        {
            "question": "How can I monitor Kubernetes metrics?",
            "expected_topic": "Prometheus",
        },
        {
            "question": "What tool can I use to provision cloud infrastructure?",
            "expected_topic": "Terraform",
        },
        {
            "question": "How are applications packaged into containers?",
            "expected_topic": "Docker",
        },
    ]

    passed = 0

    print("Improved RAG Retrieval Evaluation")
    print("=" * 60)

    for number, test_case in enumerate(test_cases, start=1):
        result = retrieve_top_document(
            test_case["question"],
            documents,
            model,
        )

        document = result["document"]

        success = (
            document["topic"].lower()
            == test_case["expected_topic"].lower()
        )

        status = "PASS" if success else "FAIL"

        if success:
            passed += 1

        print(f"\nTest {number}: {status}")
        print(f"Question: {test_case['question']}")
        print(f"Expected Topic: {test_case['expected_topic']}")
        print(f"Retrieved Topic: {document['topic']}")
        print(f"Category: {document['category']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Document: {document['content']}")

    total = len(test_cases)
    accuracy = passed / total * 100

    print("\n" + "=" * 60)
    print(f"Evaluation Result: {passed}/{total} tests passed")
    print(f"Retrieval Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
