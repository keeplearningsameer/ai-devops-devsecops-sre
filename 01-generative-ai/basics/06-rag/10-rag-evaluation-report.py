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


def retrieve(query, documents, model, use_metadata=False):
    if use_metadata:
        searchable_text = [
            f"{document['topic']} "
            f"{document['category']} "
            f"{document['content']}"
            for document in documents
        ]
    else:
        searchable_text = [
            document["content"]
            for document in documents
        ]

    document_embeddings = model.encode(searchable_text)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    best_index = similarities.argmax()

    return documents[best_index], similarities[best_index]


def evaluate(documents, model, use_metadata):
    # FIXED evaluation dataset.
    # The same questions are used for both baseline and improved retrieval.
    test_cases = [
        {
            "question": "Why is my Kubernetes pod restarting repeatedly?",
            "expected": "Kubernetes",
        },
        {
            "question": "How can I monitor Kubernetes metrics?",
            "expected": "Prometheus",
        },
        {
            "question": "What tool can I use to provision cloud infrastructure?",
            "expected": "Terraform",
        },
        {
            "question": "How are applications packaged into containers?",
            "expected": "Docker",
        },
    ]

    passed = 0

    for test in test_cases:
        document, score = retrieve(
            test["question"],
            documents,
            model,
            use_metadata=use_metadata,
        )

        success = (
            document["topic"].lower()
            == test["expected"].lower()
        )

        if success:
            passed += 1

        print(
            f"  {'PASS' if success else 'FAIL'} | "
            f"{test['expected']} | "
            f"Retrieved: {document['topic']} | "
            f"Score: {score:.4f}"
        )

    accuracy = (passed / len(test_cases)) * 100

    return passed, len(test_cases), accuracy


def main():
    print("End-to-End RAG Evaluation Report")
    print("=" * 60)

    documents = load_documents()

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n1. BASELINE RETRIEVAL")
    print("-" * 60)

    baseline_passed, baseline_total, baseline_accuracy = evaluate(
        documents,
        model,
        use_metadata=False,
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_passed}/{baseline_total} "
        f"({baseline_accuracy:.1f}%)"
    )

    print("\n2. IMPROVED RETRIEVAL")
    print("-" * 60)

    improved_passed, improved_total, improved_accuracy = evaluate(
        documents,
        model,
        use_metadata=True,
    )

    print(
        f"Improved Accuracy: "
        f"{improved_passed}/{improved_total} "
        f"({improved_accuracy:.1f}%)"
    )

    improvement = improved_accuracy - baseline_accuracy

    print("\n3. FINAL COMPARISON")
    print("-" * 60)

    print(f"Baseline Accuracy : {baseline_accuracy:.1f}%")
    print(f"Improved Accuracy : {improved_accuracy:.1f}%")
    print(f"Improvement       : {improvement:+.1f} percentage points")

    print("\nConclusion:")

    if improvement > 0:
        print(
            "Retrieval accuracy improved after adding metadata "
            "to the searchable document representation."
        )
    elif improvement == 0:
        print(
            "No accuracy improvement was observed on the fixed "
            "evaluation dataset."
        )
    else:
        print(
            "Retrieval accuracy decreased after the change. "
            "Further investigation is required."
        )


if __name__ == "__main__":
    main()
