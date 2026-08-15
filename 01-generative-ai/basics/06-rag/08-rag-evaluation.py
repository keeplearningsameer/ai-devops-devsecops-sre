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


def retrieve_top_document(query, documents, model):
    document_texts = [document["content"] for document in documents]

    document_embeddings = model.encode(document_texts)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    best_index = similarities.argmax()

    return {
        "id": documents[best_index]["id"],
        "content": documents[best_index]["content"],
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
            "expected_text": "CrashLoopBackOff",
        },
        {
            "question": "How can I monitor Kubernetes metrics?",
            "expected_text": "Prometheus",
        },
        {
            "question": "What tool can I use to provision cloud infrastructure?",
            "expected_text": "Terraform",
        },
        {
            "question": "How are applications packaged into containers?",
            "expected_text": "Docker",
        },
    ]

    passed = 0

    print("RAG Evaluation")
    print("=" * 50)

    for number, test_case in enumerate(test_cases, start=1):
        result = retrieve_top_document(
            test_case["question"],
            documents,
            model,
        )

        success = test_case["expected_text"].lower() in (
            result["content"].lower()
        )

        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\nTest {number}: {status}")
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected_text']}")
        print(f"Retrieved: {result['content']}")
        print(f"Similarity: {result['score']:.4f}")

    total = len(test_cases)

    print("\n" + "=" * 50)
    print(f"Evaluation Result: {passed}/{total} tests passed")

    accuracy = passed / total * 100

    print(f"Retrieval Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
