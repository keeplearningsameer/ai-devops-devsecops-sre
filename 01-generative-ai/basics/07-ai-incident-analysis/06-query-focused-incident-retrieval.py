from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_context(query, documents, model, top_k=3):
    document_texts = [document["content"] for document in documents]

    document_embeddings = model.encode(document_texts)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indexes:
        results.append({
            "title": documents[index]["title"],
            "category": documents[index]["category"],
            "content": documents[index]["content"],
            "similarity": float(similarities[index]),
        })

    return results


def main():
    print("Query-Focused Incident Retrieval")
    print("=" * 60)

    knowledge_base = [
        {
            "title": "CrashLoopBackOff Troubleshooting",
            "category": "kubernetes",
            "content": """
A Kubernetes pod enters CrashLoopBackOff when its container
starts and repeatedly fails. Investigation should begin with
container logs, previous logs, pod events, exit codes, application
configuration, environment variables, Secrets, ConfigMaps, and
dependent services.
""",
        },
        {
            "title": "Database Connection Failure After Deployment",
            "category": "application-configuration",
            "content": """
If an application was working before a deployment and immediately
starts failing with a database connection error, compare the new
deployment with the previous working version. Check database host,
port, environment variables, Secrets, ConfigMaps, connection strings,
service names, and configuration files introduced or changed during
the deployment.
""",
        },
        {
            "title": "Kubernetes Database Service Connectivity",
            "category": "networking",
            "content": """
A database connection refused error can occur when the Kubernetes
Service is unavailable, has no healthy endpoints, points to the wrong
target port, or when the database Pods are unhealthy. Verify the
Service, endpoints, targetPort, selectors, and database Pod status.
""",
        },
        {
            "title": "NetworkPolicy Database Access",
            "category": "network-security",
            "content": """
Kubernetes NetworkPolicies can block traffic between application Pods
and database Pods. Verify ingress and egress rules, namespace selectors,
Pod selectors, and whether TCP port 5432 is allowed between the
payments application and database.
""",
        },
        {
            "title": "Deployment Rollback",
            "category": "deployment",
            "content": """
When a production incident begins immediately after a deployment,
rolling back to the previous known-good Deployment revision is a safe
and reversible mitigation. After recovery, compare the failed and
working versions to identify the change that introduced the issue.
""",
        },
    ]

    # The original incident has many signals.
    # This query focuses specifically on the deployment/configuration signal.
    focused_query = """
The application was working correctly before the latest deployment.
Immediately after the deployment, it started failing to connect to
the database. Retrieve knowledge about deployment-related
configuration changes that can cause database connection failures,
including environment variables, Secrets, ConfigMaps, service names,
ports, and connection settings.
"""

    print("\nFocused Query:")
    print(focused_query.strip())

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    results = retrieve_context(
        focused_query,
        knowledge_base,
        model,
        top_k=3,
    )

    print("\nTop Retrieved Knowledge:")

    for number, result in enumerate(results, start=1):
        print(f"\n--- Result {number} ---")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(f"Similarity: {result['similarity']:.4f}")

    expected_title = "Database Connection Failure After Deployment"

    print("\nEvaluation:")

    if results[0]["title"] == expected_title:
        print("PASS")
        print(
            "The focused query retrieved the expected "
            "deployment-related knowledge as Result 1."
        )
    else:
        print("NOT YET")
        print(
            f"Expected Result 1: {expected_title}"
        )
        print(
            f"Actual Result 1: {results[0]['title']}"
        )


if __name__ == "__main__":
    main()
