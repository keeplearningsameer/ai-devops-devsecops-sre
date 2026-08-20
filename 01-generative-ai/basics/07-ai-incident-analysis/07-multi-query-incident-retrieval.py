from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_top_result(query, documents, model):
    document_texts = [
        document["content"]
        for document in documents
    ]

    document_embeddings = model.encode(document_texts)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    best_index = similarities.argmax()

    return {
        "query": query,
        "title": documents[best_index]["title"],
        "category": documents[best_index]["category"],
        "content": documents[best_index]["content"],
        "similarity": float(similarities[best_index]),
    }


def main():
    print("Multi-Query Incident Retrieval")
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

    focused_queries = [
        {
            "investigation_path": "Deployment and Configuration",
            "query": """
The application worked before the latest deployment but immediately
failed afterwards with a database connection refused error. Retrieve
knowledge about deployment-related configuration changes, environment
variables, Secrets, ConfigMaps, service names, ports, and connection
settings.
""",
        },
        {
            "investigation_path": "Database Connectivity",
            "query": """
Retrieve knowledge about Kubernetes database connection refused errors,
including Service availability, endpoints, target ports, selectors,
and database Pod health.
""",
        },
        {
            "investigation_path": "CrashLoopBackOff",
            "query": """
Retrieve troubleshooting knowledge for a Kubernetes pod repeatedly
restarting with CrashLoopBackOff. Focus on logs, previous logs,
events, exit codes, configuration, and dependent services.
""",
        },
        {
            "investigation_path": "Network Security",
            "query": """
Retrieve knowledge about Kubernetes NetworkPolicies blocking application
traffic to a database, including ingress, egress, Pod selectors,
namespace selectors, and TCP port access.
""",
        },
    ]

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    all_results = []

    for item in focused_queries:
        result = retrieve_top_result(
            item["query"],
            knowledge_base,
            model,
        )

        result["investigation_path"] = (
            item["investigation_path"]
        )

        all_results.append(result)

    print("\nInvestigation Results:")

    for number, result in enumerate(all_results, start=1):
        print(f"\n--- Investigation {number} ---")
        print(
            f"Path: {result['investigation_path']}"
        )
        print(f"Retrieved: {result['title']}")
        print(f"Category: {result['category']}")
        print(
            f"Similarity: {result['similarity']:.4f}"
        )

    unique_results = {}

    for result in all_results:
        title = result["title"]

        if title not in unique_results:
            unique_results[title] = result

    print("\nCombined Unique Knowledge:")
    print("-" * 60)

    for number, result in enumerate(
        unique_results.values(),
        start=1,
    ):
        print(f"\n[Source {number}]")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(
            f"Best Similarity: "
            f"{result['similarity']:.4f}"
        )
        print(result["content"].strip())

    print("\nSummary:")
    print(
        f"{len(focused_queries)} investigation paths were used."
    )
    print(
        f"{len(unique_results)} unique knowledge sources were retrieved."
    )


if __name__ == "__main__":
    main()
