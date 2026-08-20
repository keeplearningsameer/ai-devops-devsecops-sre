import os
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


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
        "title": documents[best_index]["title"],
        "category": documents[best_index]["category"],
        "content": documents[best_index]["content"],
        "similarity": float(similarities[best_index]),
    }


def main():
    print("Multi-Query RAG AI Incident Analysis")
    print("=" * 60)

    incident = {
        "service": "payments-api",
        "environment": "production",
        "severity": "high",
        "symptoms": [
            "Kubernetes pod status is CrashLoopBackOff",
            "Restart count is 8",
            "Application is unavailable",
        ],
        "application_logs": [
            "Connection refused to database at db:5432"
        ],
        "recent_changes": [
            "A new deployment was released before the incident"
        ],
        "infrastructure_changes": [
            "No recent infrastructure changes reported"
        ],
    }

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
            "path": "Deployment and Configuration",
            "query": """
The application worked before the latest deployment but immediately
failed afterwards with a database connection refused error. Retrieve
knowledge about deployment-related configuration changes, environment
variables, Secrets, ConfigMaps, service names, ports, and connection
settings.
""",
        },
        {
            "path": "Database Connectivity",
            "query": """
Retrieve knowledge about Kubernetes database connection refused errors,
including Service availability, endpoints, target ports, selectors,
and database Pod health.
""",
        },
        {
            "path": "CrashLoopBackOff",
            "query": """
Retrieve troubleshooting knowledge for a Kubernetes pod repeatedly
restarting with CrashLoopBackOff. Focus on logs, previous logs,
events, exit codes, configuration, and dependent services.
""",
        },
        {
            "path": "Network Security",
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

    retrieved_results = []

    for item in focused_queries:
        result = retrieve_top_result(
            item["query"],
            knowledge_base,
            model,
        )

        result["path"] = item["path"]
        retrieved_results.append(result)

    unique_results = {}

    for result in retrieved_results:
        title = result["title"]

        if title not in unique_results:
            unique_results[title] = result

    print("\nRetrieved Investigation Knowledge:")

    context = ""

    for number, result in enumerate(
        unique_results.values(),
        start=1,
    ):
        print(f"\n[Source {number}]")
        print(f"Path: {result['path']}")
        print(f"Title: {result['title']}")
        print(f"Category: {result['category']}")
        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        context += (
            f"\n[Source {number}]\n"
            f"Investigation Path: {result['path']}\n"
            f"Title: {result['title']}\n"
            f"Category: {result['category']}\n"
            f"{result['content'].strip()}\n"
        )

    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )

    prompt = f"""
You are an experienced Site Reliability Engineer.

Analyze the production incident below using the provided incident
evidence and retrieved operational knowledge.

INCIDENT DATA:

{json.dumps(incident, indent=2)}

RETRIEVED OPERATIONAL KNOWLEDGE:

{context}

Provide exactly these sections:

1. Incident Summary

2. Confirmed Evidence
Only include facts directly supported by the incident data.

3. Likely Root Cause
Identify the most likely explanation, but do not present it as
confirmed unless the incident evidence proves it.

4. Alternative Hypotheses
Include other plausible explanations supported by the retrieved
knowledge.

5. Supporting Sources
Mention the source numbers relevant to each major hypothesis.

6. Missing Evidence
List what should be checked to confirm or reject the hypotheses.

7. Recommended Investigation
Provide the investigation steps in priority order.

8. Recommended Mitigation
Prefer safe and reversible actions.

Rules:
- Do not invent logs, metrics, events, or configuration values.
- Clearly distinguish confirmed evidence from hypotheses.
- Use the retrieved sources as supporting context.
- Mention source numbers where relevant.
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

    print("\nAI-SRE Incident Investigation Report")
    print("=" * 60)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
