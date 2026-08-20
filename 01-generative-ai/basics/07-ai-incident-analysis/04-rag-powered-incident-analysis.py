import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


def retrieve_context(question, documents, model, top_k=3):
    document_embeddings = model.encode(documents)
    question_embedding = model.encode([question])

    similarities = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    ranked_indexes = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indexes:
        results.append({
            "document": documents[index],
            "similarity": float(similarities[index]),
        })

    return results


def main():
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
        """
        CrashLoopBackOff means that a Kubernetes container is repeatedly
        starting and failing. Common causes include application errors,
        incorrect configuration, missing environment variables, failed
        dependencies, or insufficient permissions.
        """,
        """
        A Kubernetes Service provides a stable network endpoint for
        accessing a group of Pods. If a Service has no healthy endpoints,
        applications may fail to connect to the backend.
        """,
        """
        A recent application deployment can introduce configuration changes
        such as incorrect environment variables, Secrets, ConfigMaps,
        service names, ports, or connection settings.
        """,
        """
        Kubernetes NetworkPolicies can restrict traffic between Pods.
        Incorrect ingress or egress rules may prevent an application from
        connecting to a database or other dependent service.
        """,
        """
        Rolling back to a previously working Deployment revision can be a
        safe and reversible mitigation when an incident starts immediately
        after a new deployment.
        """,
    ]

    print("RAG-Powered AI Incident Analysis")
    print("=" * 60)

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    question = """
    Why are the payments-api pods restarting and unable to connect
    to the database after a recent deployment?
    """

    results = retrieve_context(
        question,
        knowledge_base,
        model
    )

    print("\nRetrieved Operational Knowledge:")

    context = ""

    for number, result in enumerate(results, start=1):
        print(f"\n[Source {number}]")
        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )
        print(result["document"].strip())

        context += (
            f"\n[Source {number}]\n"
            f"{result['document'].strip()}\n"
        )

    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )

    prompt = f"""
You are an experienced Site Reliability Engineer.

Analyze the production incident below.

INCIDENT DATA:

{json.dumps(incident, indent=2)}

RETRIEVED OPERATIONAL KNOWLEDGE:

{context}

Provide:

1. Incident Summary

2. Confirmed Evidence
Only include facts directly present in the incident data.

3. Likely Root Cause
Use both the incident evidence and retrieved knowledge.
Do not claim a cause is confirmed unless directly supported.

4. Supporting Sources
Mention which retrieved sources support your analysis.

5. Alternative Hypotheses

6. Missing Evidence

7. Recommended Investigation
Provide diagnostic steps in priority order.

8. Recommended Mitigation
Prefer safe and reversible actions.

Rules:
- Do not invent evidence.
- Clearly separate facts from hypotheses.
- Use the retrieved operational knowledge as supporting context.
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

    print("\nAI Incident Analysis:")
    print("=" * 60)

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
