import os
import json
from openai import OpenAI


def main():
    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )

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

    prompt = f"""
You are an experienced Site Reliability Engineer.

Analyze this production incident.

Incident data:

{json.dumps(incident, indent=2)}

Provide the following sections:

1. Incident Summary
2. Confirmed Evidence
3. Likely Root Cause
4. Assumptions and Uncertainties
5. Potential Impact
6. Recommended Investigation Steps
7. Recommended Remediation

Rules:

- Do not treat assumptions as confirmed facts.
- Base your analysis primarily on the provided evidence.
- Clearly separate confirmed evidence from likely causes.
- If more information is required, mention what should be checked.
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

    print("Structured AI-Powered Incident Analysis")
    print("=" * 60)

    print("\nIncident Input:")
    print(json.dumps(incident, indent=2))

    print("\nAI Analysis:")
    print("-" * 60)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
