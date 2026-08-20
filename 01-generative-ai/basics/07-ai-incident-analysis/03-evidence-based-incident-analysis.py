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

Analyze the following production incident using an
evidence-based approach.

Incident data:

{json.dumps(incident, indent=2)}

Provide exactly these sections:

1. Incident Summary

2. Confirmed Evidence
Only include facts directly supported by the incident data.

3. Likely Root Cause
Identify the most likely cause, but do not present it as confirmed
unless there is direct evidence.

4. Confidence Level
Choose one:
- High
- Medium
- Low

Explain briefly why.

5. Alternative Hypotheses
List other possible explanations.

6. Missing Evidence
List the specific information needed to confirm or reject
the likely root cause.

7. Recommended Investigation
Provide the next diagnostic steps in priority order.

8. Recommended Mitigation
Suggest the safest immediate mitigation based on the available evidence.

Rules:
- Do not invent logs, metrics, configuration values, or events.
- Clearly distinguish facts from hypotheses.
- Prefer reversible actions when recommending immediate mitigation.
- Do not claim a root cause is confirmed without evidence.
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

    print("Evidence-Based AI Incident Analysis")
    print("=" * 60)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
