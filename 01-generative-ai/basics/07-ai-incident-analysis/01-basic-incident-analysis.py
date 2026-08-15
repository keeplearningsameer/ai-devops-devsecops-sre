import os
from openai import OpenAI


def main():
    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"].strip(),
    )

    incident = """
Incident: payments-api pods are repeatedly restarting.

Symptoms:
- Kubernetes pod status: CrashLoopBackOff
- Restart count: 8
- Application logs show:
  "Connection refused to database at db:5432"
- The application was working correctly before the latest deployment.
- No recent infrastructure changes were reported.
"""

    prompt = f"""
You are an experienced Site Reliability Engineer.

Analyze the following production incident.

{incident}

Provide:

1. Incident Summary
2. Most Likely Root Cause
3. Evidence
4. Potential Impact
5. Recommended Investigation Steps
6. Recommended Remediation
7. Risk if the issue is not fixed

Do not invent information that is not present in the incident.
Clearly distinguish evidence from assumptions.
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

    print("AI-Powered Incident Analysis")
    print("=" * 60)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
