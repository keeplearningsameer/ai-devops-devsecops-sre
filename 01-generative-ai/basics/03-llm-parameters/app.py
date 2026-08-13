import os
from openai import OpenAI


def main():
    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    incident = """
    Kubernetes pod is in CrashLoopBackOff.

    Application logs show:
    "Connection refused to database at db:5432"

    The application was working normally before the deployment.
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": """
You are an experienced Site Reliability Engineer.

Analyze production incidents carefully.

Return your response using exactly these sections:

Problem:
Likely Causes:
Investigation:
Recommended Action:
Risk:
""",
            },
            {
                "role": "user",
                "content": incident,
            },
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
