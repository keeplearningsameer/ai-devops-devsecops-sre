import os
from openai import OpenAI


def main():
    client = OpenAI(
        timeout=30.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": "Explain Kubernetes to a DevOps engineer who is new to Generative AI.",
            }
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
