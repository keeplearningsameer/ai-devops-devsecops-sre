# LLM Basics — First API Call

This is my first hands-on Generative AI application.

## What I built

A simple Python application that sends a prompt to an LLM through OpenRouter and prints the generated response.

## Architecture

Python Application
→ OpenAI Python SDK
→ OpenRouter API
→ Free LLM
→ Generated Response

## Technologies

- Python 3.12
- OpenAI Python SDK
- OpenRouter
- GitHub Codespaces

## Example Prompt

```text
Explain Kubernetes to a DevOps engineer who is new to Generative AI.

## Key Learning

- How to make an LLM API call from Python.
- How to send a prompt to an LLM.
- How to receive and print the generated response.
- How OpenAI-compatible APIs can be used with different LLM providers.

## API Key Security

The API key is not stored in the source code.

The application reads it from:

```python
os.environ["OPENROUTER_API_KEY"]

The actual API key is kept outside the Git repository.
