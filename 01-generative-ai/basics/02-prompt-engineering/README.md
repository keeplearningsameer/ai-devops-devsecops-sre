# Prompt Engineering

## Objective

Learn how system instructions and user prompts influence an LLM response.

## What I built

A Python application that instructs the LLM to behave as an experienced Site Reliability Engineer and answer a practical Kubernetes question.

## Prompt Structure

System message:

> You are an experienced Site Reliability Engineer.

User message:

> Explain Kubernetes liveness probes and readiness probes.

## Key Learning

- System prompts establish the role and behavior of the model.
- User prompts provide the specific task.
- Clear context produces more focused responses.
- Prompt engineering is important when building AI assistants for DevOps and SRE.

## Example Use Case

The same pattern can later be used for an AI-SRE incident assistant:

```text
System:
You are an experienced SRE incident assistant.

User:
Analyze this Kubernetes incident and suggest investigation steps.
