# LLM Parameters & Structured Responses

## Objective

Learn how LLM parameters and structured prompting can make AI responses more consistent and useful for automation.

## What I built

A Python application that sends a simulated Kubernetes production incident to an LLM and asks it to analyze the incident using a predefined response structure.

## Incident

The Kubernetes pod is in `CrashLoopBackOff`.

Application logs show:

```text
Connection refused to database at db:5432
