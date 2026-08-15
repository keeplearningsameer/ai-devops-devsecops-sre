# RAG — Retrieval Augmented Generation

This module contains my hands-on learning journey with Retrieval
Augmented Generation (RAG) for DevOps and SRE use cases.

## What I Built

A progressively improved RAG system that can:

- Retrieve relevant DevOps/SRE knowledge
- Use multiple documents as context
- Track source documents
- Apply relevance thresholds
- Avoid answering when context is insufficient
- Use conversation history
- Evaluate retrieved context
- Test retrieval accuracy
- Experiment with metadata-enriched retrieval
- Generate an end-to-end evaluation report

## RAG Architecture

Question
↓
Embedding Model
↓
Vector Search
↓
Retrieve Relevant Documents
↓
Evaluate Context
↓
LLM
↓
Source-aware Answer

## Exercises

| Exercise | Topic | Status |
|---|---|---|
| 6.1 | Simple RAG | ✅ |
| 6.2 | Multi-document RAG | ✅ |
| 6.3 | Source-aware RAG | ✅ |
| 6.4 | Relevance threshold | ✅ |
| 6.5 | No-context safety | ✅ |
| 6.6 | History-aware RAG | ✅ |
| 6.7 | Context evaluation | ✅ |
| 6.8 | RAG evaluation | ✅ |
| 6.9 | Retrieval improvement experiment | ✅ |
| 6.10 | End-to-end evaluation report | ✅ |

## Key Learning

### 1. RAG is more than calling an LLM

A useful RAG pipeline is:

Retrieve → Validate → Generate

The LLM should receive relevant context rather than relying only
on its general knowledge.

### 2. Retrieval quality matters

A powerful LLM cannot compensate for poor retrieved context.

The retrieval layer should therefore be tested independently.

### 3. Source awareness improves trust

The RAG response can identify which retrieved documents support
the answer.

This is particularly useful for SRE and incident troubleshooting,
where engineers need to understand where information came from.

### 4. Don't guess when context is missing

If the knowledge base does not contain enough relevant information,
the system should say that it does not have sufficient context
rather than inventing an answer.

### 5. Relevance thresholds are useful

A similarity threshold can be used as a quality gate before sending
retrieved context to the LLM.

### 6. Conversation history can improve retrieval

Previous conversation context can help interpret follow-up questions
and retrieve more relevant operational information.

### 7. RAG needs evaluation

A RAG system should be tested using a fixed evaluation dataset.

Example:

Question → Expected Document → Retrieved Document → PASS/FAIL

### 8. Retrieval improvements must be measured

An optimization should not be considered successful just because
the code changed.

The correct process is:

Measure → Identify Problem → Change → Measure Again

### Exercise 6.10 Result

The final controlled evaluation used the same evaluation dataset
for both approaches.

Baseline Retrieval Accuracy: 100%

Metadata-enriched Retrieval Accuracy: 100%

Improvement: 0 percentage points

Conclusion:

The metadata change did not produce a measurable accuracy improvement
on this fixed evaluation dataset because the baseline already retrieved
all four expected documents correctly.

## Important Engineering Lesson

Keep the evaluation dataset fixed when comparing different versions
of a RAG system.

Otherwise, changes in the dataset can make an improvement appear
larger or smaller than it really is.

## DevOps / SRE Relevance

RAG can be used to build AI-assisted SRE systems that retrieve:

- Runbooks
- Incident history
- Kubernetes troubleshooting guides
- Monitoring documentation
- Architecture documentation
- Deployment procedures
- Operational standards

A future AI-SRE assistant can follow:

Incident
↓
Retrieve relevant operational knowledge
↓
Evaluate evidence
↓
Generate troubleshooting guidance
↓
Show supporting sources
