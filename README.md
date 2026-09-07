# Advanced RAG with Qdrant & Ollama

An end-to-end local Retrieval-Augmented Generation (RAG) pipeline built with Ollama, Qdrant, semantic search, and cross-encoder re-ranking.

The project demonstrates how an LLM can answer questions using information retrieved from external documents rather than relying only on its pretrained knowledge.

## 🚀 Overview

Large Language Models do not automatically know your private documents, internal knowledge bases, or information that was created after their training.

RAG solves this problem by retrieving relevant information at query time and providing it to the LLM as context.

This project implements the following pipeline:

```text
Documents
    ↓
Chunking
    ↓
Qwen3 Embedding 0.6B
    ↓
Qdrant Vector Database
    ↓
Semantic Search
    ↓
Top-K Candidates
    ↓
Cross-Encoder Re-ranking
    ↓
Top Relevant Chunks
    ↓
Qwen 3.5 4B
    ↓
Grounded Answer
