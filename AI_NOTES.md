# AI_NOTES.md — Mini RAG / Chat Agent

## Overview
This project was developed within **one day** as part of the **"AI-First Workflow"** challenge.  
It demonstrates the end-to-end design of a **Retrieval-Augmented Generation (RAG)** chatbot powered by **Groq LLM (Llama-3.1-8B-Instant)** for real-time question answering.  

As an **AI/ML engineer**, my primary focus was on building the **backend architecture, RAG pipeline, and model integration** from scratch.  
AI tools were mainly used to accelerate **UI development**, **basic prompt drafting**, and **documentation refinement** for faster delivery.

---

## AI Tools Used
- **ChatGPT (GPT-5)** – used extensively for:
  - **UI layout and styling guidance** (Tailwind + HTMX-based chat interface).  
  - Generating **frontend component ideas** and improving user-experience flow.  
  - Assisting in **prompt phrasing** and **basic documentation formatting**.  
- **Groq LLM (Llama-3.1-8B-Instant)** – integrated as the deployed inference model for chat responses.

---

## What Was AI-Assisted
- **UI / Frontend Development** – majority of the frontend (chat window, message flow, buttons, and Tailwind classes) was AI-assisted for rapid prototyping.  
- **Prompt Writing (Initial Draft)** – AI helped create the first version of context and small-talk prompts, later fine-tuned manually.  
- **Documentation Support** – structure and formatting ideas for `README.md` and this file.  
- Minor **folder structuring guidance** for the FastAPI project layout.

---

## What Was Manually Done
- **FastAPI Backend Architecture** – implemented complete routing, file upload endpoints, and scalable API structure.  
- **Core RAG Pipeline** – designed embedding, retrieval, and document-based QA using `sentence-transformers` and `pypdf`.  
- **Prompt Routing Logic** – implemented intelligent switching between context-based and open-domain responses using confidence thresholds.  
- **Groq API Integration** – built and tested the full Llama 3.1 inference workflow with robust error handling.  
- **UI Integration & Testing** – manually connected AI-generated UI components with backend endpoints and validated real-time interactions.  
- **Debugging & Optimization** – resolved dependency issues (`python-multipart`, FAISS, scikit-learn), verified embeddings, and optimized runtime.  
- **Deployment** – created Docker configuration, `.env` handling, and Fly.io deployment setup.  
- **Version Control** – managed Git initialization, commit workflow, and public repository configuration.

---

## Reflection
This project showcases how an **AI-driven workflow** can accelerate productivity when combined with solid backend and ML engineering expertise.  
While AI significantly assisted in **UI creation and documentation**, all **functional logic, integration, and deployment** were completed manually.  

Delivering a full RAG chatbot within one day highlights the ability to work **efficiently, creatively, and technically** — a strong example of AI-augmented engineering in action.

---

**Author:** *Vaibhav Gupta*  
**Date:** *October 15, 2025*

