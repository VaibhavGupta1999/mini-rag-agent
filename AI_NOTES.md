# AI_NOTES.md — Mini RAG / Chat Agent

### Overview
This project was built with AI assistance to meet the **"AI-First Workflow"** assignment requirements.

### AI Tools Used
- **ChatGPT (GPT-5)** – used for generating code scaffolding, debugging, and UI design.
- **Groq LLM (Llama-3.1-8B-Instant)** – used inside the app as the actual model for answering questions.

### What Was AI-Assisted
- Project structure: FastAPI app layout, directory setup, and file organization.
- Core RAG logic: Embeddings, retrieval, and PDF support (using `sentence-transformers` + `pypdf`).
- Prompt engineering: The dual prompt system for small-talk + context-only answers.
- UI: Tailwind + HTMX chat interface and interactive features (Send, Help, Rebuild Index buttons).
- Debugging: Fixing dependency issues (`python-multipart`, removing FAISS/Sklearn for Windows compatibility).
- Documentation: README instructions, deployment guide, and this AI_NOTES.md summary.

### What Was Manually Done
- Testing and integration of Groq API key.
- Running and verifying embeddings on PDFs.
- Designing prompt routing and confidence threshold tuning.
- Final testing and UI polishing.
- Preparing for deployment and version control setup.

### Reflection
AI assistance significantly accelerated building this RAG agent by generating the base scaffolding and refining prompts.  
The human contribution focused on debugging, integrating real-world Groq LLM inference, and ensuring production-ready usability.

---

*Author: Vaibhav Gupta*  
*Date: 10/15/2025*
