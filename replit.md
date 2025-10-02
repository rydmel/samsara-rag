# Samsara Customer RAG Chatbot

## Overview
RAG-based chatbot that answers questions about Samsara's customers using scraped customer stories from https://www.samsara.com/customers. Built with Streamlit, OpenAI GPT-5, ChromaDB, and Pydantic Logfire observability.

## Recent Changes
- **2025-10-02**: Applied dark blue theme (#0f1b2e background) for clean, professional UI
- **2025-10-02**: Fixed Agentic RAG temperature issue (gpt-5 requires temperature=1.0)
- **2025-10-02**: Implemented Agentic RAG pattern with multi-step reasoning and query planning
- **2025-10-02**: Integrated Pydantic Logfire for real-time observability and tracing
- **2025-10-02**: Added comprehensive export functionality (CSV/JSON) for performance data

## User Preferences
- **UI Theme**: Dark blue background (#0f1b2e) with light text for clean, professional look
- **Temperature**: Always 1.0 (gpt-5 model requirement)
- **Auto-load**: Customer stories automatically load on first app launch

## Project Architecture

### Core Components
- `app.py`: Main Streamlit application with 4 tabs (Chat, Configuration, Evaluation, Knowledge Base)
- `rag_engine.py`: RAG implementation with 4 patterns (naive, parent_document, hybrid, agentic)
- `vector_store.py`: ChromaDB vector database management
- `observability.py`: Pydantic Logfire integration for tracing
- `scraper.py`: Web scraper for Samsara customer stories
- `evaluation.py`: Performance metrics and RAG evaluation

### RAG Patterns
1. **Naive**: Simple semantic search
2. **Parent Document**: Chunk-level search with parent document retrieval
3. **Hybrid**: Combines semantic + keyword (BM25) search
4. **Agentic**: Multi-step reasoning with LLM-based query planning

### Technical Stack
- **LLM**: OpenAI GPT-5 (temperature=1.0 only)
- **Vector DB**: ChromaDB with sentence-transformers embeddings
- **Framework**: Streamlit on port 5000
- **Observability**: Pydantic Logfire with LOGFIRE_TOKEN
- **Deployment**: Streamlit headless mode at 0.0.0.0:5000

## Key Configuration
- ChromaDB: Custom settings to avoid tenant errors
- Streamlit: Dark blue theme, headless=true, port 5000
- OpenAI: Uses OPENAI_API_KEY secret
- Logfire: Uses LOGFIRE_TOKEN secret for tracing
