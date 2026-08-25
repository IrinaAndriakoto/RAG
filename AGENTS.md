# AGENTS.md

Local "clone de NotebookLM" — TP RAG (Streamlit + LangChain + ChromaDB). Two source files only: `app.py` (UI + chat loop) and `ingestion.py` (load → chunk → embed → Chroma).

## Run / verify

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py   # venv is Python 3.14, no other env manager configured
.\venv\Scripts\pip.exe install -r requirements.txt # deps unpinned
```

- Ollama must run in background (`ollama serve`, model pulled via `ollama pull mistral`) for full RAG mode — `mistral` is hardcoded at app.py:190. With the sidebar toggle off ("recherche sémantique pure"), no LLM is called and Ollama isn't needed.
- First run downloads embedding model `paraphrase-multilingual-MiniLM-L12-v2` from Hugging Face (~120 MB) — needs network on first launch.
- Chroma persists to `./chroma_db` relative to CWD: run Streamlit from the repo root; delete the folder to reset the index.
- Gotcha: `ingestion.load_existing_vectorstore()` exists but is never called in `app.py` — every new session starts unindexed even if `chroma_db/` has data.

## Conventions

- UI strings, comments, and docstrings are French — keep them French.
- All cross-interaction state lives in `st.session_state` (`messages`, `vectorstore`, `indexed_files`); Streamlit reruns the whole script per interaction, so don't move state to module globals or cache the vectorstore elsewhere.
- Embedding model must stay multilingual (docs are French) — do not swap to English-only models like `all-MiniLM-L6-v2`.

## Tooling

No git repo, no tests, no lint/format/typecheck config. Verify changes by running the app manually.
