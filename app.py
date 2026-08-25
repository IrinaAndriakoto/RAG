"""
Application principale - Clone de NotebookLM local.
Étapes 1 à 4 du TP.

Lancer avec : streamlit run app.py
Prérequis   : ollama doit tourner en tâche de fond (ollama serve),
              avec un modèle déjà téléchargé (ex: ollama pull mistral).
"""

import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from ingestion import load_documents, split_documents, build_vectorstore, load_existing_vectorstore

st.set_page_config(page_title="RAG Local", page_icon="📚", layout="wide")

# --- État de session -------------------------------------------------------
# st.session_state est OBLIGATOIRE : Streamlit ré-exécute tout le script à
# chaque interaction. Sans ça, l'historique de chat et la base vectorielle
# seraient réinitialisés à chaque message envoyé.
if "messages" not in st.session_state:
    st.session_state.messages = []  # dicts {"role", "content", "sources"?}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# Recharger la base existante au démarrage si chroma_db/ contient déjà des données
if st.session_state.vectorstore is None and os.path.isdir("./chroma_db"):
    st.session_state.vectorstore = load_existing_vectorstore()


# --- Étape 4 : le prompt template ------------------------------------------
# PromptTemplate (et non un message role="system") comme demandé par
# l'énoncé. Note pour votre rapport : PromptTemplate est historiquement
# pensé pour des LLM "complétion" (une seule chaîne en entrée/sortie), pas
# pour des chat models à rôles séparés (system/user/assistant). Ici on
# formate le template en UNE SEULE chaîne, qu'on envoie ensuite comme
# unique message "human" au chat model — c'est un choix volontaire pour
# respecter l'indication technique de l'énoncé, mais sachez que
# ChatPromptTemplate (avec de vrais rôles system/human) est l'alternative
# plus idiomatique si vous utilisez un ChatModel dans un projet réel.
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Tu es un assistant qui répond UNIQUEMENT à partir du contexte "
        "fourni ci-dessous, extrait des documents de l'utilisateur.\n"
        "Règles strictes :\n"
        "- Si la réponse ne se trouve pas dans le contexte, dis "
        "clairement que tu ne sais pas, n'invente jamais.\n"
        "- Cite le nom du fichier source de chaque information que tu "
        "utilises.\n\n"
        "Contexte :\n"
        "{context}\n\n"
        "Question : {question}\n\n"
        "Réponse :"
    ),
)


def format_source_ref(doc) -> str:
    """Construit une référence lisible 'fichier.pdf (page 3)' depuis les métadonnées."""
    source = doc.metadata.get("source", "inconnu")
    page = doc.metadata.get("page")
    return f"{source} (page {page + 1})" if page is not None else source


# --- Barre latérale ---------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Charger vos documents",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🔍 Indexer les documents", disabled=not uploaded_files):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

        if not new_files:
            st.info("Ces fichiers sont déjà indexés.")
        else:
            with st.spinner(f"Extraction et vectorisation de {len(new_files)} fichier(s)..."):
                docs = load_documents(new_files)
                chunks = split_documents(docs)
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.indexed_files.extend(f.name for f in new_files)
            st.success(f"{len(chunks)} chunks indexés depuis {len(new_files)} fichier(s).")

    if st.session_state.indexed_files:
        st.caption("Fichiers indexés :")
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")

        file_to_delete = st.selectbox("Supprimer un fichier indexé", st.session_state.indexed_files)
        if st.button("🗑️ Désindexer", type="secondary"):
            st.session_state.vectorstore._collection.delete(where={"source": file_to_delete})
            st.session_state.indexed_files.remove(file_to_delete)
            if not st.session_state.indexed_files:
                st.session_state.vectorstore = None
            st.rerun()

    st.divider()
    st.header("⚙️ Mode")

    rag_mode = st.toggle(
        "Assistant RAG complet (avec LLM)",
        value=True,
        help=(
            "Activé (Étape 4) : le LLM génère une réponse contrainte au contexte.\n"
            "Désactivé (Étape 3) : recherche sémantique pure, aucun appel LLM — "
            "affiche directement les extraits bruts les plus pertinents."
        ),
    )

    top_k = st.slider("Nombre d'extraits à récupérer (k)", min_value=1, max_value=10, value=4)

    if st.button("🗑️ Effacer l'historique de conversation"):
        st.session_state.messages = []
        st.rerun()


# --- Zone principale --------------------------------------------------------
st.title("📚 RAG Local — Clone NotebookLM")

if st.session_state.vectorstore is None:
    st.info("Chargez et indexez au moins un document dans la barre latérale pour commencer.")

# Réaffichage de l'historique — y compris l'expander de transparence pour
# les messages RAG passés (sinon il disparaîtrait à chaque rerun).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 Voir les extraits utilisés comme contexte"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**{i}. {src['ref']}**")
                    st.markdown(f"> {src['content']}")

user_input = st.chat_input("Posez une question sur vos documents...")

if user_input:
    if st.session_state.vectorstore is None:
        st.warning("Veuillez d'abord indexer au moins un document.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Récupération : commune aux deux modes (Étape 3 = Étape 4.1) ------
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(user_input)

    with st.chat_message("assistant"):

        if not rag_mode:
            # ================= ÉTAPE 3 : recherche sémantique pure ======
            # Aucun appel LLM. On affiche le contenu brut des chunks et
            # le nom de fichier EXACT tel qu'il est stocké en métadonnée.
            if not retrieved_docs:
                response = "Aucun extrait pertinent trouvé dans la base."
                st.markdown(response)
            else:
                lines = ["**Extraits les plus pertinents (recherche sémantique, sans LLM) :**\n"]
                for i, doc in enumerate(retrieved_docs, start=1):
                    ref = format_source_ref(doc)
                    lines.append(f"**{i}. [{ref}]**\n> {doc.page_content.strip()}\n")
                response = "\n".join(lines)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            # ================= ÉTAPE 4 : assistant RAG complet ==========

            # Garde-fou : si rien n'est trouvé, ne PAS appeler le LLM avec
            # un contexte vide. Un contexte vide + une consigne "réponds
            # à partir du contexte" est justement la situation où un LLM
            # a le plus tendance à halluciner pour combler le vide.
            if not retrieved_docs:
                response = (
                    "Aucun extrait pertinent n'a été trouvé dans les documents indexés "
                    "pour répondre à cette question."
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()

            context = "\n\n---\n\n".join(
                f"[{format_source_ref(doc)}]\n{doc.page_content}" for doc in retrieved_docs
            )

            # Injection dynamique de {context} et {question} via PromptTemplate,
            # comme demandé par l'indication technique.
            final_prompt = RAG_PROMPT.format(context=context, question=user_input)

            llm = ChatOllama(model="mistral", temperature=0.1)

            placeholder = st.empty()
            full_response = ""
            for chunk in llm.stream(final_prompt):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            # Transparence (exigence 4 de l'Étape 4) : extraits dépliables
            # sous la réponse, pour que l'utilisateur vérifie ce qui a
            # réellement servi de contexte.
            sources = [
                {"ref": format_source_ref(doc), "content": doc.page_content.strip()}
                for doc in retrieved_docs
            ]
            with st.expander("🔍 Voir les extraits utilisés comme contexte"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**{i}. {src['ref']}**")
                    st.markdown(f"> {src['content']}")

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response, "sources": sources}
            )
