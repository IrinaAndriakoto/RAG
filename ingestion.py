"""
Pipeline d'ingestion pour le RAG local.
Étape 2 du TP : Extraction -> Chunking -> Vectorisation.
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --- Choix du modèle d'embeddings ---------------------------------------
# ATTENTION : si vos documents sont en français, N'UTILISEZ PAS un modèle
# purement anglais (ex: "all-MiniLM-L6-v2"). Ce modèle est majoritairement
# entraîné sur du texte anglais : la similarité cosinus entre une question
# française et des chunks français sera dégradée SANS erreur visible.
# On prend donc un modèle multilingue explicitement.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_local"


def load_documents(uploaded_files) -> List[Document]:
    """
    Prend une liste d'objets UploadedFile (Streamlit) et retourne une liste
    de Document LangChain, avec la métadonnée 'source' = nom réel du fichier.

    Point technique important : Streamlit fournit les fichiers en mémoire
    (BytesIO), mais PyMuPDFLoader / TextLoader attendent un CHEMIN sur
    disque. Il faut donc écrire un fichier temporaire, le charger, puis le
    supprimer.
    """
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
                loaded = loader.load()  # 1 Document par page, metadata["page"] déjà présent
            elif suffix in (".txt", ".md"):
                loader = TextLoader(tmp_path, encoding="utf-8")
                loaded = loader.load()
            else:
                # Ne JAMAIS ignorer silencieusement un format non supporté :
                # l'utilisateur doit comprendre pourquoi son fichier n'est
                # pas apparu dans la base après indexation.
                raise ValueError(f"Format non supporté : {suffix}")

            # PyMuPDFLoader met par défaut metadata["source"] = tmp_path,
            # c'est-à-dire un chemin temporaire illisible. On le remplace
            # par le vrai nom du fichier, sinon les citations affichées
            # à l'utilisateur seront incompréhensibles ("/tmp/xk3f9a.pdf").
            for doc in loaded:
                doc.metadata["source"] = uploaded_file.name

            documents.extend(loaded)
        finally:
            os.remove(tmp_path)  # ne jamais laisser de fichiers temporaires trainer sur disque

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Découpe les documents en chunks.

    Justification des valeurs par défaut (à documenter dans votre rapport
    de TP, c'est explicitement demandé) :

    - chunk_size=800 caractères (~150-200 tokens) : assez grand pour
      contenir une idée complète (2-3 phrases avec leur contexte), assez
      petit pour que le retriever reste précis. Un chunk trop long dilue
      le score de similarité (le vecteur moyenne plusieurs idées) et fait
      remonter du bruit dans les résultats de recherche.

    - chunk_overlap=120 (~15% du chunk_size) : évite qu'une phrase
      importante soit coupée exactement à la frontière entre deux chunks
      et perde le contexte qui l'entoure des deux côtés.

    - RecursiveCharacterTextSplitter plutôt qu'un découpage naïf par
      nombre de caractères fixe : il essaie de couper aux frontières
      naturelles du texte (\\n\\n, puis \\n, puis ". ", puis " ") avant de
      couper au milieu d'un mot. Ça préserve la cohérence sémantique de
      chaque chunk au lieu de trancher arbitrairement.

    Ces valeurs sont un POINT DE DÉPART raisonnable, pas une vérité
    absolue. Sur des documents très structurés (contrats, documentation
    technique avec du code), il est recommandé de tester au moins deux
    configurations (ex: 500/50 et 1000/150) et de comparer la qualité de
    récupération sur un petit jeu de questions de test — c'est ce genre
    de comparaison qui fait un bon rapport de TP, pas juste "j'ai choisi
    800 parce que ça marche".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Vectorise les chunks et les stocke dans ChromaDB, AVEC persistance sur
    disque (persist_directory). Sans ça, toute la base est perdue à chaque
    redémarrage de l'application Streamlit — problème classique que les
    étudiants découvrent en démo, trop tard.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """
    Recharge une base déjà indexée depuis le disque. À appeler au démarrage
    de l'app pour ne pas obliger l'utilisateur à ré-indexer à chaque
    lancement s'il a déjà des documents indexés d'une session précédente.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
