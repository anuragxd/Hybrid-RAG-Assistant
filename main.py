import os
import yaml
import streamlit as st
from dotenv import load_dotenv

# --- Defensive Network Settings ---
# These prevent Hugging Face from timing out on slower connections
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

# --- Custom Module Imports ---
from core.ingestion import WebIngestor
from core.preprocessing import clean_html
from core.chunking import SemanticChunker
from core.embeddings import EmbeddingEngine
from core.llm import GroqEngine

from indexing.manager import IndexManager
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import NeuralReranker
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import get_logger

# ------------------------------------------------------
# Setup & Initialization
# ------------------------------------------------------
load_dotenv()
logger = get_logger("Streamlit_UI")

# Page config
st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="🔍", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models(_config):
    """Caches the heavy ML models so they don't reload on every UI interaction."""
    st.sidebar.info("Loading ML Models into memory...")
    llm = GroqEngine(model=_config["llm"]["model"])
    embedder = EmbeddingEngine(model_name=_config["models"]["embedding"])
    reranker = NeuralReranker(model_name=_config["models"]["reranker"])
    st.sidebar.success("Models loaded!")
    return llm, embedder, reranker

config = load_config()
llm, embedder, reranker = load_models(config)

# ------------------------------------------------------
# App Layout
# ------------------------------------------------------
st.title("🧠 Hybrid Multi-Stage RAG Assistant")
st.markdown("Search a topic, build a dynamic knowledge base, and ask questions using lexical, dense, and cross-encoder retrieval.")

# Sidebar: Ingestion & Indexing Phase
with st.sidebar:
    st.header("1. Build Knowledge Base")
    topic = st.text_input("Enter a Research Topic:", placeholder="e.g., Solid State Batteries")
    num_results = st.slider("Max Web Sources", min_value=1, max_value=10, value=5)
    
    if st.button("Scrape & Index Topic", type="primary"):
        if not topic:
            st.error("Please enter a topic.")
        else:
            with st.status("Building Knowledge Base...", expanded=True) as status:
                st.write(f"🔍 Searching Tavily for: *{topic}*")
                ingestor = WebIngestor()
                search_results = ingestor.search_topic(topic, max_results=num_results)
                
                chunker = SemanticChunker(
                    chunk_size=config["retrieval"]["chunk_size"], 
                    chunk_overlap=config["retrieval"]["chunk_overlap"]
                )
                
                all_chunks = []
                progress_bar = st.progress(0)
                
                # Scrape and Chunk
                for idx, res in enumerate(search_results):
                    st.write(f"📄 Scraping: {res['url']}")
                    clean_text = clean_html(res['url'])
                    if clean_text:
                        chunks = chunker.chunk_document(clean_text, res['url'], res['title'])
                        all_chunks.extend(chunks)
                    progress_bar.progress((idx + 1) / len(search_results))
                
                st.write(f"⚙️ Indexing {len(all_chunks)} chunks into BM25 and FAISS HNSW...")
                index_manager = IndexManager(embedder, config["indexing"])
                index_manager.index_chunks(all_chunks)
                
                # --- CRITICAL FIX: Save to Disk ---
                index_manager.save("saved_index")
                st.write("💾 Knowledge Base successfully saved to disk!")
                # ----------------------------------
                
                # Setup Retriever and save to session state
                rewriter = QueryRewriter(llm)
                st.session_state.retriever = HybridRetriever(
                    index_manager=index_manager,
                    rewriter=rewriter,
                    reranker=reranker,
                    config=config["retrieval"]
                )
                status.update(label="Knowledge Base Built Successfully!", state="complete", expanded=False)

# Main Area: Retrieval & Generation Phase
st.header("2. Ask Questions")

if "retriever" not in st.session_state:
    st.info("👈 Please enter a topic and build the knowledge base from the sidebar first.")
else:
    st.success("Knowledge Base is active and ready for queries.")
    
    user_query = st.text_input("Ask a question based on the topic:")
    
    if st.button("Generate Answer", type="primary"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            # Container to show step-by-step progress
            with st.spinner("Running Multi-Stage Retrieval..."):
                retriever = st.session_state.retriever
                
                # 1. Retrieve & Rerank
                final_context_chunks = retriever.retrieve(user_query)
                
                # 2. Construct Prompt
                context_str = "\n\n".join([
                    f"[Source: {c.metadata.url}]\n{c.text}" for c in final_context_chunks
                ])
                
                prompt = f"""Use the following context to answer the question. Cite your sources using the URLs provided.
                
                Context:
                {context_str}
                
                Question: {user_query}
                
                Answer:"""
                
                # 3. Generate Answer
                answer = llm.generate(prompt)
                
            # Display Final Answer
            st.markdown("### 🤖 Assistant Answer")
            st.write(answer)
            
            # Display Retrieved Sources in an expander for transparency
            with st.expander("🔍 View Retrieved Sources & Reranker Scores"):
                for idx, chunk in enumerate(final_context_chunks):
                    st.markdown(f"**Source {idx + 1}:** [{chunk.metadata.title}]({chunk.metadata.url})")
                    st.caption(f"Cross-Encoder Rerank Score: `{chunk.rerank_score:.4f}`")
                    st.text(chunk.text)
                    st.divider()