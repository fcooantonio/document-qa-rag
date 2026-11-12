import os
import tempfile
import streamlit as st
from src.ingestion import DocumentIngestion
from src.indexer import VectorIndexer
from src.rag import RAGEngine

st.set_page_config(page_title="Document Q&A RAG", page_icon="📄", layout="wide")


@st.cache_resource
def load_embedding_model():
    return VectorIndexer(model_name="paraphrase-multilingual-MiniLM-L12-v2")


def index_documents(file_targets):
    ingestion = DocumentIngestion(chunk_size=800, chunk_overlap=100)
    all_chunks = []

    for path, original_name in file_targets:
        pages = ingestion.extract_text_from_pdf(path)
        for p in pages:
            p["metadata"]["source"] = original_name
        chunks = ingestion.chunk_document(pages)
        all_chunks.extend(chunks)

    indexer = load_embedding_model()
    indexer.build_index(all_chunks)
    return indexer, len(all_chunks)


st.title("Document Q&A — RAG Engine")
st.caption("Busca híbrida (FAISS + Lexical) com geração fundamentada e citação de páginas.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexer_ready" not in st.session_state:
    st.session_state.indexer_ready = False

with st.sidebar:
    st.header("Base de Conhecimento")
    uploaded_files = st.file_uploader(
        "Envie arquivos PDF", type=["pdf"], accept_multiple_files=True
    )

    sample_path = os.path.join("data", "raw", "exemplo-ppgeec.pdf")
    use_sample = False
    if os.path.exists(sample_path):
        use_sample = st.checkbox("Usar edital de exemplo (PPGEEC UFC)", value=True)

    if st.button("Processar e Indexar", type="primary", use_container_width=True):
        targets = []
        temp_paths = []

        if uploaded_files:
            for uf in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(uf.read())
                tmp.close()
                targets.append((tmp.name, uf.name))
                temp_paths.append(tmp.name)

        if use_sample and os.path.exists(sample_path):
            targets.append((sample_path, "exemplo-ppgeec.pdf"))

        if not targets:
            st.warning("Selecione ao menos um PDF antes de indexar.")
        else:
            with st.spinner("Extraindo texto e gerando embeddings..."):
                indexer, total_chunks = index_documents(targets)
                st.session_state.indexer = indexer
                st.session_state.indexer_ready = True
                st.session_state.messages = []
                st.success(f"Indexação concluída: {total_chunks} chunks prontos.")

            for tmp_path in temp_paths:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.divider()
    st.subheader("Parâmetros de Busca")
    top_k = st.slider("Top-K trechos recuperados", min_value=1, max_value=10, value=5)
    threshold = st.slider(
        "Limiar mínimo de similaridade", min_value=0.0, max_value=0.8, value=0.25, step=0.05
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Fontes e scores recuperados"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['metadata']['source']} — Pág. {s['metadata']['page']}** "
                        f"(Score híbrido: `{s['similarity_score']:.3f}`)"
                    )
                    st.caption(s["text"])

if prompt := st.chat_input("Faça uma pergunta sobre os documentos indexados..."):
    if not st.session_state.indexer_ready:
        st.warning("Processe um documento na barra lateral antes de perguntar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando trechos e gerando resposta..."):
                rag = RAGEngine(
                    indexer=st.session_state.indexer,
                    similarity_threshold=threshold
                )
                result = rag.answer(prompt, top_k=top_k)
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Fontes e scores recuperados"):
                        for s in result["sources"]:
                            st.markdown(
                                f"**{s['metadata']['source']} — Pág. {s['metadata']['page']}** "
                                f"(Score híbrido: `{s['similarity_score']:.3f}`)"
                            )
                            st.caption(s["text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })