import os
from src.ingestion import DocumentIngestion
from src.indexer import VectorIndexer
from src.rag import RAGEngine


def test_full_rag_pipeline():
    pdf_path = os.path.join("data", "raw", "exemplo-ppgeec.pdf")
    if not os.path.exists(pdf_path):
        print(f"Arquivo {pdf_path} não encontrado.")
        return

    ingestion = DocumentIngestion(chunk_size=800, chunk_overlap=100)
    pages = ingestion.extract_text_from_pdf(pdf_path)
    chunks = ingestion.chunk_document(pages)

    indexer = VectorIndexer()
    indexer.build_index(chunks)

    rag = RAGEngine(indexer=indexer)

    q1 = "Quais são os documentos necessários para realizar a inscrição?"
    print(f"\n[PERGUNTA 1]: {q1}")
    res1 = rag.answer(q1, top_k=5)
    print(f"[RESPOSTA]:\n{res1['answer']}\n")
    print(f"[FONTES RECUPERADAS]: {len(res1['sources'])} trechos")
    for s in res1["sources"]:
        print(f" - Pág. {s['metadata']['page']} (Score: {s['similarity_score']:.3f}): {s['text'][:100]}...")

    q2 = "Qual a receita para fazer pão de queijo mineiro?"
    print(f"\n[PERGUNTA 2]: {q2}")
    res2 = rag.answer(q2, top_k=5)
    print(f"[RESPOSTA]:\n{res2['answer']}\n")


if __name__ == "__main__":
    test_full_rag_pipeline()