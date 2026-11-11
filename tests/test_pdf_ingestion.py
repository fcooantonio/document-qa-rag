import os
from src.ingestion import DocumentIngestion

def test_real_pdf():
    pdf_path = os.path.join("data", "raw", "exempl.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Ficheiro {pdf_path} não encontrado. Coloque um PDF nessa pasta primeiro.")
        return

    ingestion = DocumentIngestion(chunk_size=400, chunk_overlap=50)

    print(f"--> A processar: {pdf_path}")
    pages = ingestion.extract_text_from_pdf(pdf_path)
    print(f"Páginas com texto extraídas: {len(pages)}")

    chunks = ingestion.chunk_document(pages)
    print(f"Total de chunks gerados: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:2]):
        print(f"--- CHUNK {chunk['chunk_id']} ---")
        print(f"Fonte: {chunk['metadata']['source']} | Página: {chunk['metadata']['page']}")
        print(f"Extensão do texto: {len(chunk['text'])} caracteres")
        print(f"Excerto: {chunk['text'][:150]}...")
        print()

if __name__ == "__main__":
    test_real_pdf()