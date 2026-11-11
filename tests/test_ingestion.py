import os
from src.ingestion import DocumentIngestion

def test_chunking_logic():
    ingestion = DocumentIngestion(chunk_size=100, chunk_overlap=20)
    fake_pages = [
        {
            "text": "A Universidade Federal do Ceará oferece o curso de Engenharia da Computação em Sobral com foco em software e sistemas embarcados.",
            "metadata": {"source": "manual_curso.pdf", "page": 1}
        }
    ]
    chunks = ingestion.chunk_document(fake_pages)
    
    assert len(chunks) > 0, "Deveria ter gerado ao menos um chunk"
    assert "source" in chunks[0]["metadata"], "Metadado de origem ausente"
    assert "page" in chunks[0]["metadata"], "Metadado de página ausente"
    print(f"Sucesso: {len(chunks)} chunks gerados com integridade de metadados.")

if __name__ == "__main__":
    test_chunking_logic()