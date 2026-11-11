import os
from typing import List, Dict, Any
from pypdf import PdfReader


class DocumentIngestion:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Lê um PDF e retorna uma lista de dicionários com texto e metadados por página."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

        reader = PdfReader(pdf_path)
        pages_content = []
        doc_name = os.path.basename(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text:
                pages_content.append({
                    "text": text,
                    "metadata": {
                        "source": doc_name,
                        "page": page_num
                    }
                })
        return pages_content

    def chunk_document(self, pages_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aplica segmentação com overlap preservando metadados de página e documento."""
        chunks = []
        chunk_id = 0

        for page in pages_content:
            text = page["text"]
            metadata = page["metadata"]
            
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "char_range": f"{start}-{min(end, len(text))}"
                    }
                })
                chunk_id += 1
                start += self.chunk_size - self.chunk_overlap

        return chunks