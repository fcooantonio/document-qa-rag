import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader


class DocumentIngestion:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"PRPPG: Edital.*?pg\.\s*\d+", "", text)
        text = re.sub(r"Rua Coronel Estanislau Frota.*?4607", "", text)
        return " ".join(text.split())

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

        reader = PdfReader(pdf_path)
        pages_content = []
        doc_name = os.path.basename(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            text = self._clean_text(raw_text)
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
        chunks = []
        chunk_id = 0

        for page in pages_content:
            text = page["text"]
            metadata = page["metadata"]
            text_len = len(text)
            start = 0

            while start < text_len:
                end = min(start + self.chunk_size, text_len)

                # Alinha o corte no último espaço para evitar quebrar palavras
                if end < text_len:
                    last_space = text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space

                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "source": metadata["source"],
                            "page": metadata["page"],
                            "char_range": f"{start}-{end}"
                        }
                    })
                    chunk_id += 1

                if end >= text_len:
                    break

                next_start = max(end - self.chunk_overlap, start + 1)
                if next_start > 0:
                    first_space = text.find(" ", next_start, end)
                    if first_space != -1:
                        next_start = first_space + 1
                start = next_start

        return chunks