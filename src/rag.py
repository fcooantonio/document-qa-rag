import os
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def _get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    return api_key


class RAGEngine:
    def __init__(
        self,
        indexer,
        model_id: str = "qwen/qwen3.8-27b",
        similarity_threshold: float = 0.25
    ):
        self.indexer = indexer
        self.model_id = model_id
        self.similarity_threshold = similarity_threshold
        self.client = Groq(api_key=_get_groq_api_key())

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        formatted_context = ""
        for i, ctx in enumerate(contexts, start=1):
            source = ctx["metadata"]["source"]
            page = ctx["metadata"]["page"]
            formatted_context += f"[Trecho {i} | Arquivo: {source}, Pág. {page}]: {ctx['text']}\n\n"

        system_prompt = (
            "Você é um assistente técnico especializado em análise documental. "
            "Responda à pergunta do usuário utilizando EXCLUSIVAMENTE os trechos fornecidos no contexto abaixo. "
            "Regras obrigatórias:\n"
            "1. Cite o nome exato do arquivo e a página ao final das afirmações, por exemplo: (edital.pdf, Pág. 8).\n"
            "2. Responda com base nas informações encontradas nos trechos. Apenas se nenhuma parte da pergunta "
            "puder ser respondida pelo contexto, responda: "
            "'Não encontrei informações suficientes nos documentos fornecidos para responder a essa pergunta.'\n"
            "3. Não invente dados, datas ou interpretações externas ao texto."
        )

        user_prompt = f"Contexto:\n{formatted_context}\nPergunta: {query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved = self.indexer.search(query, top_k=top_k)

        valid_contexts = [
            c for c in retrieved if c["similarity_score"] >= self.similarity_threshold
        ]

        if not valid_contexts:
            return {
                "answer": "Não encontrei trechos com relevância suficiente nos documentos para responder a essa pergunta.",
                "sources": []
            }

        messages = self._build_prompt(query, valid_contexts)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": valid_contexts
        }