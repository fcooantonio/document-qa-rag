---
title: Document QA RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Document Q&A — RAG Engine

Sistema de Recuperação e Geração Aumentada (RAG) para consulta em documentos PDF técnicos e normativos, com busca híbrida (vetorial + lexical), controle de limiar de relevância e citação explícita de arquivo e página.

**Aplicação ao vivo:** [document-app-rag.streamlit.app](https://document-app-rag-tet2jtrbcqq5zzr88bfrub.streamlit.app/)

---

## Arquitetura do Sistema

O projeto foi estruturado de forma modular para separar ingestão de dados, indexação vetorial, orquestração da LLM e camada de interface:

```text
document-qa-rag/
├── app.py                   # Interface web interativa (Streamlit)
├── src/
│   ├── ingestion.py         # Extração de PDF, limpeza via regex e chunking alinhado por palavras
│   ├── indexer.py           # Indexação FAISS (IndexFlatIP) e re-ranking híbrido
│   └── rag.py               # Construção de prompt com guardrails e cliente Groq
├── tests/
│   ├── test_pdf_ingestion.py
│   ├── test_indexer.py
│   └── test_rag_e2e.py      # Validação ponta a ponta (escopo válido vs. fora de contexto)
├── data/
│   └── raw/                 # Documento de exemplo (Edital PPGEEC/UFC)
└── requirements.txt
```

### Fluxo de Processamento

1. **Ingestão (`src/ingestion.py`)**: Lê o PDF página a página via `pypdf`, remove ruídos institucionais recorrentes (rodapés e carimbos administrativos) e segmenta o texto em blocos de `800` caracteres com `100` caracteres de sobreposição (*overlap*), alinhando os cortes nos limites de palavras para evitar fragmentação semântica. Cada chunk preserva os metadados `source`, `page` e `char_range`.
2. **Busca Híbrida (`src/indexer.py`)**:
   * **Busca Densa**: Gera embeddings de 384 dimensões utilizando o modelo multilíngue `paraphrase-multilingual-MiniLM-L12-v2`, normalizados pela norma $L_2$ em um índice `faiss.IndexFlatIP` (equivalente à similaridade de cosseno).
   * **Re-ranking Lexical**: Combina o score vetorial ($75\%$) com a sobreposição normalizada de tokens sem acentos ($25\%$), aumentando a precisão na recuperação de termos específicos de editais e especificações técnicas.
3. **Geração Fundamentada (`src/rag.py`)**: Filtra trechos abaixo do limiar mínimo de similaridade (`similarity_threshold`) antes de acionar a LLM (`qwen/qwen3.8-27b` via API da Groq). O *system prompt* restringe a resposta estritamente ao contexto recuperado e exige a citação no formato `(arquivo.pdf, Pág. X)`.

---

## Execução Local

### 1. Clonar o repositório e ativar o ambiente virtual

```bash
git clone https://github.com/fcooantonio/document-qa-rag.git
cd document-qa-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar a variável de ambiente

Crie um arquivo `.env` na raiz do projeto com a sua chave da API da Groq:

```env
GROQ_API_KEY=gsk_sua_chave_aqui
```

### 3. Executar a aplicação

```bash
streamlit run app.py
```

Para executar os testes de validação do pipeline:

```bash
python3 -m tests.test_rag_e2e
```

---

## Decisões Técnicas e Trade-offs

* **Embeddings Multilíngues**: Durante os testes com documentos em português, o modelo padrão `all-MiniLM-L6-v2` apresentou queda de recall devido ao viés para inglês. A migração para `paraphrase-multilingual-MiniLM-L12-v2` elevou a separação entre trechos relevantes e consultas fora de escopo.
* **Consultas Pontuais vs. Agregação Global**: Como o recuperador opera sobre os *Top-K* trechos mais similares, o sistema apresenta alta precisão em perguntas factuais e normativas (prazos, critérios, regras específicas), mas possui restrição natural em perguntas de contagem ou listagem exaustiva distribuída ao longo de dezenas de páginas (ex.: *"Quantos requisitos existem no total?"*), onde técnicas como *Parent-Document Retrieval* ou *Map-Reduce* seriam necessárias como evolução futura.