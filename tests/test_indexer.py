from src.indexer import VectorIndexer


def test_indexer_semantic_filtering():
    sample_chunks = [
        {
            "chunk_id": 0,
            "text": "O Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPGEEC) da UFC Sobral tem inscrições abertas.",
            "metadata": {"source": "edital.pdf", "page": 1}
        },
        {
            "chunk_id": 1,
            "text": "Critérios de avaliação para aprovação de bolsas de iniciação científica e acadêmica.",
            "metadata": {"source": "edital.pdf", "page": 5}
        },
        {
            "chunk_id": 2,
            "text": "Lista de ingredientes culinários para preparo de bolos de fubá com goiabada.",
            "metadata": {"source": "receitas.pdf", "page": 12}
        }
    ]

    indexer = VectorIndexer()
    indexer.build_index(sample_chunks)

    query = "Onde ocorrem as inscrições do mestrado da UFC em Sobral?"
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == 0
    assert results[0]["similarity_score"] > 0.4

    print(f"OK - Chunk {results[0]['chunk_id']} | Score: {results[0]['similarity_score']:.4f}")


if __name__ == "__main__":
    test_indexer_semantic_filtering()