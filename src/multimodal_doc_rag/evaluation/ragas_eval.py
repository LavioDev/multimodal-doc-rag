"""RAGAS context metrics, configured to use the project's OpenRouter reader."""


def evaluate_rag(records: list[dict[str, str]]) -> dict[str, float]:
    """Evaluate text contexts; RAGAS judge calls use OpenRouter and API_KEY."""
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics.collections import answer_relevancy, context_precision, faithfulness

    from multimodal_doc_rag.config import get_settings

    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError("Set API_KEY to run RAGAS through OpenRouter")
    if not records:
        raise ValueError("RAGAS input contains no records")
    if any(not row.get("reference", row.get("ground_truth")) for row in records):
        raise ValueError("Each RAGAS record needs reference or ground_truth for context precision")
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            temperature=0,
        )
    )

    class E5Embeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            from multimodal_doc_rag.index import embed_texts

            return embed_texts(texts, prefix="passage")

        def embed_query(self, text: str) -> list[float]:
            from multimodal_doc_rag.index import embed_texts

            return embed_texts([text], prefix="query")[0]

    embeddings = LangchainEmbeddingsWrapper(E5Embeddings())
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row.get("contexts", []),
                "reference": row.get("reference", row.get("ground_truth", "")),
            }
            for row in records
        ]
    )
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    return {
        metric: float(frame[metric].dropna().mean())
        for metric in frame.columns
        if metric in {"faithfulness", "answer_relevancy", "context_precision"}
        and frame[metric].notna().any()
    }
