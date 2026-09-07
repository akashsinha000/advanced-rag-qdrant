from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, results, top_k=2):
    pairs = [
        [query, result.payload["text"]]
        for result in results
    ]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (result, float(score))
        for result, score in ranked[:top_k]
    ]