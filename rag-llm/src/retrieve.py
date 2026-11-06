
import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[1] / "embeddings" / "patients"


def retrieve(patient_id, query, top_k=3):
    # Restrict patient identifiers to safe directory names.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient_id):
        raise ValueError("Invalid patient ID")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be positive")

    folder = BASE / patient_id
    index_path = folder / "index.faiss"
    metadata_path = folder / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found for patient: {patient_id}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata["patient_id"] != patient_id:
        raise ValueError("Patient metadata mismatch")

    chunks = metadata["chunks"]

    # Only load indexes created by our trusted pipeline.
    index = faiss.read_index(str(index_path))

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count mismatch")

    print("Loading embedding model...")

    model = SentenceTransformer(
        metadata["model_name"]
    )

    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    if query_vector.shape[1] != index.d:
        raise ValueError("Embedding dimension mismatch")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        if chunk["patient_id"] != patient_id:
            raise ValueError("Cross-patient chunk detected")

        results.append({
            "score": float(score),
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient-id",
        required=True
    )

    parser.add_argument(
        "--query",
        required=True
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    args = parser.parse_args()

    results = retrieve(
        patient_id=args.patient_id,
        query=args.query,
        top_k=args.top_k
    )

    print("\n--- CARECOPILOT SEMANTIC RETRIEVAL ---")
    print("Patient:", args.patient_id)
    print("Question:", args.query)
    print("Results:", len(results))

    for i, result in enumerate(results, start=1):
        print(f"\n--- RESULT {i} ---")
        print("Similarity:", round(result["score"], 4))
        print("Source:", result["source_file"])
        print("Page:", result["page"])
        print("Text:", result["text"])
