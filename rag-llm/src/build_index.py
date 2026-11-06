
import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path("rag-llm")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(patient_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient_id):
        raise ValueError("Invalid patient ID")

    chunks_path = (
        BASE / "knowledge_base" / "patients"
        / patient_id / "chunks" / "chunks.jsonl"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}"
        )

    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)

                if chunk["patient_id"] != patient_id:
                    raise ValueError("Patient mismatch")

                if chunk["text"].strip():
                    chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid chunks available")

    texts = [chunk["text"] for chunk in chunks]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    vectors = np.asarray(vectors, dtype="float32")

    dimension = vectors.shape[1]

    # Normalized vectors + inner product =
    # cosine similarity search.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    output = BASE / "embeddings" / "patients" / patient_id
    output.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(output / "index.faiss")
    )

    metadata = {
        "patient_id": patient_id,
        "model_name": MODEL_NAME,
        "dimension": dimension,
        "chunks": chunks
    }

    (output / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n--- CARECOPILOT FAISS INDEX ---")
    print("Patient:", patient_id)
    print("Embedding model:", MODEL_NAME)
    print("Total chunks:", len(chunks))
    print("Vector dimension:", dimension)
    print("FAISS vectors:", index.ntotal)
    print("Index saved:", output / "index.faiss")
    print("Metadata saved:", output / "metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()

    build_index(args.patient_id)
