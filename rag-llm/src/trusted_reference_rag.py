"""CareCopilot Phase 3: separate, source-attributed public medical-reference RAG.

Commands:
    python rag-llm/src/trusted_reference_rag.py --self-test
    python rag-llm/src/trusted_reference_rag.py build
    python rag-llm/src/trusted_reference_rag.py verify
    python rag-llm/src/trusted_reference_rag.py search --query "What is dizziness and vertigo?" --top-k 3

Stores only general, manually curated paraphrases of MedlinePlus health-topic
summaries. It does not fetch web pages, contain patient information, contact
Groq, or offer diagnoses. Review source pages periodically for changes.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "rag-llm" / "embeddings" / "trusted_medical"
ACTIVE = STORE / "CURRENT.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SOURCE_NAME = "MedlinePlus, National Library of Medicine"
REVIEWED_ON = "2026-10-20"  # Manual prototype curation, NOT the website's update date.

# Each short summary is a human-authored paraphrase of the linked health-topic
# overview. It is NOT a full article copy or patient-specific medical advice.
RECORDS = [
    {
        "topic": "Dizziness and vertigo",
        "keywords": "dizziness, vertigo, lightheaded, spinning, unsteady, balance",
        "url": "https://medlineplus.gov/dizzinessandvertigo.html",
        "summary": (
            "Dizziness can feel like lightheadedness or unsteadiness, while vertigo "
            "involves a sensation of spinning. Possible causes mentioned in the "
            "health-topic overview include dehydration, standing up too quickly, "
            "some medicines, and inner-ear problems. The symptom alone does not "
            "establish its cause."
        ),
    },
    {
        "topic": "Cough",
        "keywords": "cough, coughing, acute cough, chronic cough, airways",
        "url": "https://medlineplus.gov/cough.html",
        "summary": (
            "Coughing is a protective reflex that helps keep the throat and airways "
            "clear. A cough can be short-lived or persistent and has a variety of "
            "possible causes, including infections and allergies. A mention of "
            "cough in a report is not a diagnosis."
        ),
    },
    {
        "topic": "Headache",
        "keywords": "headache, head pain, migraine, tension headache",
        "url": "https://medlineplus.gov/headache.html",
        "summary": (
            "Headaches are common and include different types such as tension "
            "headaches and migraines. A headache can have different causes, so "
            "a report of head pain by itself does not identify the cause."
        ),
    },
    {
        "topic": "Nausea and vomiting",
        "keywords": "nausea, vomiting, sick to stomach, emesis",
        "url": "https://medlineplus.gov/nauseaandvomiting.html",
        "summary": (
            "Nausea means feeling sick to the stomach, as if vomiting may occur. "
            "It can have many causes, including motion sickness, medicines, "
            "and different medical conditions. Its presence alone does not "
            "identify the cause."
        ),
    },
    {
        "topic": "Fatigue and tiredness",
        "keywords": "fatigue, tiredness, lack of energy, weariness",
        "url": "https://medlineplus.gov/fatigue.html",
        "summary": (
            "Fatigue is a feeling of tiredness, weariness or lack of energy. "
            "It may relate to lifestyle factors, medicines, or physical or mental "
            "health conditions. Fatigue is a symptom, not a diagnosis."
        ),
    },
]


def make_documents():
    """Validate and transform the curated source manifest."""
    documents = []
    seen_urls = set()
    for row in RECORDS:
        parsed = urlsplit(row["url"])
        if parsed.scheme != "https" or parsed.hostname != "medlineplus.gov":
            raise ValueError("Non-allowlisted medical reference URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Unexpected data in medical reference URL")
        if row["url"] in seen_urls:
            raise ValueError("Duplicate reference URL")
        seen_urls.add(row["url"])
        if not all(row.get(k, "").strip() for k in ("topic", "keywords", "summary")):
            raise ValueError("Incomplete curated reference")
        if len(row["summary"]) > 900:
            raise ValueError("Reference summary too long")
        doc_id = hashlib.sha256(row["url"].encode("utf-8")).hexdigest()[:16]
        documents.append({
            "ref_id": doc_id,
            "topic": row["topic"],
            "keywords": row["keywords"],
            "summary": row["summary"],
            "source_url": row["url"],
            "source_name": SOURCE_NAME,
            "reviewed_on": REVIEWED_ON,
            "content_type": "manually_curated_paraphrase_of_public_health_topic_summary",
            "text": f"Topic: {row['topic']}. Keywords: {row['keywords']}. "
                    f"General reference: {row['summary']}",
        })
    return documents


def fingerprint(docs):
    raw = json.dumps(docs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def read_active():
    if not ACTIVE.exists():
        raise FileNotFoundError("Trusted reference index not built. Run 'build' first.")
    pointer = json.loads(ACTIVE.read_text(encoding="utf-8"))
    version = pointer.get("version")
    if not isinstance(version, str) or not re.fullmatch(r"snapshot_[0-9a-f]{16}", version):
        raise ValueError("Invalid trusted-reference snapshot pointer")
    folder = STORE / version
    return folder / "index.faiss", folder / "metadata.json", version


def build():
    # Expensive dependencies are imported only for build/search/verification.
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    docs = make_documents()
    version = "snapshot_" + fingerprint(docs)
    STORE.mkdir(parents=True, exist_ok=True)
    target = STORE / version
    if target.exists():
        if not (target / "index.faiss").is_file() or not (target / "metadata.json").is_file():
            raise ValueError("Existing snapshot is incomplete: " + str(target))
    else:
        print("Loading existing sentence embedding model...")
        model = SentenceTransformer(MODEL)
        vectors = model.encode(
            [d["text"] for d in docs],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(docs) or vectors.shape[1] != 384:
            raise ValueError("Unexpected embedding output shape")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        metadata = {
            "index_type": "general_medical_reference_not_patient_data",
            "model_name": MODEL,
            "dimension": int(vectors.shape[1]),
            "documents": docs,
        }
        # Build into a temporary directory; never mutate active snapshots.
        with tempfile.TemporaryDirectory(prefix="ref_stage_", dir=STORE) as work:
            workdir = Path(work)
            faiss.write_index(index, str(workdir / "index.faiss"))
            (workdir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Move two completed files as a single new directory.
            os.replace(workdir, target)
    # Pointer is the only mutable active-index file, replaced atomically.
    tmp_pointer = STORE / "CURRENT.tmp"
    tmp_pointer.write_text(json.dumps({"version": version}), encoding="utf-8")
    os.replace(tmp_pointer, ACTIVE)
    print("\n--- TRUSTED MEDICAL REFERENCE INDEX BUILT ---")
    print("References:", len(docs))
    print("Snapshot:", version)
    print("Source: MedlinePlus / NLM")
    print("Patient reports modified: NO")
    print("Groq requests: 0")


def verify():
    import faiss
    docs = make_documents()
    index_path, meta_path, version = read_active()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("index_type") != "general_medical_reference_not_patient_data":
        raise AssertionError("Incorrect index collection type")
    if metadata["documents"] != docs:
        raise AssertionError("Index does not match current reviewed source manifest")
    if metadata["model_name"] != MODEL or metadata["dimension"] != 384:
        raise AssertionError("Embedding model/dimensions mismatch")
    if version != "snapshot_" + fingerprint(docs):
        raise AssertionError("Unexpected active snapshot version")
    index = faiss.read_index(str(index_path))  # only load locally generated FAISS files
    if index.ntotal != len(docs) or index.d != metadata["dimension"]:
        raise AssertionError("Index and reference metadata mismatch")
    if len({d["ref_id"] for d in docs}) != len(docs):
        raise AssertionError("Duplicate reference IDs")
    print("\n--- TRUSTED REFERENCE VERIFICATION ---")
    print("References:", len(docs))
    print("FAISS vectors:", index.ntotal)
    print("Dimensions:", index.d)
    print("Patient-specific data:", "NONE")
    print("Source attribution:", SOURCE_NAME)
    print("VERIFICATION: PASS")


def search(query, top_k):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    index_path, meta_path, _ = read_active()
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    docs = metadata["documents"]
    if index.ntotal != len(docs) or index.d != metadata["dimension"]:
        raise ValueError("Reference index inconsistent; run verify")
    model = SentenceTransformer(metadata["model_name"])
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    vector = np.ascontiguousarray(vector, dtype="float32")
    distances, indices = index.search(vector, min(top_k, index.ntotal))
    print("\n--- TRUSTED MEDICAL REFERENCE SEARCH ---")
    print("Question:", query)
    for order, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if idx < 0:
            continue
        d = docs[int(idx)]
        print(f"\n[REF{order}] {d['topic']}")
        print("Similarity:", round(float(score), 4))
        print("Summary:", d["summary"])
        print("Source:", d["source_name"])
        print("URL:", d["source_url"])
        print("Curated on:", d["reviewed_on"])
    print("\nReminder: general education only, not a patient diagnosis or triage system.")


def self_test():
    docs = make_documents()
    assert len(docs) == 5
    assert len({d["ref_id"] for d in docs}) == 5
    assert all(d["source_name"] == SOURCE_NAME for d in docs)
    assert all("patient_id" not in d for d in docs)
    assert len(fingerprint(docs)) == 16
    assert {d["topic"] for d in docs} == {
        "Dizziness and vertigo", "Cough", "Headache", "Nausea and vomiting", "Fatigue and tiredness"
    }
    print("SELF-TEST: PASS (5 references, unique IDs, approved URLs, no patient fields)")


def main():
    parser = argparse.ArgumentParser(description="Separate public-medical-reference RAG index")
    parser.add_argument("command", nargs="?", choices=["build", "verify", "search"])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.command == "build":
        build()
    elif args.command == "verify":
        verify()
    elif args.command == "search":
        if not args.query:
            parser.error("search requires --query")
        search(args.query, args.top_k)
    else:
        parser.error("Use build, verify, search or --self-test")


if __name__ == "__main__":
    main()
