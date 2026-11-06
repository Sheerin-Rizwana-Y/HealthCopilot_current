
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path("rag-llm/knowledge_base/patients")


def chunk_text(text, chunk_size=120, overlap=25):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    words = text.split()
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            yield chunk

        if start + chunk_size >= len(words):
            break


def process_patient(patient_id):
    if (
        not patient_id
        or not all(c.isalnum() or c in "_-" for c in patient_id)
        or len(patient_id) > 64
        or not patient_id.isascii()
    ):
        raise ValueError("Invalid patient ID")

    patient_dir = BASE / patient_id
    extracted_dir = patient_dir / "extracted"

    files = sorted(extracted_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No extracted reports for {patient_id}"
        )

    chunks = []

    for file in files:
        report = json.loads(file.read_text(encoding="utf-8"))

        if report["patient_id"] != patient_id:
            raise ValueError("Patient ID mismatch")

        document_id = report["document_id"]

        for page in report["pages"]:
            for index, text in enumerate(
                chunk_text(page["text"])
            ):
                chunk_id = hashlib.sha256(
                    f"{document_id}:{page['page']}:{index}".encode()
                ).hexdigest()[:16]

                chunks.append({
                    "chunk_id": chunk_id,
                    "patient_id": patient_id,
                    "document_id": document_id,
                    "source_file": report["original_filename"],
                    "page": page["page"],
                    "text": text
                })

    output_dir = patient_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / "chunks.jsonl"

    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n--- CARECOPILOT DOCUMENT CHUNKING ---")
    print("Patient:", patient_id)
    print("Reports:", len(files))
    print("Chunks:", len(chunks))
    print("Output:", output)

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0]["text"][:400])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    args = parser.parse_args()

    process_patient(args.patient_id)
