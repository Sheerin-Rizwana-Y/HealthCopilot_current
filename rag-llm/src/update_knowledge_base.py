"""Staged synthetic PDF -> patient FAISS index updater.

Stages the complete rebuild, verifies it, and uses backups for recoverable
publication failures. Not safe for concurrent readers or machine power loss.
Only synthetic demo_patient PDFs in rag-llm/sample_reports are accepted.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import pymupdf


ROOT = Path(__file__).resolve().parents[2]
RAG = ROOT / "rag-llm"
SRC = RAG / "src"
SAMPLES = RAG / "sample_reports"
PATIENT_ID = "demo_patient"
MARKER = "SYNTHETIC MEDICAL REPORT - TEST ONLY"


def validate_pdf(pdf_arg: str, patient_id: str) -> tuple[Path, str]:
    if patient_id != PATIENT_ID:
        raise ValueError("Only synthetic demo_patient is supported here.")

    pdf_path = Path(pdf_arg).resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Choose an existing PDF file.")
    if not pdf_path.is_relative_to(SAMPLES.resolve()):
        raise ValueError("PDF must be inside rag-llm/sample_reports.")
    if not re.fullmatch(r"demo_[A-Za-z0-9_.-]+\.pdf", pdf_path.name):
        raise ValueError("Only demo_*.pdf filenames are allowed.")
    if not 0 < pdf_path.stat().st_size <= 5 * 1024 * 1024:
        raise ValueError("PDF must be nonempty and at most 5 MB.")

    try:
        with pymupdf.open(pdf_path) as document:
            if document.needs_pass or len(document) == 0:
                raise ValueError("Encrypted or empty PDF is not supported.")
            text = "\n".join(page.get_text("text") for page in document)
    except pymupdf.FileDataError as exc:
        raise ValueError("Unable to read PDF.") from exc

    if MARKER not in text or "Patient: Demo Patient" not in text:
        raise ValueError("This test accepts synthetic Demo Patient PDFs only.")
    return pdf_path, hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def read_existing_ids(extracted_dir: Path) -> set[str]:
    ids = set()
    for path in sorted(extracted_dir.glob("report_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("patient_id") != PATIENT_ID:
            raise ValueError(f"Invalid patient association: {path.name}")
        pages = record.get("pages")
        if not isinstance(pages, list) or MARKER not in "\n".join(
            page.get("text", "") for page in pages
        ):
            raise ValueError(f"Non-synthetic report found: {path.name}")
        identifier = record.get("document_id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise ValueError(f"Invalid document hash: {path.name}")
        if identifier in ids:
            raise ValueError("Duplicate document IDs in extracted directory.")
        ids.add(identifier)
    return ids


def check_index(index_dir: Path, expected_ids: set[str]) -> tuple[int, int]:
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = metadata["chunks"]
    if metadata.get("patient_id") != PATIENT_ID:
        raise ValueError("Index belongs to another patient.")
    if not chunks or index.ntotal != len(chunks) or index.d != metadata["dimension"]:
        raise ValueError("FAISS index and metadata are inconsistent.")
    if any(c.get("patient_id") != PATIENT_ID for c in chunks):
        raise ValueError("Cross-patient chunk in index.")
    if len({c.get("chunk_id") for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs in FAISS metadata.")
    actual_ids = {c["document_id"] for c in chunks}
    if actual_ids != expected_ids:
        raise ValueError("Indexed document IDs do not match extracted reports.")
    return len(chunks), index.d


def run_script(filename: str, stage_root: Path, *args: str) -> None:
    print(f"[Stage] Running {filename} ...", flush=True)
    subprocess.run(
        [sys.executable, str(SRC / filename), *args],
        cwd=stage_root,
        check=True,
    )


def publish_with_rollback(
    staged_files: list[tuple[Path, Path]], backup_dir: Path, simulate_failure: bool
) -> None:
    # Back up ALL targets before replacing any of them.
    saved = []
    for number, (source, target) in enumerate(staged_files):
        if not source.is_file():
            raise FileNotFoundError(f"Missing staged output: {source}")
        backup = backup_dir / f"{number:02d}.bak"
        if target.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            saved.append((target, backup))
        else:
            saved.append((target, None))

    try:
        for number, (source, target) in enumerate(staged_files):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            # Exercise rollback after the FAISS index has been published.
            if simulate_failure and number == 3:
                raise RuntimeError("Simulated mid-publish failure (testing rollback)")
    except Exception as exc:
        rollback_errors = []
        for target, backup in saved:
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup, target)
            except Exception as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "Publication failed AND rollback had errors: "
                + "; ".join(rollback_errors)
            ) from exc
        print("[Recovery] Previous production files restored.", flush=True)
        raise


def update(pdf_arg: str, patient_id: str, force: bool, simulate_failure: bool) -> None:
    pdf_path, digest = validate_pdf(pdf_arg, patient_id)
    patient_dir = RAG / "knowledge_base" / "patients" / patient_id
    extracted_dir = patient_dir / "extracted"
    index_dir = RAG / "embeddings" / "patients" / patient_id
    previous_ids = read_existing_ids(extracted_dir)

    if digest in previous_ids and not force:
        try:
            chunks, dim = check_index(index_dir, previous_ids)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError, OSError):
            print("[Repair] Duplicate extracted report, but index needs rebuilding.")
        else:
            print("\nDUPLICATE: report already ingested and indexed; rebuild skipped.")
            print(f"Reports: {len(previous_ids)} | Chunks: {chunks} | Dimension: {dim}")
            return

    staging_parent = RAG / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kb_", dir=staging_parent) as tmp:
        stage_root = Path(tmp)
        stage_patient = stage_root / "rag-llm" / "knowledge_base" / "patients" / patient_id
        stage_index = stage_root / "rag-llm" / "embeddings" / "patients" / patient_id

        for folder_name in ("extracted", "reports"):
            production_folder = patient_dir / folder_name
            if production_folder.exists():
                shutil.copytree(production_folder, stage_patient / folder_name)

        run_script("ingest_pdf.py", stage_root, "--pdf", str(pdf_path), "--patient-id", patient_id)
        run_script("chunk_documents.py", stage_root, "--patient-id", patient_id)
        run_script("build_index.py", stage_root, "--patient-id", patient_id)

        staged_ids = read_existing_ids(stage_patient / "extracted")
        if staged_ids != (previous_ids | {digest}):
            raise ValueError("Staging introduced or lost document IDs.")
        chunks, dimension = check_index(stage_index, staged_ids)
        print(f"[Verification] Staged {len(staged_ids)} reports / {chunks} chunks / {dimension} dimensions.")

        pdf_name = f"report_{digest[:12]}.pdf"
        json_name = f"report_{digest[:12]}.json"
        staged_files = [
            (stage_patient / "reports" / pdf_name, patient_dir / "reports" / pdf_name),
            (stage_patient / "extracted" / json_name, extracted_dir / json_name),
            (stage_patient / "chunks" / "chunks.jsonl", patient_dir / "chunks" / "chunks.jsonl"),
            (stage_index / "index.faiss", index_dir / "index.faiss"),
            (stage_index / "metadata.json", index_dir / "metadata.json"),
        ]
        publish_with_rollback(staged_files, stage_root / "backups", simulate_failure)

    production_ids = read_existing_ids(extracted_dir)
    final_chunks, final_dim = check_index(index_dir, production_ids)
    print("\n--- KNOWLEDGE BASE UPDATE VERIFIED ---")
    print("Patient:", patient_id)
    print("Total reports:", len(production_ids))
    print("FAISS vectors:", final_chunks)
    print("Vector dimensions:", final_dim)
    print("Report indexed: YES")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--simulate-failure", action="store_true")
    args = parser.parse_args()

    if args.simulate_failure and not args.force_rebuild:
        parser.error("--simulate-failure requires --force-rebuild")
    try:
        update(args.pdf, args.patient_id, args.force_rebuild, args.simulate_failure)
    except Exception as exc:
        print("\nUPDATE FAILED:", type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
