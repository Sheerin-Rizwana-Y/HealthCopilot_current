
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pymupdf as fitz

BASE = Path("rag-llm/knowledge_base/patients")


def make_demo_pdf():
    folder = BASE / "demo_patient" / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "demo_report.pdf"

    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "SYNTHETIC MEDICAL REPORT - TEST ONLY",
        "Patient: Demo Patient",
        "History: Previously reported fever.",
        "Symptoms: Dizziness and tiredness.",
        "Heart Rate: 96 bpm",
        "SpO2: 98 percent",
        "Temperature: 37.2 C",
        "Note: This is fictional test data."
    ]

    for i, line in enumerate(lines):
        page.insert_text(
            (50, 60 + i * 25),
            line,
            fontsize=11
        )

    doc.save(path)
    doc.close()
    return path


def ingest(pdf_path, patient_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient_id):
        raise ValueError("Invalid patient ID")

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("PDF file not found or invalid extension")

    pages = []

    with fitz.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError("Password-protected PDF is not supported")

        for i, page in enumerate(doc):
            text = page.get_text("text", sort=True).strip()
            pages.append({
                "page": i + 1,
                "text": text
            })

    if not any(p["text"] for p in pages):
        raise ValueError(
            "No extractable text. Scanned PDFs need OCR."
        )

    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    patient_folder = BASE / patient_id

    reports = patient_folder / "reports"
    extracted = patient_folder / "extracted"
    reports.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    report_name = f"report_{digest[:12]}.pdf"
    stored_pdf = reports / report_name

    if not stored_pdf.exists():
        shutil.copy2(pdf_path, stored_pdf)

    result = {
        "patient_id": patient_id,
        "document_id": digest,
        "original_filename": pdf_path.name,
        "stored_report": report_name,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages
    }

    output = extracted / f"report_{digest[:12]}.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n--- CARECOPILOT PDF INGESTION ---")
    print("Patient ID:", patient_id)
    print("Pages:", len(pages))
    print("Text characters:", sum(len(p["text"]) for p in pages))
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--patient-id")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        pdf = make_demo_pdf()
        patient = "demo_patient"
    else:
        if not args.pdf or not args.patient_id:
            parser.error(
                "Provide --pdf and --patient-id, or use --demo"
            )
        pdf = args.pdf
        patient = args.patient_id

    ingest(pdf, patient)
