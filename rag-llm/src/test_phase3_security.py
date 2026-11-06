"""CareCopilot Phase 3 / Step 7.12 -- offline safety regression tests.

Run from project root:
    python rag-llm/src/test_phase3_security.py

All synthetic fixtures are created in temporary directories. Tests do not
contact Groq/Hugging Face or modify the existing patient FAISS files.

Important: Passing these tests is NOT authentication, privacy certification,
or proof that real patient data are safe for a cloud LLM.
"""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import update_knowledge_base as updater
import retrieve as retrieval_module
import explain_timeline as timeline_module


ROOT = Path(__file__).resolve().parents[2]
MARKER = "SYNTHETIC MEDICAL REPORT - TEST ONLY"
GOOD_ID = "a" * 64
BAD_ID = "b" * 64


class DummyIndex:
    def __init__(self, total=1, dimension=384):
        self.ntotal = total
        self.d = dimension

    def search(self, _vector, _k):
        return (
            np.array([[0.8]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )


class DummyEmbeddingModel:
    def __init__(self, _model_name):
        pass

    def encode(self, _texts, **_kwargs):
        return np.ones((1, 384), dtype=np.float32)


class TestPhase3Safety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="carecopilot_security_")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def extracted_record(self, *, patient="demo_patient", marker=True, doc_id=GOOD_ID):
        body = (MARKER + "\nPatient: Demo Patient") if marker else "Real patient record"
        return {
            "patient_id": patient,
            "document_id": doc_id,
            "original_filename": "demo_report.pdf",
            "pages": [{"page": 1, "text": body}],
        }

    def write_extracted(self, directory, name, record):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(record), encoding="utf-8")

    def write_fake_index(self, *, patient="demo_patient", chunk_patient="demo_patient", doc_id=GOOD_ID,
                         chunk_ids=None):
        folder = self.base / "indexes" / "demo_patient"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_bytes(b"fake-index-not-loaded")
        if chunk_ids is None:
            chunk_ids = ["chunk-001"]
        chunks = [
            {"patient_id": chunk_patient, "document_id": doc_id,
             "chunk_id": chunk_id, "text": MARKER, "source_file": "demo_report.pdf", "page": 1}
            for chunk_id in chunk_ids
        ]
        (folder / "metadata.json").write_text(json.dumps({
            "patient_id": patient, "model_name": "mock-miniLM", "dimension": 384,
            "chunks": chunks,
        }), encoding="utf-8")
        return folder

    def test_01_rejects_other_patient_for_upload(self):
        with self.assertRaises(ValueError):
            updater.validate_pdf("nonexistent.pdf", "another_patient")

    def test_02_rejects_pdf_outside_approved_sample_folder(self):
        outside = self.base / "demo_outside.pdf"
        outside.write_bytes(b"fake pdf")
        with self.assertRaises(ValueError):
            updater.validate_pdf(str(outside), "demo_patient")

    def test_03_rejects_non_synthetic_pdf_inside_sample_folder(self):
        sample = self.base / "sample_reports"
        sample.mkdir()
        candidate = sample / "demo_untrusted.pdf"
        document = updater.pymupdf.open()
        try:
            page = document.new_page()
            page.insert_text((55, 80), "Patient: Demo Patient. Not a synthetic report.")
            document.save(str(candidate))
        finally:
            document.close()
        with patch.object(updater, "SAMPLES", sample):
            with self.assertRaisesRegex(ValueError, "synthetic"):
                updater.validate_pdf(str(candidate), "demo_patient")

    def test_04_rejects_cross_patient_extracted_record(self):
        directory = self.base / "extracted"
        self.write_extracted(directory, "report_foreign.json", self.extracted_record(patient="other_patient"))
        with self.assertRaisesRegex(ValueError, "association"):
            updater.read_existing_ids(directory)

    def test_05_rejects_non_synthetic_extracted_record(self):
        directory = self.base / "extracted"
        self.write_extracted(directory, "report_real.json", self.extracted_record(marker=False))
        with self.assertRaisesRegex(ValueError, "Non-synthetic"):
            updater.read_existing_ids(directory)

    def test_06_rejects_duplicate_document_ids(self):
        directory = self.base / "extracted"
        self.write_extracted(directory, "report_one.json", self.extracted_record())
        self.write_extracted(directory, "report_two.json", self.extracted_record())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            updater.read_existing_ids(directory)

    def test_07_rejects_malformed_document_digest(self):
        directory = self.base / "extracted"
        self.write_extracted(directory, "report_bad.json", self.extracted_record(doc_id="not-a-sha256"))
        with self.assertRaisesRegex(ValueError, "hash"):
            updater.read_existing_ids(directory)

    def test_08_rejects_index_owned_by_other_patient(self):
        folder = self.write_fake_index(patient="other_patient")
        with patch.object(updater.faiss, "read_index", return_value=DummyIndex()):
            with self.assertRaisesRegex(ValueError, "another patient"):
                updater.check_index(folder, {GOOD_ID})

    def test_09_rejects_cross_patient_index_chunk(self):
        folder = self.write_fake_index(chunk_patient="other_patient")
        with patch.object(updater.faiss, "read_index", return_value=DummyIndex()):
            with self.assertRaisesRegex(ValueError, "Cross-patient"):
                updater.check_index(folder, {GOOD_ID})

    def test_10_rejects_missing_or_mismatched_indexed_document(self):
        folder = self.write_fake_index(doc_id=BAD_ID)
        with patch.object(updater.faiss, "read_index", return_value=DummyIndex()):
            with self.assertRaisesRegex(ValueError, "document IDs"):
                updater.check_index(folder, {GOOD_ID})

    def test_11_rejects_cross_patient_faiss_retrieval_metadata(self):
        folder = self.write_fake_index(patient="other_patient")
        with patch.object(retrieval_module, "BASE", folder.parent):
            with self.assertRaisesRegex(ValueError, "Patient metadata mismatch"):
                retrieval_module.retrieve("demo_patient", "dizziness", top_k=1)

    def test_12_rejects_cross_patient_retrieved_chunk(self):
        folder = self.write_fake_index(chunk_patient="other_patient")
        with (patch.object(retrieval_module, "BASE", folder.parent),
              patch.object(retrieval_module.faiss, "read_index", return_value=DummyIndex()),
              patch.object(retrieval_module, "SentenceTransformer", DummyEmbeddingModel)):
            with self.assertRaisesRegex(ValueError, "Cross-patient chunk"):
                retrieval_module.retrieve("demo_patient", "dizziness", top_k=1)

    def test_13_rejects_unsafe_patient_identifier(self):
        with self.assertRaises(ValueError):
            retrieval_module.retrieve("../another_patient", "dizziness", top_k=1)

    def test_14_cloud_gate_rejects_non_synthetic_report(self):
        path = self.base / "rag-llm" / "knowledge_base" / "patients" / "demo_patient" / "extracted"
        self.write_extracted(path, "report_fake.json", self.extracted_record(marker=False))
        with patch.object(timeline_module, "ROOT", self.base):
            with self.assertRaisesRegex(ValueError, "Non-synthetic"):
                timeline_module.verify_demo_sources("demo_patient")

    def test_15_cloud_gate_rejects_unapproved_filename(self):
        path = self.base / "rag-llm" / "knowledge_base" / "patients" / "demo_patient" / "extracted"
        record = self.extracted_record()
        record["original_filename"] = "unexpected_document.pdf"
        self.write_extracted(path, "report_fake.json", record)
        with patch.object(timeline_module, "ROOT", self.base):
            with self.assertRaisesRegex(ValueError, "Unexpected report"):
                timeline_module.verify_demo_sources("demo_patient")

    def test_16_rollback_restores_all_existing_files(self):
        stage = self.base / "stage"
        production = self.base / "live"
        backup = self.base / "backups"
        stage.mkdir()
        production.mkdir()
        entries = []
        for n in range(5):
            source = stage / f"output_{n}.dat"
            target = production / f"output_{n}.dat"
            source.write_bytes(f"new-{n}".encode())
            target.write_bytes(f"old-{n}".encode())
            entries.append((source, target))
        with self.assertRaisesRegex(RuntimeError, "Simulated"):
            updater.publish_with_rollback(entries, backup, simulate_failure=True)
        for n, (_source, target) in enumerate(entries):
            self.assertEqual(target.read_bytes(), f"old-{n}".encode())


if __name__ == "__main__":
    print("\n--- CARECOPILOT PHASE 3 SECURITY / FAILURE REGRESSION ---")
    print("Mode: LOCAL TEST FIXTURES ONLY | Groq calls: 0 | Patient index writes: 0\n")
    unittest.main(verbosity=2)
