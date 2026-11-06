
import unittest

from explain_timeline import build_comparison
from compare_symptoms import normalize, extract_report_symptoms
from report_timeline import extract_report_date


PATIENT_ID = "demo_patient"


class TestCareCopilotHistory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.facts, cls.sources = build_comparison(
            PATIENT_ID
        )

        cls.comparisons = {
            normalize(record["symptom"]): record
            for record in cls.facts["caregiver_comparisons"]
        }

        cls.sources_by_ref = {
            source["reference"]: source
            for source in cls.sources
        }

    def test_01_reports_and_dates(self):
        self.assertEqual(len(self.sources), 2)

        reports = {
            s["file"]: s for s in self.sources
        }

        self.assertEqual(
            reports[
                "demo_followup_2026-10-10.pdf"
            ]["date"],
            "2026-10-10"
        )

        self.assertIsNone(
            reports["demo_report.pdf"]["date"]
        )

    def test_02_caregiver_symptoms(self):
        self.assertEqual(
            set(self.comparisons),
            {"dizziness", "cough"}
        )

        self.assertEqual(
            self.comparisons["dizziness"]["severity"],
            "moderate"
        )

        self.assertEqual(
            self.comparisons["cough"]["severity"],
            "mild"
        )

    def test_03_report_matches(self):
        expected = {
            "dizziness": "demo_report.pdf",
            "cough": "demo_followup_2026-10-10.pdf"
        }

        for symptom, filename in expected.items():
            with self.subTest(symptom=symptom):
                record = self.comparisons[symptom]

                self.assertTrue(record["match_found"])

                self.assertTrue(
                    any(
                        m["file"] == filename
                        for m in record["matches"]
                    )
                )

    def test_04_chronology(self):
        dizziness = self.comparisons["dizziness"]
        cough = self.comparisons["cough"]

        self.assertTrue(
            any(
                m["chronology"] == "Report date unknown"
                for m in dizziness["matches"]
            )
        )

        self.assertTrue(
            any(
                m["chronology"] ==
                "Report dated before caregiver-reported onset"
                for m in cough["matches"]
            )
        )

    def test_05_additional_report_symptoms(self):
        additional = {
            normalize(item["symptom"])
            for item in self.facts[
                "additional_report_symptoms"
            ]
        }

        self.assertEqual(
            additional,
            {"headache", "tiredness"}
        )

    def test_06_citation_integrity(self):
        for record in self.facts["caregiver_comparisons"]:
            for match in record["matches"]:
                reference = match["reference"]

                self.assertIn(
                    reference,
                    self.sources_by_ref
                )

                source = self.sources_by_ref[reference]

                self.assertEqual(
                    match["file"],
                    source["file"]
                )

                self.assertGreaterEqual(
                    match["page"],
                    1
                )

                self.assertTrue(
                    any(
                        s["page"] == match["page"]
                        for s in source["symptoms"]
                    )
                )

    def test_07_extraction_and_missing_dates(self):
        symptoms = extract_report_symptoms(
            "Symptoms: Cough and headache. "
            "Heart Rate: 88 bpm"
        )

        self.assertEqual(
            {normalize(s) for s in symptoms},
            {"cough", "headache"}
        )

        self.assertIsNone(
            extract_report_date([
                {"text": "Symptoms: Dizziness."}
            ])
        )


if __name__ == "__main__":
    print("\n--- CARECOPILOT HISTORY EVALUATION ---")
    unittest.main(verbosity=2)
