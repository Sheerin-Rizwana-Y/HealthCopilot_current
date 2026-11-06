
import argparse
import re

from retrieve import retrieve
from symptom_records import load_symptoms


# Labels that may follow a Symptoms field in a report.
STOP_LABELS = (
    r"Patient|History|Symptoms?|Heart\s*Rate|HR|"
    r"SpO2|Temperature|Note|Diagnosis|"
    r"Medications?|Date"
)

SYMPTOM_PATTERN = re.compile(
    rf"\bSymptoms?\s*:\s*(.*?)"
    rf"(?=\s+(?:{STOP_LABELS})\s*:|\Z)",
    flags=re.IGNORECASE | re.DOTALL
)


def normalize(text):
    """Normalize symptom names for exact matching."""
    return " ".join(
        re.findall(r"[a-z0-9]+", text.casefold())
    )


def extract_report_symptoms(text):
    """Extract explicitly labeled symptoms from text."""
    symptoms = []

    for match in SYMPTOM_PATTERN.finditer(text):
        field = match.group(1)

        # Handles: dizziness and tiredness
        # Also handles: dizziness, tiredness
        items = re.split(
            r"\s*(?:[,;]|\band\b)\s*",
            field,
            flags=re.IGNORECASE
        )

        for item in items:
            symptom = item.strip().strip(" .")

            if symptom:
                symptoms.append(symptom)

    return symptoms


def compare_symptoms(patient_id, top_k=5):
    records = load_symptoms(patient_id)

    if not records:
        print("No caregiver symptom records found.")
        return

    for record in records:
        if record.get("patient_id") != patient_id:
            raise ValueError("Patient record mismatch")

    # Retrieve report context once for all symptoms.
    symptom_names = sorted({
        record["symptom"].strip()
        for record in records
    })

    query = (
        "Which symptoms are listed in the "
        "patient's medical reports? "
        + ", ".join(symptom_names)
    )

    print("\n[1] Loading caregiver symptom records...")
    print("Records:", len(records))

    print("\n[2] Retrieving medical report passages...")

    results = retrieve(
        patient_id=patient_id,
        query=query,
        top_k=top_k
    )

    # Map each normalized symptom to source references.
    report_symptoms = {}

    for result in results:
        extracted = extract_report_symptoms(
            result["text"]
        )

        for symptom in extracted:
            key = normalize(symptom)

            source = {
                "symptom": symptom,
                "file": result["source_file"],
                "page": result["page"]
            }

            report_symptoms.setdefault(key, [])

            if source not in report_symptoms[key]:
                report_symptoms[key].append(source)

    print("\n--- CARECOPILOT SYMPTOM COMPARISON ---")
    print("Patient:", patient_id)
    print("Caregiver records:", len(records))
    print("Retrieved passages:", len(results))

    recorded_names = set()

    for i, record in enumerate(records, start=1):
        symptom = record["symptom"]
        key = normalize(symptom)
        recorded_names.add(key)

        print(f"\n--- SYMPTOM {i} ---")
        print("Caregiver symptom:", symptom)
        print("Severity:", record["severity"])
        print("Onset:", record["onset_at"])

        matches = report_symptoms.get(key, [])

        if matches:
            print("Status: MATCH FOUND IN REPORT")

            for source in matches:
                print(
                    f"Source: {source['file']}, "
                    f"page {source['page']}"
                )
        else:
            print(
                "Status: NOT FOUND IN RETRIEVED "
                "REPORT PASSAGES"
            )

    # Symptoms mentioned in reports but not in
    # the available caregiver symptom entries.
    report_only = [
        sources[0]["symptom"]
        for key, sources in report_symptoms.items()
        if key not in recorded_names
    ]

    print("\n--- ADDITIONAL REPORT SYMPTOMS ---")

    if report_only:
        for symptom in report_only:
            print("-", symptom)
    else:
        print("None found in retrieved passages.")

    print("\nImportant:")
    print(
        "A missing match does not prove that a "
        "symptom never occurred."
    )
    print(
        "This is text comparison, not clinical "
        "diagnosis or medical risk assessment."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient-id",
        required=True
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    compare_symptoms(
        patient_id=args.patient_id,
        top_k=args.top_k
    )
