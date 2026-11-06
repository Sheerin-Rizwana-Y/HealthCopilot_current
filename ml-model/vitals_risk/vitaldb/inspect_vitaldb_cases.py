import pandas as pd

CASES_URL = "https://api.vitaldb.net/cases"


def main():
    print("Downloading VitalDB case information...")

    cases = pd.read_csv(CASES_URL)

    print("\nDataset shape:")
    print(cases.shape)

    print("\nAvailable columns:")
    for column in cases.columns:
        print(f" - {column}")

    print("\nFirst 5 cases:")
    print(cases.head().to_string())

    print("\nMissing values:")
    print(cases.isna().sum().to_string())

    # Load the compatible cases we identified earlier
    compatible = pd.read_csv(
        "datasets/vitals/compatible_cases.csv"
    )

    compatible_cases = cases[
        cases["caseid"].isin(compatible["caseid"])
    ].copy()

    print("\n--------------------------------")
    print("CARECOPILOT COMPATIBLE CASES")
    print("--------------------------------")

    print("Compatible clinical cases:",
          len(compatible_cases))

    print("\nFirst 5 compatible cases:")
    print(compatible_cases.head().to_string())

    output_path = (
        "datasets/vitals/compatible_case_metadata.csv"
    )

    compatible_cases.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
    