import pandas as pd

TRACKLIST_URL = "https://api.vitaldb.net/trks"


def main():
    print("Downloading VitalDB track list...")

    tracks = pd.read_csv(TRACKLIST_URL)

    print("\nTrack-list shape:", tracks.shape)
    print("\nColumns:")
    print(tracks.columns.tolist())

    # Signals required for CareCopilot V1
    required_tracks = {
        "heart_rate": "Solar8000/HR",
        "spo2": "Solar8000/PLETH_SPO2",
        "temperature": "Solar8000/BT",
    }

    print("\nRequired CareCopilot tracks:")
    for name, track in required_tracks.items():
        count = (tracks["tname"] == track).sum()
        print(f"{name:12s}: {track:25s} -> {count} tracks")

    # Find cases containing each signal.
    case_sets = {}

    for name, track in required_tracks.items():
        case_sets[name] = set(
            tracks.loc[tracks["tname"] == track, "caseid"]
        )

    # Intersection = cases having ALL required signals.
    common_cases = (
        case_sets["heart_rate"]
        & case_sets["spo2"]
        & case_sets["temperature"]
    )

    common_cases = sorted(common_cases)

    print("\n--------------------------------")
    print("CARECOPILOT V1 DATASET CHECK")
    print("--------------------------------")

    print("Cases with HR:         ", len(case_sets["heart_rate"]))
    print("Cases with SpO2:       ", len(case_sets["spo2"]))
    print("Cases with temperature:", len(case_sets["temperature"]))
    print("Cases containing ALL 3:", len(common_cases))

    print("\nFirst 20 compatible case IDs:")
    print(common_cases[:20])

    # Save compatible case IDs.
    output = pd.DataFrame({"caseid": common_cases})

    output_path = "datasets/vitals/compatible_cases.csv"
    output.to_csv(output_path, index=False)

    print(f"\nSaved compatible cases to:\n{output_path}")


if __name__ == "__main__":
    main()