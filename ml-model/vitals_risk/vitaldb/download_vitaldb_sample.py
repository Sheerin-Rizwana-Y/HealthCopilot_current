import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

OUTPUT_DIR = Path("datasets/vitals/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load VitalDB track list
tracks_df = pd.read_csv("https://api.vitaldb.net/trks")

compatible = pd.read_csv("datasets/vitals/compatible_cases.csv")

sample_cases = compatible["caseid"].head(50).tolist()

print(f"Total cases selected: {len(sample_cases)}")

wanted_tracks = {
    "hr": "Solar8000/HR",
    "spo2": "Solar8000/PLETH_SPO2",
    "temperature": "Solar8000/BT",
}

for caseid in sample_cases:

    print(f"\nCase {caseid}")

    case_tracks = tracks_df[
        tracks_df["caseid"] == caseid
    ]

    for short_name, track_name in wanted_tracks.items():

        match = case_tracks[
            case_tracks["tname"] == track_name
        ]

        if match.empty:
            print(f"  {short_name}: TRACK NOT FOUND")
            continue

        tid = match.iloc[0]["tid"]

        url = f"https://api.vitaldb.net/{tid}"

        output_file = OUTPUT_DIR / (
            f"case_{caseid}_{short_name}.csv"
        )

        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  Already downloaded: {output_file}")
            continue

        print(
            f"  Downloading {short_name} "
            f"(tid={tid})..."
        )

        try:
            urlretrieve(url, output_file)

            print(
                f"  Saved: {output_file} "
                f"({output_file.stat().st_size / 1024:.1f} KB)"
            )

        except Exception as error:
            print(f"  FAILED: {error}")

print("\nSample download finished.")