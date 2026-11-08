"""Turn Synthea CSV files into chronological patient token sequences.

Run: python prepare.py
Requires: data/synthea/patients.csv, encounters.csv, conditions.csv,
          medications.csv
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path("data/synthea")
OUTPUT = Path("data/processed.json")
SEED = 42
MAX_EVENTS = 96  # Enough for a lightweight initial experiment.


def read_rows(filename):
    path = DATA / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the Synthea CSV sample and place the "
            "unpacked CSV files directly inside data/synthea/."
        )
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    patients = read_rows("patients.csv")
    patient_ids = {row["ID"] for row in patients}
    events = defaultdict(list)

    # Keep only three event families for the first, small experiment.
    for filename, kind in [
        ("encounters.csv", "ENCOUNTER"),
        ("conditions.csv", "CONDITION"),
        ("medications.csv", "MEDICATION"),
    ]:
        rows = read_rows(filename)
        for row in rows:
            patient = row.get("PATIENT", "")
            code = row.get("CODE", "").strip()
            date = row.get("START", "") or row.get("DATE", "")
            if patient in patient_ids and code and date:
                events[patient].append((date, f"{kind}:{code}"))
        print(f"Read {len(rows):,} rows from {filename}")

    # No names, addresses, raw patient identifiers, or other demographics
    # are put into the model's token sequences.
    sequences = []
    for patient in sorted(patient_ids):
        history = sorted(events[patient], key=lambda item: (item[0], item[1]))
        tokens = ["<BOS>"] + [code for _, code in history[:MAX_EVENTS]] + ["<EOS>"]
        if len(tokens) >= 12:
            sequences.append(tokens)

    if len(sequences) < 8:
        raise ValueError("Need at least 8 patients with 10 or more events.")

    random.Random(SEED).shuffle(sequences)
    canary = sequences[0]  # This entire synthetic patient is excluded from background training.
    held_out = sequences[1:max(2, round(len(sequences) * 0.1))]
    train = sequences[max(2, round(len(sequences) * 0.1)):]

    tokens = sorted({token for seq in sequences for token in seq if token not in {"<PAD>", "<BOS>", "<EOS>"}})
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + tokens
    payload = {
        "vocab": vocab,
        "train": train,
        "held_out": held_out,
        "canary": canary,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Prepared {len(train)} training patients, {len(held_out)} held-out patients, and 1 canary")
    print(f"Vocabulary: {len(vocab)} event tokens")
    print(f"Pilot canary length: {len(canary)} tokens")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
