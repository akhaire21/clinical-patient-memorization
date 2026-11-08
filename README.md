# Clinical patient memorization — first experiment

A deliberately small starting point for our 6.3930 project. This repository does **three** things:

1. Converts Synthea CSV records into patient event sequences.
2. Trains a tiny causal transformer (a small GPT-style model).
3. Compares one held-out synthetic patient included in training **0 times** versus **20 times**.

**This is a pipeline smoke test, not a validated privacy or memorization result.** We will add carefully designed canaries, more baselines, multiple random seeds, patient diversity, and model-size experiments after the first run works.

## 1. Download the dataset

Download **100 Sample Synthetic Patient Records, CSV** from the official Synthea website:

https://synthetichealth.github.io/downloads.html

Direct ZIP: https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip

Unzip the CSV files so the folder looks like this:

```text
clinical-patient-memorization/
├── data/
│   └── synthea/
│       ├── patients.csv
│       ├── encounters.csv
│       ├── conditions.csv
│       ├── medications.csv
│       └── ... (other Synthea CSVs can stay here)
├── prepare.py
├── train.py
├── requirements.txt
└── README.md
```

Do **not** upload or commit the CSV files to GitHub. The `.gitignore` already excludes them. Use synthetic data only.

## 2. Install dependencies

From this folder in your terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate using `.venv\\Scripts\\activate` instead.

## 3. Prepare the patient timelines

```bash
python prepare.py
```

This creates `data/processed.json`. It takes event codes from encounters, conditions, and medications, sorts them by date, and creates one sequence for each patient. A single synthetic patient is held out to serve as our *pilot* canary. A separate group of patients is held out for evaluation.

## 4. Train the baseline and repeated-canary models

```bash
python train.py --copies 0 --steps 150
python train.py --copies 20 --steps 150
```

Both runs keep the **same model architecture**, initialization seed, and training-step budget. Results are printed and saved as `outputs/results_copies_0.json` and `outputs/results_copies_20.json`. Model checkpoints are also saved under `outputs/`.

The terminal reports:
- Training loss (next-token prediction loss).
- Average loss on the second half of the held-out canary, given its prefix.
- Average loss on a different held-out patient.
- The percentage of the first 20 continuation tokens reproduced by greedy generation.

A **lower canary loss in the 20-copy run than the zero-copy run** is a preliminary signal that the model has learned more about that particular history. It is not, by itself, proof of exact extraction or privacy leakage.

## 5. Create your Git repository

```bash
git init
git add .
git commit -m "Initial Synthea preprocessing and GPT canary baseline"
```

## What we will add next (not in this first version)

- Repeat each experiment with multiple random seeds.
- Add a bigram baseline and a randomly initialized model baseline.
- Evaluate multiple *designed* canaries with clinically plausible, distinctive histories.
- Evaluate multiple repetition counts (0, 1, 2, 5, 10, 20, and 100).
- Compare standard Synthea patients against more diverse synthetic cohorts.
- Measure continuation ranking and extraction more rigorously.
- Sweep model sizes and, later, examine internal activations.

Important limitation: The first canary is an existing **synthetic** Synthea patient held out from the background training set, rather than a purpose-designed rare patient. Its sequence might be predictable from common clinical patterns. Also, the 100-patient sample is too small for strong scientific conclusions. This first repository is for getting the pipeline working.
