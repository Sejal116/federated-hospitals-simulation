# Federated Hospitals Simulation

A small experiment on how hospitals can train a shared medical AI model without sharing patient data. It compares training at one hospital alone, training on pooled data, and federated learning, then tests two privacy questions: what privacy noise costs in accuracy, and whether an attacker can tell who was in the training data.

## Background

This project follows my published literature review:

Gupta, S. (2026). "AI for Healthcare: How Can Hospitals Share Data to Build Better Models?" *International Journal of Innovative Research in Computer and Communication Engineering (IJIRCCE)*, 14(8). DOI: [10.15680/IJIRCCE.2026.1408009](https://doi.org/10.15680/IJIRCCE.2026.1408009)

The review surveyed federated learning, synthetic data, differential privacy and homomorphic encryption, and noted that it did not test any method experimentally. This repository is a small experiment in that direction, covering federated learning and differential privacy. It does not test synthetic data or homomorphic encryption.

## What it does

- Uses scikit-learn's built-in Wisconsin breast cancer dataset (569 patients, 30 measurements each).
- Holds out 20% of patients as a test set (114 patients), and splits the other 455 between 3 simulated hospitals with deliberately different patient mixes.
- Implements logistic regression, federated averaging, and differential-privacy-style training (per-patient gradient clipping plus Gaussian noise) from scratch in NumPy.
- Repeats every experiment on many random splits and reports averages and spread.

## Results

All numbers are averages over 20 random splits, shown as mean ± standard deviation. Of 30 splits tried, 20 were usable: a split was skipped if any hospital had fewer than 5 patients of either class.

### 1. Hospital alone vs pooled vs federated

| Method | Accuracy | AUC |
|---|---|---|
| Hospital alone (average of 3) | 0.957 ± 0.016 | 0.992 ± 0.006 |
| All data pooled | 0.978 ± 0.013 | 0.994 ± 0.005 |
| Federated averaging | 0.978 ± 0.013 | 0.994 ± 0.005 |

![Accuracy comparison](results/accuracy.png)

Federated training matched pooled training, without any patient data leaving a hospital. Pooled training beat the hospital-alone models on all 20 splits, by 2.1 percentage points on average. The error bars overlap because the spread between splits is similar in size to the gap.

### 2. Cost of privacy noise (federated)

| Noise level | Accuracy |
|---|---|
| No clipping, no noise | 0.978 |
| 0 (clipping only) | 0.971 |
| 10 | 0.970 |
| 25 | 0.967 |
| 50 | 0.957 |
| 100 | 0.944 |
| 200 | 0.907 |
| 400 | 0.825 |
| 800 | 0.731 |

![Accuracy vs noise](results/privacy.png)

Clipping alone cost about 0.7 percentage points. Accuracy only starts to drop clearly from a noise level of about 50. My first attempt tested noise levels up to 16 and showed no effect at all, so I widened the range. Each hospital has only 100 to 200 patients, and the noise is divided by that number.

### 3. Can an attacker tell who was in the training data?

A loss-based membership inference attack: guess "member" for patients the model is least surprised by. Attack AUC of 0.5 means the attacker is guessing.

| Setting | Attack AUC |
|---|---|
| A: federated model, no noise | 0.488 ± 0.028 |
| A: federated model, noise 50 | 0.489 ± 0.026 |
| A: federated model, noise 200 | 0.495 ± 0.026 |
| B: overfit hospital model, no noise | 0.576 ± 0.071 |
| B: overfit hospital model, noise 50 | 0.564 ± 0.063 |
| B: overfit hospital model, noise 200 | 0.548 ± 0.055 |

![Attack results](results/attack.png)

- **Experiment A** uses the simple model from the other experiments. The attack does no better than guessing, so this model leaks almost nothing and the noise has nothing to improve.
- **Experiment B** uses a model I deliberately made prone to overfitting (squared features, 495 inputs, long training, one small hospital) to check that the attack works when there is something to find. It leaks a little, and more noise lowers the attack's success. The spread is large, so this is suggestive, not proven.

## Limitations

- One small, easy dataset. Hospital-alone models already score about 96%, so the possible gain from sharing data is small.
- A simulation: the "hospitals" are random slices of one public dataset, not real hospitals.
- Simple model (logistic regression). Results may not carry over to deep learning or medical images.
- Noise levels are not privacy budgets. No formal privacy accounting (epsilon) is computed, so none of these results is a formal privacy guarantee.
- The feature scaler is fitted on all training data, which pools some statistics across hospitals. Real hospitals could not do this.
- In experiment B, each hospital is scored separately, so its 60 results come from 20 splits and are not fully independent.
- Splits where a hospital lacks a class were skipped, which slightly favours more even splits.

## Run it yourself

Tested with Python 3.10.

```
git clone https://github.com/Sejal116/federated-hospitals-simulation.git
cd federated-hospitals-simulation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python src/baselines.py
python src/run_experiments.py
python src/run_privacy.py
python src/run_attack.py
```

Run the scripts from the main folder, since they save results to `results/`. Random seeds are fixed, so the results are reproducible. Small differences in the last digit can appear with different library versions.

## Project structure

```
src/split.py            load data, train/test split, split into hospitals
src/models.py           logistic regression and evaluation
src/federated.py        federated averaging
src/privacy.py          clipped and noised training
src/attack.py           membership inference attack
src/baselines.py        single-run comparison
src/run_experiments.py  repeated comparison, results table and chart
src/run_privacy.py      privacy noise experiment
src/run_attack.py       attack experiments
tests/                  automatic tests
results/                saved tables and charts
```

## Author

Sejal Gupta. Licensed under the MIT License.