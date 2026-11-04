"""Estimate what joining a federated network could mean for one hospital.

The estimate is read from results/size_results.csv, which comes from the
simulation in this repository. It is an illustration based on one small
public dataset, not a prediction for a real hospital.

Usage, from the main folder:
    python src/impact.py --patients-in-training 120 --screened-per-year 5000
"""
import argparse

import pandas as pd


def load_size_results(path="results/size_results.csv"):
    return pd.read_csv(path)


def band_for(size_results, training_patients):
    """Pick the size band that matches the hospital's training data size."""
    for _, row in size_results.iterrows():
        if row.band_low <= training_patients < row.band_high:
            return row
    raise ValueError("No size band matches this hospital size.")


def estimate_impact(size_results, training_patients, screened_per_year):
    """Return plain numbers for one hospital.

    training_patients: how many labelled patient records the hospital has
        to train its own model.
    screened_per_year: how many patients the model would be used on in a year.
    """
    if training_patients <= 0 or screened_per_year < 0:
        raise ValueError("Patient numbers must be positive.")

    row = band_for(size_results, training_patients)
    wrong_alone = row.error_alone * screened_per_year
    wrong_shared = row.error_shared * screened_per_year
    return {
        "error_alone": row.error_alone,
        "error_shared": row.error_shared,
        "wrong_alone": wrong_alone,
        "wrong_shared": wrong_shared,
        "fewer_wrong": wrong_alone - wrong_shared,
        "share_better_off": row.share_better_off,
        "share_worse_off": row.share_worse_off,
        "simulated_hospitals": int(row.hospitals),
    }


def describe(result, screened_per_year):
    """Turn the numbers into a short message a doctor can read."""
    lines = [
        f"Out of {screened_per_year:,} patients a year:",
        f"  Own model only:  about {result['wrong_alone']:.0f} wrong results "
        f"({result['error_alone']:.1%})",
        f"  Shared model:    about {result['wrong_shared']:.0f} wrong results "
        f"({result['error_shared']:.1%})",
        f"  Difference:      about {result['fewer_wrong']:.0f} fewer wrong results",
        f"In the simulation, {result['share_better_off']:.0%} of similar "
        f"hospitals did better in the shared model and "
        f"{result['share_worse_off']:.0%} did worse "
        f"({result['simulated_hospitals']} simulated hospitals).",
        "This is based on one small public dataset. Test on your own data first.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients-in-training", type=int, required=True)
    parser.add_argument("--screened-per-year", type=int, required=True)
    args = parser.parse_args()

    table = load_size_results()
    result = estimate_impact(table, args.patients_in_training,
                             args.screened_per_year)
    print(describe(result, args.screened_per_year))
