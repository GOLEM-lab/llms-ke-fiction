#!/usr/bin/env python3
import os
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

# ——————————————————————————————
#   Adjust these two paths if needed
# ——————————————————————————————
INPUT_ROOT  = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/tuning_output"
OUTPUT_ROOT = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/tuning_output/correlation"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Only these files will be evaluated:
INPUT_FILES = [
    "stories_llama/tuned/parachute_stories_llama.csv",
    "stories_deepseek/tuned/parachute_stories_deepseek.csv",
    "specialised_summary_llama/tuned/parachute_specialised_summary_llama.csv",
    "specialised_summary_deepseek/tuned/parachute_specialised_summary_deepseek.csv",
    "generic_summary_llama/tuned/parachute_generic_summary_llama.csv",
    "generic_summary_deepseek/tuned/parachute_generic_summary_deepseek.csv",
]

# ——————————————————
#   Core configuration
# ——————————————————
ID_COLUMN = "work_id"
ENTITY_CONFIG = [
    {"col": "Victim",         "flags": ["Victim_TP","Victim_FP","Victim_FN","Victim_TN"]},
    {"col": "Perpetrator",    "flags": ["Perpetrator_TP","Perpetrator_FP","Perpetrator_FN","Perpetrator_TN"]},
    {"col": "Mode of Demise", "flags": ["Mode of Demise_TP","Mode of Demise_FP","Mode of Demise_FN","Mode of Demise_TN"]},
]

class MissingColumnsError(Exception):
    pass

def strategy_binary_flag_correlations(df, log):
    log("\n=== Strategy A: Binary-flag correlations ===")
    for lab in ["TP","FP","FN","TN"]:
        manual, auto = [], []
        for et in ENTITY_CONFIG:
            col, acol = f"{et['col']}_{lab}", f"{et['col']}_{lab}_auto"
            if col not in df.columns or acol not in df.columns:
                raise MissingColumnsError(f"Missing columns {col!r} or {acol!r}")
            manual.extend(df[col].fillna(0).astype(int).tolist())
            auto.extend( df[acol].fillna(0).astype(int).tolist())

        pearson_r, _  = pearsonr(manual, auto)
        spearman_r, _ = spearmanr(manual, auto)
        mcc           = matthews_corrcoef(manual, auto)
        kappa         = cohen_kappa_score(manual, auto)
        log(f"{lab}: Pearson r = {pearson_r:.3f}, "
            f"Spearman ρ = {spearman_r:.3f}, "
            f"MCC = {mcc:.3f}, Cohen’s κ = {kappa:.3f}")

def strategy_multiclass(df, log):
    log("\n=== Strategy B: Multi-class classification ===")
    labels = ["TP","FP","FN","TN"]
    records = []
    for idx, row in df.iterrows():
        uid = row.get(ID_COLUMN, idx)
        for et in ENTITY_CONFIG:
            ent = et['col']
            manu = next((lab for lab, flag in zip(labels, et['flags'])
                         if row.get(flag, 0) == 1), "TN")
            auto = next((lab for lab, flag in zip(labels, et['flags'])
                         if row.get(f"{flag}_auto", 0) == 1), "TN")

            records.append({
                ID_COLUMN:         uid,
                "Entity":          ent,
                "manual_label":    manu,
                "auto_label":      auto,
                "Gold_Value":      row.get(ent, ""),
                "Predicted_Value": row.get(f"{ent}_P", "")
            })

    rec_df = pd.DataFrame(records)

    cm = confusion_matrix(rec_df.manual_label, rec_df.auto_label, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    log("\nConfusion matrix (manual ↓ × auto →):")
    log(cm_df.to_string())

    kappa = cohen_kappa_score(rec_df.manual_label, rec_df.auto_label)
    log(f"\nMulti-class Cohen’s κ = {kappa:.3f}")

    log("\nClassification report (manual vs auto):")
    log(classification_report(rec_df.manual_label,
                              rec_df.auto_label,
                              labels=labels, digits=3))
    return rec_df

def extract_misclassifications(rec_df, out_dir, log):
    mis_df = rec_df[rec_df.manual_label != rec_df.auto_label]
    if mis_df.empty:
        log("\nNo misclassifications to save.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for (m, a), grp in mis_df.groupby(["manual_label","auto_label"]):
        fname = f"misclass_manual_{m}_auto_{a}.csv"
        path  = os.path.join(out_dir, fname)
        grp.to_csv(path, sep=';', index=False)
        log(f"Saved {len(grp)} cases manual={m} auto={a} → {path}")

def process_file(full_path):
    # derive subfolder under correlation, e.g.
    # input:  .../tuning_output/stories_llama/tuned/parachute_...csv
    # rel:    stories_llama/tuned/parachute_...csv
    rel     = os.path.relpath(full_path, INPUT_ROOT)
    subdir  = os.path.dirname(rel)
    out_dir = os.path.join(OUTPUT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "evaluation_report.txt")

    with open(report_path, 'w') as report_f:
        def log(msg):
            print(f"[{os.path.basename(full_path)}] {msg}")
            report_f.write(msg + "\n")

        try:
            log(f"Loading CSV")
            df = pd.read_csv(full_path, sep=';')

            strategy_binary_flag_correlations(df, log)
            rec_df = strategy_multiclass(df, log)
            extract_misclassifications(rec_df,
                                       os.path.join(out_dir, "misclassifications"),
                                       log)
            log(f"Done. All outputs in {out_dir}")

        except MissingColumnsError as e:
            log(f"SKIPPED: {e}")
        except Exception as e:
            log(f"ERROR ({type(e).__name__}): {e}")

def main():
    for rel in INPUT_FILES:
        full = os.path.join(INPUT_ROOT, rel)
        if not os.path.isfile(full):
            print(f"[MISSING] {full}", file=sys.stderr)
            continue
        process_file(full)

if __name__ == "__main__":
    main()
