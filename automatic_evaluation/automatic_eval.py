#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------
EVAL_METHOD                = "parachute"
BASE_INPUT_DIR             = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/input/gold_pred_mapped"
BASE_OUTPUT_DIR            = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/output"

# selection strategies for thresholds: "tuned", "lower", or "balanced"
FUZZY_SELECTION_STRATEGY    = "tuned"
SEMANTIC_SELECTION_STRATEGY = "tuned"

# toggle: whether to load previously saved thresholds instead of recomputing
LOAD_SAVED_THRESHOLDS       = True
THRESHOLDS_JSON             = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/tuning_output/stories_llama/stories_llama_threshold.json"

# entities configuration
ENTITY_CONFIG = [
    {"col": "Victim",         "pred": "Victim_P",         "flags": ["Victim_TP","Victim_FP","Victim_FN","Victim_TN"]},
    {"col": "Perpetrator",    "pred": "Perpetrator_P",    "flags": ["Perpetrator_TP","Perpetrator_FP","Perpetrator_FN","Perpetrator_TN"]},
    {"col": "Mode of Demise", "pred": "Mode of Demise_P", "flags": ["Mode of Demise_TP","Mode of Demise_FP","Mode of Demise_FN","Mode of Demise_TN"]},
]

# threshold search grids
FUZZY_THRESHOLDS    = [i/100 for i in range(0, 101, 5)]
SEMANTIC_THRESHOLDS = [i/100 for i in range(20, 101, 5)]  # start at 0.20

# load semantic model + cache
SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
SEM_CACHE = {}

# values considered as missing for rule-based TN
MISSING_VALUES = {"", "undefined", "unspecified", "unnamed", "none mentioned", "unknown", "-", "---"}

# -------- Helper functions --------
def normalize_text(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if s.endswith("'s"):
        s = s[:-2]
    for ch in "'\".,!?;:":
        s = s.replace(ch, "")
    return s

def is_missing(val):
    if pd.isna(val):
        return True
    return str(val).strip().lower() in MISSING_VALUES

def fuzzy_match(a, b, thr):
    if not a or not b:
        return False
    return (fuzz.ratio(a, b) / 100) >= thr

def compute_cosine(a, b):
    if a not in SEM_CACHE:
        SEM_CACHE[a] = SEM_MODEL.encode(a, show_progress_bar=False)
    if b not in SEM_CACHE:
        SEM_CACHE[b] = SEM_MODEL.encode(b, show_progress_bar=False)
    return float(util.cos_sim(SEM_CACHE[a], SEM_CACHE[b]))

def resolve_cell(gold, pred, fuzzy_thr, sem_thr):
    if is_missing(gold):
        return ("rule", (0,0,0,1)) if is_missing(pred) else ("rule", (0,1,0,0))
    if is_missing(pred):
        return ("rule", (0,0,1,0))

    g_norm, p_norm = normalize_text(gold), normalize_text(pred)
    if g_norm == p_norm:
        return ("rule", (1,0,0,0))
    if fuzzy_match(g_norm, p_norm, fuzzy_thr):
        return ("fuzzy", (1,0,0,0))

    cos = compute_cosine(gold, pred)
    if cos >= sem_thr:
        return ("semantic", (1,0,0,0))
    else:
        return ("semantic", (0,1,1,0))

def tune_thresholds(df, config, thr_list, mode):
    recs = []
    for et in config:
        gold, pred, flags = et["col"], et["pred"], et["flags"]
        sub = df[~df[gold].apply(is_missing) & ~df[pred].apply(is_missing)]
        for thr in thr_list:
            match = total = 0
            for _, row in sub.iterrows():
                human = tuple(int(row[f]) for f in flags)
                method, auto = resolve_cell(
                    row[gold], row[pred],
                    fuzzy_thr=(thr if mode=="fuzzy" else 1.0),
                    sem_thr =(thr if mode=="semantic" else 1.1)
                )
                if method != mode:
                    continue
                total += 1
                if auto == human:
                    match += 1
            recs.append({
                "Entity": gold,
                "Threshold": thr,
                "Accuracy": match/total if total else np.nan
            })
    return pd.DataFrame(recs)

def compute_prf(df, flags):
    tp, fp, fn = df[flags[0]].sum(), df[flags[1]].sum(), df[flags[2]].sum()
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f = 2*p*r/(p+r) if (p+r) else 0
    return p, r, f

# -------- Pipeline Function --------
def run_pipeline(input_csv, model_name):
    base = os.path.splitext(os.path.basename(input_csv))[0]
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_name, SEMANTIC_SELECTION_STRATEGY)
    os.makedirs(output_dir, exist_ok=True)

    # Derive all file paths
    output_csv          = os.path.join(output_dir, f"{EVAL_METHOD}_{base}.csv")
    merged_report_csv   = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_merged_report.csv")
    sim_metrics_csv     = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_similarity_metrics.csv")
    fuzzy_tuning_csv    = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_fuzzy_tuning.csv")
    semantic_tuning_csv = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_semantic_tuning.csv")

    cosine_plot         = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_cosine_distributions.png")
    fuzzy_plot          = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_fuzzy_tuning_plot.png")
    semantic_plot       = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_semantic_tuning_plot.png")
    resolution_plot     = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_resolution_plot.png")
    overall_plot        = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_overall_resolution.png")
    prec_plot           = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_prf_precision.png")
    recall_plot         = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_prf_recall.png")
    f1_plot             = os.path.join(output_dir, f"{EVAL_METHOD}_{base}_prf_f1.png")

    # 0) Load input
    df = pd.read_csv(input_csv, sep=",")

    for extra in ("Classification", "Reason"):
        if extra not in df.columns:
            df[extra] = ""

    for et in ENTITY_CONFIG:
        for flag in et["flags"]:
            df[f"{flag}_auto"] = 0
        df[f"{et['col']}_res"] = ""

    # 1) Collect similarity metrics
    sims = []
    for et in ENTITY_CONFIG:
        for _, row in df.iterrows():
            g, p = row[et["col"]], row[et["pred"]]
            if is_missing(g) or is_missing(p):
                continue
            sims.append({
                "Entity":   et["col"],
                "Cosine":   compute_cosine(g, p),
                "Fuzzy":    fuzz.ratio(normalize_text(g), normalize_text(p)) / 100,
                "Human_TP": int(row[et["flags"][0]] == 1)
            })
    sims_df = pd.DataFrame(sims)
    sims_df.to_csv(sim_metrics_csv, sep=";", index=False)

    # 2) Plot cosine histograms
    fig, ax = plt.subplots()
    for ent, grp in sims_df.groupby("Entity"):
        n, bins, patches = ax.hist(grp["Cosine"], bins=30, alpha=0.5, label=ent)
        total = n.sum()
        for count, patch in zip(n, patches):
            if count > 0:
                pct = count/total * 100
                x = patch.get_x() + patch.get_width()/2
                y = patch.get_height()
                ax.text(x, y, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    n, bins, patches = ax.hist(sims_df["Cosine"], bins=30, alpha=0.3, label="Overall")
    total = n.sum()
    for count, patch in zip(n, patches):
        if count > 0:
            pct = count/total * 100
            x = patch.get_x() + patch.get_width()/2
            y = patch.get_height()
            ax.text(x, y, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Cosine Similarity Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cosine_plot)
    plt.close(fig)

    # 3) Compute 5th-percentile rec-lower limits
    rec_fuz_entity  = sims_df[sims_df.Human_TP==1].groupby("Entity")["Fuzzy"].quantile(0.05).to_dict()
    rec_fuz_overall = float(sims_df[sims_df.Human_TP==1]["Fuzzy"].quantile(0.05))
    rec_sem_entity  = sims_df[sims_df.Human_TP==1].groupby("Entity")["Cosine"].quantile(0.05).to_dict()
    rec_sem_overall = float(sims_df[sims_df.Human_TP==1]["Cosine"].quantile(0.05))

    # 4) Load or compute thresholds
    if LOAD_SAVED_THRESHOLDS:
        with open(THRESHOLDS_JSON, "r") as f:
            saved = json.load(f)

        best_f_overall   = saved["overall"]["tuned_fuzzy"]
        rec_fuz_overall  = saved["overall"]["rec_fuzzy_lower"]
        final_f_overall  = saved["overall"]["final_fuzzy"]
        best_s_overall   = saved["overall"]["tuned_semantic"]
        rec_sem_overall  = saved["overall"]["rec_sem_lower"]
        final_s_overall  = saved["overall"]["final_semantic"]

        # enforce same overall thresholds for every entity
        final_f = {ent: final_f_overall for ent in saved["entities"]}
        final_s = {ent: final_s_overall for ent in saved["entities"]}

    else:
        fuzz_df = tune_thresholds(df, ENTITY_CONFIG, FUZZY_THRESHOLDS,    "fuzzy")
        sem_df  = tune_thresholds(df, ENTITY_CONFIG, SEMANTIC_THRESHOLDS, "semantic")
        fuzz_df.to_csv(fuzzy_tuning_csv, sep=";", index=False)
        sem_df.to_csv(semantic_tuning_csv, sep=";", index=False)

        # plot tuning curves
        for mode, tune_df, plot_path in [
            ("Fuzzy", fuzz_df, fuzzy_plot),
            ("Semantic", sem_df, semantic_plot)
        ]:
            plt.figure()
            for ent, grp in tune_df.groupby("Entity"):
                plt.plot(grp["Threshold"], grp["Accuracy"], "o-", label=ent)
            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            plt.title(f"{mode} Threshold Tuning")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        best_f         = fuzz_df.groupby("Entity")["Accuracy"]\
                                .idxmax().apply(lambda i: fuzz_df.loc[i,"Threshold"])\
                                .to_dict()
        best_s         = sem_df.groupby("Entity")["Accuracy"]\
                                .idxmax().apply(lambda i: sem_df.loc[i,"Threshold"])\
                                .to_dict()
        best_f_overall = float(fuzz_df.groupby("Threshold")["Accuracy"]\
                                        .mean().idxmax())
        best_s_overall = float(sem_df.groupby("Threshold")["Accuracy"]\
                                        .mean().idxmax())

        final_f, final_s = {}, {}
        for ent in best_f:
            bf, lf = best_f[ent], rec_fuz_entity.get(ent, np.nan)
            bs, ls = best_s[ent], rec_sem_entity.get(ent, np.nan)
            final_f[ent] = bf if FUZZY_SELECTION_STRATEGY=="tuned" else (lf if FUZZY_SELECTION_STRATEGY=="lower" else max(bf,lf))
            final_s[ent] = bs if SEMANTIC_SELECTION_STRATEGY=="tuned" else (ls if SEMANTIC_SELECTION_STRATEGY=="lower" else max(bs,ls))

        # enforce overall
        final_f = {ent: best_f_overall for ent in final_f}
        final_s = {ent: best_s_overall for ent in final_s}

        thresholds = {
            "fuzzy_selection":    FUZZY_SELECTION_STRATEGY,
            "semantic_selection": SEMANTIC_SELECTION_STRATEGY,
            "entities":           {},
            "overall": {
                "tuned_fuzzy":     best_f_overall,
                "rec_fuzzy_lower": rec_fuz_overall,
                "final_fuzzy":     best_f_overall,
                "tuned_semantic":  best_s_overall,
                "rec_sem_lower":   rec_sem_overall,
                "final_semantic":  best_s_overall
            }
        }
        for ent in best_f:
            thresholds["entities"][ent] = {
                "tuned_fuzzy":     best_f[ent],
                "rec_fuzzy_lower": rec_fuz_entity.get(ent, np.nan),
                "final_fuzzy":     final_f[ent],
                "tuned_semantic":  best_s[ent],
                "rec_sem_lower":   rec_sem_entity.get(ent, np.nan),
                "final_semantic":  final_s[ent]
            }
        with open(THRESHOLDS_JSON, "w") as f:
            json.dump(thresholds, f, indent=2)

    # 5) Full resolution pass
    for idx, row in df.iterrows():
        for et in ENTITY_CONFIG:
            col, pred, flags = et["col"], et["pred"], et["flags"]
            f_thr, s_thr = final_f[col], final_s[col]
            method, (tp, fp, fn, tn) = resolve_cell(row[col], row[pred], f_thr, s_thr)
            df.at[idx, f"{flags[0]}_auto"] = tp
            df.at[idx, f"{flags[1]}_auto"] = fp
            df.at[idx, f"{flags[2]}_auto"] = fn
            df.at[idx, f"{flags[3]}_auto"] = tn
            df.at[idx, f"{col}_res"]       = method

    # 6) Build merged report
    rows = []
    for et in ENTITY_CONFIG + [{"col": "Overall", "flags": []}]:
        ent = et["col"]
        if ent != "Overall":
            cnts = df[f"{ent}_res"].value_counts().to_dict()
            p_m, r_m, f_m = compute_prf(df, et["flags"][:3])
            p_a, r_a, f_a = compute_prf(df, [f"{et['flags'][0]}_auto",
                                             f"{et['flags'][1]}_auto",
                                             f"{et['flags'][2]}_auto"])
        else:
            cnts = pd.Series(df[[e["col"]+"_res" for e in ENTITY_CONFIG]].values.ravel()).value_counts().to_dict()
            TPm = sum(df[e["flags"][0]].sum() for e in ENTITY_CONFIG)
            FPm = sum(df[e["flags"][1]].sum() for e in ENTITY_CONFIG)
            FNm = sum(df[e["flags"][2]].sum() for e in ENTITY_CONFIG)
            TPa = sum(df[f"{e['flags'][0]}_auto"].sum() for e in ENTITY_CONFIG)
            FPa = sum(df[f"{e['flags'][1]}_auto"].sum() for e in ENTITY_CONFIG)
            FNa = sum(df[f"{e['flags'][2]}_auto"].sum() for e in ENTITY_CONFIG)
            p_m = TPm/(TPm+FPm) if TPm+FPm else 0
            r_m = TPm/(TPm+FNm) if TPm+FNm else 0
            f_m = 2*p_m*r_m/(p_m+r_m) if (p_m+r_m) else 0
            p_a = TPa/(TPa+FPa) if TPa+FPa else 0
            r_a = TPa/(TPa+FNa) if TPa+FNa else 0
            f_a = 2*p_a*r_a/(p_a+r_a) if (p_a+r_a) else 0

        rows.append({
            "Entity": ent,
            "Resolved_by_rule":     cnts.get("rule",    0),
            "Resolved_by_fuzzy":    cnts.get("fuzzy",   0),
            "Resolved_by_semantic": cnts.get("semantic",0),
            "Manual_P":  p_m,
            "Manual_R":  r_m,
            "Manual_F1": f_m,
            "Auto_P":    p_a,
            "Auto_R":    r_a,
            "Auto_F1":   f_a,
            "Tuned_Fuzzy_Threshold":     best_f.get(ent, np.nan)     if 'best_f' in locals() else np.nan,
            "Rec_Fuzzy_Lower":           rec_fuz_entity.get(ent, rec_fuz_overall),
            "Final_Fuzzy_Threshold":     final_f.get(ent, final_f_overall),
            "Tuned_Semantic_Threshold":  best_s.get(ent, np.nan)     if 'best_s' in locals() else np.nan,
            "Rec_Sem_Lower":             rec_sem_entity.get(ent, rec_sem_overall),
            "Final_Semantic_Threshold":  final_s.get(ent, final_s_overall),
        })

    report_df = pd.DataFrame(rows)
    report_df["Delta_P"]  = report_df["Auto_P"]  - report_df["Manual_P"]
    report_df["Delta_R"]  = report_df["Auto_R"]  - report_df["Manual_R"]
    report_df["Delta_F1"] = report_df["Auto_F1"] - report_df["Manual_F1"]

    report_df.to_csv(merged_report_csv, sep=";", index=False)
    df.to_csv(output_csv, sep=";", index=False)

    # 7) Plot resolution breakdown
    report_df.set_index("Entity")[["Resolved_by_rule","Resolved_by_fuzzy","Resolved_by_semantic"]].plot.bar(stacked=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Per-Entity Resolution Breakdown")
    plt.tight_layout()
    plt.savefig(resolution_plot)
    plt.close()

    overall = report_df.set_index("Entity").loc["Overall",
                ["Resolved_by_rule","Resolved_by_fuzzy","Resolved_by_semantic"]]
    overall.plot.bar()
    plt.ylabel("Count")
    plt.title("Overall Resolution Breakdown")
    plt.tight_layout()
    plt.savefig(overall_plot)
    plt.close()

    # 8) PRF comparison plots with delta labels
    for metric, col_manual, col_auto, plot_path, ylabel, title in [
        ("Precision", "Manual_P", "Auto_P", prec_plot, "Precision", "Precision: Manual vs Auto"),
        ("Recall",    "Manual_R", "Auto_R", recall_plot, "Recall",    "Recall: Manual vs Auto"),
        ("F1",        "Manual_F1","Auto_F1", f1_plot,     "F1",        "F1 Score: Manual vs Auto"),
    ]:
        df_pr = report_df.set_index("Entity")[[col_manual, col_auto]]
        fig, ax = plt.subplots()
        df_pr.plot.bar(ax=ax)
        for i in range(len(df_pr)):
            manual = df_pr.iloc[i,0]; auto = df_pr.iloc[i,1]
            delta = auto - manual
            bar = ax.patches[2*i+1]
            x = bar.get_x() + bar.get_width()/2
            y = bar.get_height()
            ax.text(x, y, f"Δ {delta:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    print(f"[✓] Done: {model_name}/{base} → outputs in {output_dir}")


if __name__ == "__main__":
    for model_name in os.listdir(BASE_INPUT_DIR):
        model_dir = os.path.join(BASE_INPUT_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue

        pattern = os.path.join(model_dir, "*_aligned.csv")
        for input_csv in glob.glob(pattern):
            run_pipeline(input_csv, model_name)
