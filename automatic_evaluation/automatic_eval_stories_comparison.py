#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd

# --- CONFIG ---
BASE_OUTPUT_DIR   = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/output"
OUTPUT_MD         = os.path.join(BASE_OUTPUT_DIR, "comparison.md")
ENTITIES          = ["Victim", "Perpetrator", "Mode of Demise"]
METRICS           = ["P", "R", "F1"]
# substrings to group input types
GROUP_SUBSTRINGS  = ["stories", "specialised summary", "generic summary"]

# helper to clean & title‐case the input type
def parse_input_type(fn, model):
    name, _ = os.path.splitext(fn)
    name = re.sub(r"_merged_report$", "", name)
    parts = name.split("_", 1)
    base  = parts[1] if len(parts) > 1 else parts[0]
    toks  = base.split("_")
    if toks and toks[-1] == "aligned": toks.pop()
    if toks and toks[-1] == model:     toks.pop()
    if toks and toks[0].upper() == "KE": toks.pop(0)
    s     = "_".join(toks) or "unknown"
    s     = re.sub(r"[_]", " ", s)
    s     = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return s.title()

# 1) gather all merged_report paths
paths = glob.glob(os.path.join(BASE_OUTPUT_DIR, "*", "*", "*_merged_report.csv"))

# 2) read & annotate
rows = []
for path in paths:
    model = os.path.basename(os.path.dirname(os.path.dirname(path)))
    fn    = os.path.basename(path)
    itype = parse_input_type(fn, model)
    df    = pd.read_csv(path, sep=";")
    for ent in ENTITIES:
        row = df[df.Entity == ent].iloc[0]
        rows.append({
            "Model":      model,
            "Input Type": itype,
            "Entity":     ent,
            "P":          row.Auto_P,
            "R":          row.Auto_R,
            "F1":         row.Auto_F1
        })

all_df = pd.DataFrame(rows)

# 3) open output file and write grouped tables
with open(OUTPUT_MD, "w") as outf:
    outf.write("# Auto P/R/F1 by Category\n\n")
    for cat in GROUP_SUBSTRINGS:
        mask     = all_df["Input Type"].str.lower().str.contains(cat)
        group_df = all_df[mask]
        if group_df.empty:
            continue
        outf.write(f"## Category: {cat.title()}\n\n")
        # pivot (averaging over duplicate Model×Entity pairs)
        pivot = group_df.pivot_table(
            index="Model",
            columns="Entity",
            values=METRICS,
            aggfunc="mean"
        )
        # reorder entities in desired order
        pivot = pivot.reindex(columns=ENTITIES, level=1)
        # flatten to "Entity Metric"
        pivot.columns = [f"{ent} {met}" for met, ent in pivot.columns]
        # find maxima per column
        maxes = pivot.max()
        # format percentages & bold maxima
        fmt = lambda x, m: f"**{x*100:.1f}%**" if x == m else f"{x*100:.1f}%"
        for col in pivot.columns:
            m = maxes[col]
            pivot[col] = pivot[col].apply(lambda x: fmt(x, m))
        # write this table
        outf.write(pivot.to_markdown() + "\n\n")

print(f"✅ Written grouped comparison tables to {OUTPUT_MD}")
