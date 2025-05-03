import pandas as pd
import os
import re

# =====================================================
# 1) CONFIG SECTION
# =====================================================
INPUT_FOLDER      = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/input/df"
OUTPUT_FOLDER     = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/input/gold_pred_mapped"
GOLD_FILE         = "/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/input/gold_standard_300425.csv"
INCLUDE_SUMMARY_P = True

FINAL_COLUMNS_ORDER = [
    "work_id",
    "Title",
    "Link",
    "(Major) Character Death Evidence",
    "Presentation_Mode",
    "Narrator_POV",
    "Victim",
    "Victim_P",
    "Victim_TP",
    "Victim_FP",
    "Victim_FN",
    "Victim_TN",
    "Perpetrator",
    "Perpetrator_P",
    "Perpetrator_TP",
    "Perpetrator_FP",
    "Perpetrator_FN",
    "Perpetrator_TN",
    "Mode of Demise",
    "Mode of Demise_P",
    "Mode of Demise_TP",
    "Mode of Demise_FP",
    "Mode of Demise_FN",
    "Mode of Demise_TN",
]


# =====================================================
# 2) UTILITY FUNCTIONS
# =====================================================
def clean_work_id(work_id):
    if not isinstance(work_id, str):
        work_id = str(work_id)
    work_id = work_id.replace('\ufeff', '')
    return re.sub(r'\.0$', '', work_id)

def clean_summary(summary):
    if pd.isna(summary):
        return ''
    return summary.replace('\n', ' ').replace('\r', ' ')

def is_empty(value):
    if pd.isna(value):
        return True
    v = str(value).strip().lower()
    return v in ['', '---', 'unspecified', 'unknown']

def partial_match(gold_value, predicted_value):
    if is_empty(gold_value) or is_empty(predicted_value):
        return False
    g, p = str(gold_value).lower(), str(predicted_value).lower()
    return (g in p) or (p in g)


# =====================================================
# 3) CORE PROCESS
# =====================================================
def process_files(input_folder, output_folder, gold_df):
    for root, _, files in os.walk(input_folder):
        # compute where in the tree we are
        rel_dir = os.path.relpath(root, input_folder)
        # build the matching tuning_output directory
        if rel_dir == '.':
            out_dir = output_folder
        else:
            out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            if not fname.endswith(".csv"):
                continue

            input_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]

            # 1) load silver
            silver_df = pd.read_csv(input_path)
            silver_df['work_id'] = silver_df['work_id'].apply(clean_work_id)
            silver_df = silver_df[silver_df['work_id'].isin(gold_df['work_id'])]
            if silver_df.empty:
                print(f"[SKIP] no matching work_ids for {input_path}")
                continue

            # 2) prep a fresh gold copy
            curr = gold_df.copy()
            for col in ['Mode of Demise','Victim','Perpetrator']:
                pcol = f"{col}_P"
                if pcol not in curr.columns:
                    curr[pcol] = ''
            if 'summary_P' not in curr.columns:
                curr['summary_P'] = ''

            # 3) align each silver row into curr
            for _, srow in silver_df.iterrows():
                wid   = srow['work_id']
                vic   = srow.get('Victim','')
                mode  = srow.get('Mode of Demise','')
                perp  = srow.get('Perpetrator','')
                summ  = clean_summary(srow.get('summary',''))

                subset = curr[curr['work_id']==wid]
                vic_match = subset[ subset['Victim'].apply(lambda x: partial_match(x, vic)) ]

                if not vic_match.empty:
                    idx0 = vic_match.index[0]
                    curr.at[idx0, 'Victim_P']         = vic
                    curr.at[idx0, 'Mode of Demise_P'] = mode
                    curr.at[idx0, 'Perpetrator_P']    = perp
                    curr.at[idx0, 'summary_P']        = summ
                else:
                    # no partial victim match → new row
                    new = {c: None for c in curr.columns}
                    new.update({
                        'work_id': wid,
                        'Victim_P': vic,
                        'Mode of Demise_P': mode,
                        'Perpetrator_P': perp,
                        'summary_P': summ
                    })
                    curr = pd.concat([curr, pd.DataFrame([new])], ignore_index=True)

            # 4) ensure TP/FP/FN/TN columns
            for col in ['Mode of Demise','Victim','Perpetrator']:
                for m in ['TP','FP','FN','TN']:
                    cname = f"{col}_{m}"
                    if cname not in curr.columns:
                        curr[cname] = 0

            # 5) populate metrics
            # for i, row in curr.iterrows():
            #     for col in ['Mode of Demise','Victim','Perpetrator']:
            #         gval = row[col]
            #         pval = row[f"{col}_P"]
            #         empty_g = is_empty(gval)
            #         empty_p = is_empty(pval)
            #
            #         if not empty_g and not empty_p:
            #             if partial_match(gval, pval):
            #                 curr.at[i, f"{col}_TP"] = 1
            #             else:
            #                 curr.at[i, f"{col}_FP"] = 1
            #                 curr.at[i, f"{col}_FN"] = 1
            #         elif empty_g and not empty_p:
            #             curr.at[i, f"{col}_FP"] = 1
            #         elif not empty_g and empty_p:
            #             curr.at[i, f"{col}_FN"] = 1
            #         else:
            #             curr.at[i, f"{col}_TN"] = 1


            # 6) rename for tuning_output schema
            curr.rename(
                columns={
                    "Presentation mode": "Presentation_Mode",
                    "Narrator POV":      "Narrator_POV"
                },
                inplace=True
            )

            # 7) select & order columns
            cols = FINAL_COLUMNS_ORDER.copy()
            if INCLUDE_SUMMARY_P:
                cols.append("summary_P")
            existing = [c for c in cols if c in curr.columns]
            aligned_df = curr[existing]

            # 8) save aligned
            aligned_df.to_csv(os.path.join(out_dir, f"{base}_aligned.csv"), index=False)

            # 9) field-level metrics
            metrics = []
            for col in ['Mode of Demise','Victim','Perpetrator']:
                tp = curr[f"{col}_TP"].sum()
                fp = curr[f"{col}_FP"].sum()
                fn = curr[f"{col}_FN"].sum()
                prec = tp/(tp+fp) if tp+fp else 0
                rec  = tp/(tp+fn) if tp+fn else 0
                f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0
                metrics.append({
                    'Column': col, 'TP': tp, 'FP': fp, 'FN': fn,
                    'Precision': prec, 'Recall': rec, 'F1 Score': f1
                })
            pd.DataFrame(metrics).to_csv(
                os.path.join(out_dir, f"{base}_field_metrics.csv"), index=False
            )

            # 10) work_id-level metrics
            wid_mets = []
            for wid, grp in curr.groupby('work_id'):
                tp = fp = fn = 0
                for col in ['Mode of Demise','Victim','Perpetrator']:
                    tp += grp[f"{col}_TP"].sum()
                    fp += grp[f"{col}_FP"].sum()
                    fn += grp[f"{col}_FN"].sum()
                prec = tp/(tp+fp) if tp+fp else 0
                rec  = tp/(tp+fn) if tp+fn else 0
                f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0
                wid_mets.append({
                    'work_id': wid, 'TP': tp, 'FP': fp, 'FN': fn,
                    'Precision': prec, 'Recall': rec, 'F1 Score': f1
                })
            pd.DataFrame(wid_mets).to_csv(
                os.path.join(out_dir, f"{base}_workid_metrics.csv"), index=False
            )

            print(f"[DONE] {input_path}  →  {out_dir}")


# =====================================================
# 4) MAIN
# =====================================================
def main():
    gold_df = pd.read_csv(GOLD_FILE, sep=";")
    gold_df['work_id'] = gold_df['work_id'].apply(clean_work_id)
    process_files(INPUT_FOLDER, OUTPUT_FOLDER, gold_df)

if __name__ == "__main__":
    main()
