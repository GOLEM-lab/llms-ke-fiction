import pandas as pd
import json
import re
from pathlib import Path

# ----------------------------------------
# 1) CONFIG SECTION
# ----------------------------------------
INPUT_FOLDER = Path("/home/arianna/PycharmProjects/llms-ke-fiction/llm_output_cleaned")
OUTPUT_BASE_FOLDER = Path("/home/arianna/PycharmProjects/llms-ke-fiction/automatic_evaluation/input/df")
FILENAME_REGEX = r".*_(?P<llm>[^_]+)\.jsonl$"
WORK_ID_COLUMN   = "work_id"
LLM_COLUMN       = "LLM_response"
SUMMARY_COLUMN   = "full_story"
STRIP_THINK_BLOCKS = True

# ----------------------------------------
# 2) UTILITY FUNCTIONS
# ----------------------------------------

def try_fix_json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt to close any unclosed braces
        if not json_str.strip().endswith('}'):
            json_str = json_str.strip() + '}'
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_llm_json(llm_response: str):
    """
    Extract all JSON objects from fenced ```json``` or ```jsonl``` blocks,
    and from any inline JSON or JSONL in the text.
    """
    death_events = []

    # 1) Handle fenced code blocks
    fence_blocks = re.findall(
        r"```jsonl?\s*(.*?)```",
        llm_response,
        flags=re.DOTALL | re.IGNORECASE
    )
    for block in fence_blocks:
        # find any JSON object in the block
        for obj_str in re.findall(r"\{.*?\}", block, flags=re.DOTALL):
            obj = try_fix_json(obj_str)
            if obj is not None:
                death_events.append(obj)

    # 2) Remove all fenced blocks from the text to avoid double-parsing
    cleaned = re.sub(
        r"```jsonl?\s*.*?```",
        "",
        llm_response,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 3) Parse any inline JSON objects
    decoder = json.JSONDecoder()
    idx = 0
    n = len(cleaned)
    while idx < n:
        try:
            obj, end_idx = decoder.raw_decode(cleaned, idx)
            death_events.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            snippet = cleaned[idx:]
            obj = try_fix_json(snippet)
            if obj is not None:
                death_events.append(obj)
            break
        # skip whitespace
        while idx < n and cleaned[idx].isspace():
            idx += 1

    return death_events


def process_llm_responses(df, work_id_col, llm_col, summary_col):
    all_rows = []
    for _, row in df.iterrows():
        llm_response = row.get(llm_col, "") or ""

        if STRIP_THINK_BLOCKS and isinstance(llm_response, str):
            llm_response = strip_think_blocks(llm_response)

        events = extract_llm_json(llm_response)
        if not events:
            continue

        for event in events:
            # If top-level keys are death_event_X
            for key, val in event.items():
                if key.startswith("death_event") and isinstance(val, dict):
                    all_rows.append({
                        "work_id":        row.get(work_id_col, ""),
                        "summary":        row.get(summary_col, "").replace("\n", " "),
                        "Mode of Demise": val.get("Mode of Demise", ""),
                        "Murder":         val.get("Murder", ""),
                        "Victim":         val.get("Victim", ""),
                        "Perpetrator":    val.get("Perpetrator", "")
                    })
            # If event itself is a death event dict
            if all(k in event for k in ["Mode of Demise", "Victim"]):
                all_rows.append({
                    "work_id":        row.get(work_id_col, ""),
                    "summary":        row.get(summary_col, "").replace("\n", " "),
                    "Mode of Demise": event.get("Mode of Demise", ""),
                    "Murder":         event.get("Murder", ""),
                    "Victim":         event.get("Victim", ""),
                    "Perpetrator":    event.get("Perpetrator", "")
                })

    return pd.DataFrame(
        all_rows,
        columns=["work_id", "summary", "Mode of Demise", "Murder", "Victim", "Perpetrator"]
    )

# ----------------------------------------
# 3) MAIN SCRIPT
# ----------------------------------------

def main():
    OUTPUT_BASE_FOLDER.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(FILENAME_REGEX)

    for infile in INPUT_FOLDER.glob("*.jsonl"):
        m = pattern.match(infile.name)
        if not m:
            print(f"Skipping unmatched file: {infile.name}")
            continue

        llm_name = m.group("llm")
        llm_folder = OUTPUT_BASE_FOLDER / llm_name
        llm_folder.mkdir(parents=True, exist_ok=True)
        outfile = llm_folder / infile.with_suffix(".csv").name

        df = pd.read_json(infile, lines=True)
        final_df = process_llm_responses(df, WORK_ID_COLUMN, LLM_COLUMN, SUMMARY_COLUMN)
        final_df.to_csv(outfile, index=False)
        print(f"Processed {infile.name} → {outfile}")

if __name__ == "__main__":
    main()