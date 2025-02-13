import os
import pandas as pd
from litellm import completion
from tqdm import tqdm

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################
MODEL_NAME = "ollama/deepseek-r1:70b"
API_BASE = "http://localhost:11434"

# Input, output, and error log paths
INPUT_CSV_PATH = ""
OUTPUT_CSV_PATH = ""
ERROR_LOG_PATH = ""

# Column to store your generated summaries
SUMMARY_COLUMN_NAME = "automatic_summary_prompt1_deepseek-r170b"

# LLM token limit and a rough counting function
MAX_TOKENS = 128000


###############################################################################
#                             HELPER FUNCTIONS                                #
###############################################################################
def preprocess_story(story: str) -> str:
    """
    Remove empty line breaks, extra spaces, and trim the story text.
    """
    return " ".join(story.split()).strip()


def count_tokens(text: str) -> int:
    """
    Simple approximation: tokens ~ number of words.
    """
    return len(text.split())


def ollama_query_prompt1(story: str) -> str:
    """
    Query the LLM for a summary of the given story (prompt #1).
    """
    msg = (
        "You are an expert in short stories summarization. "
        "Create a summary of the provided short story:"
        f"\"{story}.\" "
        "Rely STRICTLY on the provided text. It is FORBIDDEN to include any information "
        "that is not present in the text. "
        "Your output is a coherent and cohesive summary that encapsulates "
        "the essence of the given short story in a few sentences. "
        "Make sure to capture all the events of the story. "
        "Return the generated summary only. "
        "Do not write anything else."
    )

    # Send the completion request to the LLM
    response = completion(
        model=MODEL_NAME,
        messages=[{"content": msg, "role": "user"}],
        api_base=API_BASE,
    )

    response_content = response.choices[0]["message"]["content"]
    return response_content


###############################################################################
#                                 MAIN SCRIPT                                 #
###############################################################################
def main():
    """
    Main function to:
      1) Load CSV data
      2) Possibly merge partial results (if OUTPUT_CSV_PATH exists)
      3) Generate missing summaries
      4) Save results progressively
      5) Print a sample story's summary (for work_id == "125672", if present)
    """

    # -----------------------------------------------------------------
    # 1) Load the input CSV
    # -----------------------------------------------------------------
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV_PATH}")

    fanfics_deathprj = pd.read_csv(INPUT_CSV_PATH, low_memory=False)

    # Create the summary column if it doesn't exist
    if SUMMARY_COLUMN_NAME not in fanfics_deathprj.columns:
        fanfics_deathprj[SUMMARY_COLUMN_NAME] = None

    # -----------------------------------------------------------------
    # 2) Merge partial results if output CSV already exists
    # -----------------------------------------------------------------
    if os.path.exists(OUTPUT_CSV_PATH):
        partial_df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
        # Merge partial results by 'work_id'
        if "work_id" in fanfics_deathprj.columns and "work_id" in partial_df.columns:
            partial_dict = partial_df.set_index("work_id")[SUMMARY_COLUMN_NAME].to_dict()
            fanfics_deathprj.set_index("work_id", inplace=True)
            for w_id, summary_val in partial_dict.items():
                if w_id in fanfics_deathprj.index:
                    if pd.isna(fanfics_deathprj.at[w_id, SUMMARY_COLUMN_NAME]) and not pd.isna(summary_val):
                        fanfics_deathprj.at[w_id, SUMMARY_COLUMN_NAME] = summary_val
            fanfics_deathprj.reset_index(inplace=True)
        else:
            print("WARNING: 'work_id' column not found for merging partial results.")

    # -----------------------------------------------------------------
    # 3) Open error log (append mode) and iterate
    # -----------------------------------------------------------------
    error_log = open(ERROR_LOG_PATH, "a", encoding="utf-8")

    for index, row in tqdm(
        fanfics_deathprj.iterrows(),
        total=fanfics_deathprj.shape[0],
        desc="Processing stories"
    ):
        # Skip if we already have a summary
        if pd.notnull(row[SUMMARY_COLUMN_NAME]):
            continue

        work_id = row.get("work_id", index)  # fallback to index if no 'work_id'
        story = row["body"] if "body" in row else ""

        # Preprocess and (optionally) check token limit
        preprocessed_story = preprocess_story(story)
        prompt_length = count_tokens(preprocessed_story)

        # Uncomment if you want to skip extremely long stories
        """
        if prompt_length > MAX_TOKENS:
            error_msg = (f"Error: Story with work_id {work_id} "
                         f"exceeds token limit ({prompt_length} tokens)\n")
            error_log.write(error_msg)
            continue
        """

        # Query the LLM and store the result
        try:
            summary = ollama_query_prompt1(preprocessed_story)
            fanfics_deathprj.at[index, SUMMARY_COLUMN_NAME] = summary
        except Exception as e:
            error_log.write(
                f"Error: LLM query failed for work_id {work_id} with error: {e}\n"
            )
            continue

        # -----------------------------------------------------------------
        # 4) Save after each row (progressive saving)
        # -----------------------------------------------------------------
        fanfics_deathprj.to_csv(OUTPUT_CSV_PATH, index=False)

    error_log.close()
    print("Processing complete.")
    print(f"Results saved to '{OUTPUT_CSV_PATH}'")


if __name__ == "__main__":
    main()
