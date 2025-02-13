import os
import pandas as pd
from litellm import completion
from tqdm import tqdm

###############################################################################
#                              CONFIGURATION                                  #
###############################################################################
# Adjust these paths and parameters as needed.
MODEL_NAME = "ollama/deepseek-r1:70b"
API_BASE = "http://localhost:11434"

# Paths to input CSV, output CSV, error log, etc.
INPUT_CSV_PATH = ""
OUTPUT_CSV_PATH = ""
ERROR_LOG_PATH = ""

# Name of the column in which to store the generated summaries.
SUMMARY_COLUMN_NAME = "automatic_summary_prompt2_deepseek-r1-70b_specialised"

# LLM token limit / approximation
MAX_TOKENS = 128000


###############################################################################
#                             HELPER FUNCTIONS                                #
###############################################################################
def preprocess_story(story: str) -> str:
    """
    Remove empty line breaks, extra spaces, and trim the story to clean it up.
    """
    return " ".join(story.split()).strip()


def count_tokens(text: str) -> int:
    """
    Simple approximation: tokens ~ number of words.
    """
    return len(text.split())


def ollama_query_prompt2(story: str) -> str:
    """
    Query the LLM with the given story, returning the summary text only.
    """
    msg = (
        "You are an expert in short stories summarization. "
        "Create a summary of the provided short story:\n"
        f"\"{story}.\"\n"
        "Make sure to include information about all the characters death mentioned, "
        "specifying who is/are the murderer(s), what is/are the mode(s) of demise, "
        "who is/are the victim(s), who is/are the perpetrator(s). "
        "Rely STRICTLY on the provided text. It is FORBIDDEN to include any information "
        "that is not present in the text. "
        "Your output is a coherent and cohesive summary that encapsulates "
        "the essence of the given short story in a few sentences. "
        "Make sure to capture all the events of the story. "
        "Return the generated summary only. "
        "Do not write anything else."
    )

    response = completion(
        model=MODEL_NAME,
        messages=[{"content": msg, "role": "user"}],
        api_base=API_BASE,
    )
    return response.choices[0]["message"]["content"]


###############################################################################
#                                 MAIN SCRIPT                                 #
###############################################################################
def main():
    """
    Main script that:
      1) Loads the input CSV
      2) Preprocesses each story
      3) Checks token limit
      4) Generates summaries (if not already present)
      5) Saves results progressively
    """
    # ----------------
    # 1) Load the data
    # ----------------
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV_PATH}")

    # Read the input CSV into a DataFrame
    fanfics_deathprj = pd.read_csv(INPUT_CSV_PATH, low_memory=False)

    # If the summary column does not exist, create it. Otherwise, we'll keep existing data.
    if SUMMARY_COLUMN_NAME not in fanfics_deathprj.columns:
        fanfics_deathprj[SUMMARY_COLUMN_NAME] = None

    # --------------------------------------------------------------------
    # 2) If an output CSV already exists, read it to skip processed rows
    #    (so if there's a crash or interruption, we pick up where we left)
    # --------------------------------------------------------------------
    if os.path.exists(OUTPUT_CSV_PATH):
        partial_df = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
        # Merge partial results back into the main DataFrame
        # - We'll match by a unique key, presumably 'work_id' is unique.
        if "work_id" in fanfics_deathprj.columns:
            # Use 'work_id' as a key to update rows:
            partial_dict = partial_df.set_index("work_id")[SUMMARY_COLUMN_NAME].to_dict()
            fanfics_deathprj.set_index("work_id", inplace=True)
            # Update only rows that are NaN in the main DataFrame
            for w_id, summary_val in partial_dict.items():
                if w_id in fanfics_deathprj.index:
                    if pd.isna(fanfics_deathprj.at[w_id, SUMMARY_COLUMN_NAME]) and not pd.isna(summary_val):
                        fanfics_deathprj.at[w_id, SUMMARY_COLUMN_NAME] = summary_val
            fanfics_deathprj.reset_index(inplace=True)
        else:
            print(
                "WARNING: 'work_id' column not found for merging partial results. "
                "Will not be able to skip already-processed rows."
            )

    # ----------------------------------------------
    # 3) Prepare to log errors (stories over token limit, etc.)
    # ----------------------------------------------
    error_log = open(ERROR_LOG_PATH, "a", encoding="utf-8")

    # ----------------------------------------------
    # 4) Iterate over rows and generate summaries
    # ----------------------------------------------
    for index, row in tqdm(
        fanfics_deathprj.iterrows(),
        total=fanfics_deathprj.shape[0],
        desc="Processing stories"
    ):
        # Skip if we already have a summary in the target column
        if pd.notnull(row[SUMMARY_COLUMN_NAME]):
            continue

        work_id = row.get("work_id", index)  # Fallback to index if no 'work_id'
        story = row["body"] if "body" in row else ""
        preprocessed_story = preprocess_story(story)
        token_count = count_tokens(preprocessed_story)

        # Check token limit
        if token_count > MAX_TOKENS:
            error_msg = (
                f"Error: Story with work_id {work_id} exceeds token limit "
                f"({token_count} tokens)\n"
            )
            error_log.write(error_msg)
            # Skip this story
            continue

        # Query the LLM to get the summary
        try:
            summary = ollama_query_prompt2(preprocessed_story)
        except Exception as e:
            error_log.write(
                f"Error: LLM query failed for work_id {work_id} with error: {str(e)}\n"
            )
            continue

        # Store the summary back into the DataFrame
        fanfics_deathprj.at[index, SUMMARY_COLUMN_NAME] = summary

        # ----------------------------------------------
        # 5) Progressively save after each row
        #    This ensures we don't lose progress if we crash.
        # ----------------------------------------------
        fanfics_deathprj.to_csv(OUTPUT_CSV_PATH, index=False)

    # Close error log
    error_log.close()

    print("Processing complete.")
    print(f"Results saved to '{OUTPUT_CSV_PATH}'")


if __name__ == "__main__":
    main()
