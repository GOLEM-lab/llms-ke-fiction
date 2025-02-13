import os
import re
import pandas as pd
from tqdm import tqdm
from litellm import completion
import json

# ======================== CONFIGURATION STEP ========================
# Set the paths and configuration variables here
CONFIG = {
    "model": "ollama/deepseek-r1:70b",
    "api_base": "http://localhost:11434",
    "output_file": "",
    "error_log_file": "",
    "input_csv": "",
    "sample_size": None,  # Set to a number to process only that many samples (e.g., 2), or None for all
    "story_row": ""  # Updated to match the column name
}

# ===================================================================

# Function to clean and post-process the LLM response
def clean_llm_response(response):
    response = re.sub(r"Here are the answers:\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"Let me know.*", "", response, flags=re.IGNORECASE)
    response = ' '.join(response.split()).strip()
    return response

json_string = (
    "{ \"work_id\": \"<work_id>\","
    " \"death_event_1\": {\"Character Death\": \"\", \"Mode of Demise\": \"\", \"Victim\": \"\", \"Perpetrator\": \"\"},"
    " \"death_event_2\": {\"Character Death\": \"\", \"Mode of Demise\": \"\", \"Victim\": \"\", \"Perpetrator\": \"\"} }"
)

# Function to interact with LLM for a single story
def ollama_query_QA_prompt1(work_id, story):
    msg = (
        "Answer the following questions based on the provided story summary.\n"
        f"Story summary: \"{story}\".\n"
        "For each death event reported in the summary, answer the following:\n"
        "1) Character Death (Yes/No)\n"
        "2) Mode of Demise\n"
        "3) Victim\n"
        "4) Perpetrator\n"
        "Your answers must be made using a single word or as few words as possible. For example:\n"
        "\"Character Death\":\"yes\"; \"Mode of Demise\":\"tearing to pieces\"; \"Victim\":\"Orpheus\"; \"Perpetrator\":\"Maenads\".\n"
        "There may be multiple death events in the summary. Please provide separate answers for each event.\n"
        "You MUST STRICTLY RELY on the PROVIDED SUMMARY ONLY. You MUST NOT provide answers based on any information outside the text."
        "Each group of answers for each death event should be output in JSONL format, as in the following example:\n"
        + json_string + "\n"
        f"For this summary, use \"{work_id}\" as the work identifier.\n"
        "You MUST Return the generated JSONL only."
        "Do not write anything else."
    )

    try:
        response = completion(
            model=CONFIG["model"],
            messages=[{"content": msg, "role": "user"}],
            api_base=CONFIG["api_base"],
        )
        response_content = response.choices[0]["message"]["content"]
        return clean_llm_response(response_content)

    except Exception as e:
        print(f"Error during API call for work_id {work_id}: {str(e)}")
        return None

# Check if the presentation_mode_output file exists
if os.path.exists(CONFIG["output_file"]):
    existing_results_df = pd.read_json(CONFIG["output_file"], orient='records', lines=True)
    processed_work_ids = set(existing_results_df['work_id'].unique())
else:
    processed_work_ids = set()

# Load the dataset
fanfics_deathprj = pd.read_csv(CONFIG["input_csv"])

# If sample_size is specified, take only the first n samples
if CONFIG["sample_size"]:
    fanfics_deathprj = fanfics_deathprj.head(CONFIG["sample_size"])

# Open the error log file in append mode
with open(CONFIG["error_log_file"], 'a') as error_log:
    for index, row in tqdm(fanfics_deathprj.iterrows(), total=fanfics_deathprj.shape[0], desc="Processing stories"):
        work_id = row['work_id']

        # Skip if already processed
        if work_id in processed_work_ids:
            continue

        # Access the story row using CONFIG
        story = row[CONFIG["story_row"]]

        # Call the function to get the LLM's response
        answers = ollama_query_QA_prompt1(work_id, story)

        # If answers is None, log the error and continue
        if answers is None:
            error_log.write(f"Error: Failed to generate response for work_id {work_id}\n")
            continue

        # Prepare the result as a dictionary
        result = {
            'work_id': work_id,
            'QuestionAnswering_prompt1_deepseek-r170b': story,
            'LLM_response': answers
        }

        # Save the result to the presentation_mode_output file
        with open(CONFIG["output_file"], 'a') as f:
            f.write(json.dumps(result) + '\n')

        # Add the work_id to processed_work_ids
        processed_work_ids.add(work_id)

print(f"Processing complete. Results saved to '{CONFIG['output_file']}'")
