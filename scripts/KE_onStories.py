import os
import re
import pandas as pd
from tqdm import tqdm
from litellm import completion
import json

# Function to clean and post-process the LLM response (remove unwanted text and line breaks)
def clean_llm_response(response):
    # Remove any prefix like "Here are the answers:" or similar phrases
    response = re.sub(r"Here are the answers:\s*", "", response, flags=re.IGNORECASE)
    # Remove trailing phrases like "Let me know if you need any further assistance!"
    response = re.sub(r"Let me know.*", "", response, flags=re.IGNORECASE)
    # Remove extra line breaks and unnecessary whitespace
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
        "Answer the following questions based on the provided short story.\n"
        f"Short story: \"{story}\".\n"
        "For each death event reported in the story, answer the following:\n"
        "1) Character Death (Yes/No)\n"
        "2) Mode of Demise\n"
        "3) Victim\n"
        "4) Perpetrator\n"
        "Your answers must be made using a single word or as few words as possible. For example:\n"
        "\"Character Death\":\"yes\"; \"Mode of Demise\":\"tearing to pieces\"; \"Victim\":\"Orpheus\"; \"Perpetrator\":\"Maenads\".\n"
        "There may be multiple death events in the story. Please provide separate answers for each event.\n"
        "You MUST STRICTLY RELY on the PROVIDED STORY ONLY. You MUST NOT provide answers based on any information outside the text."
        "Each group of answers for each death event should be presentation_mode_output in JSONL format, as in the following example:\n"
        + json_string + "\n"
        f"For this story, use \"{work_id}\" as the work identifier.\n"
        "You MUST Return the generated JSONL only."
        "Do not write anything else."
    )

    # Interact with the LLM model
    try:
        response = completion(
            model="ollama/deepseek-r1:70b",
            messages=[{"content": msg, "role": "user"}],
            api_base="http://localhost:11434",
        )

        # Extract the content from the LLM response
        response_content = response.choices[0]["message"]["content"]

        # Clean and return the response
        return clean_llm_response(response_content)

    except Exception as e:
        print(f"Error during API call for work_id {work_id}: {str(e)}")
        return None

# Define the presentation_mode_output and error log file paths
output_file = ''
error_log_file = ''

# Check if the presentation_mode_output file exists
if os.path.exists(output_file):
    # Load existing results to avoid reprocessing
    existing_results_df = pd.read_json(output_file, orient='records', lines=True)
    processed_work_ids = set(existing_results_df['work_id'].unique())
else:
    processed_work_ids = set()

# Load the dataset
fanfics_deathprj = pd.read_csv("")

#take only 2 samples
#fanfics_deathprj = fanfics_deathprj.head(2)

# Open the error log file in append mode
with open(error_log_file, 'a') as error_log:
    # Iterate over each row in the DataFrame with a progress bar
    for index, row in tqdm(fanfics_deathprj.iterrows(), total=fanfics_deathprj.shape[0], desc="Processing stories"):
        work_id = row['work_id']

        # Skip if already processed
        if work_id in processed_work_ids:
            continue  # Skip already processed stories

        story = row['body']  # Extract the story

        # Call the function to get the LLM's response
        answers = ollama_query_QA_prompt1(work_id, story)

        # If answers is None, log the error and continue
        if answers is None:
            error_log.write(f"Error: Failed to generate response for work_id {work_id}\n")
            continue

        # Prepare the result as a dictionary
        result = {
            'work_id': work_id,
            'full_story': story,
            'LLM_response': answers
        }

        # Save the result to the presentation_mode_output file
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

        # Add the work_id to processed_work_ids
        processed_work_ids.add(work_id)

print(f"Processing complete. Results saved to '{output_file}'")
