import json
import openai
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing

openai.organization = "org-cHHrGYoOFHuN8rXScgYupSCk"
openai.api_key = "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"

# openai.organization = "org-8LzkZF7K2vZgan52qUQgtUrZ"
# openai.api_key = "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"

def run_gpt4_query(filled_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": filled_prompt},
        ],
        temperature=0,
    )
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))
def generate_story(text, concept, story, prompt_file):
    """Explains a given text for a specific audience.

    Args:
        text (str): The input text to be explained.
        prompt_file (str): The file path to the prompt file.

    Returns:
        str: The explanation of the input text.

    """
    # Read prompt template
    prompt_template = open(prompt_file).read()

    prompt = prompt_template.replace("{TEXT}", text)
    prompt = prompt.replace("{CONCEPT}", concept)
    # replace the beginning of the simplification
    # story = story.replace("Concept Simplified:", "").strip("\n")
    prompt = prompt.replace("{STORY}", story)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n"
    # print(prompt)
    response = run_gpt4_query(prompt)
    response = response["choices"][0]['message']['content'].strip("\n")
    return response

def main():

    question_type = "limitation_question"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./law/stories/102_doctrines_stories.csv")
    parser.add_argument("--prompt_file_path", default="./prompts/{}.txt".format(question_type), type=str)
    parser.add_argument("--output_folder", type=str, default="./law/outputs/")
    parser.add_argument("--model", default="gpt-4", type=str)
    args = parser.parse_args()

    df_text = pd.read_csv(args.input_file, encoding="utf-8", delimiter="\t")
    df_text = df_text.iloc[:]
    print(df_text.shape)
    print(df_text)
    output_folder = args.output_folder

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # map audience to its full string
    short_prompt_folder_name = "102-doctrines-nonexperts"
    Path(os.path.join(output_folder, short_prompt_folder_name)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder, short_prompt_folder_name, question_type)).mkdir(parents=True, exist_ok=True)

    # normal call to debug
    # for concept_name, text, story in tqdm(zip(df_text.concept.to_list(), df_text.intro_text.to_list(), df_text.story.to_list())):
    #     concept_name = " ".join(concept_name.split("_"))
    #     response = generate_story(text, concept_name, story, args.prompt_file_path)
    #     print(response)
    #     break

    pool = multiprocessing.Pool()

    responses = []

    for concept_name, text, story in tqdm(zip(df_text.concept.to_list(), df_text.intro_text.to_list(), df_text.story.to_list())):
        concept_name_string = " ".join(concept_name.split("_"))
        response = pool.apply_async(generate_story, args=(text, concept_name, story, args.prompt_file_path))
        responses.append([concept_name, text, story, response])

    for concept_name, text, story, response in tqdm(responses):
        json_obj = {"id": concept_name, "text": text, "story": story}
        json_obj["annotation"] = response.get()
        json_obj = json.dumps(json_obj, indent=4)
        with open(os.path.join(output_folder, short_prompt_folder_name, question_type, "{}.json".format(concept_name)), "w", encoding='UTF-8') as out:
            out.write(json_obj)

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()