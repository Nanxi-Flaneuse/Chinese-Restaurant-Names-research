import json
import openai
from openai import OpenAI

import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
import argparse
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_random_exponential
import multiprocessing
import re

client = OpenAI(api_key="")
openai.organization = ""



# def is_answer_in_valid_form(answer):
#     """Check if the GPT's answer is in the expected format.

#     This is the format we want:
#         Readability: 1

#     Note: 4.5 will be extracted as 4.
#     """
#     answer = answer.strip("\n").strip()
#     if re.search("^[0-1]+$", answer):
#         # print("true --------------")
#         return True
#     return False

def run_gpt4_query(filled_prompt, model):

    # GPT API
    # print('gpt3.5 running')
    sys = "You are a bilingual translator who translates Chinese into English and you know both languages very well."
    response = client.chat.completions.create(model= model, #"gpt-3.5-turbo", # #
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": filled_prompt},
    ],
    temperature=0)
    # print('retrieved answer')
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))

### this function prompts the GPT model to generate a response when given a prompt
def translate(restaurant_cn, prompt_file, model):
    """Explains a given text for a specific audience.

    Args:
        text (str): The input text to be explained.
        prompt_file (str): The file path to the prompt file.

    Returns:
        str: The explanation of the input text.

    """
    # Read prompt template
    prompt_template = open(prompt_file).read()
    prompt = prompt_template.replace("{CN_NAME}", restaurant_cn)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n"

    # comment out below for prompt generation testing
    # print("before running chatgpt")
    response = run_gpt4_query(prompt, model)
    # print("after running chatgpt")
        # print('response generated ----------------------')
    response = response.choices[0].message.content.strip("\n")
    # print(response)
    return response
    # while True:
    #     print('here---------------')
    #     # print(prompt)
    #     response = run_gpt4_query(prompt)
    #     # print('response generated ----------------------')
    #     response = response["choices"][0]['message']['content'].strip("\n")
    #     return response


def main():

    # question_type = "prompt_en_Positivity_json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./sample.csv") #model_training/
    parser.add_argument("--prompt_file_path", default="./prompts/GPT4/prompt_1.txt", type=str)
    parser.add_argument("--output_folder", type=str, default="./outputs") #/model_training
    parser.add_argument("--model", default="gpt-4-0125-preview", type=str)
    parser.add_argument('--output_file',type=str, default = 'gpt4.csv')
    parser.add_argument('--col_name',type=str, default = 'gpt4_translation')
    args = parser.parse_args()

    df_text = pd.read_csv(args.input_file)#, encoding="utf-8", delimiter="\t")
    df_text = df_text.iloc[:]
    print(df_text.shape)
    # print(df_text)
    output_folder = args.output_folder

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    Path(os.path.join(output_folder)).mkdir(parents=True, exist_ok=True)


    pool = multiprocessing.Pool()

    responses = []
    results = pd.DataFrame(columns=['English_Name','Chinese_Name','nanxi_translation','rukun_translation', args.col_name])

    for restaurant_en, restaurant_cn, nanxi_translation, rukun_translation in tqdm(zip(df_text.English_Name.to_list(), df_text.Chinese_Name.to_list(), df_text.nanxi_translation.to_list(),df_text.rukun_translation.to_list())):
        prompt = args.prompt_file_path
        response = pool.apply_async(translate, args=(restaurant_cn, prompt, args.model))
        responses.append([restaurant_en, restaurant_cn, nanxi_translation, rukun_translation, response])
        print('raw response -------------------------------------')
        print(responses)

### QUESTION: How do I collect the boolean-value categories of each restaurant? Or can I ask gpt to export a csv file?
    for restaurant_en, restaurant_cn, nanxi_translation, rukun_translation, response in tqdm(responses):
        results.loc[len(results.index)] = [restaurant_en, restaurant_cn, nanxi_translation, rukun_translation, response.get()]

    results.to_csv(os.path.join(output_folder, args.output_file),encoding='utf-8')

    pool.close()
    pool.join()

if __name__ == "__main__":
    # print(is_answer_in_valid_form('1'))
    main()
    # print("running code")
    # print(translate('新茶园','./prompts/prompt_0.txt','gpt-3.5-turbo'))