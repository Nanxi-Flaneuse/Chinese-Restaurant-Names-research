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
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from seqeval.metrics.sequence_labeling import get_entities
import torch
from cjkfuzz import fuzz 
from cjkfuzz import process
import collections
# add tokens here


# helper function that identifies the existance of personal name in phrase
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

### Chinese NER
tokenizer_cn = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
model_cn = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']


#list of Chinese idioms
df_idioms = pd.read_csv('utilities/chinese_idioms.csv')

def classify_research_value(text, language = "English_Name"):
    """Returns True if text contains the specified value, Otherwise returns False"""
    if language == "English_Name":
        ner_results = nlp(text)
        for item in ner_results:
            if "PER" in item['entity']:
                return True
        return False
    elif language == "Chinese_Name":
        
        # helper function
        def get_entity(sentence):
            tokens = tokenizer_cn.tokenize(sentence)
            inputs = tokenizer_cn.encode(sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = model_cn(inputs).logits
            predictions = torch.argmax(outputs, dim=2)
            char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
            # print(sentence)
            # print(char_tags)

            pred_labels = [i[1] for i in char_tags]
            entities = []
            line_entities = get_entities(pred_labels)
            for i in line_entities:
                word = sentence[i[1]: i[2] + 1]
                entity_type = i[0]
                entities.append((word, entity_type))

            # print("Sentence entity:")
            # print(entities)
            return entities
        
        results = get_entity(text)
        for item in results:
            if "PER" in item[1]:
                return True
        return False

# detects how similar a word is to Chinese idiom
def find_idiom(text,df):
    score = process.extract(text, df['word'])[0][0]
    return score #> 0.8 or score == 0.8

def is_answer_in_valid_form(answer):
    """Check if the GPT's answer is in the expected format.

    This is the format we want:
        Readability: 1

    Note: 4.5 will be extracted as 4.
    """
    answer = answer.strip("\n").strip()
    # print("asnwer ---------------------------------------------")
    # print(answer)
    # return True
    if re.search("^[0-1]+$", answer):
        # print("true --------------")
        return True
    return False

def run_gpt4_query(filled_prompt, lang, model):
    print('running gpt -----------------------')
    if lang == "en":
        sys = "Your job is a computational social scientist interested in the names of Chinese restaurants in the U.S. "
    else:
        sys = "您的工作是一名计算社会科学家，对美国的中餐馆名称感兴趣。"
    # print(sys)
    response = client.chat.completions.create(model=model,
    messages=[
        {"role": "system", "content": sys},
        {"role": "user", "content": filled_prompt},
    ],
    temperature=0)
    print('getting response')
    return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15))

### this function prompts the GPT model to generate a response when given a prompt
def generate_categorization(restaurant_en, restaurant_cn, prompt_file, language, model,category = None):
    """Explains a given text for a specific audience.

    Args:
        text (str): The input text to be explained.
        prompt_file (str): The file path to the prompt file.

    Returns:
        str: The explanation of the input text.

    """
    # Read prompt template
    prompt_template = open(prompt_file).read()
    print('prompt read')
    if language == "en":
        prompt = prompt_template.replace("{EN_NAME}", restaurant_en)
        # print("restaurant name:",restaurant_en)
        if 'lexicon' in prompt_file:
            print('lexicon')
            if category == "Personal_Name":
                if not classify_research_value(restaurant_en):
                    prompt = prompt.replace("{TF}", "doesn't")
                else:
                    prompt = prompt.replace("{TF}", "does")
       
    else:
        prompt = prompt_template.replace("{CN_NAME}", restaurant_cn)
        # print("restaurant name:",restaurant_cn)
        if 'lexicon' in prompt_file:
            # print('lexicon')
            if category == "Personal_Name":
                if not classify_research_value(restaurant_cn,language="Chinese_Name"):
                    prompt = prompt.replace("{TF}", "不")
                else:
                    prompt = prompt.replace("{TF}", "")
            elif category == "Pun_Creative":
                prompt = prompt.replace("{SIMILARITY}", str(find_idiom(restaurant_cn,df_idioms)))
    # prompt = prompt.replace("{CN_NAME}", restaurant_cn)
    prompt = prompt.strip("\n").strip()
    prompt = prompt + "\n"
    print(prompt)
    print('prompt generated')
    ## Nanxi testing
    # return True
    # comment out below for prompt generation testing
    while True:
        # print('here---------------')
        response = run_gpt4_query(prompt, lang = language, model = model)
        print('response generated ----------------------')
        response = response.choices[0].message.content.strip("\n")
        print(response)
        # return response
        if is_answer_in_valid_form(response):
            # print(response)
            return response
        else:
            print("====>>> Answer not right, re-submitting request...")
            print(response)


def main():

    # question_type = "prompt_en_Positivity_json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../data_cleaning/output/validation_en.csv") #model_training/
    parser.add_argument("--prompt_file_path", type=str)
    parser.add_argument("--output_folder", type=str) #/model_training
    parser.add_argument("--model", default="gpt-4-0125-preview", type=str)
    parser.add_argument('--output_file',type=str)
    parser.add_argument("--category",type=str)
    # parser.add_argument("--category_cn",type=str,default="氛围")
    parser.add_argument("--language", type = str, default="en")
    parser.add_argument("--prompt_type",type = str)
    args = parser.parse_args()
    ### QUESTION: IS df_test the trianing file?
    df_text = pd.read_csv(args.input_file)#, encoding="utf-8", delimiter="\t")
    df_text = df_text.iloc[:]
    print(df_text.shape)
    # print(df_text)
    output_folder = args.output_folder

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # map audience to its full string
    # short_prompt_folder_name = "102-doctrines-nonexperts"
    # Path(os.path.join(output_folder, short_prompt_folder_name)).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(output_folder, short_prompt_folder_name, question_type)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder)).mkdir(parents=True, exist_ok=True)
    # Path(os.path.join(output_folder, question_type)).mkdir(parents=True, exist_ok=True)

    # normal call to debug
    # for concept_name, text, story in tqdm(zip(df_text.concept.to_list(), df_text.intro_text.to_list(), df_text.story.to_list())):
    #     concept_name = " ".join(concept_name.split("_"))
    #     response = generate_story(text, concept_name, story, args.prompt_file_path)
    #     print(response)
    #     break

    pool = multiprocessing.Pool()

    responses = []
    results = pd.DataFrame(columns=['sample_id','national_id','English_Name','Chinese_Name',args.category])

    for restaurant_en, restaurant_cn, sample_id, national_id in tqdm(zip(df_text.English_Name.to_list(), df_text.Chinese_Name.to_list(), df_text.sample_id.to_list(),df_text.national_id.to_list())):
        # concept_name_string = " ".join(concept_name.split("_"))
        if args.language == "en":
            prompt = "./prompts/English/binary/{}.txt".format(args.prompt_type+args.prompt_file_path)
        else:
            prompt = "./prompts/Chinese/binary/{}.txt".format(args.prompt_type+args.prompt_file_path)
        # prompt = "./prompts/English/{}.txt".format(args.prompt_file_path)
        response = pool.apply_async(generate_categorization, args=(restaurant_en, restaurant_cn, prompt, args.language, args.model,args.category))
        # print(response)
        responses.append([sample_id, national_id, restaurant_en, restaurant_cn, response])
        # print('raw response -------------------------------------')
        # print(responses)

### QUESTION: How do I collect the boolean-value categories of each restaurant? Or can I ask gpt to export a csv file?
    for sample_id, national_id, restaurant_en, restaurant_cn, response in tqdm(responses):
        # json_obj = {"sample_id": sample_id, "national_id": national_id, "English_Name": restaurant_en, "Chinese_Name":restaurant_cn}
        # json_obj["Positivity"] = response.get()
        # print(type(response))
        # print(response)
        results.loc[len(results.index)] = [sample_id, national_id, restaurant_en, restaurant_cn, response.get()]
        # json_obj = json.dumps(json_obj, indent=4)
        # with open(os.path.join(output_folder, short_prompt_folder_name, question_type, "{}.json".format(concept_name)), "w", encoding='UTF-8') as out:
        # with open(os.path.join(output_folder, args.output_file), "w", encoding='UTF-8') as out:

            # out.write(json_obj)
    ########## SWITCH THE CODE WHEN NEEDED. THE SECOND LINE IS FOR USING THE BEST MODEL FOR EACH CATEGORY
    results.to_csv(os.path.join(output_folder, args.prompt_type+args.output_file),encoding='utf-8')
    # results.to_csv(os.path.join(output_folder, args.output_file),encoding='utf-8')

    pool.close()
    pool.join()

if __name__ == "__main__":
    # print(is_answer_in_valid_form('1'))
    # main()
#     prompt='''You will decide if a given restaurant name belongs to the Creative category or no. If a name classifies as Creative, return 1; if not, return 0. 
# Your response can only be 0 or 1, no other explanation is needed.
# You will decide whether the name of the restaurant "Woking Hard" belongs to the Creative category. As long as a name contains one word that classifies as Creative, the entire name classifies as 
# Creative. 

# Here is the criteria for classifying Creative:

# Creative refers to names that contain puns or words used creatively. Here are some examples:
#     - "wok with me" (sounds like “walk with me”), "pho king good" (sounds like “fucking good”), "rice to meet you" (sounds like “nice to meet you”), "Wok N Talk" ('wok' ryhmes with 'talk'), etc'''
#     prompt='''您将决定给定餐厅名称是否属于"人名"类别。如果名称属于"人名"，回应1；如果不是，回应0。
# 您的回应只能是0或1，无需其他解释。
# 请回答餐厅名称南京盐水鸭是否属于"人名"。如果一个名称中包含至少一个被归类为"人名"的词，整个名称就被归类为"人名"。

# 这是归类"人名"的标准：
# 人名包括姓氏，名字，全名或者昵称。

# 根据我们的中文NER模型判断，南京盐水鸭不 存在人名。你可以将此数据作为参考，但该模型的预测不一定准确。
# 加油！如果你成功完成任务会得到一百美元的消费奖励！'''
#     response = run_gpt4_query(prompt, 'cn','gpt-3.5-turbo-0125')
#     response = response.choices[0].message.content.strip("\n")
#     print(response)
    answer = generate_categorization('Panda Express','新茶园','./prompts/English/binary/3.5/few_shot/rule_based/prompt_en_positivity.txt','en','gpt-3.5-turbo-0125')
    print(answer)