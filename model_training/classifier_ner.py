from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from seqeval.metrics.sequence_labeling import get_entities
import torch
import re


##### English NER ##################################
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# example = "Sumiao's Hunan Kitchen"

# ner_results = nlp(example)
# def find_name(result):
#     for item in result:
#         if "PER" in item['entity']:
#             return True
#     return False
# print(find_name(ner_results))
# print(ner_results)

##### Chinese NER ##################################
tokenizer_cn = AutoTokenizer.from_pretrained("shibing624/bert4ner-base-chinese")
model_cn = AutoModelForTokenClassification.from_pretrained("shibing624/bert4ner-base-chinese")
label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']

sentence = "小明餐厅"


def get_entity(sentence):
    tokens = tokenizer_cn.tokenize(sentence)
    inputs = tokenizer_cn.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model_cn(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]
    print(sentence)
    print(char_tags)

    pred_labels = [i[1] for i in char_tags]
    entities = []
    line_entities = get_entities(pred_labels)
    for i in line_entities:
        word = sentence[i[1]: i[2] + 1]
        entity_type = i[0]
        entities.append((word, entity_type))

    print("Sentence entity:")
    print(entities)
    return entities


# result = get_entity(sentence)
def find_name(results):
    for item in results:
        if "PER" in item[1]:
            return True
    return False
print("Name in sentence:")
print(find_name(get_entity(sentence)))