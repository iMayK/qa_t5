from __future__ import print_function

import torch

import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm


from collections import Counter
import string
import re
import argparse
import json
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


#model = T5ForConditionalGeneration.from_pretrained('models/tpu') # because its loaded on xla by default
#tokenizer = T5Tokenizer.from_pretrained('models/tpu')
#valid_dataset = torch.load('valid_data.pt')
#dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

# answers = []
# for batch in tqdm(dataloader):
#   outs = model.generate(input_ids=batch['input_ids'], 
#                         attention_mask=batch['attention_mask'],
#                         max_length=16,
#                         early_stopping=True)
#   outs = [tokenizer.decode(ids) for ids in outs]
#   answers.extend(outs)

# json.dump(answers,open("predicted_eval.json","w"))
# print("wrote answers")

# predictions = []
# references = []
# for ref, pred in zip(valid_dataset, answers):
#   predictions.append(pred)
#   references.append(ref['answers'][0]['text'])

# evaluate(references, predictions)

def get_answer(question, context,tokenizer,model):
  input_text = "question: %s  context: %s" % (question, context)

  features = tokenizer.encode_plus(input_text,pad_to_max_length=True, max_length=512)
  #print(features)

  output = model.generate(input_ids=torch.tensor(features['input_ids']).unsqueeze(0), 
               attention_mask=torch.tensor(features['attention_mask']).unsqueeze(0),max_length=16,
                        early_stopping=True)

  return tokenizer.decode(output[0])

question = "who is the president of USA ?"
context = "Joe Biden is the president of USA."
context = context.replace("\n","")
model = T5ForConditionalGeneration.from_pretrained('models/tpu') # because its loaded on xla by default
tokenizer = T5Tokenizer.from_pretrained('models/tpu')
print(get_answer(question,context,tokenizer,model))
tdata = json.load(open("test_data/test_v_with_SSS.json"))
answers ={}
for k,v in tdata.items():
    question = v["question"]
    # for i, cont in enumerate(v["context"]):
    context = " ".join(v["context"]).replace("\n","")
    #context = context.replace("\n","")
    answers[str(k)] = get_answer(question,context,tokenizer,model)
    print(len(answers))
json.dump(answers,open("test_data/answers_v_with_SSS.json","w"))

# predicted_answers = json.load(open("test_data/answers.json"))
# id_pred_answers = {}
# for i,answer in predicted_answers.items():
#     id_pred_answers[i] = normalize_answer(answer)
# gt_answers = json.load(open("test_data/gt_ans.json"))["annotations"]
# id_gt_answers = {}
# for anot in gt_answers:
#     answers = anot["answers"]
#     answers_list = []
#     for ans in answers:
#         answers_list.append(ans["answer"])
#     id_gt_answers[str(anot["question_id"]).strip()] = list(set(answers_list))
# #print(id_gt_answers.keys())

# final ={}
# for k,v in id_pred_answers.items():
#     gt_pred ={}
#     gt_pred["pred"] = v
#     gt_pred["gt"] = id_gt_answers[str(k)]
#     final[k] = gt_pred
# correct = 0
# total = 0
# for k,v in final.items():
    
#     if v["pred"] in v["gt"]:
#         correct+=1
#     total+=1
# accuracy = correct/total

# print(accuracy)
    
    

#print(final)


    
    








