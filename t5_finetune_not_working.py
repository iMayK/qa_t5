# import torch
# from nlp import load_dataset
# from transformers import T5Tokenizer
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

# # process the examples in input and target text format and the eos token at the end 
# def add_eos_to_examples(example):
#     example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
#     example['target_text'] = '%s </s>' % example['answers']['text'][0]
#     return example

# # tokenize the examples
# def convert_to_features(example_batch):
#     input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
#     target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16)

#     encodings = {
#         'input_ids': input_encodings['input_ids'], 
#         'attention_mask': input_encodings['attention_mask'],
#         'target_ids': target_encodings['input_ids'],
#         'target_attention_mask': target_encodings['attention_mask']
#     }

#     return encodings

# # load train and validation split of squad
# train_dataset  = load_dataset("./squad.py",ignore_verifications=True,split="train")
# valid_dataset =  load_dataset("./squad.py",split="validation")

# # map add_eos_to_examples function to the dataset example wise 
# train_dataset = train_dataset.map(add_eos_to_examples)
# # map convert_to_features batch wise
# train_dataset = train_dataset.map(convert_to_features, batched=True)

# valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
# valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


# # set the tensor type and the columns which the dataset should return
# columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
# train_dataset.set_format(type='torch', columns=columns)
# valid_dataset.set_format(type='torch', columns=columns)

# torch.save(train_dataset, 'train_data.pt')
# torch.save(valid_dataset, 'valid_data.pt')
# print("saving Done")




import json

args_dict = {
  "num_cores": 8,
  'training_script': 'train_t5_squad.py',
  "model_name_or_path": 't5-base',
  "max_len": 512 ,
  "target_max_len": 16,
  "output_dir": './models/tpu',
  "overwrite_output_dir": True,
  "per_gpu_train_batch_size": 8,
  "per_gpu_eval_batch_size": 8,
  "gradient_accumulation_steps": 4,
  "learning_rate": 1e-4,
  "tpu_num_cores": 8,
  "num_train_epochs": 4,
  "do_train": True
}

with open('args.json', 'w') as f:
  json.dump(args_dict, f)


