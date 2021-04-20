import json
qdata = json.load(open("test_data/vk_v_with_SSS.json"))
processed_data = {}
id_quest ={}
for k,v in qdata.items():
    id_quest[k]= v

context_data = json.load(open("test_data/vk_Google_v_with_SSS.json"))
id_context = {}

for k,v in context_data.items():
    id_context[k] = v
for k,v in id_quest.items():
    data ={}
    data["question"] = id_quest[k]
    if k not in id_context.keys():
        continue
    data["context"] = id_context[k]
    processed_data[k] = data
json.dump(processed_data,open("test_data/test_v_with_SSS.json","w"))



