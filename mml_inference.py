from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("/home/all/ljc/code/gen_ner_train/bert_base_chinese")

model = AutoModelForMaskedLM.from_pretrained("/home/all/ljc/code/gen_ner_train/bert_base_chinese")

text_masked = "我们一起去打[MASK]吧.真的我很[MASK]跟[MASK]一起打球"
input_ids = tokenizer.encode(text_masked, return_tensors="pt")

# Predict the masked token
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]
    mask_indexs = (input_ids[0] == tokenizer.mask_token_id).nonzero().squeeze().tolist()
    if type(mask_indexs) == int:
        mask_indexs = [mask_indexs]
    text_filled = text_masked
    for masked_index in mask_indexs:
        masked_index = masked_index
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.decode([predicted_index])
        text_filled = text_filled.replace("[MASK]", predicted_token, 1)
print(text_masked)
print(text_filled)



