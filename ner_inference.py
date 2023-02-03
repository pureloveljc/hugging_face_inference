import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForTokenClassification, AutoTokenizer
pretrained_model ='/home/all/ljc/code/gen_ner_train/test_voc_3'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
print(model)
# sequence='我要从上海到北京。'
# "我要到北京，从上海出发，买一张机票"
# sequence='我要买一张南京到上海的机票。'
# 文本任务输入512个
sequence='我要从南京飞去杭州的机票。再从杭州出发到北京，再从北京坐火车到上海'
# print(tokenizer.decode(tokenizer.encode(sequence)))
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
print(tokens)
inputs = tokenizer.encode(sequence, return_tensors="pt")
outputs = model(inputs)
outputs= outputs.logits
predictions = torch.argmax(outputs, dim=2)
for token, prediction in zip(tokens, predictions[0].numpy()):
     print((token, model.config.id2label[prediction]))
     # print(model.config.id2label[prediction])

# padding和attenttion_mask  扩展成10 ->15， 多加5个0,
