import torch
from configuration_bert import BertConfig
from modeling_bert import BertForMaskedLM
from transformers import AutoTokenizer


model_path = "chinese-bert-wwm-ext"
model = BertForMaskedLM.from_pretrained(model_path)

for k,v in model.named_parameters():
    print(k, v.shape)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入：有一个 [MASK]
text = "中国的首都是[MASK][MASK]。"
inputs = tokenizer(text, return_tensors="pt")

# 前向
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 找到 [MASK] 的位置
print(logits.shape)
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# 对应位置的 logits -> softmax
mask_logits = logits[0, mask_token_index, :]
predicted_id = torch.argmax(mask_logits, dim=-1)
print(predicted_id)
print("Predicted word:", tokenizer.decode(predicted_id))