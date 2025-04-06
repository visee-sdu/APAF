from transformers import AutoTokenizer
from transformers import XLMRobertaTokenizer
import json
import jsonlines

# 加载本地分词器
bert_tokenizer = AutoTokenizer.from_pretrained('/home/huangbw/navigation/baseline/bert_config/bert-base-uncased')
roberta_tokenizer = AutoTokenizer.from_pretrained("/home/huangbw/navigation/baseline/bert_config/xlm-roberta-base")

input_file = '/home/huangbw/navigation/baseline/datasets/R2R/annotations/R2R_scalevln_ft_aug_enc.json'
output_file = '/home/huangbw/navigation/baseline/datasets/R2R/annotations/R2R_scalevln_ft_aug_enc_xlmr.jsonl'

idx = 0
with open(input_file, 'r') as file:
    datas = json.load(file)
    processed_data = []

    for data in datas:
        # 使用本地 BERT 分词器将 instr_encoding 转回文本
        # instr_text = bert_tokenizer.decode(data['instr_encodings'][0], skip_special_tokens=True)
        instr_text = data['instructions'][0]
        
        # 使用 XLM-Roberta 分词器进行重新编码
        new_encoding = roberta_tokenizer.encode(instr_text, add_special_tokens=True)
        
        # 更新 instr_encoding 字段
        data['instr_encoding'] = new_encoding
        data['instruction_id'] = "{:09}".format(idx)
        data['language'] = "en-IN"
        data['instruction'] = instr_text
        del data['instructions']
        del data['instr_encodings']
        
        # 添加到处理后的数据列表中
        processed_data.append(data)
        idx += 1

# 将处理后的数据写入新的 JSONL 文件
with jsonlines.open(output_file, mode="w") as writer:
    writer.write_all(processed_data)