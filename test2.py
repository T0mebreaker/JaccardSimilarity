import transformers

# 定义函数生成BERT模型的三种输入形式
def generate_bert_inputs(tokenizer, sentence, max_length):
    # Step 1: 分词并编码文本
    encoded_input = tokenizer.encode_plus(
        sentence,                       # 输入文本
        add_special_tokens=True,        # 添加特殊标记（[CLS]和[SEP]）
        max_length=max_length,          # 设置最大长度
        padding='max_length',           # 对文本进行填充
        truncation=True,                # 如果文本超过最大长度则截断
        return_tensors='pt'             # 返回PyTorch张量
    )
    
    # Step 2: 获取编码后的输入
    input_ids = encoded_input['input_ids']           # 输入标识符
    token_type_ids = encoded_input['token_type_ids'] # 标记类型标识符
    attention_mask = encoded_input['attention_mask'] # 注意力掩码
    
    return input_ids, token_type_ids, attention_mask

# 要编码的句子
sentence = "在湖北师范大学读书"

# 使用中文BERT分词器
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-chinese')

# 生成BERT模型所需的三种输入形式
input_ids, token_type_ids, attention_mask = generate_bert_inputs(tokenizer, sentence, max_length=32)

# 打印结果
print("Input IDs:", input_ids)
print("Token Type IDs:", token_type_ids)
print("Attention Mask:", attention_mask)
