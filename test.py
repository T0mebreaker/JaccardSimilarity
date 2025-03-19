from transformers import BertTokenizer
# BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # bert分词器

sentence = "在湖北师范大学读书"
encode_ids = tokenizer.encode(sentence) # encode 默认为True 加[CLS][SEP]
encode_words = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))   # encode 默认为True 加[CLS][SEP]


print(f"word_list   : {sentence.split()}")                 # 单词列表 (不进行分词)
print(f"tokenize    : {tokenizer.tokenize(sentence) }")    # 单词列表 (进行分词)
print(f"encode_words: {encode_words}")                     # 单词列表 (进行分词) [CLS]+sentence+[SEP]
print(f"encode_ids  : {tokenizer.encode(sentence)}")       # 词id列表 进行分词   101 + ids + 102
print(f"encode_plus : {tokenizer.encode_plus(sentence)}")  # dict 类型 三个key:value, {input_ids:词id列表(进行分词) token_type_ids:分句列表0(分句) attention_mask:掩码列表1(掩码)}
print("=" * 100)

encode_words_true =  tokenizer.encode(sentence, add_special_tokens=True)    # encode 默认为True 加[CLS][SEP]
encode_words_false = tokenizer.encode(sentence, add_special_tokens=False)  # encode False    不加[CLS][SEP]
print(f"encode_words_true : {encode_words_true}")
print(f"encode_words_false: {encode_words_false}")
