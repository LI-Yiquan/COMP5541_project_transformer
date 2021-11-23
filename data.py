# coding=utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import io
import tensorflow_datasets as tfds
import jieba
from nltk.tokenize import word_tokenize





print(tfds.list_builders())

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
config = tfds.translate.wmt.WmtConfig(
    description="WMT 2019 translation task dataset.",
    version="0.0.3",
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v13"],
        tfds.Split.VALIDATION: ["newsdev2017"],
    }
)

builder = tfds.builder("wmt_translate", config=config)
#print(builder.info)
builder.download_and_prepare()
datasets = builder.as_dataset(as_supervised=True)
train_dataset = datasets['train']
val_dataset = datasets['validation']


#打印测试
zh_vocab={}
zh_count=0
en_vocab={}
en_count=0
enc_input=[]
dec_input =[]
dec_output=[]
for zh, en in train_dataset.take(5):
    print('zh: {}'.format(zh.numpy().decode('utf-8')))
    seg_list = jieba.cut(zh.numpy().decode('utf-8'))
    print(u"[分词]: ", "/ ".join(seg_list))
    print('en: {}'.format(en.numpy().decode('utf-8')))
    print(word_tokenize(en.numpy().decode('utf-8')))
for zh, en in train_dataset.take(5):
    seg_list = jieba.cut(zh.numpy().decode('utf-8'))
    this_enc=[]
    for i in seg_list:
        if i not in zh_vocab:
            zh_vocab[i]=zh_count
            this_enc.append(zh_count)
            zh_count+=1
        else:
            this_enc.append(zh_vocab[i])
    enc_input.append(this_enc)
    seg_list =word_tokenize(en.numpy().decode('utf-8'))
    this_dec=[]
    for i in seg_list:
        if i not in en_vocab:
            en_vocab[i]=en_count
            this_dec.append(en_count)
            en_count+=1
        else:
            this_dec.append(en_vocab[i])
    dec_input.append(this_dec)
    dec_output.append(this_dec)
print(enc_input)
print(dec_input)
print(dec_output)

#训练集生成
zh_vocab={}
zh_count=0
en_vocab={}
en_count=0
enc_input=[]
dec_input =[]
dec_output=[]
for zh, en in train_dataset.take(1000):
    seg_list = jieba.cut(zh.numpy().decode('utf-8'))
    this_enc=[]
    for i in seg_list:
        if i not in zh_vocab:
            zh_vocab[i]=zh_count
            this_enc.append(zh_count)
            zh_count+=1
        else:
            this_enc.append(zh_vocab[i])
    enc_input.append(this_enc)
    seg_list =word_tokenize(en.numpy().decode('utf-8'))
    this_dec=[]
    for i in seg_list:
        if i not in en_vocab:
            en_vocab[i]=en_count
            this_dec.append(en_count)
            en_count+=1
        else:
            this_dec.append(en_vocab[i])
    dec_input.append(this_dec)
    dec_output.append(this_dec)

enc_input_Y=[]
dec_input_Y =[]
dec_output_Y=[]
for zh, en in val_dataset.take(10):
    seg_list = jieba.cut(zh.numpy().decode('utf-8'))
    this_enc=[]
    for i in seg_list:
        if i not in zh_vocab:
            zh_vocab[i]=zh_count
            this_enc.append(zh_count)
            zh_count+=1
        else:
            this_enc.append(zh_vocab[i])
    enc_input_Y.append(this_enc)
    seg_list =word_tokenize(en.numpy().decode('utf-8'))
    this_dec=[]
    for i in seg_list:
        if i not in en_vocab:
            en_vocab[i]=en_count
            this_dec.append(en_count)
            en_count+=1
        else:
            this_dec.append(en_vocab[i])
    dec_input_Y.append(this_dec)
    dec_output_Y.append(this_dec)

print(zh_vocab)
print(en_vocab)
print(zh_count)
print(en_count)
#BLEU分数
#score = sentence_bleu(y_hat, y)

# If you need NumPy arrays
# np_datasets = tfds.as_numpy(datasets)