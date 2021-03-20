import requests
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer

# bert词表
dict_path='/mnt/disk6-old/project/auto-sentiment/bert-chinese_wwm/vocab.txt'

def read_token_vocab(file):
    with open(file) as f:
        vocabs = f.read().split('\n')
        token_dict = dict(zip(vocabs, range(len(vocabs))))
    return token_dict

tokenizer = Tokenizer(read_token_vocab(dict_path))

def regulation():
    """
    """
    pass


def predict(df):
    """use model2 first

    parameters:
    df: dataframe
        df after expand
    """
    x1, x2 = [], []
    if not ("content" in df.columns and "ex" in df.columns):
        raise Exception("df does the correct fields......")
    for content, ex in zip(df["content"], df["ex"]):
        a, b = tokenizer.encode(content, ex)
        x1.append(a)
        x2.append(b)
    x1 = pad_sequences(x1, padding='post')
    x2 = pad_sequences(x2, padding='post')

    assert len(x1) == len(x2)
    inputs_dict = [{"input0": x1[i].tolist(),
                    "input1": x2[i].tolist()} for i in range(len(x1))]

    data = {
        "signature_name": "predict",
        "instances": inputs_dict
    }
    r = requests.post("http://172.16.1.51:8501/v1/models/cor/versions/1:predict", json=data)
    res = np.array(json.loads(r.text)['predictions']).argmax(-1)
    return res

if __name__ == '__main__':
    model_data = pd.DataFrame([{"content": "这是一段我也不知道啥意思的内容", "ex": "反正这是个字符串"},
                               {"content": "这是一段我也不知道啥意思的内容容", "ex": "反正这是个字符串串"}])

    print(predict(model_data))