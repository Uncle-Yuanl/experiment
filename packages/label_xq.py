import pandas as pd
import numpy as np
from itertools import product
import argparse
import multiprocessing
import os
import re
from time import time
import subprocess

def relexicon(string, dic):
    for key, value in dic.items():
        if re.search(re.escape(value), string):
            return key
    return ''


def label(queue, dfori, field, dic):
    dfori["attr"] = dfori[field].apply(relexicon, args=(dic,))
    queue.put(dfori)


if __name__ == '__main__':
    dfori = pd.read_csv('./cor_restmp_xuqiu_fangan.csv', sep=str('\t'), error_bad_lines=False, nrows=10)
    dfori = dfori.dropna()

    lexicon = pd.read_excel('./lexicon/final_lexicon.xlsx')
    xqcls = "需求.问题期许.便秘问题"
    lexicon_xuqiu = lexicon[lexicon["TagName"].str.contains(xqcls)]
    xqdic = {t: k for (t, k) in zip(lexicon_xuqiu["TagName"], lexicon_xuqiu["Keywords"])}
    xqdic["腹泻"] = "test"
    print(xqdic)

    queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(1)
    s = time()
    pw = pool.apply_async(func=label, args=(queue, dfori, 'a', xqdic))
    pw.wait()
    print("Data process completed...... Time cost：", time() - s)
    pool.close()
    pool.join()
    if queue.empty():
        print("Process failed, no data return......")
    else:
        dfori = queue.get()
        df_tmp = dfori[dfori.attr.str.len() != 0]
        print(len(df_tmp))
        # 方案与需求label的映射
        df_drop = df_tmp.drop_duplicates('a')
        strlab = {k: v for k, v in zip(df_tmp['b'], df_tmp['attr'])}
        print('映射表元素个数为：'.format(len(strlab)))
        # 有对应关系的方案列表
        dfmap = pd.DataFrame(zip(strlab.keys(), strlab.values()), columns=['方案', '需求label'])
        dfmap.to_csv('./map_fx_test.csv', sep=str('\t'), index=False)
        print("Write file completed......")