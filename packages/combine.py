from xtools import *
from time import time
import collections
import pandas as pd
import numpy as np
import os
import jieba
import codecs
import pickle
import math
import subprocess
from sklearn.cluster import KMeans


class Combine():
    def __init__(self):
        pass

    def _query(self, tablename):
        mh = MarcpointHive()
        print("开始检索数据，请等待......")
        start = time()
        sql = "select content, attr from {}".format(tablename)
        df = mh.query(sql)
        print("数据检索完毕...耗时：{} s".format(time() - start))
        print("数据量为：", df.shape)
        return df

    def transferdata(self, tablename, abpath=None):
        """通过export2csv工具，将表以csv格式传输至绝对路径

        parameters:
        -----------
        tablename: string
            表名
        abpath: string
            数据存放的绝对路径
        """
        if not tablename.endswith("tags"):
            raise Exception("请传入tags表")
        if not tablename.startswith("das."):
            tablename = "das." + tablename
        if not abpath:
            abpath = "/mnt/disk3/CIData/" + tablename + ".csv"
        if not os.path.exists(abpath):
            c = "su ops -c 'sh /mnt/disk4/tools/Export2CSV.sh " + tablename + ' ' + abpath + "'"
            print(c)
            subprocess.run(c, shell=True)
        df = pd.read_csv(abpath, error_bad_lines=False, dtype=str)
        if "ATTR" in df.columns and "attr" not in df.columns:
            df['attr'] = df["ATTR"]
        df = df.dropna(subset=['attr'])
        if len(df) == 0:
            raise Exception("Please use comprehensive lexicon to label the data !")
        return df

    def df2cluster(self, df, k):
        """statistic topk 问题期许

        parameter:  dataframe
            df after transferdata
        k:  int
            top k 问题需求
        """
        def __sel_qixu(attr):
            return "|".join([x.split(".")[-1] for x in attr.split("|") if "问题期许" in x])
        df = df[df.attr.str.contains("问题期许")]
        qixu_list = "|".join(list(df.attr.apply(__sel_qixu))).split("|")
        qixu_num = collections.Counter(qixu_list).most_common()
        dfqixu = pd.DataFrame(qixu_num[:k], columns=["问题期许", "count"])
        return dfqixu

    def cluster(self, n_clusters, filename=None, df=None, tofile=False):
        """return result of cluster with given file or df
        """
        path = '/mnt/disk2/data/YuanHAO/对应关系应用/files/{}'.format(filename)
        if not os.path.exists(path) and not isinstance(df, pd.DataFrame):
            raise Exception("Please upload cluster file to ./files folder OR specify the df")
        if not filename:
            # ws = df["组合需求"].apply(lambda x: x.split('.')[-1]).tolist()
            ws = df["问题期许"].tolist()
        else:
            ws = pd.read_excel(path, header=None)[0].tolist()

        with codecs.open('/mnt/disk2/data/YuanHAO/对应关系应用/tongyong.pickle', "rb") as f:
            VECTOR = pickle.load(f)
        W_norm = VECTOR['W_norm']
        vocab = VECTOR['vocab']
        wsvec = []
        newws = []
        print('****不在词典里的词有****')

        _fix = '的|了|有点|多|没有|长了|不|的|特别|不错|好|好处|不|没有|无|不会|可以|一直|不用|超级|慢慢|会'.split('|')
        for x in ws:
            cut_x = x
            for _ in _fix:
                cut_x = cut_x.strip(_)

            if cut_x in vocab:
                vec = W_norm[vocab[cut_x]]
                if sum(np.isnan(vec)) != 0 or sum(pd.DataFrame(vec)[0].apply(math.isinf)) != 0:
                    print(x, ': nan or inf')
                    continue
                wsvec.append(vec)
                newws.append(x)
            else:
                cut_x = list(jieba.cut(cut_x))
                tmp = [W_norm[vocab[w]] for w in cut_x if w in vocab]
                if len(tmp) == 0:
                    print(x)
                else:
                    wsvec.append(sum(tmp) / len(tmp))
                    newws.append(x)

        n_clusters = n_clusters
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(wsvec)
        labels = kmeans_model.labels_

        outputwords = []
        outputvecs = []
        for i in range(n_clusters):
            tmpword, tmpvec = [], []
            if not isinstance(df, pd.DataFrame):
                print('\nClass {}'.format(i))
            for j in range(len(wsvec)):
                if labels[j] == i:
                    if not isinstance(df, pd.DataFrame):
                        print(newws[j], end=',')
                    tmpword.append(newws[j])
                    tmpvec.append(wsvec[j])
            outputvecs.append(tmpvec)
            outputwords.append(tmpword)
        respath = None
        if tofile and filename:
            respath = '/mnt/disk2/data/YuanHAO/对应关系应用/files/{}聚类结果.txt'.format(filename)
        elif tofile and isinstance(df, pd.DataFrame):
            respath = '/mnt/disk2/data/YuanHAO/对应关系应用/files/聚类结果.txt'
        if respath:
            with open(respath, 'w', encoding='utf-8') as f:
                for cls, wl in enumerate(outputwords):
                    f.write("--------------class: {}---------------".format(cls) + '\n')
                    for word in wl:
                        f.write(word + '\n')

    def __func(self, tags):
        """tags即attr字段内容
        更新之后attr字段中的需求格式为： 需求.问题期许/部位/程度.$类别.xx
        """
        if '问题期许' not in tags:
            return ""
        tags = tags.split('|')
        lsw, lsb, lsc = [], [], []
        for tag in tags:
            if tag.startswith('需求.问题期许'):
                lsw.append(tag)
            elif tag.startswith('需求.部位'):
                lsb.append(tag)
            elif tag.startswith('需求.程度'):
                lsc.append(tag)
        ls = []
        for w in lsw:
            if len(lsb) == 0:
                if len(lsc) == 0:
                    ls.append("需求." + ".".join(w.split('.')[-2:]))
                for c in lsc:
                    ls.append("需求." + ".".join(w.split('.')[-2:]) + '_' + c.split('.')[-1])
            for b in lsb:
                if len(lsc) == 0:
                    if b.split('.')[-1] in w.split('.')[-1]:
                        ls.append("需求." + ".".join(w.split('.')[-2:]))
                    else:
                        ls.append("需求." + w.split('.')[-2] + "." + b.split('.')[-1] + '_' + w.split('.')[-1])
                for c in lsc:
                    if b.split('.')[-1] in w.split('.')[-1]:
                        ls.append("需求." + ".".join(w.split('.')[-2:]) + '_' + c.split('.')[-1])
                    else:
                        ls.append("需求." + w.split('.')[-2] + "." + b.split('.')[-1] \
                                  + '_' + w.split('.')[-1] + '_' + c.split('.')[-1])
        return '|'.join(ls)

    def combine(self, df):
        """combine after having final tags
        """
        df = df[['content', 'attr']]
        df.attr = df.attr.astype('str')
        df = df.drop_duplicates()
        total = '|'.join(list(df[df.attr.str.contains('问题期许')].attr.apply(self.__func))).split('|')
        df2 = pd.DataFrame(collections.Counter(total).most_common())
        df2.columns = ['组合需求', 'count']
        df2 = df2[~df2["组合需求"].str.contains("需求.其他")]
        return df2

    def judge(self, tags, clslist, cls=None):
        if not cls:
            for cls in clslist:
                if cls in tags:
                    return True
        else:
            if cls in tags:
                return True
        return False

    def countclass(self, comb_res, origin_df):
        """return ratio of every class
        """
        clsdic_ratio = {}
        self.clsdic_df = {}
        # totalcount = df['count'].sum()  # no sum of count but the num of id which attr contains cls
        clslist = comb_res['组合需求'].apply(lambda x: x.split('.')[1]).unique().tolist()

        totalcount = len(origin_df[origin_df.attr.apply(self.judge, args=(clslist, ))])
        for cls in clslist:
            # dfcls = comb_res[comb_res['组合需求'].str.contains(cls)]  # no count but distinct id
            df_cls = origin_df[origin_df.attr.apply(self.judge, args=(clslist, cls,))]
            self.clsdic_df[cls] = df_cls
            clsdic_ratio[cls] = round(len(df_cls) / totalcount * 100, 2)
        return sorted(clsdic_ratio.items(), key=lambda x: (x[1], x[0]), reverse=True)

    def judge_combine(self, tags, cls, comb_xuqiu):
        """judge whether the combined requirement in the attr of this id
        """
        xuqiu_list = comb_xuqiu.split('_')
        state = True
        if len(xuqiu_list) == 1:
            xuqiu = cls + "." + xuqiu_list[0]
            if xuqiu not in tags:
                state = False
        elif len(xuqiu_list) == 2:
            for xuqiu in xuqiu_list:
                if xuqiu not in tags:
                    state = False
        elif len(xuqiu_list) == 3:
            xuqiu_list = ["部位" + "." + xuqiu_list[0],
                            cls + "." + xuqiu_list[1],
                          "程度" + "." + xuqiu_list[2]]
            for xuqiu in xuqiu_list:
                if xuqiu not in tags:
                    state = False
        return state

    def search(self, comb_res, cls, k):
        """search the top k in combined xuqiu under given cls
        """
        df_cls = self.clsdic_df[cls]
        totalcount = len(df_cls)
        xq_cls = comb_res[comb_res['组合需求'].str.contains(cls)].reset_index(drop=True)
        ratiodic = {}
        for pd in xq_cls.itertuples():
            if pd[0] == k:
                break
            # count the num of id which attr contains specific requirement
            # requirement maybe in the form of A_B_C
            xuqiu_str = pd[1].split('.')[-1]
            df_comb = df_cls[df_cls.attr.apply(self.judge_combine, args=(cls, xuqiu_str, ))]
            xuqiu_rat = round(len(df_comb) / totalcount * 100, 2)
            ratiodic[xuqiu_str] = xuqiu_rat
        return ratiodic


        # totalcount = df['count'].sum()
        # dfcls = df[df['组合需求'].str.contains(cls)].reset_index(drop=True)
        # clscount = dfcls['count'].sum()
        # ratiocls = round(clscount / totalcount * 100, 2)
        # ratiodic = {}
        # for pd in dfcls.itertuples():
        #     if pd[0] == k:
        #         break
        #     xuqiu_str = pd[1].split('.')[-1]
        #     xuqiu_rat = round(int(pd[2]) / clscount * 100, 2)
        #     ratiodic[xuqiu_str] = xuqiu_rat
        # return ratiocls, ratiodic



