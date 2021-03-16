from xtools import *
from time import time
import collections
import pandas as pd
import os
import re
import subprocess

lexicon = pd.read_excel('./lexicon/词云0303-综合词库_全部_2021-03-10 19-38-48.xlsx', usecols=['TagName', 'Keywords'])
lexicon = lexicon[lexicon['TagName'].str.startswith('需求')]


class Combine():
    def __init__(self, tagsname):
        if not tagsname.endswith("tags"):
            raise Exception("请传入tags表")
        if not tagsname.startswith("das."):
            tagsname = "das." + tagsname
        self.tablename = tagsname

    def _query(self):
        self.mh = MarcpointHive()
        print("开始检索数据，请等待......")
        start = time()
        sql = "select content, attr from {}".format(self.tablename)
        df = self.mh.query(sql)
        print("数据检索完毕...耗时：{} s".format(time() - start))
        print("数据量为：", df.shape)
        return df

    def transferdata(self, abpath=None):
        """通过export2csv工具，将表以csv格式传输至绝对路径

        parameters:
        -----------
        tablename: string
            表名
        abpath: string
            数据存放的绝对路径
        """
        if not abpath:
            abpath = "/mnt/disk3/CIData/" + self.tablename + ".csv"
        if not os.path.exists(abpath):
            c = "su ops -c 'sh /mnt/disk4/tools/Export2CSV.sh " + self.tablename + ' ' + abpath + "'"
            print(c)
            subprocess.run(c, shell=True)
        df = pd.read_csv(abpath, error_bad_lines=False, dtype=str)
        if "ATTR" in df.columns and "attr" not in df.columns:
            df['attr'] = df["ATTR"]
        return df

    def __func(self, tags):
        """tags即attr字段内容
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
                    ls.append("需求." + w.split('.')[-1])
                for c in lsc:
                    ls.append("需求." + w.split('.')[-1] + '_' + c.split('.')[-1])
            for b in lsb:
                if len(lsc) == 0:
                    if b.split('.')[-1] in w.split('.')[-1]:
                        ls.append("需求." + w.split('.')[-1])
                    else:
                        ls.append("需求." + b.split('.')[-1] + '_' + w.split('.')[-1])
                for c in lsc:
                    if b.split('.')[-1] in w.split('.')[-1]:
                        ls.append("需求." + w.split('.')[-1] + '_' + c.split('.')[-1])
                    else:
                        ls.append("需求." + b.split('.')[-1] + '_' + w.split('.')[-1] + '_' + c.split('.')[-1])
        return '|'.join(ls)

    def label(self, df):
        """use lexicon to label the df
        this df should have the filed 'xuqiu' and not have field 'tags'
        """
        print('Start labelling......')
        for _, rowd in df.iterrows():
            attr = []
            for _, rowl in lexicon.iterrows():
                tmp = re.findall(re.escape(rowl['Keywords']), rowd['xuqiu'])
                if tmp:
                    attr.extend(tmp)
            rowd['attr_new'] = '|'.join(attr)
        print('Labelling completed......')
        return df

    def _combineM1(self, df):
        """combine after cluster
        """
        if 'attr_new' not in df.columns:
            raise Exception("Labelling failded......")
        df['attr_new'] = df['attr_new'].astype('str')
        df = df[['content', 'attr_new']]
        df = df.drop_duplicates()
        total = '|'.join(list(df[df['attr_new'].str.contains('问题期许')]['attr_new'].apply(self.__func))).split('|')
        df1 = pd.DataFrame(collections.Counter(total).most_common())
        df1.columns = ['组合需求', 'count']
        return df1

    def combineM2(self, df):
        """combine after having tags
        """
        df.attr = df.attr.astype('str')
        df = df[['content', 'attr']]
        df = df.drop_duplicates()
        total = '|'.join(list(df[df.attr.str.contains('问题期许')].attr.apply(self.__func))).split('|')
        df2 = pd.DataFrame(collections.Counter(total).most_common())
        df2.columns = ['组合需求', 'count']
        return df2
