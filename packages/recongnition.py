import jieba.posseg as psg
import pandas as pd
from tqdm import tqdm
from time import time
import re
import collections
import numpy as np
import jieba
import math
import codecs
import pickle
from xtools import *
import subprocess
import os

# 根据具体的项目来修改
# _fix = '的|了|有点|多|没有|长了|不|的|特别|不错|好|好处|不|没有|无|不会|可以|一直|不用|超级|慢慢|会|很|现在|我家|我|宝宝|你家|你|有|今天'.split('|')
_fix = '的|了|有点|多|没有|长了|不|的|特别|不错|好|好处|不|没有|无|不会|可以|一直|不用|超级|慢慢|会|很|现在,|，|啊'.split('|')

class Recongnition():
    def __init__(self, tagsname, sim=0.75):
        """创建数据库工具、读取向量、成员变量

        parameters:
        ----------
        tags_name: string
            必须是tags表
        sim: int
            词相似度阈值，默认0.75
        """
        if not tagsname.endswith("tags"):
            raise Exception("请传入tags表")
        if not tagsname.startswith("das."):
            tagsname = "das." + tagsname
        self.tablename = tagsname

        with codecs.open('/mnt/disk2/data/YuanHAO/对应关系应用/tongyong.pickle', "rb") as f:
            VECTOR = pickle.load(f)
        self.W_norm = VECTOR['W_norm']
        self.vocab = VECTOR['vocab']

        self.diccjcount = {}
        self.dicxqcount = {}

        self.dicmap = {}

        self.sim = sim

    def _query(self):
        self.mh = MarcpointHive()
        print("开始检索数据，请等待......")
        start = time()
        sql = "select content, changjing, attr from {}".format(self.tablename)
        df = self.mh.query(sql)
        print("数据检索完毕...耗时：{} s".format(time() - start))
        self.df = df
        if "ATTR" in self.df.columns and "attr" not in self.df.columns:
            self.df['attr'] = self.df["ATTR"]
        print("数据量为：", df.shape)
        return self.df

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
        self.df = pd.read_csv(abpath, error_bad_lines=False, sep=',', escapechar='\\', quotechar='"')
        if "ATTR" in self.df.columns and "attr" not in self.df.columns:
            self.df['attr'] = self.df["ATTR"]

    def __func(self, tags, comb_only=False):
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
        if comb_only:
            return ''.join([x for x in ls if len(x.split('.')[-1].split('_')) > 1])
        else:
            return '|'.join(ls)

    def __combine(self, comb_only):
        """
        将部位、问题期许、程度进行组合
        """
        print('开始生成组合需求......')
        self.df.attr = self.df.attr.astype('str').apply(self.__func, args=(comb_only,))


    def __select_cjwords(self, classtag):
        """
        筛选数据，场景为事件（动词），且属于“人”、“时间”、“空间”

        parameters:
        -----------
            type: string

        returns:
        --------
            type: dataframe
        """
        self.df.changjing = self.df.changjing.astype('str')
        self.df.attr = self.df.attr.astype('str')

        # self.df = self.df[self.df.attr.apply(lambda x: bool(re.search(classtag, x)))]
        self.df = self.df[self.df.attr.str.contains(classtag)]
        print(self.df.shape)

        self.obj = collections.Counter([x for x in '|'.join(list(self.df.attr)).split('|') if bool(re.search(classtag, x))])

        events = []
        for x in list(self.df.changjing):
            ls = list(psg.cut(x))
            event = 0
            for y in ls:
                if y.flag == 'v':
                    event = 1
                    break
            events.append(event)
        self.df['event'] = events
        self.df['ren'] = self.df.attr.apply(lambda x: int('场景.人' in x))
        self.df['shi'] = self.df.attr.apply(lambda x: int('场景.时间' in x))
        self.df['kong'] = self.df.attr.apply(lambda x: int('场景.空间' in x))
        self.df['rsk'] = self.df.ren.astype('str') + self.df.shi.astype('str') + self.df.kong.astype('str')

        self.rskset = '111'
        self.df = self.df[(self.df.event == 1) | (self.df.rsk.str.contains(self.rskset))]
        self.df = self.df[['content', 'changjing', 'attr', 'event', 'ren', 'shi', 'kong', 'rsk']].drop_duplicates()
        print("筛选后的df：", self.df.shape)
        self.df.changjing = self.df.changjing.apply(lambda x: self.__simplifycj(x))

        dftmp = pd.DataFrame(self.df.changjing.value_counts()).reset_index()
        dftmp.columns = ['场景', '条数']
        for i in range(dftmp.shape[0]):
            self.diccjcount[dftmp.iloc[i]['场景']] = dftmp.iloc[i]['条数']
        self.lscjwords = list(dftmp[dftmp['条数'] > 1]['场景'].unique())

    def __simplifycj(self, cjs):
        tmp = []
        cjs = set(cjs.split('|'))
        for cj in cjs:
            for _ in _fix:
                cj = cj.strip(_)
            if len(cj) == 0:
                continue
            tmp.append(cj)
        return '|'.join(sorted(list(set(tmp))))

    def __update_dicmap(self):
        """以dataframe格式创建两个词的相似度
        更新self.dicmap
        """
        print('待判断的场景数', len(self.lscjwords))
        print('****不在词典里的词有****')

        wsvec = []
        newws = []
        subvocab = {}

        for x in set('|'.join(self.lscjwords).split('|')):
            cut_x = x
            for _ in _fix:
                cut_x = cut_x.strip(_)

            if cut_x in self.vocab:
                vec = self.W_norm[self.vocab[cut_x]]
                if sum(np.isnan(vec)) != 0 or sum(pd.DataFrame(vec)[0].apply(math.isinf)) != 0:
                    print(x, ': nan or inf')
                    continue
                wsvec.append(vec)
                newws.append(x)
            else:
                cut_x = list(jieba.cut(cut_x))
                tmp = [self.W_norm[self.vocab[w]] for w in cut_x if w in self.vocab]
                if len(tmp) == 0:
                    print(x)
                else:
                    wsvec.append(sum(tmp) / len(tmp))
                    newws.append(x)
        subvocab = {w: idx for idx, w in enumerate(newws)}

        def cal_sim(word1, word2):
            vec1 = wsvec[subvocab[word1]]
            vec2 = wsvec[subvocab[word2]]
            return np.dot(vec1, vec2)

        lssim = []
        for i in tqdm(range(len(newws))):
            for j in range(i + 1, len(newws)):
                lssim.append([newws[i], newws[j], cal_sim(newws[i], newws[j])])

        dfsim = pd.DataFrame(lssim).reset_index(drop=True)
        dfsim.columns = ['word1', 'word2', 'sim']
        dfsim = dfsim.sort_values(by='sim', ascending=False)

        dfsim = dfsim[dfsim.sim > self.sim]
        for i in range(dfsim.shape[0]):
            self.dicmap[dfsim.iloc[i].word2] = dfsim.iloc[i].word1

    def addpair(self, word1, word2):
        """公开函数，允许手动更新self.dicmap
        parameters:
        -----------
        word1: string
        word2: string
        """
        self.dicmap[word1] = word2

    def __mergecj(self, cjs):
        tmp = []
        cjs = set(cjs.split('|'))
        for cj in cjs:
            for _ in _fix:
                cj = cj.replace(_, '')
            if len(cj) == 0:
                continue
            if cj in self.dicmap:
                cjnew = self.dicmap[cj]
                while cjnew in self.dicmap:
                    cjnew = self.dicmap[cjnew]
                tmp.append(cjnew)
            else:
                tmp.append(cj)
        return '|'.join(sorted(list(set(tmp))))

    def proprocess(self, classtag):
        """根据jylb生成cjwords和dicmap
        """
        if not hasattr(self, "df"):
            raise Exception("请先检索数据......")

        self.__combine(comb_only=False)
        print("生成组合需求完毕......")
        print(self.df.shape)

        self.__select_cjwords(classtag)
        print("更新场景词完毕......")

        # update dicmap
        self.__update_dicmap()
        print("更新映射表完毕......")

    def candidate(self, fulltag):
        dfmerged = self.df
        dfmerged.changjing = dfmerged.changjing.apply(self.__mergecj)

        ls = []
        totalxqs = []
        for xq, xqcount in self.obj.most_common():
            # print('【', xq, '】数据量：', xqcount, end='，')
            self.dicxqcount[xq] = xqcount
            if xqcount < 50:
                continue
            dfxq = dfmerged[dfmerged.attr.apply(lambda x: bool(re.search(xq, x)))]
            dfxq = dfxq[(dfxq.event == 1) | (dfxq.rsk.str.contains(self.rskset))]
            dftmp = pd.DataFrame(dfxq.changjing.value_counts()).reset_index()
            dftmp.columns = ['cj', 'count']
            sumcj = sum(list(dftmp['count']))

            # if dftmp.shape[0] < 2 and sumcj <= 10:
            #     continue

            totalxqs.append(xq)
            for i in range(dftmp.shape[0]):
                ls.append(
                    [dftmp.iloc[i].cj, dftmp.iloc[i]['count'], round(dftmp.iloc[i]['count'] / sumcj * 100, 4), xq])

        dfxqc = pd.DataFrame(ls)
        dfxqc.columns = ['场景', '条数', '占比(%)', '需求']

        totaldf = dfxqc
        self.totaldf = totaldf
        dftest = totaldf[totaldf['条数'] > 1]
        dftgi = dftest.pivot_table(index='场景', columns='需求', values='条数', aggfunc='sum',
                                   margins=True).reset_index().fillna(0)
        self.dftgi = dftgi
        tmp = dftgi[['场景', fulltag, 'All']].sort_values(by=fulltag, ascending=False).reset_index(drop=True)
        # tmp = tmp[:20]
        tmp = tmp[tmp[fulltag] >= int(tmp.iloc[20][fulltag])]
        lstgi = []
        for i in range(1, tmp.shape[0]):
            lstgi.append([fulltag, tmp.iloc[i]['场景'],
                          round(tmp.iloc[i][fulltag] * tmp.iloc[0]['All'] / tmp.iloc[0][fulltag] / tmp.iloc[i]['All'] * 100),
                          tmp.iloc[i][fulltag]])
        dft = pd.DataFrame(lstgi)
        dft.columns = ['需求', '场景', '场景关联指数', 'count']
        candidate = dft[dft['count'] > 0].sort_values(by='场景关联指数', ascending=False)
        return candidate

    def content(self, keyword):
        """检索备选词的content，判断词是否合理

        parameters:
        -----------
        keyword: string
            备选场景词

        reutrn:
        -------
            type: pd.Series
        """
        return self.df[self.df.changjing.str.contains(keyword)].content

    def find(self, jylb, cjlist):
        """将输入的场景词进行占比计算

        parameters:
        -----------
        cjlist：list
            场景列表，元素是用户通过候选场景进行手动输入的，与候选中的词可以有一定出入
        """
        def filtercj(x):
            for cs in cjlist:
                for c in cs.split('|'):
                    if c in x:
                        return True
            return False

        def filteracj(x, cj):
            for c in cj.split('|'):
                if c in x:
                    return True
            return False

        tmp = self.dftgi[['场景', jylb, 'All']].sort_values(by=jylb, ascending=False).reset_index(drop=True)

        lstgi = []
        for cj in cjlist:
            xc = tmp[tmp['场景'].apply(lambda x: filteracj(x, cj))]
            tgi = xc[jylb].sum() * tmp.iloc[0]['All'] / tmp.iloc[0][jylb] / xc['All'].sum() * 100
            #     print(xc,cj,tgi)
            lstgi.append([jylb, cj, round(tgi), xc[jylb].sum()])
        dft = pd.DataFrame(lstgi)
        dft.columns = ['需求', '场景', '场景关联指数', 'count']

        print('TOP{}场景占比'.format(len(cjlist)), self.totaldf[(self.totaldf['需求'] == jylb) & (self.totaldf['场景'].apply(lambda x: filtercj(x)))]['占比(%)'].sum(),
              '%')
        res = dft[dft['count'] > 0].sort_values(by='场景关联指数', ascending=False)
        return res



