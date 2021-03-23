import pandas as pd
import numpy as np
from xtools import *
from itertools import product
import argparse
import multiprocessing
import os
from time import time
import subprocess

from corplugin import predict, regulation

class Correspondence():
    def __init__(self, tagsname, mission, **kwargs):
        if not tagsname.endswith("tags"):
            raise Exception("请传入tags表")
        if not tagsname.startswith("das."):
            tagsname = "das." + tagsname
        self.tablename = tagsname
        self.mission = mission

    def _query(self):
        mh = MarcpointHive()
        print("开始检索数据，请等待......")
        start = time()
        field1, field2 = self.mission.split('_')[0], self.mission.split('_')[1]
        sql = 'select id, content, {}, {} from {}'.format(field1, field2, self.tablename)
        print(sql)
        df = mh.query(sql)
        df = df.dropna()
        print("数据检索完毕...耗时：{} s".format(time() - start))
        print("数据量为：", df.shape)
        return df

    def transferdata(self):
        """if data exists in csv format, read csv,
        otherwise retrieve via sql
        """
        abpath = '/mnt/disk3/CIData/{}.csv'.format(self.tablename)
        if os.path.exists(abpath):
            return pd.read_csv(abpath, keep_default_na=False, error_bad_lines=False, dtype=str).dropna()
        else:
            try:
                c = "su ops -c 'sh /mnt/disk4/tools/Export2CSV.sh " + self.tablename + ' ' + abpath + "'"
                print(c)
                subprocess.run(c, shell=True)
                return pd.read_csv(abpath, keep_default_na=False, error_bad_lines=False, dtype=str)
            except:
                mh = MarcpointHive()
                field1, field2 = self.mission.split('_')[0], self.mission.split('_')[1]
                sql = 'select content, {}, {}, from {}'.format(field1, field2, self.tablename)
                return mh.query(sql)

    def __expand(self, df):
        """expand each piece of data
        """
        print("Data expansion start......")
        field1, field2 = self.mission.split('_')[0], self.mission.split('_')[1]
        if not (field1 in df.columns and field2 in df.columns):
            raise Exception("Table {} does not have field: {} or {}".format(self.tablename, field1, field2))
        df = df[['id', 'content', field1, field2]]
        df = df[(df['content'] != '') & (df[field1] != '') & (df[field2] != '')]     
        
        # add field
        comblist = []
        for idx, row in df.iterrows():
            field1list = row[field1].split('|')
            field2list = row[field2].split('|')
            loop_val = [field1list, field2list]
            comblist.append(list(product(*loop_val)))
        df[self.mission] = comblist
        
        # expand
        i = 0
        df_list = []
        for idx, row in df.iterrows():
            pair_list = row[self.mission]
            count_total = len(pair_list)
            for count, pair in enumerate(pair_list):
                i += 1
                df_list.append(pd.DataFrame({'count': '{}/{}'.format(count + 1, count_total),
                                             'id': row['id'],
                                             'content': row['content'],
                                             'a': pair[0],
                                             'b': pair[1],
                                             'aclass': field1,
                                             'bclass': field2}, index=[i]))
        df_res = pd.concat(df_list)
        df_res = df_res.dropna()
        df_res['ex'] = df_res['a'] + "#" + df_res['b']
        print("Data expansion completed...... Total: {}".format(df_res.shape))
        return df_res

    def __predict(self, df):
        """return data filtered by various methods
        """
        print("Data prediction start......")
        pred = np.array([])
        total_loop = len(df) // 1000 + 1
        for loop in range(total_loop):
            if loop % 10 == 0:
                print(loop)
            try:
                tmp = predict(df[loop * 1000:(loop + 1) * 1000])
            except:
                tmp = np.array([0] * 1000)
                print('-----------{}------------'.format(loop))
            pred = np.concatenate([pred, tmp])
        pred = pred.astype(int)
        df['pred'] = pred
        df_res = df[df['pred'] == 1]
        print("Data prediction completed...... Total: {}".format(df_res.shape))
        return df_res

    def filter(self, queue, df):
        """select only paired data
        """
        dfexpand = self.__expand(df)
        dfres = self.__predict(dfexpand)
        queue.put(dfres)

    def label(self, ):
        """

        :return:
        """
        pass

    def statistic(self, df):
        """

        """


if __name__ == '__main__':
    # add command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("tablename", type=str,
                        help="determine the data source, start with das. end with _tags")
    parser.add_argument("mission", type=str,
                        help="determine the mission type, 'xuqiu_fangan' or 'fangan_driver'")
    parser.add_argument("--numprocess", type=int, default=10,
                        help="determine the number of processes")
    args = parser.parse_args()
    cor = Correspondence(args.tablename, args.mission)

    # retrieve data by sql
    start = time()
    # df = cor.transferdata()
    df = cor._query()
    print("Data acquisition completed... Time cost: ", time() - start)
    print("Data to process: {}".format(df.shape))

    # filter data with multiprocess
    queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(args.numprocess)
    s = time()
    pw = pool.apply_async(func=cor.filter, args=(queue, df))
    pw.wait()
    print("Data selection completed...... Time cost：", time() - s)
    pool.close()
    pool.join()
    if queue.empty():
        print("Process failed, no data return......")
    else:
        # print("Total data: ", len(queue.get()))
        # print(queue.get())

        # statistics and calculate
        print("Start writing to file......")
        df = queue.get()
        df.to_csv('/mnt/disk2/data/YuanHAO/对应关系应用/cor_restmp_{}.csv'.format(args.mission),
                  sep=str('\t'), index=False)
        print("Start statistics......")













