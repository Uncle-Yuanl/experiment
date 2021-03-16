import multiprocessing
import time
import os
import pandas as pd

class Test():
    def makedf(self, total):
        print(os.getpid())
        dic = {"content": ["测试内容的content文本"] * total,
               "ex": ["测试内容的对应关系文本"] * total}
        return pd.DataFrame(dic)

    def looptest(self, df):
        numlist = []
        print(df.shape)
        num = 0
        for id, row in df.iterrows():
            print(num)
            a = row["content"][:2]
            b = row["ex"][:2]
            numlist.append(a + b)
            num += 1
        df['num'] = numlist
        print('finish')
        return df

    def filter(self, queue, total):
        # print(os.getpid())
        df = self.makedf(total)
        df = self.looptest(df)
        queue.put(df)


if __name__ == '__main__':
    queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(5)
    t = Test()
    s = time.time()
    res = pool.apply_async(func=t.filter, args=(queue, 10))
    res.wait()
    print("耗时：", time.time() - s)
    pool.close()
    pool.join()
    print(queue.get())
    # print(len(queue.get()))

