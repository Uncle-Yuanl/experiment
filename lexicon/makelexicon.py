import pandas as pd


if __name__ == '__main__':
    # 部位 + 程度
    df_zonghe = pd.read_excel('./词云0303-综合词库_全部_2021-03-10 19-38-48.xlsx')
    df_buwei = df_zonghe[df_zonghe['TagName'].str.startswith("需求.部位")]
    df_chengdu = df_zonghe[df_zonghe['TagName'].str.startswith("需求.程度")]
    df_chengdu['TagName'] = df_chengdu['TagName'].apply(lambda x: ".".join(x.split(".")[:2] + x.split(".")[-1]))

    # 类别 + "问题期许"
    df_TA = pd.read_excel('./奶粉TA词库 - 更新0223-全all.xlsx')
    bools = df_TA['TagName'].str.startswith("需求")
    df_qixu = df_TA[bools]
    df_res = df_TA[~bools]

    # 更新df_qixu
    df_qixu['TagName'] = df_qixu['TagName'].apply(lambda x: "需求.问题期许" + ".".join(x.split(".")[-2:]))

    # 拼接
    dfres = pd.concat([df_qixu, df_res, df_buwei, df_chengdu])

    # 写入文件
    dfres.to_excel('./final_lexicon.xlsx', index=False)