import pandas as pd

# df = pd.read_excel('./聚类.xlsx', header=None)[0].tolist()
#
# print(df)

outputwords = [["过敏", "牙齿过敏"],
        ["变白", "牙齿变白", "牙齿出现裂缝", "牙齿发黄", "牙齿黄", "牙齿敏感", "牙齿松动", "牙齿疼痛", "牙齿有斑点",
         "牙敏感", "牙齿健康", "根部蛀牙", "焕白牙齿", "坚固牙齿", "健白牙齿", "美白", "美白牙齿", "强健牙齿", "清洁牙齿",
         "三重美白", "伤害牙釉质", "牙齿变黑", "牙齿变黄", "牙齿酸痛", "牙齿自然白", "牙垢", "预防牙垢", "早晚美白", "蛀牙"],
        ["一嚼东西就疼", "遇冷就疼", "口气", "唇齿留香", "刺激", "黑黄牙", "清新口气",
         "去黄", "去火", "去烟渍", "夏日抗敏新体", "牙锈会自然脱落"],
        ["龋齿", "牙黄", "牙结石", "牙菌斑", "牙髓发炎", "牙髓感染", "牙龈炎", "牙周炎", "牙渍", "口干", "牙本质敏感"],
        ["护齿", "护齿健龈", "清火"],
        ["固齿", "护龈", "护龈清火", "健齿", "锈斑", "炫白"],
        ["持久清新", "除菌", "干净", "清洁", "清洁力", "清爽", "杀菌", "深层洁净"],
        ["出血", "红肿", "口腔溃疡", "溃疡", "上火", "牙龈出血", "牙龈发炎", "牙龈敏感", "牙龈上火", "牙龈萎缩",
         "牙龈肿", "牙龈肿大", "牙龈痛", "牙龈肿痛", "牙龈肿胀", "发炎", "肿痛", "牙龈健康", "刺激牙龈", "强健牙龈",
         "舒缓牙龈", "牙龈年轻", "牙龈问题", "养护牙龈"],
        ["口臭", "口腔上火", "口腔问题", "口腔异味", "异味", "改善口腔环境", "呵护口腔", "缓解口腔问题",
         "健康口腔", "口腔健康", "平衡口腔菌群", "清洁口腔", "清洁口腔细菌", "清新口腔", "调节口腔酸碱平衡",
         "修复口腔受损组织", "牙周组织健康"],
        ["氟斑牙", "黑牙", "黄牙", "烟牙"]]

with open('./test.txt', 'w', encoding='utf-8') as f:
    for cls, wl in enumerate(outputwords):
        f.write("class: {}".format(cls) + '\n')
        for word in wl:
            f.write(word + '\n')