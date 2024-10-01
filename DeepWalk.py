import networkx as nx
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import random
df = pd.read_csv("../data/seealsology-data.tsv", sep="\t")
# print(df.shape) # 5202行3列
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())

# print(len(G)) # 3720个点

def get_random_walk(node, length):
    random_walk = [node]
    for i in range(length - 1):
        temp = list(G.neighbors(node)) # 去重
        temp = list(set(temp) - set(random_walk)) # 去重
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(random_node)
    return random_walk

# print(get_random_walk('digital humanities', 10))

gamma = 10
walk_length = 5
all_nodes = list(G.nodes)
random_walks = []

for n in tqdm(all_nodes):
    for i in range(gamma):
        random_walks.append(get_random_walk(n, walk_length))

# print(random_walks[1])

model = Word2Vec(vector_size=8, # Embedding 维数
                 window=4, # 窗口大小
                 sg=1, # skip gram
                 hs=0, # 不加分层softmax
                 negative=10, # 负采样
                 alpha=0.03, # 初始学习率
                 min_alpha=0.0007, # 最低学习率
                 seed=14)
# print(random_walks)

model.build_vocab(random_walks, progress_per=2)

model.train(random_walks, total_examples=model.corpus_count, epochs=50, report_delay=1)

print(model.wv.get_vector('digital humanities')) # 输出结点嵌入
print(model.wv.get_vector('digital rhetoric')) # 输出结点嵌入

print(model.wv.similar_by_word('digital humanities')) # 找相似结点