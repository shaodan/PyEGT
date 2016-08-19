# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from evolution import Evolution, CoEvolution
import rule
import game
import adapter


# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
G = nx.random_regular_graph(5, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10,10))
# G = nx.star_graph(10)
# G = nx.random_graphs.watts_strogatz_graph(1000, 4, 0.3)
# G = nx.random_graphs.barabasi_albert_graph(5000, 100, 10)
# G = nx.random_graphs.powerlaw_cluster_graph(1000, 10, 0.2)
# G = nx.convert_node_labels_to_integers(nx.davis_southern_women_graph())
# G = ["/../../DataSet/ASU/Douban-dataset/data/edges.csv", ',']
# G = {"path":"/../wechat/barabasi_albert_graph(5000,100)_adj.txt", "fmt":"adj"}
# G = "/../wechat/facebook.txt"


# 博弈类型
g = game.PDG()
# g = game.PGG(3)

# 学习策略
u = rule.BirthDeath()
# u = rule.DeathBirth()
# u = rule.Fermi()
# u = rule.HeteroFermi(g.delta)

# 演化
# e = Evolution(G, g, u, has_mut=True)

# 共演
a = adapter.Preference(3)
e = CoEvolution(G, g, u, a, has_mut=False)

e.evolve(10000)
e.show()

# 分析节点最终fit和结构参数的关系
# e.degree_distribution()


# 重复实验，得到关系图
def repeat(times):
    a = [0] * times
    for i in xrange(100):
        e.evolve(20000)
        print(i)
        a[i] = e.cooperate[-1]
    plt.plot(a, 'r-')
