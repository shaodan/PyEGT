# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import evolution
import rule
import game
import adapter


# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(4, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10,10))
# G = nx.star_graph(10)
# G = nx.random_graphs.watts_strogatz_graph(1000, 4, 0.3)
# G = nx.random_graphs.barabasi_albert_graph(100, 5, 10)
# G = nx.random_graphs.powerlaw_cluster_graph(1000, 10, 0.2)
# G = nx.convert_node_labels_to_integers(nx.davis_southern_women_graph())
# G = ["/../../DataSet/ASU/Douban-dataset/data/edges.csv", ',']
# G = {"path":"/../wechat/barabasi_albert_graph(5000,100)_adj.txt", "fmt":"adj"}
G = "/../wechat/facebook.txt"

# 网络结构绘图
# pos=nx.spring_layout(G)
# nx.draw_networkx(G,pos,node_size=20)
# plt.show()
# plt.savefig("graph.png")
# exit(1)

# 博弈类型
g = game.PDG()
# g = game.PGG(3)

# 学习策略
# u = rule.BirthDeath()
u = rule.DeathBirth()
# u = rule.Fermi()
# u = rule.HeteroFermi(g.delta)

# 演化
e = evolution.Evolution(G, g, u)
e.evolve(20000)

# 重复实验，得到关系图
def repeat(times):
    a = [0] * times
    for i in xrange(100):
        e.evolve(20000, 20000)
        print(i)
        a[i] = e.proportion[-1]

# 共演
# a = adapter.Preference(3)
# e = evolution.CoEvolution(G, g, u, a)
# e.evolve(2000)

# 画图
e.show()
# 分析节点最终fit和结构参数的关系
# e.degree_distribution()
