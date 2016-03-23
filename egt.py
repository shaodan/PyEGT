# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import os
import networkx as nx
import matplotlib.pyplot as plt
import evolution
import update
import game
import rewire


def process(graph, game, update_rule):
    e = evolution.Evolution(graph, game, update_rule)
    e.evolve(1000, 100)
    return e.proportion[-1]

# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(3, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10,10))
# G = nx.star_graph(10)
G = nx.random_graphs.watts_strogatz_graph(1000, 4, 0.3)
# G = nx.random_graphs.barabasi_albert_graph(1000, 10, 100)
# G = nx.random_graphs.powerlaw_cluster_graph(1000, 10, 0.2)
# G = nx.convert_node_labels_to_integers(nx.davis_southern_women_graph())
# douban = nx.read_edgelist(os.path.dirname(os.path.realpath(__file__))+'/dataset/ASU/Douban-dataset/data/edges.csv', delimiter=',', nodetype=int, data=False)
# G = nx.relabel_nodes(douban, {len(douban): 0}, copy=False)

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
# u = update.BirthDeath()
u = update.DeathBirth()

# 演化
e = evolution.Evolution(G, g, u)
e.evolve(10000)

# 重复实验，得到关系图
# a = [0] * 100
# for i in xrange(100):
#     e.evolve(1000, 100)
#     a[i] = e.proportion[-1]

# 共演
# r = rewire.Rewire(3)
# p.coevolve(g, u, r, 1000)

# 画图
e.show()
# 分析节点最终fit和结构参数的关系
# e.show_degree()