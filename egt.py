# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import networkx as nx
import evolution
import update
import game
import rewire

# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(5, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10,10))
# G = nx.star_graph(10)
G = nx.random_graphs.barabasi_albert_graph(100, 5, 10)
# douban = nx.read_edgelist('dataset/ASU/Douban-dataset/data/edges.csv', delimiter=',', nodetype=int, data=False)
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
e.evolve(1000, 100)
# 分析节点最终fit和结构参数的关系
e.show_degree()

# 重复实验，得到关系图
# a = [0] * 100
# for i in xrange(100):
#     e.evolve(1000, 100)
#     a[i] = e.proportion[-1]

# 共演
# r = rewire.Rewire(3)
# p.coevolve(g, u, r, 1000)

# 画图
# p.show()
