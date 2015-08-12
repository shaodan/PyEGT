# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.06.28 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import population, update, game

## 生成网络
# G = ba(N, 5, 3, 1);
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(5, 1000)
G = nx.random_graphs.barabasi_albert_graph(100,5,3)

## 博弈类型
g = game.PDG()
# g = game.PGG(3)

## 学习策略
# u = update.BirthDeath()
u = update.DeathBirth()

## 演化
p = population.Population(G)
p.evolve(g, u, 100, 10)
p.show()
