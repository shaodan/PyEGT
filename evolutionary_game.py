# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.06.28 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import population, update, game

#生成网络
# G = ba(N, 5, 3, 1);
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(5, 1000)
G = nx.random_graphs.barabasi_albert_graph(100,5,3)
g = game.PDG()
# g = game.PGG(3)
u = update.BirthDeath()
# u = update.DeathBirth()
p = population.Population(G, g, u)
p.evolve(10000, 1000)
p.show()
