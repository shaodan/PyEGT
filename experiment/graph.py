# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt

# transMat=np.zeros(shape=(3,3))
# transMat[0][0]=0.5
# transMat[0][1]=0.5
# transMat[1][1]=0.5
# transMat[1][2]=0.5
# transMat[2][0]=0.25
# transMat[2][2]=0.75
# dist=[1,0,0]
# # Choose an initial distribution; all cooperators
# for i in xrange(100):
#     dist=np.dot(dist,transMat)
# # Multiply distribution and matrix together
# # until the distribution is stable
# print dist


G = nx.random_graphs.barabasi_albert_graph(1000,3)
print G.degree(0)
print G.degree()
print nx.degree_histogram(G)

degree = nx.degree_histogram(G)
x = range(len(degree))
y = [z / float(sum(degree)) for z in degree]

plt.loglog(x, y, color="blue", linewidth=2)
plt.show() 