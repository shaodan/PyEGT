# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.06.28 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#~ import game

#参数设定
N  = 1000              # 网络规模
K = 10000            # 演化循环
C = 10                  # 平均出度
cycle = 1000            # 更新周期
m_rate = 0.05      # 突变概率
strengh = 0.2      # 选择强度

#生成网络
# network = nx.davis_southern_women_graph()
network = nx.random_regular_graph(C, N)
# network = nx.random_graphs.barabasi_albert_graph(N,C)


#节点策略
# 博弈策略, 0.合作 1.背叛
s = np.random.randint(2, size=N)

# 演化记录
# r_s = np.zeros((1,K), dtype=np.int)
r_s = [0] * K

# pdg收益矩阵
R_p = 1;T_p = 1.5;S_p = 0;P_p = 0.1
payoff_matrix = np.array([[(R_p, R_p), (S_p, T_p)], [(T_p,S_p), (P_p,P_p)]], dtype=np.double)
# todo: s是int类型的，fitness是list，anchor是int
def pdg(G, s, fitness, anchor):
    # R = 1;T = 1.5;S = 0;P = 0.1
    # payoff_matrix = np.array([[R, T], [S, P]], dtype=np.double)
    if not isinstance(anchor, int):
         # 第一次，计算所有节点的收益
        fitness.fill(0)
        for edge in G.edges_iter():
            a = edge[0]
            b = edge[1]
            p = payoff_matrix[s[a]][s[b]]
            fitness[a] += p[0]
            fitness[b] += p[1]
    elif anchor>=0 :    
        # 只用计算新节点和其邻居节点的收益
        f = 0 # 新节点收益从0计算
        neigh_iter = G.neighbors_iter(anchor)
        for neigh in neigh_iter:
            p = payoff_matrix[s[anchor]][s[neigh]]
            f += p[0]           # 新节点累加
            new_payoff = p[1]   # 邻居节点计算新的收益
            # 0合作，1背叛
            p = payoff_matrix[1-s[anchor]][s[neigh]]
            old_payoff = p[1]   # 邻居节点计算原来的收益
            fitness[neigh] += new_payoff - old_payoff
        fitness[anchor] = f
    else:
        # 节点策略没有变化
        pass

def pgg(G, s, fitness, anchor=None):
    # 获利倍数
    r = 3.0
    # 可能会有负的fitness
    if True or not isinstance(anchor, int):
        # 第一次，计算所有节点的收益
        fitness.fill(0)
        # 第一种每个group投入1
        degrees = np.array(G.degree().values())
        for node in G.nodes_iter():
            degree = degrees[node]
            fitness[node] += (s[node] - 1) * (degree+1)
            neighs = G.neighbors(node)
            neighs.append(node)
            b = r * (s[neighs]==0).sum() / (degree+1)
            for neigh in neighs:
                fitness[neigh] += b
        # 第二种每个group投入1/(k+1)
        # degrees = np.array(G.degree().values())
        # inv = (1.0-s) / (degrees)
        # for node in G.nodes_iter():
        #     fitness[node] += s[node] - 1
        #     neighs = G.neighbors(node)
        #     neighs.append(node)
        #     b = r * inv[neighs].sum() / (degrees[node]+1)
        #     for neigh in neighs:
        #         fitness[neigh] += b
    elif anchor>=0 :
        # 只用计算新节点和其邻居节点的收益
        # 0合作，1背叛
        # 现在0背叛到合作(+)，1合作到背叛(-)
        # sign = 
        fitness[anchor]
        neigh_iter = G.neighbors_iter(anchor)
        for neigh in neigh_iter:
            # p = 
            f += p[0]           # 新节点累加
            new_payoff = p[1]   # 邻居节点计算新的收益
            p = payoff_matrix[1-s[anchor]][s[neigh]]
            old_payoff = p[1]   # 邻居节点计算原来的收益
            fitness[neigh] += new_payoff - old_payoff
        fitness[anchor] = f
    else:
        # 节点策略没有变化
        pass

# from timeit import Timer
# t1=Timer("pgg(G,s,fitness,None)","from __main__ import pgg,pdg;N=1000;import networkx as nx;import numpy as np;G=nx.random_regular_graph(5, N);fitness=np.empty(N, dtype=np.double);s = np.random.randint(2, size=N)")
# print t1.timeit(300)/300
# exit()

def BD(G, fitness):
    # N = fitness.size
    # p = fitness / fitness.sum()
    p = fitness * strengh + 1
    p = p / p.sum()
    birth = np.random.choice(N,replace=False,p=p)
    neigh = G.neighbors(birth)
    death = np.random.choice(neigh,replace=False)
    return birth, death

def DB(G, fitness):
    # N = fitness.size
    death = np.random.randint(N)
    neigh = G.neighbors(death)
    p = fitness[neigh] * strengh + 1
    p = p / p.sum()
    birth = np.random.choice(neigh,replace=False,p=p)
    return birth, death

# Main process
fitness = np.empty(N, dtype=np.double)
# fitness = [0] * N
death = None
for i in xrange(K):
    # 根据网络结构进行博弈
    # pdg(network, s, fitness, death)
    pgg(network, s, fitness, death)
    # fitness = pgg(network, s)

    # 策略的模拟、扩散
    # (birth,death) = DB(network, fitness)
    (birth,death) = BD(network, fitness)

    # 突变的影响
    if (np.random.random() > m_rate) :
        new_s = s[birth]
    else:
        new_s = np.random.randint(2)

    if s[death]==new_s:
        death = -1
    else:
        s[death] = new_s
    
    # 可以优化，通过r_s[i-1]直接计算
    r_s[i]= (s==0).sum()

    if (i+1)%cycle == 0:
        print('turn:'+str(i+1))


plt.figure()
plt.plot(r_s)
plt.show()