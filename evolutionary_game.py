# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.06.28 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#~ import game

#参数设定
N  = 10              # 网络规模
K = 10000            # 演化循环
C = 5                   # 平均出度
cycle = 1000             # 更新周期
m_rate = 0.01      # 

#生成网络
# network = ba(N, 5, 3, 1);
# H = nx.davis_southern_women_graph()
network = nx.random_regular_graph(C, N)
#~ network = nx.random_graphs.barabasi_albert_graph(N,5,3)


#节点策略
# 博弈策略, 1.合作 2.背叛
s = np.random.randint(2, size=N)
# 演化策略
# 1.随机 2.度优先 3.共同邻居 4.适应度优先
S = 3
s_e = np.random.randint(S, size=N)

# 演化记录
record = np.zeros((S,K), dtype=np.int)
# r_s = np.zeros((1,K), dtype=np.int)
r_s = [0] * K

R_p = 1;T_p = 1.5;S_p = 0;P_p = 0.1
payoff_matrix = np.array([[(R_p, R_p), (S_p, T_p)], [(T_p,S_p), (P_p,P_p)]], dtype=np.double)


# todo: s是int类型的，fitness是list，anchor是int
def pdg(G, s, fitness, anchor):
    # R = 1;T = 1.5;S = 0;P = 0.1
    # payoff_matrix = np.array([[R, T], [S, P]], dtype=np.double)
    fitness.fill(0)
    for edge in G.edges_iter():
        a = edge[0]
        b = edge[1]
        p = payoff_matrix[s[a]][s[b]]
        fitness[a] += p[0]
        fitness[b] += p[1]
    return fitness

def BD(G, fitness):
    # N = fitness.size
    p = fitness / fitness.sum()
    birth = np.random.choice(N, replace=False,p=p)
    neigh = G.neighbors(birth)
    death = np.random.choice(neigh)
    return birth, death

def DB(G, fitness):
    # N = fitness.size
    death = np.random.randint(N)
    neigh = G.neighbors(death)
    if (len(neigh) == 0):
        print "==========A=========="
        print death
        print neigh
        return death, death
    p = fitness[neigh]
    p = p / p.sum()
    if ((p<0).sum() > 0 ):
        print "==========B=========="
        print death
        print neigh
        print fitness
        print p
        exit()
    birth = np.random.choice(neigh,replace=False,p=p)
    return birth, death
        
def rewire(G, s_e, anchor):
    change_list = [anchor]
    if anchor==None:
        pass
    else:
        k = G.degree(anchor)
        if s_e==0:   # 随机选择
            p = np.ones(N)
        elif s_e==1: # 度优先
            p = np.array(G.degree().values(),dtype=np.float64)
        elif s_e==2: # 相似度
            p = np.array([len(list(nx.common_neighbors(G,anchor,x))) for x in G.nodes_iter()],dtype=np.float64)
        elif s_e==3:
            pass
        elif s_e==4:
            pass
        p[anchor] = 0
        p = p / float(p.sum())
        new_neigh = np.random.choice(N,k,replace=False,p=p)
        G.remove_edges_from(G.edges(anchor))
        for node in new_neigh:
            # if node >= anchor:
            #     node += 1
            G.add_edge(anchor, node)


# Main process
fitness = np.empty(N, dtype=np.double)
death = None
for i in xrange(K):
    # 根据网络结构进行博弈
    pdg(network, s, fitness, death)
    # fitness = pgg(network, s)

    # 策略的模拟、扩散
    (birth,death) = DB(network, fitness)
    # (birth,death) = BD(network, fitness)

    # 突变的影响
    if (np.random.random() > 0.01) :
        new_s = s[birth]
        new_s_e = s_e[birth]
    else:
        new_s = np.random.randint(2)
        new_s_e = np.random.randint(S)

    flag = False
    flag2 = False
    if s[death]==new_s:
        flag = True
    else:
        s[death] = new_s
    if s_e[death] == new_s_e:
        flag2=True
    else:
        s_e[death] = new_s_e
    
    # 可以优化，通过r_s[i-1]直接计算
    r_s[i]= (s==1).sum()
    for m in xrange(S):
        record[m][i] = (s_e==m).sum()

    # 根据策略改变网络结构
    rewire(network,s_e[death],death)

    if flag :
        death = -1

    if (i+1)%cycle == 0:
        print('turn:'+str(i+1))


plt.figure()
plt.plot(r_s)
plt.show()
color = 'brgcmykw';
symb = '.ox+*sdph';
plt.plot(range(k),record,)
plt.show()