# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

class Coevlv:
    def __init__(self, S):
        self.S = S
        self.N = None

    def rewire(self, G, s_e, anchor):
        pass

    def draw(self):
        plt.figure(2)
        for i in xrange(self.S):
            plt.plot(self.evl[:][i], self.color[i], label=self.label[i])
        plt.title('Coevolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()

class Rewire(Coevlv):

    def __init__(self, S):
        super(S, self).__init__()
        self.color = 'brgcmykw'
        # self.symb = '.ox+*sdph'
        self.label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']


    def rewire(self, G, s_e, anchor):
        if self.N == None:
            self.N = len(G)
        change_list = [anchor]
        if anchor==None:
            pass
        else:
            k = G.degree(anchor)
            if s_e==0:   # 随机选择
                p = np.ones(self.N)
            elif s_e==1: # 度优先
                p = np.array(G.degree().values(),dtype=np.float64)
            elif s_e==2: # 相似度
                p = np.array([len(list(nx.common_neighbors(G,anchor,x))) for x in G.nodes_iter()],dtype=np.float64)
                # 防止没有足够公共节点的
                p = p + 1
            elif s_e==3:
                pass
            elif s_e==4:
                pass
            p[anchor] = 0
            p = p / float(p.sum())
            new_neigh = np.random.choice(self.N,k,replace=False,p=p)
            G.remove_edges_from(G.edges(anchor))
            for node in new_neigh:
                # if node >= anchor:
                #     node += 1
                G.add_edge(anchor, node)

if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    fitness = np.random.randint(1,3, size=100) * 1.0
