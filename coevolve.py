# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Rule(object):
    def __init__(self, order):
        self.size = order
        self.color = None
        self.label = None

    def rewire(self, graph, s_e, anchor):
        pass

    def draw(self):
        plt.figure(2)
        for i in xrange(self.size):
            plt.plot(self.evl[:][i], self.color[i], label=self.label[i])
        plt.title('Co-evolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()


class Rewire(Rule):

    def __init__(self, order):
        super(self.__class__, self).__init__(order)
        self.color = 'brgcmykw'
        # self.symb = '.ox+*sdph'
        self.label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']

    def rewire(self, graph, s_e, anchor):
        change_list = [anchor]
        size = len(graph)
        if anchor is None:
            pass
        else:
            p = []
            k = G.degree(anchor)
            if s_e == 0:    # 随机选择
                p = np.ones(size)
            elif s_e == 1:  # 度优先
                p = np.array(G.degree().values(), dtype=np.float64)
            elif s_e == 2:  # 相似度
                p = np.array([len(list(nx.common_neighbors(G, anchor, x))) for x in G.nodes_iter()], dtype=np.float64)
                # 防止没有足够公共节点的
                p += 1
            elif s_e == 3:
                pass
            elif s_e == 4:
                pass
            p[anchor] = 0
            p /= float(p.sum())
            new_neigh = np.random.choice(size, k, replace=False, p=p)
            G.remove_edges_from(G.edges(anchor))
            for node in new_neigh:
                # if node >= anchor:
                #     node += 1
                G.add_edge(anchor, node)

    def rewire_new(self, G, s_e, anchor):
        # rewire only one link
        if s_e == 0:
            p = np.ones(self.N)
        elif s_e == 1:
            p = np.array(G.degree().values(), dtype=np.float64)
        else:
            pass
        p[anchor] = 0
        p /= float(p.sum())
        new_neigh = np.random.choice(self.N, 1, replace=False, p=p)
        k = G.degree(anchor)
        G.remove_edges_from(G.edges(anchor)[np.random.choice(k, 1, False)])
        G.add_edge(anchor, new_neigh)


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 100)
    fitness = np.random.randint(1,3, size=100) * 1.0
