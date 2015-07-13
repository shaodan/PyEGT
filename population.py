# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.07.11 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import game


class Population:
    def __init__(self, graph, game, update):
        self.graph = graph
        self.game = game
        self.N = len(graph)
        self.strategies = np.random.randint(2, size=N)
        game.set_graph(graph)
        game.set_strategies(self.strategies)
        # self.S = update.size
        # self.evolve_strategies = np.random.randint(S, size=N)

    def evolve(self, turns, cycle=None):
        # 演化记录
        self.cop = [0] * turns
        self.elv = np.zeros((self.S,turns), dtype=np.int)
        # 输出间隔
        if cycle == None:
            cycle = turns/100;
        # 循环
        for i in xrange(turns):
            self.game.interact(self.graph, self., fitness, death)

            # (birth,death) = self.(network, fitness)

            if (np.random.random() > 0.01) :
                new_s = self.strategies[birth]
                # new_s_e = self.evolve_strategies[birth]
            else:
                new_s = np.random.randint(2)
                # new_s_e = np.random.randint(self.S)
            
            # 可以优化，通过r_s[i-1]直接计算
            self.cop[i-1]= (self.strategies==1).sum()
            # for m in xrange(S):
            #     record[m][i] = (s_e==m).sum()

            rewire(network,s_e[death],death)

            if (i+1)%cycle == 0:
                print('turn:'+str(i+1))

    def show(self):
        plt.figure(1)
        plt.plot(self.cop)
        plt.show()
        plt.close()
        # plt.figure(2)
        # color = 'brgcmykw';
        # symb = '.ox+*sdph';
        # label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
        # for i in xrange(S):
        #     plt.plot(range(K),record[:][i],color[i]+symb[i], label=label[i])
        # plt.title('Evolutionary game');
        # plt.xlabel('step');
        # plt.ylabel('strategies');
        # plt.legend();
        # plt.show()

if __name__ == "main":
    G = nx.random_regular_graph(5, 10)
    s = np.random.randint(2, size=10)
    g = game.PDG()
    u = update.BD()
    p = Population(G, g, u)
    p.evolve()