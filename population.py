# -*- coding: utf-8 -*-
''' -*- Author: shaodan -*- '''
''' -*-  2015.07.11 -*- '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import game


class Population:
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)
        # 需要记录收益数组、策略数组
        self.fitness = np.empty(self.N, dtype=np.double)
        self.strategies = np.random.randint(2, size=self.N)

    # 演化过程：game博弈类型、update更新规则、turns迭代轮数、cycle提示轮数
    def evolve(self, game, update, turns, cycle=None):
        # 演化记录
        self.cop = [0] * turns
        # 输出间隔
        if cycle == None:
            cycle = turns/100;
        # 循环
        death = None
        for i in xrange(turns):
            game.interact(self.graph, self.strategies, self.fitness)
            (birth,death) = update.update(self.graph, self.fitness)

            if (np.random.random() > 0.01) :
                new_s = self.strategies[birth]
            else:
                new_s = np.random.randint(2)

            self.strategies[death] = new_s
            
            # 统计绘图
            if i == 0:
                self.cop[0]= (self.strategies==0).sum()
            else:
                self.cop[i] = self.cop[i-1] + 1 - 2*new_s

            if (i+1)%cycle == 0:
                print('turn:'+str(i+1))

    # 共演过程：game博弈类型、update更新规则、coevolv共演规则, turns迭代轮数、cycle提示轮数
    def coevolve(self, game, update, coevlv, turns, cycle=None):
        # 演化记录
        self.cop = [0] * turns
        self.S = coevlv.strategies.size
        self.elv = np.zeros((self.S,turns), dtype=np.int)
        self.evolve_strategies = np.random.randint(S, size=N)
        # 输出间隔
        if cycle == None:
            cycle = turns/100;
        # 循环
        death = None
        for i in xrange(turns):
            game.interact(self.graph, self.strategies, self.fitness)
            (birth,death) = update.update(self.graph, self.fitness)

            if (np.random.random() > 0.01) :
                new_s = self.strategies[birth]
                # new_s_e = self.evolve_strategies[birth]
            else:
                new_s = np.random.randint(2)
                # new_s_e = np.random.randint(self.S)

            self.strategies[death] = new_s
            
            # 统计绘图
            if i == 0:
                self.cop[0]= (self.strategies==0).sum()
            else:
                self.cop[i] = self.cop[i-1] + 1 - 2*new_s
            # for m in xrange(S):
            #     record[m][i] = (s_e==m).sum()

            # rewire(network,s_e[death],death)

            if (i+1)%cycle == 0:
                print('turn:'+str(i+1))

    def show(self):
        plt.figure(1)
        plt.plot(self.cop)
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
        plt.show()

if __name__ == "main":
    G = nx.random_regular_graph(5, 10)
    g = game.PDG()
    u = update.BD()
    p = Population(G)
    p.evolve(g,u,10000)