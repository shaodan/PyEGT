# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import update
import game


class Population:

    def set_game(self, game):
        self.game = game
        # 0合作， 1背叛
        self.strategies = np.random.randint(2, size=self.N)
        self.S = None

    def __init__(self, graph, mutation=False):
        self.graph = graph
        self.N = len(graph)
        # 需要记录收益数组、策略数组
        self.fitness = np.empty(self.N, dtype=np.double)
        self.set_game(None)
        self.mutation = mutation
        self.rec = None
        self.S = None
        self.evl = None
        self.evolve_strategies = None

    # def __init__(self, graph, game):
    #     self.set_graph(graph)
    #     self.set_game(game)

    # 演化过程：game博弈类型、update更新规则、turns迭代轮数、cycle提示轮数
    def evolve(self, game, update, turns, cycle=None):
        # 演化记录
        self.rec = [0] * turns
        # 输出间隔
        if cycle is None:
            cycle = turns/100
        # 循环
        death = None
        update.set_param(self.graph, self.fitness)
        game.set_param(self.graph, self.strategies, self.fitness)
        for i in xrange(turns):
            game.interact()
            (birth, death) = update.update()

            if np.random.random() > 0.01:
                new_s = self.strategies[birth]
            else:
                new_s = np.random.randint(2)

            # 统计绘图
            if i == 0:
                self.rec[0] = (self.strategies == 0).sum()
            else:
                self.rec[i] = self.rec[i-1] + self.strategies[death] - new_s

            # 更新策略
            self.strategies[death] = new_s

            if (i+1)%cycle == 0:
                print('turn:'+str(i+1))

    # 共演过程：game博弈类型、update更新规则、coevolv共演规则, turns迭代轮数、cycle提示轮数
    def coevolve(self, game, update, coevlv, turns, cycle=None):
        # 演化记录
        self.rec = [0] * turns
        self.S = coevlv.S
        self.evl = np.zeros((self.S,turns), dtype=np.int)
        self.evolve_strategies = np.random.randint(self.S, size=self.N)
        # 输出间隔
        if cycle is None:
            cycle = turns/10
            if cycle < 1:
                cycle = 10
        # 循环
        death = None
        update.set_graph(self.graph, self.fitness)
        for i in xrange(turns):
            game.interact(self.graph, self.strategies, self.fitness)
            (birth, death) = update.update()

            if np.random.random() > 0.01:
                new_s = self.strategies[birth]
                new_s_e = self.evolve_strategies[birth]
            else:
                new_s = np.random.randint(2)
                new_s_e = np.random.randint(self.S)

            # 统计绘图
            if i == 0:
                self.rec[0]= (self.strategies==0).sum()
            else:
                self.rec[i] = self.rec[i-1] + self.strategies[death] - new_s
            for m in xrange(self.S):
                self.evl[m][i] = (self.evolve_strategies==m).sum()

            # 更新策略
            self.strategies[death] = new_s
            self.evolve_strategies[death] = new_s_e

            coevlv.rewire_new(self.graph, self.evolve_strategies[death], death)

            if (i+1)%cycle == 0:
                print('turn:'+str(i+1))

    def show(self):
        plt.figure(1)
        plt.plot(self.rec)
        # x_old = range(len(self.rec))
        # x = np.linspace(x_old[0],x_old[-1],300)
        # y = spline(x_old,self.rec,x)
        # plt.plot(x,y)
        plt.title('Evolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Cooperation Ratio')

        if self.S is not None:
            plt.figure(2)
            color = 'brgcmykw'
            # symb = '.ox+*sdph'
            label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
            for i in xrange(self.S):
                plt.plot(self.evl[:][i], color[i], label=label[i])
            plt.title('Coevolutionary Game')
            plt.xlabel('Step')
            plt.ylabel('Strategies')
            plt.legend()
        plt.show()

if __name__ == "main":
    G = nx.random_regular_graph(5, 10)
    g = game.PDG()
    u = update.BD()
    p = Population(G)
    p.evolve(g, u, 10000)
