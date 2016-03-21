# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import datetime, os
import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import update
import game


class Evolution(object):

    def __init__(self, graph, game_type, update_rule):
        self.population = graph
        self.game = game_type
        self.rule = update_rule
        self.size = len(graph)
        # 迭代次数，中断续演
        self.generation = 0
        self.fitness = np.empty(self.size, dtype=np.double)
        # 策略: 0合作， 1背叛
        self.strategy = np.random.randint(2, size=self.size)
        # initialize variants
        self.has_mut = False
        self.proportion = None

    # 演化过程
    def evolve(self, turns, profile=None):
        # 演化记录
        self.proportion = [0] * turns
        # 输出间隔
        if profile is None:
            profile = turns/100
            if profile < 1:
                profile = 10
        # 循环
        for i in xrange(turns):
            self.game.interact(self.population, self.strategy, self.fitness)
            (birth, death) = self.rule.update(self.population, self.fitness)

            if np.random.random() > 0.01:
                new_s = self.strategy[birth]
            else:
                new_s = np.random.randint(2)

            # 统计
            if i == 0:
                self.proportion[0] = (self.strategy == 0).sum()
            else:
                self.proportion[i] = self.proportion[i - 1] + self.strategy[death] - new_s

            # 更新策略
            self.strategy[death] = new_s

            # 记录总演化轮数
            self.generation += 1
            if self.generation % profile == 0:
                print('turn:'+str(self.generation))

    def show(self):
        plt.figure(1)
        plt.plot(self.proportion)
        # x_old = range(len(self.log))
        # x = np.linspace(x_old[0],x_old[-1],300)
        # y = spline(x_old,self.log,x)
        # plt.plot(x,y)
        plt.title('Evolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Cooperation Ratio')

        plt.show()

    def show_degree(self):
        # plt.figure(1)
        # plt.plot(self.population.degree().values(), self.fitness, marker='*')
        # plt.show()
        plt.scatter(self.population.degree().values(), self.fitness)
        plt.show(block=False)

    def save(self):
        current = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = 'data/'+current
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_pajek(path)
        sio.savemat(path +'/data.mat', mdict={'fitness': self.fitness,
                                             'strategy': self.strategy,
                                             'log': self.proportion})

    def load(self, path):
        self.load_pajek(path)
        mat = sio.loadmat(path +'/data.mat', mdict={'fitness': self.fitness,
                                                   'strategy': self.strategy,
                                                   'log': self.proportion})

    def save_pajek(self, path):
        nx.write_pajek(self.population, path+'/graph.net')

    def load_pajek(self, path):
        self.population = nx.read_pajek(path+'/graph.net')


class CoEvolution(Evolution):

    def __init__(self, graph, game_type, update_rule, coevolve_rule):
        super(self.__class__, self).__init__(graph, game_type, update_rule)
        self.coevolve = coevolve_rule
        self.s_size = 3
        self.evl = None
        self.evolve_strategies = None

    # 共演过程
    def evolve(self, turns, profile=None):
        super(self.__class__, self).evolve(turns, profile)
        # 演化记录
        self.proportion = [0] * turns
        self.s_size = coevlv.S
        self.evl = np.zeros((self.s_size, turns), dtype=np.int)
        self.evolve_strategies = np.random.randint(self.s_size, size=self.size)
        # 输出间隔
        if profile is None:
            profile = turns/10
            if profile < 1:
                profile = 10
        # 循环
        death = None
        update.set_graph(self.population, self.fitness)
        for i in xrange(turns):
            game.interact(self.population, self.strategy, self.fitness)
            (birth, death) = update.update()

            if np.random.random() > 0.01:
                new_s = self.strategy[birth]
                new_s_e = self.evolve_strategies[birth]
            else:
                new_s = np.random.randint(2)
                new_s_e = np.random.randint(self.s_size)

            # 统计绘图
            if i == 0:
                self.proportion[0]= (self.strategy == 0).sum()
            else:
                self.proportion[i] = self.proportion[i - 1] + self.strategy[death] - new_s
            for m in xrange(self.s_size):
                self.evl[m][i] = (self.evolve_strategies==m).sum()

            # 更新策略
            self.strategy[death] = new_s
            self.evolve_strategies[death] = new_s_e

            coevlv.rewire_new(self.population, self.evolve_strategies[death], death)

            if (i+1)%profile == 0:
                print('turn:'+str(i+1))

    def show(self):
        super(self.__class__, self).show()

        plt.figure(2)
        color = 'brgcmykw'
        # symb = '.ox+*sdph'
        label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
        for i in xrange(self.s_size):
            plt.plot(self.evl[:][i], color[i], label=label[i])
        plt.title('Coevolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 10)
    g = game.PDG()
    u = update.BirthDeath()
    p = Evolution(G, g, u)
    p.evolve(10000)
