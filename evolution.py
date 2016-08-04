# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import os
import datetime
import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from population import Population
import rule
import game


class Evolution(object):

    def __init__(self, graph, gametype, updaterule, has_mut=True):
        assert isinstance(gametype, game.Game)
        assert isinstance(updaterule, rule.Rule)
        self.population = Population(graph)
        self.game = gametype.bind(self.population)
        self.rule = updaterule.bind(self.population)
        # 迭代次数，中断续演
        self.gen = 0
        # 是否突变
        self.has_mut = has_mut
        # 合作率记录
        self.proportion = None

    def next_generation(self):
        # play game
        self.game.play(self.death)

        # update strategy
        (birth, death) = self.rule.update()
        if self.has_mut and np.random.random() <= 0.01:
            new_strategy = np.random.randint(2)
        else:
            new_strategy = self.population.strategy[birth]

        if new_strategy == self.population.strategy[death]:
            increase = 0
            death = []
        else:
            self.population.strategy[death] = new_strategy
            increase = new_strategy*2-1

        self.death = death
        # record cooperation rate path
        return self.population.cooperation_rate(increase)

    def evolve(self, turns, profile=None):
        self.death = None
        self.proportion = [0] * turns

        if profile is None:
            profile = turns/10
            if profile < 1:
                profile = 10

        for i in xrange(turns):
            self.proportion[i] = self.next_generation()
            self.gen += 1
            if self.gen % profile == 0:
                self.death = None
                # print('accumulative error:'+self.game.error(death))
                death = None
                print('turn:%d/%d'%(self.gen, turns))
        # self.show()

    # 同步演化
    def evolve_syn(self, turns, profile=None):
        assert isinstance(self.rule, rule.Imitation)
        pass

    # 变化趋势
    def show(self, wait=False):
        f = plt.figure(1)
        plt.plot(self.proportion)
        # x_old = range(len(self.log))
        # x = np.linspace(x_old[0],x_old[-1],300)
        # y = spline(x_old,self.log,x)
        # plt.plot(x,y)
        plt.title('Evolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Cooperation Ratio')

        if wait:
            f.show()
        else:
            plt.show()

    # 度和收益相关性
    def show_degree(self):
        # plt.figure(1)
        # plt.plot(self.population.degree().values(), self.fitness, marker='*')
        # plt.show()
        plt.scatter(self.population.degree().values(), self.population.fitness)
        plt.show(block=True)

    # 保存演化结果
    def save(self):
        time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.dirname(os.path.realpath(__file__)) + time
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_pajek(path)
        sio.savemat(path+'/data.mat', mdict={'gen': self.gen,
                                             'fit': self.population.fitness,
                                             'stg': self.population.strategy,
                                             'log': self.proportion})

    # 读取演化结果
    def load(self, path):
        self.load_pajek(path)
        mat = sio.loadmat(path+'/data.mat', mdict={'gen': self.gen,
                                                   'fit': self.population.fitness,
                                                   'stg': self.population.strategy,
                                                   'log': self.proportion})
        self.population.fitness = mat['fitness']
        self.population.strategy = mat['strategy']
        self.proportion = mat['log']

    def save_pajek(self, path):
        nx.write_pajek(self.population, path+'/graph.net')

    def load_pajek(self, path):
        self.population = nx.read_pajek(path+'/graph.net')


class StaticStrategy(Evolution):

    def __init__(self, graph, gametype, updaterule):
        super(self.__class__, self).__init__(graph, gametype, updaterule)


    def evolve(self, turns, profile=None):
        super(self.__class__, self).evolve(turns, profile)


class CoEvolution(Evolution):

    def __init__(self, graph, gametype, updaterule, adapter):
        super(self.__class__, self).__init__(graph, gametype, updaterule)
        assert(isinstance(adapter, adapter.Adapter))
        self.coevolve = adapter
        self.s_size = adapter.order
        self.preference = None
        self.evl = None

    def next_generation(self):
        proportion = super(self.__class__, self).next_generation()
        new_s_e = np.random.randint(self.s_size) if self.has_mut else self.preference[birth]

        for m in xrange(self.s_size):
            self.evl[m][i] = (self.preference == m).sum()
        old, new = self.coevolve.adapt_once(self.population, self.preference[death], death)
        return proportion

    def evolve_next(self, turns, profile=None):
        self.preference = np.random.randint(self.s_size, size=self.population.size)
        self.evl = np.zeros((self.s_size, turns), dtype=np.int)
        super(self.__class__, self).evolve(turns, profile)

    # 共演过程
    def evolve(self, turns, profile=None):
        # 演化记录
        self.proportion = [0] * turns
        self.evl = np.zeros((self.s_size, turns), dtype=np.int)
        self.preference = np.random.randint(self.s_size, size=self.population.size)
        # 输出间隔
        if profile is None:
            profile = turns/10
            if profile < 1:
                profile = 10
        # 循环
        death = None
        for i in xrange(turns):
            self.game.play(death)
            (birth, death) = self.rule.update()

            if self.has_mut and np.random.random() > 0.01:
                new_s = self.population.strategy[birth]
                new_s_e = self.preference[birth]
            else:
                new_s = np.random.randint(2)
                new_s_e = np.random.randint(self.s_size)

            # 统计绘图
            if i == 0:
                self.proportion[0] = self.population.cooperation_rate()
            else:
                self.proportion[i] = self.proportion[i-1] + self.population.strategy[death]-new_s

            for m in xrange(self.s_size):
                self.evl[m][i] = (self.preference == m).sum()

            # 更新策略
            self.population.strategy[death] = new_s
            self.preference[death] = new_s_e

            old, new = self.coevolve.adapt_once(self.population, self.preference[death], death)
            self.rewired = (old, new)

            if (i+1) % profile == 0:
                print('turn:'+str(i+1))

    def show(self, wait=False):
        super(self.__class__, self).show(True)
        f = plt.figure(2)
        color = 'brgcmykw'
        # symb = '.ox+*sdph'
        label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
        for i in xrange(self.s_size):
            plt.plot(self.evl[:][i], color[i], label=label[i])
        plt.title('CoEvolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 10)
    g = game.PDG()
    u = rule.BirthDeath()
    e = Evolution(G, g, u)
    e.evolve(10000)
    e.show()
