# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import os
import datetime
import numpy as np
import networkx as nx
import scipy.io as sio
import matplotlib.pyplot as plt
from population import Population, DynamicPopulation
import rule
import game
import adapter


class EvolutionState(object):
    def __int__(self):
        self.gen = 0
        self.birth = None
        self.death = None
        self.mut = False

        self.fitness = None
        self.strategy = None


class CoEvolutionState(EvolutionState):
    def __int__(self):
        super(self.__class__, self).__init__()
        self.rewire = None
        self.dynamic = None


class Evolution(object):
    """ Evolutionary Game agent-based Simulation """
    def __init__(self, has_mut=True):
        # 是否突变
        self.has_mut = has_mut
        self.population = None
        self.game = None
        self.rule = None

        # 迭代次数，中断续演
        self.gen = 0
        # 合作率记录
        self.cooperate = None

        self.death = None
        self.rewire = None

    def init_state(self):
        self.es = EvolutionState()
        if self.population is not None:
            pass

    def set_population(self, p):
        assert isinstance(p, Population)
        self.population = p
        return self

    def set_game(self, g):
        assert isinstance(g, game.Game)
        self.game = g
        return self

    def set_rule(self, r):
        assert isinstance(r, rule.Rule)
        self.rule = r
        return self

    def bind_process(self):
        self.population.init_strategies(self.game)
        self.game.bind(self.population)
        self.rule.bind(self.population)

    def next_generation(self):
        self.game.play(self.death, self.rewire)
        birth, death = self.rule.update()

        if self.has_mut and np.random.random() <= 0.01:
            new_s = np.random.randint(self.game.order)
            mutate = True
        else:
            new_s = self.population.strategy[birth]
            mutate = False

        if new_s == self.population.strategy[death]:
            increase = 0
        else:
            self.population.strategy[death] = new_s
            increase = 1-new_s*2

        self.death = death
        return mutate

    def prepare(self, turns, sampling):
        self.gen = 0
        self.death = None
        self.rewire = None

        # todo 全部记录
        if sampling <= 0 or turns <= sampling:
            period = 1
        else:
            period = turns / sampling
        # x = np.linspace(0, turns, sampling + 1, dtype=int)
        # x[-1] = turns - 1
        # y = self.cooperate[x]
        self.cooperate = []
        return period

    def record(self, cop):
        self.cooperate.append((self.gen, cop))

    def evolve(self, turns, sampling=20, quiet=False, autostop=True):
        # self.bind_process()
        period = self.prepare(turns, sampling)
        start_time = datetime.datetime.now()

        # fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': [], 'yticks': []})
        # j = 0
        for i in xrange(turns):
            self.gen += 1
            if self.gen % period == 0:
                delta = self.cooperate[-1]-self.cooperate[-2]
                if autostop and self.time_to_stop(delta):
                    print("stop at turn: %d, delta: %d" % (self.gen, delta))
                    break
                # 精度修正 accuracy correction
                err = self.game.correct_error(self.death, self.rewire)
                # self.population.check_cache()                if not quiet:
                print('turn: %d/%d' % (self.gen, turns))
                print(self.death, self.rewire, ' error:', err, ' delta:', delta)
                self.death, self.rewire = -1, None

                # grid = self.population.dynamics.reshape((30, -1))
                # fig.subplots_adjust(hspace=0.3, wspace=0.05)
                # ax = axes.flat[j]
                # im = ax.imshow(grid, interpolation='nearest')
                # ax.set_title("Generation %d" % self.gen)
                # j += 1

            inc, _ = self.next_generation()
            self.record()
            if inc == 0:
                self.death = -1
        end_time = datetime.datetime.now()
        # cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        # cbar.ax.set_yticklabels(['RD', 'K', 'CNN'])  # vertically oriented colorbar
        print("Evolution Duration " + str(end_time - start_time))

    def evolve_syn(self, turns, profile=None):
        """
        同步演化，更新策略只能是Imitation
        """
        assert isinstance(self.rule, rule.Imitation)
        self.evolve(turns, profile)

    def time_to_stop(self, delta):
        """
        收敛判断：合作率变化小于1/512
        """
        if abs(delta) < (len(self.population) >> 9):
            return True
        return False

    # 合作率曲线
    def show(self, *args, **kwargs):
        # f = plt.figure(1)
        sampling = 20
        l = len(self.cooperate)
        if sampling <= 0:
            x = range(l)
            y = self.cooperate
        else:
            x = np.linspace(0, l, sampling+1, dtype=int)
            x[-1] = l-1
            y = self.cooperate[x]
        plt.plot(x, y, *args, **kwargs)
        # x_old = range(len(self.log))
        # from scipy.interpolate import spline
        # x = np.linspace(x_old[0], x_old[-1], 300)
        # y = spline(x_old,self.log,x)
        # plt.plot(x,y)
        # plt.title('Evolutionary Game')
        # plt.xlabel('Step')
        # plt.ylabel('Cooperation Ratio')
        # plt.ylim([0, len(self.population)])

        # if wait:
        #     f.show()
        # else:
        #     plt.show()

    # 保存演化结果
    def save(self):
        time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.dirname(os.path.realpath(__file__)) + time
        if not os.path.exists(path):
            os.makedirs(path)
        nx.write_pajek(self.population, path+'/graph.net')
        sio.savemat(path+'/data.mat', mdict={'gen': self.gen,
                                             'fitness': self.population.fitness,
                                             'strategy': self.population.strategy,
                                             'cooperate': self.cooperate})

    # 读取演化结果
    def load(self, path):
        self.population = nx.read_pajek(path+'/graph.net')
        mat = sio.loadmat(path+'/data.mat')
        self.gen = mat['gen']
        self.population.fitness = mat['fitness']
        self.population.strategy = mat['strategy']
        self.cooperate = mat['cooperate']


class StaticStrategyEvolution(Evolution):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.adapter = None
        self.prefer = None
        self.rewire = None

    def set_adapter(self, a):
        assert (isinstance(a, adapter.Adapter))
        self.adapter = a
        return self

    def bind_process(self):
        super(self.__class__, self).bind_process()
        self.population.init_dynamics(self.adapter)
        self.adapter.bind(self.population)

    def evolve(self, turns, **kwargs):
        self.rewire = None
        self.prefer = np.zeros((turns, self.adapter.category), dtype=np.int)
        super(self.__class__, self).evolve(turns, **kwargs)

    def next_generation(self, i):
        if i == 0:
            self.game.entire_play()
        else:
            self.game.rewire_play(self.rewire)
        birth, death = self.rule.update()
        # todo 策略如果进行突变会怎么样
        # if self.has_mut and np.random.random() <= 0.01:
        #     new_d = np.random.randint(self.adapter.category)
        #     mutate = True
        # else:
        new_d = self.population.dynamics[birth]
        mutate = False
        # todo 无论是否更新，都重连？
        # if birth == death:
        #     return 0, False
        # todo 无论有没有改变，都进行重连
        old_d = self.population.dynamics[death]
        if new_d == old_d:
            old_d = None
        else:
            self.population.dynamics[death] = new_d

        self.prefer[i] = self.population.prefer(old_d, new_d)
        old, new = self.adapter.adapt2(death, birth)
        if old < 0:
            self.rewire = None
        else:
            self.rewire = death, old, new

        self.death = None
        return 0, mutate

    def show(self, fmt, label, *args, **kwargs):
        sampling = 20
        l = len(self.cooperate)
        x = np.linspace(0, l, sampling+1, dtype=int)
        x[-1] = l-1
        y = self.prefer[x]
        colors = 'bgrcmykw'
        for i in range(self.adapter.category):
            plt.plot(x, y[:, i], colors[i]+fmt[1:], label=label+self.adapter.category_words[i])


class CoEvolution(Evolution):

    def __init__(self, has_mut=True):
        super(self.__class__, self).__init__(has_mut)
        self.adapter = None
        self.prefer = None
        self.rewire = None

    def set_population(self, p):
        assert isinstance(p, DynamicPopulation)
        self.population = p
        return self

    def set_adapter(self, a):
        assert(isinstance(a, adapter.Adapter))
        self.adapter = a
        # a.bind(self.population)

    def next_generation(self, i):
        inc, mutate = super(self.__class__, self).next_generation(i)
        new_p = np.random.randint(self.adapter.category) if mutate else self.population.dynamics[self.birth]
        old_p = self.population.dynamics[self.death]
        if new_p == old_p:
            old_p = None
        else:
            self.population.dynamics[self.death] = new_p
        self.prefer[i] = self.population.prefer(old_p, new_p)
        old, new = self.adapter.adapt_once(self.death)
        if old != new:
            self.rewire = (self.death, old, new)
        else:
            self.rewire = None
        return inc, mutate

    def evolve(self, turns, **kwargs):
        self.rewire = None
        self.prefer = np.zeros((turns, self.adapter.category), dtype=np.int)
        super(self.__class__, self).evolve(turns, **kwargs)

    def show(self, fmt, label, wait=False, *args, **kwargs):
        super(self.__class__, self).show(True)
        f = plt.figure(2)
        color = 'brgcmykw'
        # symb = '.ox+*sdph'
        label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
        for i in xrange(self.adapter.category):
            plt.plot(self.prefer[:, i], color[i], label=label[i])
        plt.title('CoEvolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()
        # plt.show()
        f.show()


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 10)
    pp = Population(G)
    gg = game.PDG()
    rr = rule.BirthDeath()
    e = Evolution()
    e.set_population(pp).set_game(gg).set_rule(rr)
    e.evolve(10000)
    e.show()
