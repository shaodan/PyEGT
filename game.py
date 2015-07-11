# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx

class Game:
    '''base class of game'''
    name = "base_game"

    def interact(self, strategies):
        pass

class PGG(Game):
    name = "public_goods_game"

    def __init__(self, r=3, c=1):
        self.r = r
        self.c = c

    def interact(self, strategies):
        

class PDG(Game):
    name = "prisoner's dilemma game"

    def __init__(self, r=1, t=1.5, s=0.5, p=0):
        self.payoff_matrix = np.array([[(r, r), (s, t)], [(t,s), (p,p)]], dtype=np.double)

    def interact(self, strategies):
        # todo: check strategies only have elements 0 or 1
        return self.Matrix[strategies[0]*2 + strategies[1]]

class RPG(Game):
    name = "Rational Player Game"

    def __init__(self):
        pass

    def interact(self, strategies):
        pass



# TEST CODE HERE
if __name__ == '__main__':
    g = PGG()
    print(g.interact([1,0,1,0,1,1]))