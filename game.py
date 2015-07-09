# -*- coding: utf-8 -*-


class Game:
    '''base class of game'''
    Name = "base_game"

    def interact(self, strategies):
        pass

class PGG(Game):
    Name = "public_good_game"

    def __init__(self, r=3, c=1):
        self.r = r
        self.c = c

    def interact(self, strategies):
        n = len(strategies)
        b = sum(strategies) * self.c * self.r / n
        return [b if x==0 else b-self.c for x in strategies]

class PDG(Game):
    Name = "prisoner's dilemma game"

    def __init__(self, r=1, t=1.5, s=0.5, p=0):
        self.Matrix = [(p, p), (t, s), (s, t), (r, r)]


    def interact(self, strategies):
        # todo: check strategies only have elements 0 or 1
        return self.Matrix[strategies[0]*2 + strategies[1]]

class RPG(Game):
    Name = "Rational Player Game"

    def __init__(self):
        pass

    def interact(self, strategies):
        pass



# TEST CODE HERE
if __name__ == '__main__':
    g = PGG()
    print(g.interact([1,0,1,0,1,1]))