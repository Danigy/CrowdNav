from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])

class ActionDiscrete():
    def __init__(self, action):
        stop = 0
        forward = 1
        backward = 2
        left = 3
        right = 4


