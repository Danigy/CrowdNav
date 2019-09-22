class Obstacle(object):
    def __init__(self, config, section):
        """
        Base class for static obstacles

        """
        self.radius = None
        self.vertices = None
        self.px = None
        self.py = None
