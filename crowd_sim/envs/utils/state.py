class FullState(object):
    def __init__(self, px, py, vx, vy, radius, personal_space, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.personal_space = personal_space
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.personal_space, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.personal_space, self.gx, self.gy,
                                          self.v_pref, self.theta]])


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius, personal_space, theta=0.0):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.personal_space = personal_space
        self.theta = theta

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.personal_space, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.personal_space, self.theta]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
