import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
import numpy as np



@dataclass
class State:
    def __init__(self, torque, ratio, radius_wheels, mass):
        self.torque = torque
        self.ratio = ratio
        self.radius_wheels = radius_wheels
        self.mass = mass

    def accelrate():
        


def step(state:State) -> State:
    accleration = (
        state.torque * state.ratio) / state.rad_wheels
    force = state.mass * accleration

def animate (i):
    pass


fig = plt.figure(figsize=(3,3), dpi=150)
ax = fig.add_subplot(111)
ax.grid()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# these lines are so the animation doesnt zoom in or out
plt.pause(3)
ani = animation.FuncAnimation(fig, animate, interval=0)
plt.show()