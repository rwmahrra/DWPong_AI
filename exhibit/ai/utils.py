import time
import numpy as np
import os
import cv2
import imageio
import csv
import matplotlib.pyplot as plt

from config import Config

"""
Various utility helper methods to consolidate reusable code
"""

class Timer:
    """
    Simple helper class for performance testing
    """
    timers = {}

    @staticmethod
    def start(name):
        Timer.timers[name] = time.time()

    @staticmethod
    def stop(name):
        t = time.time() - Timer.timers[name]
        print(f'Finished {name} in {round(t, 5)}s')


def discount_rewards(r, gamma=0.99):
    """
    take 1D float array of rewards and compute discounted reward
    adapted from https://github.com/keon/policy-gradient/blob/master/pg.py
    """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def preprocess(state):
    if Config.instance().ENV_TYPE == Config.instance().CUSTOM or Config.instance().ENV_TYPE == Config.instance().HIT_PRACTICE:
        return preprocess_custom(state)
    elif Config.instance().ENV_TYPE == Config.instance().ATARI:
        return preprocess_gym(state)
    else:
        raise NotImplementedError


def preprocess_gym(I):
    """
    Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    adapted from https://github.com/keon/policy-gradient/blob/master/pg.py
    """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)


def preprocess_custom(I):
    state = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    h, w = state.shape
    state = cv2.resize(state, (w // 2, h // 2))
    state[state < 250] = 0
    state[state == 255] = 1
    return state.astype(np.float)


def encode_action(action):
    if action == "UP":
        return np.asarray([1, 0], dtype=np.float32)
    elif action == "DOWN":
        return np.asarray([0, 1], dtype=np.float32)
    elif action == "NONE":
        return np.asarray([0, 0], dtype=np.float32)
    else:
        raise NameError(f"Action {action} does not exist")


def encode_gradient(action, prob):
    if action == "UP":
        return np.asarray([1, 0], dtype=np.float32) - prob
    elif action == "DOWN":
        return np.asarray([0, 1], dtype=np.float32) - prob
    elif action == "NONE":
        return np.asarray([0, 0], dtype=np.float32) - prob
    else:
        raise NameError(f"Action {action} does not exist")


def normalize_states(states):
    states /= 255
    return states


def get_last_file():
    id = get_resume_index()
    if id:
        return os.path.join("models", f"{id}.h5")
    else:
        return None


def save_video(states, path, fps=30):
    pass#imageio.mimsave(path, states, fps=fps)


def write(value, file):
    text_file = open(file, "a+")
    text_file.write(value + '\n')
    text_file.close()


def get_resume_index():
    files = [f for f in os.listdir("models") if os.path.isfile(os.path.join("models", f))]
    ids = [int(os.path.split(f)[1].split('.')[0]) for f in files]
    max = -1
    for id in ids:
        if id > max:
            max = id
    if max == -1:
        return None
    return max


def plot_loss(path=None, show=False, include_left=True):
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    if include_left:
        with open('./analytics/agent_bottom.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in plots:
                x1.append(i)
                y1.append(float(row[0]))
                i += 1
        plt.plot(x1, y1, label='Bottom Agent')

    with open('./analytics/agent_top.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in plots:
            x2.append(i)
            y2.append(float(row[0]))
            i += 1
    plt.plot(x2, y2, label='Top Agent')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Episode For Policy Gradient Agent')
    plt.legend()
    if show:
        plt.show()
    if path:
        plt.savefig(path)
    plt.cla()


def plot_score(path=None, show=False):
    x = []
    yl = []
    yr = []

    with open('analytics/scores.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in plots:
            x.append(i)
            yl.append(float(row[0]))
            yr.append(float(row[1]))
            i += 1

    plt.plot(x, yl, label='Bottom Agent')
    plt.plot(x, yr, label='Top Agent')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Agent Score By Episode')
    plt.legend()
    if show:
        plt.show()
    if path:
        plt.savefig(path)
    plt.cla()


def plot_duration(path=None, show=False):
    x = []
    y = []

    with open('analytics/durations.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in plots:
            x.append(i)
            y.append(float(row[0]))
            i += 1

    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.title('Frames Count By Episode')
    plt.legend()
    if show:
        plt.show()
    if path:
        plt.savefig(path)
    plt.cla()
