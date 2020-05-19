import os, shutil
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="ADL HW3")
try:
    from argument import add_arguments
    parser = add_arguments(parser)
except:
    pass
args = parser.parse_args()


def moving_average(data, window=10):
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def plot(data, title):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(title)
    plt.close()


pg = np.load(args.rw_pg_path)
dqn = np.load('rw/rw_dqn.npy')
duel = np.load('rw/rw_duel_bak.npy')

pg = moving_average(pg, 20)
dqn = moving_average(dqn, 100)
duel = moving_average(duel, 100)

plot(pg, "pg_reward")
# plot(dqn, "dqn_reward")


plt.figure()

plt.plot(dqn, label="dqn")
plt.plot(duel, label="duel")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('dqn diff')
plt.legend()
plt.savefig('dqn_dif.png')
plt.close()