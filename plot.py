import os, shutil
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse


def moving_average(data, window=10):
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

if not os.path.exists("imgs"):
    os.mkdir("imgs")

# plot pg & ppo
pg = np.load('rw/rw_pg.npy')
ppo = np.load('rw/rw_ppo.npy')

window = 20
pg  = moving_average(pg, window)
ppo = moving_average(ppo, window)

# Only pg
plt.figure()
plt.plot(pg, label="pg")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('pg reward')
plt.legend()
plt.savefig('imgs/pg_reward.png')
plt.close()

# ppo & pg
plt.figure()
plt.plot(pg, label="pg")
plt.plot(ppo, label="ppo")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('pg & ppo dif')
plt.legend()
plt.savefig('imgs/pg_dif.png')
plt.close()


# plot dqn & duel
dqn1 = np.load('rw/rw_dqn_1.npy')
dqn10 = np.load('rw/rw_dqn_10.npy')
dqn100 = np.load('rw/rw_dqn_100.npy')
dqn1000 = np.load('rw/rw_dqn_1000.npy')

# duel1 = np.load('rw/rw_duel_1.npy')
# duel10 = np.load('rw/rw_duel_10.npy')
duel100 = np.load('rw/rw_duel_100.npy')
# duel1000 = np.load('rw/rw_duel_1000.npy')

window = 200
dqn1 = moving_average(dqn1, window)
dqn10 = moving_average(dqn10, window)
dqn100 = moving_average(dqn100, window)
dqn1000 = moving_average(dqn1000, window)

# duel1 = moving_average(duel1, window)
# duel10 = moving_average(duel10, window)
duel100 = moving_average(duel100, window)
# duel1000 = moving_average(duel1000, window)

# P1
plt.figure()
plt.plot(dqn1000, label="dqn")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('dqn reward')
plt.legend()
plt.savefig('imgs/dqn_reward.png')
plt.close()

# P2
plt.figure()
plt.plot(dqn1, label="dqn_1")
plt.plot(dqn10, label="dqn_10")
plt.plot(dqn100, label="dqn100")
plt.plot(dqn1000, label="dqn_1000")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('dqn target frequent diff')
plt.legend()
plt.savefig('imgs/dqn_diff_p2.png')
plt.close()

# P3
plt.figure()
plt.plot(duel100, label="duel")
plt.plot(dqn100, label="dqn")
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('dqn & duel diff')
plt.legend()
plt.savefig('imgs/dqn_diff_p3.png')
plt.close()