import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0, 0, 0],
	'dr': [0.0000, 0.5490, 0.3765],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],
	'lmk': [0.1, 0.1490, 0.3765]
}


fig_width = 5.84
fig_height = 4.38

# keyframe has larger time range than ground truth

gt_data = pd.read_csv("result/sim/easy/gt.csv")
dr_data = pd.read_csv("result/sim/easy/dr.csv")
opt_data = pd.read_csv("result/sim/easy/opt.csv")
em_data = pd.read_csv("result/sim/easy/em.csv")
boem_data = pd.read_csv("result/sim/easy/boem.csv")
lmk_data = pd.read_csv("result/sim/easy/lmk.csv")


N = 161
gt_data = gt_data[0:N]
dr_data = dr_data[0:N]
opt_data = opt_data[0:N]
em_data = em_data[0:N]
boem_data = boem_data[0:N]

print(gt_data['timestamp'][N-1])


fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)

#ax = fig.gca(projection='3d')
ax = Axes3D(fig, rect=(0.0, 0.05, 1, 0.9))


# ax = fig.add_axes((0.1, 0.1, 0.1, 0.1))


line_width = 1.2

ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], linewidth=line_width, label='ground truth')
ax.plot(dr_data['p_x'], dr_data['p_y'], dr_data['p_z'], '--', color = plot_color['dr'], linewidth=line_width, label='dead reckoning')
ax.plot(lmk_data['p_x'], lmk_data['p_y'], lmk_data['p_z'],'.', color = plot_color['lmk'], label='landmarks')
ax.plot(opt_data['p_x'], opt_data['p_y'], opt_data['p_z'], '--', color = plot_color['opt'], linewidth=line_width, label='opt.')
ax.plot(em_data['p_x'], em_data['p_y'], em_data['p_z'], '--', color = plot_color['em'], linewidth=line_width, label='EM')
ax.plot(boem_data['p_x'], boem_data['p_y'], boem_data['p_z'], '--', color = plot_color['boem'], linewidth=line_width, label='BOEM')

ax.view_init(41, 47)


# trajectory only
ax.set_xlim(-10.3, 10.3)
ax.set_ylim(-10.3, 10.3)
ax.set_zlim(-2.5, 2.5)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')


ax.legend(loc='upper right', ncol=2)

plt.savefig("result/sim/easy/trajectory.pdf")

plt.show()