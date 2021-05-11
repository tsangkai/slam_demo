import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





plot_color = {
	'gt': [0, 0, 0],
	'pre': [0.0000, 0.5490, 0.3765],
	'post': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],
	'lmk': [0.1, 0.1490, 0.3765]
}

fig_width = 5.84
fig_height = 4.38




### reprojection

pre_rot_error = pd.read_csv("result/test/reprojection/pre_rot_error.csv")
post_rot_error = pd.read_csv("result/test/reprojection/post_rot_error.csv")
pre_pos_error = pd.read_csv("result/test/reprojection/pre_pos_error.csv")
post_pos_error = pd.read_csv("result/test/reprojection/post_pos_error.csv")


fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

ax1.set_title('rotation error')

ax1.plot(pre_rot_error['noise'], pre_rot_error.mean(axis = 1), '-x', color = plot_color['pre'])
ax1.fill_between(pre_rot_error['noise'], pre_rot_error.mean(axis = 1)-0.5*pre_rot_error.std(axis = 1),
				                         pre_rot_error.mean(axis = 1)+0.5*pre_rot_error.std(axis = 1), color = plot_color['pre'], alpha = 0.5)
ax1.set(ylabel='pre. opt.')



ax2.plot(post_rot_error['noise'], post_rot_error.mean(axis = 1), '-x', color = plot_color['post'])
ax2.fill_between(post_rot_error['noise'], post_rot_error.mean(axis = 1)-0.5*post_rot_error.std(axis = 1),
				                          post_rot_error.mean(axis = 1)+0.5*post_rot_error.std(axis = 1), color = plot_color['post'], alpha = 0.5)
ax2.set(ylabel='post. opt.')

ax2.set(xlabel='observation noise std')

plt.show()



### s1 fixed

pre_rot_error = pd.read_csv("result/test/s1_fixed/pre_rot_error.csv")
post_rot_error = pd.read_csv("result/test/s1_fixed/post_rot_error.csv")
pre_pos_error = pd.read_csv("result/test/s1_fixed/pre_pos_error.csv")
post_pos_error = pd.read_csv("result/test/s1_fixed/post_pos_error.csv")


# rotation error
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

ax1.set_title('rotation error')

ax1.plot(pre_rot_error['noise'], pre_rot_error.mean(axis = 1), '-x', color = plot_color['pre'])
ax1.fill_between(pre_rot_error['noise'], pre_rot_error.mean(axis = 1)-0.5*pre_rot_error.std(axis = 1),
				                         pre_rot_error.mean(axis = 1)+0.5*pre_rot_error.std(axis = 1), color = plot_color['pre'], alpha = 0.5)
ax1.set(ylabel='pre. opt.')



ax2.plot(post_rot_error['noise'], post_rot_error.mean(axis = 1), '-x', color = plot_color['post'])
ax2.fill_between(post_rot_error['noise'], post_rot_error.mean(axis = 1)-0.5*post_rot_error.std(axis = 1),
				                          post_rot_error.mean(axis = 1)+0.5*post_rot_error.std(axis = 1), color = plot_color['post'], alpha = 0.5)
ax2.set(ylabel='post. opt.')

ax2.set(xlabel='gyro noise density [rad/s/sqrt(Hz)]')

plt.show()


# position error
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

ax1.set_title('position error')

ax1.plot(pre_pos_error['noise'], pre_pos_error.mean(axis = 1), '-x', color = plot_color['pre'])
ax1.fill_between(pre_pos_error['noise'], pre_pos_error.mean(axis = 1)-0.5*pre_pos_error.std(axis = 1),
				                         pre_pos_error.mean(axis = 1)+0.5*pre_pos_error.std(axis = 1), color = plot_color['pre'], alpha = 0.5)
ax1.set(ylabel='pre. opt. [m]')



ax2.plot(post_pos_error['noise'], post_pos_error.mean(axis = 1), '-x', color = plot_color['post'])
ax2.fill_between(post_pos_error['noise'], post_pos_error.mean(axis = 1)-0.5*post_pos_error.std(axis = 1),
				                          post_pos_error.mean(axis = 1)+0.5*post_pos_error.std(axis = 1), color = plot_color['post'], alpha = 0.5)
ax2.set(ylabel='post. opt. [m]')

ax2.set(xlabel='gyro noise density [rad/s/sqrt(Hz)]')

plt.show()



### s0 fixed

pre_rot_error = pd.read_csv("result/test/s0_fixed/pre_rot_error.csv")
post_rot_error = pd.read_csv("result/test/s0_fixed/post_rot_error.csv")
pre_pos_error = pd.read_csv("result/test/s0_fixed/pre_pos_error.csv")
post_pos_error = pd.read_csv("result/test/s0_fixed/post_pos_error.csv")


# rotation error
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

ax1.set_title('rotation error')

ax1.plot(pre_rot_error['noise'], pre_rot_error.mean(axis = 1), '-x', color = plot_color['pre'])
ax1.fill_between(pre_rot_error['noise'], pre_rot_error.mean(axis = 1)-0.5*pre_rot_error.std(axis = 1),
				                         pre_rot_error.mean(axis = 1)+0.5*pre_rot_error.std(axis = 1), color = plot_color['pre'], alpha = 0.5)
ax1.set(ylabel='pre. opt.')



ax2.plot(post_rot_error['noise'], post_rot_error.mean(axis = 1), '-x', color = plot_color['post'])
ax2.fill_between(post_rot_error['noise'], post_rot_error.mean(axis = 1)-0.5*post_rot_error.std(axis = 1),
				                          post_rot_error.mean(axis = 1)+0.5*post_rot_error.std(axis = 1), color = plot_color['post'], alpha = 0.5)
ax2.set(ylabel='post. opt.')

ax2.set(xlabel='gyro noise density [rad/s/sqrt(Hz)]')

plt.show()


# position error
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(fig_width, fig_height)

ax1.set_title('position error')

ax1.plot(pre_pos_error['noise'], pre_pos_error.mean(axis = 1), '-x', color = plot_color['pre'])
ax1.fill_between(pre_pos_error['noise'], pre_pos_error.mean(axis = 1)-0.5*pre_pos_error.std(axis = 1),
				                         pre_pos_error.mean(axis = 1)+0.5*pre_pos_error.std(axis = 1), color = plot_color['pre'], alpha = 0.5)
ax1.set(ylabel='pre. opt. [m]')



ax2.plot(post_pos_error['noise'], post_pos_error.mean(axis = 1), '-x', color = plot_color['post'])
ax2.fill_between(post_pos_error['noise'], post_pos_error.mean(axis = 1)-0.5*post_pos_error.std(axis = 1),
				                          post_pos_error.mean(axis = 1)+0.5*post_pos_error.std(axis = 1), color = plot_color['post'], alpha = 0.5)
ax2.set(ylabel='post. opt. [m]')

ax2.set(xlabel='gyro noise density [rad/s/sqrt(Hz)]')

plt.show()