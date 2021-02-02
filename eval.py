import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

plot_color = {
	'gt': [0, 0, 0],
	'opt': [0.8627, 0.2980, 0.2745],
	'em': [0.8471, 0.6824, 0.2784],
	'boem': [0.2633, 0.4475, 0.7086],  # navy
	'vo': [0.0000, 0.5490, 0.3765], # spruce
}


fig_width = 11.326
fig_height = 7.0

# keyframe has larger time range than ground truth

vo_data = pd.read_csv("trajectory_dr.csv")
est_opt_data = pd.read_csv("trajectory.csv")
est_em_data = pd.read_csv("trajectory_em.csv")
est_boem_data = pd.read_csv("trajectory_boem.csv")

gt_data = pd.read_csv("data/gt.csv")
# landmark_data = pd.read_csv("landmark.csv")


# offset initial time
init_time = est_opt_data['timestamp'][0]
vo_data['timestamp'] = vo_data['timestamp'] - init_time
est_opt_data['timestamp'] = est_opt_data['timestamp'] - init_time
est_em_data['timestamp'] = est_em_data['timestamp'] - init_time
est_boem_data['timestamp'] = est_boem_data['timestamp'] - init_time
gt_data['timestamp'] = gt_data['timestamp'] - init_time





fig = plt.figure(1)
fig.set_size_inches(fig_width, fig_height)

ax = fig.gca(projection='3d')


# ax.scatter(landmark_data['p_x'], landmark_data['p_y'], landmark_data['p_z'])

ax.plot(gt_data['p_x'], gt_data['p_y'], gt_data['p_z'], color = plot_color['gt'], label='gt')
ax.plot(vo_data['p_x'], vo_data['p_y'], vo_data['p_z'], color = plot_color['vo'], label='VO')
ax.plot(est_opt_data['p_x'], est_opt_data['p_y'], est_opt_data['p_z'], color = plot_color['opt'], label='est. opt.')
ax.plot(est_em_data['p_x'], est_em_data['p_y'], est_em_data['p_z'], color = plot_color['em'], label='est. EM')
ax.plot(est_boem_data['p_x'], est_boem_data['p_y'], est_boem_data['p_z'], color = plot_color['boem'], label='est. BOEM')

ax.view_init(39, 3)


# trajectory only
ax.set_xlim(-9.5,0.5)
ax.set_ylim(-0.5,9.5)
ax.set_zlim(-1.7,0.9)


ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.legend()

plt.savefig('result/result_2.pdf')  

plt.show()



# rotation plot
fig = plt.figure(2)
fig.set_size_inches(fig_width, fig_height)

ax_q_w = plt.subplot(411)
plt.plot(gt_data['timestamp'], gt_data['q_w'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['q_w'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['q_w'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['q_w'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['q_w'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('q_w')
plt.setp(ax_q_w.get_xticklabels(), visible=False)

plt.legend()

ax_q_x = plt.subplot(412)
plt.plot(gt_data['timestamp'], gt_data['q_x'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['q_x'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['q_x'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['q_x'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['q_x'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('q_x')
plt.setp(ax_q_x.get_xticklabels(), visible=False)


ax_q_y = plt.subplot(413)
plt.plot(gt_data['timestamp'], gt_data['q_y'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['q_y'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['q_y'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['q_y'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['q_y'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('q_y')
plt.setp(ax_q_y.get_xticklabels(), visible=False)


ax_q_z = plt.subplot(414)
plt.plot(gt_data['timestamp'], gt_data['q_z'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['q_z'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['q_z'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['q_z'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['q_z'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('q_z')
plt.xlabel('time [s]')

plt.savefig('result/rotation.pdf')  

plt.show()


# velocity plot
fig = plt.figure(3)
fig.set_size_inches(fig_width, fig_height)


ax_v_x = plt.subplot(311)
plt.plot(gt_data['timestamp'], gt_data['v_x'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['v_x'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['v_x'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['v_x'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['v_x'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('v_x')
plt.setp(ax_v_x.get_xticklabels(), visible=False)
plt.legend()


ax_v_y = plt.subplot(312)
plt.plot(gt_data['timestamp'], gt_data['v_y'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['v_y'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['v_y'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['v_y'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['v_y'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('v_y')
plt.setp(ax_v_y.get_xticklabels(), visible=False)

ax_v_z = plt.subplot(313)
plt.plot(gt_data['timestamp'], gt_data['v_z'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['v_z'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['v_z'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['v_z'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['v_z'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('v_z')

plt.xlabel('time [s]')

plt.savefig('result/velocity.pdf')  

plt.show()


# position plot
fig = plt.figure(4)
fig.set_size_inches(fig_width, fig_height)


ax_p_x = plt.subplot(311)
plt.plot(gt_data['timestamp'], gt_data['p_x'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['p_x'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['p_x'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['p_x'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['p_x'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('p_x')
plt.setp(ax_p_x.get_xticklabels(), visible=False)
plt.legend()


ax_p_y = plt.subplot(312)
plt.plot(gt_data['timestamp'], gt_data['p_y'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['p_y'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['p_y'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['p_y'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['p_y'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('p_y')
plt.setp(ax_p_y.get_xticklabels(), visible=False)

ax_p_z = plt.subplot(313)
plt.plot(gt_data['timestamp'], gt_data['p_z'], color = plot_color['gt'], label='gt')
plt.plot(vo_data['timestamp'], vo_data['p_z'], color = plot_color['vo'], label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_data['p_z'], color = plot_color['opt'], label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_data['p_z'], color = plot_color['em'], label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_data['p_z'], color = plot_color['boem'], label='est. BOEM')
plt.ylabel('p_z')

plt.xlabel('time [s]')

plt.savefig('result/position.pdf')  

plt.show()


# error plot
fig = plt.figure(5)
fig.set_size_inches(fig_width, fig_height)

vo_error = np.zeros_like(gt_data['p_x']);
est_opt_error = np.zeros_like(gt_data['p_x']);
est_em_error = np.zeros_like(gt_data['p_x']);
est_boem_error = np.zeros_like(gt_data['p_x']);

for i in range(len(gt_data['p_x'])):
	vo_error[i] = math.sqrt( (gt_data['p_x'][i]-vo_data['p_x'][i])**2 + (gt_data['p_y'][i]-vo_data['p_y'][i])**2 + (gt_data['p_z'][i]-vo_data['p_z'][i])**2)
	est_opt_error[i]  = math.sqrt( (gt_data['p_x'][i]-est_opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_opt_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_opt_data['p_z'][i])**2)
	est_em_error[i]   = math.sqrt( (gt_data['p_x'][i]-est_em_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_em_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_em_data['p_z'][i])**2)
	est_boem_error[i] = math.sqrt( (gt_data['p_x'][i]-est_boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-est_boem_data['p_y'][i])**2 + (gt_data['p_z'][i]-est_boem_data['p_z'][i])**2)

line_width = 2

plt.plot(vo_data['timestamp'], vo_error, color = plot_color['vo'], linewidth=line_width, label='VO')
plt.plot(est_opt_data['timestamp'], est_opt_error, color = plot_color['opt'], linewidth=line_width, label='est. opt.')
plt.plot(est_em_data['timestamp'], est_em_error, color = plot_color['em'], linewidth=line_width, label='est. EM')
plt.plot(est_boem_data['timestamp'], est_boem_error, color = plot_color['boem'], linewidth=line_width, label='est. BOEM')


plt.legend()

plt.xlabel('time [s]')
plt.ylabel('error [m]')
plt.ylim([0,2])

plt.savefig('result/error.pdf')  

plt.show()


print(np.mean(vo_error))
print(np.mean(est_opt_error))
print(np.mean(est_em_error))
print(np.mean(est_boem_error))