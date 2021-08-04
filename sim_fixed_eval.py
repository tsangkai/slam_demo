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
	'lmk': [0.1, 0.1490, 0.3765],
}

fig_width = 5.84
fig_height = 4.00 #4.38

# error plot
num_realizations = int(sys.argv[1])
gt_data = pd.read_csv("result/sim_fixed/gt.csv")

pdr_error_array = np.zeros((len(gt_data['p_x']), num_realizations))
po_error_array = np.zeros_like(pdr_error_array)
pe_error_array = np.zeros_like(pdr_error_array)
pb_error_array = np.zeros_like(pdr_error_array)

qdr_error_array = np.zeros_like(pdr_error_array)
qo_error_array = np.zeros_like(pdr_error_array)
qe_error_array = np.zeros_like(pdr_error_array)
qb_error_array = np.zeros_like(pdr_error_array)

def add_boarder(lim):
	coeff = 0.02
	diff = lim[1] - lim[0]
	return (lim[0]-coeff*diff, lim[1]+coeff*diff)

def quat_diff(q1, q2):
	[w1, x1, y1, z1] = q1
	[w2, x2, y2, z2] = q2

	del_q = w1*w2 + x1*x2 + y1*y2 + z1*z2
	del_x = w1*x2 - x1*w2 + y1*z2 - z1*y2
	del_y = w1*y2 - y1*w2 - x1*z2 + z1*x2
	del_z = w1*z2 - w2*z1 + x1*y2 - y1*x2
	del_xyz = [del_x, del_y, del_z]
	theta = 2*np.arctan2(np.linalg.norm(del_xyz), np.abs(del_q))
	theta = (180*theta/np.pi)
	return theta

for k in range(num_realizations):
	dr_data = pd.read_csv("result/sim_fixed/dr_%s.csv" % (k,))
	opt_data = pd.read_csv("result/sim_fixed/opt_%s.csv" % (k,))
	em_data = pd.read_csv("result/sim_fixed/em_%s.csv" % (k,))
	boem_data = pd.read_csv("result/sim_fixed/boem_%s.csv" % (k,))
	for i in range(len(gt_data['p_x'])):
		pdr_error_array[i,k]= math.sqrt( (gt_data['p_x'][i]-dr_data['p_x'][i])**2 + (gt_data['p_y'][i]-dr_data['p_y'][i])**2
									  + (gt_data['p_z'][i]-dr_data['p_z'][i])**2)
		po_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-opt_data['p_y'][i])**2
											+ (gt_data['p_z'][i]-opt_data['p_z'][i])**2)
		pe_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-em_data['p_x'][i])**2 + (gt_data['p_y'][i]-em_data['p_y'][i])**2
										   + (gt_data['p_z'][i]-em_data['p_z'][i])**2)
		pb_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-boem_data['p_y'][i])**2
											 + (gt_data['p_z'][i]-boem_data['p_z'][i])**2)

		qdr_error_array[i,k] += quat_diff([dr_data['q_w'][i], dr_data['q_x'][i], dr_data['q_y'][i], dr_data['q_z'][i]],
									 [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qo_error_array[i, k] += quat_diff([opt_data['q_w'][i], opt_data['q_x'][i], opt_data['q_y'][i], opt_data['q_z'][i]],
								  [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qe_error_array[i, k] += quat_diff([em_data['q_w'][i], em_data['q_x'][i], em_data['q_y'][i], em_data['q_z'][i]],
										  [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qb_error_array[i, k] += quat_diff([boem_data['q_w'][i], boem_data['q_x'][i], boem_data['q_y'][i], boem_data['q_z'][i]],
										   [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])

dr_error_q = np.mean(qdr_error_array, axis=1)
opt_error_q = np.mean(qo_error_array, axis=1)
em_error_q = np.mean(qe_error_array, axis=1)
boem_error_q = np.mean(qb_error_array, axis=1)

dr_error_p = np.mean(pdr_error_array, axis=1)
opt_error_p = np.mean(po_error_array, axis=1)
em_error_p = np.mean(pe_error_array, axis=1)
boem_error_p = np.mean(pb_error_array, axis=1)


dr_error_qs = np.std(qdr_error_array, axis=1)/2
opt_error_qs = np.std(qo_error_array, axis=1)/2
em_error_qs = np.std(qe_error_array, axis=1)/2
boem_error_qs = np.std(qb_error_array, axis=1)/2

dr_error_ps = np.std(pdr_error_array, axis=1)/2
opt_error_ps = np.std(po_error_array, axis=1)/2
em_error_ps = np.std(pe_error_array, axis=1)/2
boem_error_ps = np.std(pb_error_array, axis=1)/2

def plot_with_std():
	fig, (ax1, ax2) = plt.subplots(2)
	fig.set_size_inches(fig_width, fig_height)
	line_width = 1.2
	ax1.fill_between(gt_data['timestamp'], dr_error_q - dr_error_qs,
					 dr_error_q + dr_error_qs, color = plot_color['dr'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], opt_error_q - opt_error_qs,
					 opt_error_q + opt_error_qs, color = plot_color['opt'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], em_error_q - em_error_qs,
					 em_error_q + em_error_qs, color = plot_color['em'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], boem_error_q - boem_error_qs,
					 boem_error_q + boem_error_qs, color = plot_color['boem'], alpha = 0.5, linewidth=0.0)

	ax1.plot(gt_data['timestamp'], dr_error_q, color = plot_color['dr'], linewidth=line_width, label='dead reckoning')
	ax1.plot(gt_data['timestamp'], opt_error_q, color = plot_color['opt'], linewidth=line_width, label='opt.')
	ax1.plot(gt_data['timestamp'], em_error_q, color = plot_color['em'], linewidth=line_width, label='EM')
	ax1.plot(gt_data['timestamp'], boem_error_q, color = plot_color['boem'], linewidth=line_width, label='BOEM')

	ax1.legend(loc = 1)

	ax1.set(ylabel='rotation error [deg]')
	ax1.set_ylim(add_boarder((0, 1.4)))


	ax2.fill_between(gt_data['timestamp'], dr_error_p - dr_error_ps,
					 dr_error_p + dr_error_ps, color = plot_color['dr'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], opt_error_p - opt_error_ps,
					 opt_error_p + opt_error_ps, color = plot_color['opt'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], em_error_p - em_error_ps,
					 em_error_p + em_error_ps, color = plot_color['em'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], boem_error_p - boem_error_ps,
					 boem_error_p + boem_error_ps, color = plot_color['boem'], alpha = 0.5, linewidth=0.0)

	ax2.plot(gt_data['timestamp'], dr_error_p, color = plot_color['dr'], linewidth=line_width, label='dead reckoning')
	ax2.plot(gt_data['timestamp'], opt_error_p, color = plot_color['opt'], linewidth=line_width, label='opt.')
	ax2.plot(gt_data['timestamp'], em_error_p, color = plot_color['em'], linewidth=line_width, label='EM')
	ax2.plot(gt_data['timestamp'], boem_error_p, color = plot_color['boem'], linewidth=line_width, label='BOEM')


	ax2.set(ylabel='position error [m]')
	ax2.set(xlabel='time [s]')
	# ax2.legend(loc = 1)

	ax2.set_ylim(add_boarder((0, 0.1)))

	fig.savefig('dynamic.pdf')

	return plt.show()

plot_with_std()






