import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import visualization



fig_width = 5.84
fig_height = 4.00 #4.38

# error plot
num_realizations = int(sys.argv[1])
gt_data = pd.read_csv("result/sim/fixed/gt.csv")

pdr_error_array = np.zeros((len(gt_data['p_x']), num_realizations))
po_error_array = np.zeros_like(pdr_error_array)
pe_error_array = np.zeros_like(pdr_error_array)
pb_error_array = np.zeros_like(pdr_error_array)

qdr_error_array = np.zeros_like(pdr_error_array)
qo_error_array = np.zeros_like(pdr_error_array)
qe_error_array = np.zeros_like(pdr_error_array)
qb_error_array = np.zeros_like(pdr_error_array)


for k in range(num_realizations):
	dr_data = pd.read_csv("result/sim/fixed/dr_%s.csv" % (k,))
	opt_data = pd.read_csv("result/sim/fixed/opt_%s.csv" % (k,))
	em_data = pd.read_csv("result/sim/fixed/em_%s.csv" % (k,))
	boem_data = pd.read_csv("result/sim/fixed/boem_%s.csv" % (k,))
	for i in range(len(gt_data['p_x'])):
		pdr_error_array[i,k]= math.sqrt( (gt_data['p_x'][i]-dr_data['p_x'][i])**2 + (gt_data['p_y'][i]-dr_data['p_y'][i])**2
									  + (gt_data['p_z'][i]-dr_data['p_z'][i])**2)
		po_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-opt_data['p_x'][i])**2 + (gt_data['p_y'][i]-opt_data['p_y'][i])**2
											+ (gt_data['p_z'][i]-opt_data['p_z'][i])**2)
		pe_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-em_data['p_x'][i])**2 + (gt_data['p_y'][i]-em_data['p_y'][i])**2
										   + (gt_data['p_z'][i]-em_data['p_z'][i])**2)
		pb_error_array[i, k] = math.sqrt( (gt_data['p_x'][i]-boem_data['p_x'][i])**2 + (gt_data['p_y'][i]-boem_data['p_y'][i])**2
											 + (gt_data['p_z'][i]-boem_data['p_z'][i])**2)

		qdr_error_array[i,k] += visualization.quat_diff([dr_data['q_w'][i], dr_data['q_x'][i], dr_data['q_y'][i], dr_data['q_z'][i]],
									 [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qo_error_array[i, k] += visualization.quat_diff([opt_data['q_w'][i], opt_data['q_x'][i], opt_data['q_y'][i], opt_data['q_z'][i]],
								  [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qe_error_array[i, k] += visualization.quat_diff([em_data['q_w'][i], em_data['q_x'][i], em_data['q_y'][i], em_data['q_z'][i]],
										  [gt_data['q_w'][i], gt_data['q_x'][i], gt_data['q_y'][i], gt_data['q_z'][i]])
		qb_error_array[i, k] += visualization.quat_diff([boem_data['q_w'][i], boem_data['q_x'][i], boem_data['q_y'][i], boem_data['q_z'][i]],
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
					 dr_error_q + dr_error_qs, color = visualization.color['dr'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], opt_error_q - opt_error_qs,
					 opt_error_q + opt_error_qs, color = visualization.color['opt'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], em_error_q - em_error_qs,
					 em_error_q + em_error_qs, color = visualization.color['em'], alpha = 0.5, linewidth=0.0)
	ax1.fill_between(gt_data['timestamp'], boem_error_q - boem_error_qs,
					 boem_error_q + boem_error_qs, color = visualization.color['boem'], alpha = 0.5, linewidth=0.0)

	ax1.plot(gt_data['timestamp'], dr_error_q, color = visualization.color['dr'], linewidth=line_width, label='dead reckoning')
	ax1.plot(gt_data['timestamp'], opt_error_q, color = visualization.color['opt'], linewidth=line_width, label='opt.')
	ax1.plot(gt_data['timestamp'], em_error_q, color = visualization.color['em'], linewidth=line_width, label='EM')
	ax1.plot(gt_data['timestamp'], boem_error_q, color = visualization.color['boem'], linewidth=line_width, label='BOEM')

	ax1.legend(loc = 1)

	ax1.set(ylabel='rotation error [deg]')
	ax1.set_ylim(visualization.add_boarder((0, 1.4)))


	ax2.fill_between(gt_data['timestamp'], dr_error_p - dr_error_ps,
					 dr_error_p + dr_error_ps, color = visualization.color['dr'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], opt_error_p - opt_error_ps,
					 opt_error_p + opt_error_ps, color = visualization.color['opt'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], em_error_p - em_error_ps,
					 em_error_p + em_error_ps, color = visualization.color['em'], alpha = 0.5, linewidth=0.0)
	ax2.fill_between(gt_data['timestamp'], boem_error_p - boem_error_ps,
					 boem_error_p + boem_error_ps, color = visualization.color['boem'], alpha = 0.5, linewidth=0.0)

	ax2.plot(gt_data['timestamp'], dr_error_p, color = visualization.color['dr'], linewidth=line_width, label='dead reckoning')
	ax2.plot(gt_data['timestamp'], opt_error_p, color = visualization.color['opt'], linewidth=line_width, label='opt.')
	ax2.plot(gt_data['timestamp'], em_error_p, color = visualization.color['em'], linewidth=line_width, label='EM')
	ax2.plot(gt_data['timestamp'], boem_error_p, color = visualization.color['boem'], linewidth=line_width, label='BOEM')


	ax2.set(ylabel='position error [m]')
	ax2.set(xlabel='time [s]')
	# ax2.legend(loc = 1)

	ax2.set_ylim(visualization.add_boarder((0, 0.1)))

	fig.savefig('dynamic.pdf')

	return plt.show()

plot_with_std()






