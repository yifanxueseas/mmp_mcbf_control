import time
from math import cos, sin

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes

import cvxpy as cp
from scipy.integrate import solve_ivp

from gpdf_w_rh import p_dis_grad
from onMan_approximation import OnMan_Approx


def nominal_ctrl(current_state: np.ndarray, target_position: np.ndarray, nominal_speed: float, dt: float, k_p=1):
	"""Generate nominal control signal.

	Args:
		current_state: The current state, [x, y, theta].
		target_position: The target position, [x, y].
		dt: The time step.
		k_p: The proportional gain. Default to 1.

	Returns:
		nomial_action: The nominal action, [v, omega].
	"""
	v_xy = k_p*(target_position-current_state[:2])
	speed = float(np.linalg.norm(v_xy))
	if speed > nominal_speed:
		v_xy = nominal_speed/speed * v_xy
		speed = nominal_speed
	theta_current = (current_state[2]+np.pi) % (2*np.pi) - np.pi
	theta_goal = np.arctan2(v_xy[1], v_xy[0])
	delta_theta = theta_goal - theta_current
	if abs(delta_theta) > np.pi:
		delta_theta = -np.sign(delta_theta)*(2*np.pi-abs(delta_theta))
	nominal_action = [speed, delta_theta/dt]
	return nominal_action


def plot_env_standard(ax: Axes, color):
	map_shape = [100,100]
	_x = np.linspace(-3, 3, 100)
	_y = np.linspace(-3, 3, 100)

	X, Y = np.meshgrid(_x, _y)
	dis_mat = np.zeros(X.shape)
	all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
	dis_mat, normal = om.h_grad_vector(all_xy_coords, onM=6)
	dis_mat = dis_mat - 0.1
	dis_mat = dis_mat.reshape(map_shape[0], map_shape[1])-margin
	ax.contour(X, Y, dis_mat,[0], colors=color, linewidths=1.5)
	ax.contourf(X, Y, dis_mat,[0,0.1], colors=['orange','white'], extend='min', alpha=.3)


#larger alpha means larger safety boundary as well as more dramatic reaction, could lead to bounding back and stop sometimes
def f(x):
	return np.zeros((3, 1))

def g(x):
	g = np.zeros((3, 2))
	g[0][0] = cos(x[2])
	g[1][0] = sin(x[2])
	g[2][1] = 1
	return g

def alpha(x): 
	return 0.5*x


def safe_ctrl(x: np.ndarray, u_nom: np.ndarray):
	x = x.reshape(1,3)
	u_mod = cp.Variable(len(u_nom))
	dx = f(x.flatten()) + g(x.flatten()) @ u_mod

	p_w, grad_w = p_dis_grad(om.gpdf_cbf.gpdf_model, om.gpdf_cbf.pc_coords, x)
	p_seat1, grad_seat1 = p_dis_grad(om.gpdf_seat1.gpdf_model, om.gpdf_seat1.pc_coords, x)
	p_seat2, grad_seat2 = p_dis_grad(om.gpdf_seat2.gpdf_model, om.gpdf_seat2.pc_coords, x)
	p_seat3, grad_seat3 = p_dis_grad(om.gpdf_seat3.gpdf_model, om.gpdf_seat3.pc_coords, x)
	p_onM, grad_onM = p_dis_grad(om.gpdf_onM.gpdf_model, om.gpdf_onM.pc_coords, x)
	p_c, grad_c, dtp = om.p_dis_grad_c(x)
	min_idx = np.argmin(p_c)
	p_seat_min = min(p_seat1, p_seat2, p_seat3)[0,0]

	obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
	if MCBF and np.linalg.norm(x[0,:2]-target)>0.5 and (p_c[min_idx]>1 or dtp[min_idx]>=0):
		if p_onM<3 and x[0,0]<-12 and x[0,0]>-16 and abs(x[0,1])<4 and not (target == [-14,0]).all():
			e_vec = om.geodesic_approx_phi_3D(x, grad_onM, 20, target.reshape(1,2), 0.2,uni_dir=False,onM=1)

			v_b_e = e_vec.flatten()@dx
			if x[0,0]>-14.5 or abs(x[0,1])<1.5:
				ve = 0.3
				constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+ [v_b_e>=ve]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2] +[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]\
			+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
			 +  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]

			else:
				ve = 0.5
				constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+ [v_b_e>=ve]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=4]+[u_mod[1]>=-4] +[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]\
			+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0] \
			+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
		elif p_onM<3 and x[0,0]<-14 and x[0,0]>-17 and abs(x[0,1])<4 and (target == [-14,0]).all():
			e_vec = om.geodesic_approx_phi_3D(x, grad_onM, 20, target.reshape(1,2), 0.05,uni_dir=False,onM=1)
			v_b_e = e_vec.flatten()@dx
			constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+ [v_b_e>=0.1]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=4]+[u_mod[1]>=-4] +[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]\
			+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0] \
			+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
			
		elif p_seat_min<1 and ((x[0,0]>-10 and (target == [-14,0]).all())or (x[0,0]>-11 and not (target == [-14,0]).all())):
			if p_seat_min==p_seat1:
				phi = om.geodesic_approx_phi_3D(x, grad_seat1, 20, target.reshape(1,2), 0.03,uni_dir=False,onM=2)
				v_b_e = phi.flatten()@dx
				constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0] + [v_b_e>=0.15]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]\
				+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
				 +  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
			elif p_seat_min==p_seat2:
				# print("here32")
				phi = om.geodesic_approx_phi_3D(x, grad_seat2, 20,target.reshape(1,2), 0.03,uni_dir=False, onM=3)
				v_b_e = phi.flatten()@dx
				constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0] + [v_b_e>=0.15]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]\
				+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
				+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
			elif p_seat_min==p_seat3:
				phi = om.geodesic_approx_phi_3D(x, grad_seat3, 20,target.reshape(1,2), 0.03,uni_dir=False,onM=4)
				v_b_e = phi.flatten()@dx
				constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0] + [v_b_e>=0.15]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]\
				+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
				 +  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
			# print(phi)
		else:
			# print("here")
			constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]\
			+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
			+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]
	else:
		# print("here")
		constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]\
		+[grad_seat1.flatten() @ dx +alpha(p_seat1-margin)>=0] + [grad_seat2.flatten() @ dx +alpha(p_seat2-margin)>=0] + [grad_seat3.flatten() @ dx +alpha(p_seat3-margin)>=0]\
		+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]

	prob = cp.Problem(obj, constraints)
	try:
		prob.solve()
	except cp.SolverError:
		prob.solve(solver=cp.SCS)
	except:
		prob.solve(solver=cp.ECOS)

	if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
		return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
		if(cp.sum_squares(u_mod)<0.05 and cp.sum_squares(x[0:2])>0.5):
			print("no solution")
	else:
		print("infeasible")
		return np.zeros(2,), prob.status


def safe_ctrl_c(x: np.ndarray, u_nom: np.ndarray):
	x = x.reshape(1,3)
	u_mod = cp.Variable(len(u_nom))
	dx = f(x.flatten()) + g(x.flatten()) @ u_mod

	p_c, grad_c, dtp = om.p_dis_grad_c(x)
	min_idx = np.argmin(p_c)

	obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
	if MCBF:
		e_vec = om.geodesic_approx_phi_3D_c(x, grad_c[min_idx].reshape(1,3), 20, target.reshape(1,2), 0.05,uni_dir=False)
		v_b_e = e_vec.flatten()@dx
		constraints = [u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]+[v_b_e>=0.2]

	else:
		constraints = [u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]+  [grad_c @ dx +alpha(p_c.flatten()-margin)+dtp>=0]

	prob = cp.Problem(obj, constraints)
	try:
		prob.solve()
	except cp.SolverError:
		prob.solve(solver=cp.SCS)
	except:
		prob.solve(solver=cp.ECOS)

	if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
		return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
		if(cp.sum_squares(u_mod)<0.05 and cp.sum_squares(x[0:2])>0.5):
			print("no solution")
	else:
		print("infeasible")
		return np.zeros(2,), prob.status


def safe_ctrl_onM(x: np.ndarray, u_nom: np.ndarray):
	x = x.reshape(1,3)
	u_mod = cp.Variable(len(u_nom))
	dx = f(x.flatten()) + g(x.flatten()) @ u_mod

	p_w, grad_w = p_dis_grad(om.gpdf_cbf.gpdf_model, om.gpdf_cbf.pc_coords, x)
	p_onM, grad_onM = p_dis_grad(om.gpdf_onM.gpdf_model, om.gpdf_onM.pc_coords, x)
	p_onM = p_onM + 0.1

	obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
	if MCBF and np.linalg.norm(x[0,:2]-target)>0.5:
		if p_onM<1 and abs(x[0,1])<1.2 and x[0,0]<1:
			e_vec = om.geodesic_approx_phi_3D(x, grad_onM, 30, target.reshape(1,2), 0.1,uni_dir=False,onM=1,even=False)

			v_b_e = e_vec.flatten()@dx
			ve = 0.2
			constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+ [v_b_e>=ve]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=4]+[u_mod[1]>=-4] +[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]

		else:
			constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2]+[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]
	else:
		constraints = [grad_onM.flatten() @ dx +alpha(p_onM-margin)>=0]+[u_mod[0]>=-1]+[u_mod[0]<=1]+[u_mod[1]<=2]+[u_mod[1]>=-2] +[grad_w.flatten() @ dx +alpha(p_w-margin)>=0]

	prob = cp.Problem(obj, constraints)
	try:
		prob.solve()
	except cp.SolverError:
		prob.solve(solver=cp.SCS)
	except:
		prob.solve(solver=cp.ECOS)

	if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
		return (u_mod.value, (prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)))
	else:
		print("infeasible")
		return np.zeros(2,), prob.status


def update(t, x, v, omega):
	dx = v*cos(x[2])
	dy = v*sin(x[2])
	dtheta = omega
	return [dx,dy,dtheta]

def shift_theta(x):
	"""Shift the angle to [-pi, pi]."""
	if abs(x[2])>np.pi:
		while x[2]>np.pi:
			x[2] = x[2] -2*np.pi
		while x[2]<-np.pi:
			x[2] = x[2] + 2*np.pi
	return x


def terminal_condition(x, target):
	return np.linalg.norm(x[:2]-target) < 0.4

def one_step(current_state, target, nominal_speed: float, dt: float, control_method_index=0):
	"""One step of the simulation for a single robot.

	Args:
		current_state: The current state of the robot, [x, y, theta].
		control_method_index: Use which control method. Default to 0.

	Note:
		0: safe_ctrl
		1: safe_ctrl_c
		2: safe_ctrl_onM
	"""
	start_time = time.time()

	u_nom = nominal_ctrl(current_state, target, nominal_speed=nominal_speed, dt=dt)
	if control_method_index == 0:
		u_mod, prob_status = safe_ctrl(current_state, u_nom)
	elif control_method_index == 1:
		u_mod, prob_status = safe_ctrl_c(current_state, u_nom)
	elif control_method_index == 2:
		u_mod, prob_status = safe_ctrl_onM(current_state, u_nom)
	else:
		raise ValueError(f"Invalid control method index: {control_method_index}")

	execution_time = time.time()-start_time
	return u_mod, prob_status, execution_time


def animate(n_iter):
	"""Call repeatedly by FuncAnimation."""
	if n_iter % 20 == 0:
		print(n_iter)
	control_method_index=2

	fig.clear()
	ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)
	ax.set_xlim(-3,3)
	ax.set_ylim(-3,3)
	plot_env_standard(ax, 'k')
	ax.set_aspect("equal", adjustable="box")
	for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(15)

	c_bar = None
	for n_rob in range(rob_num):
		reach_target = terminal_condition(x_list[:, n_iter, n_rob], target)
		if reach_target and (iterations[n_rob] == it_max):
			iterations[n_rob] = n_iter
			isSuccess[n_rob] = 1

		elif (not reach_target) and (iterations[n_rob] == it_max):
			u_mod, prob_status, step_execution_time = one_step(x_list[:, n_iter, n_rob], target, speed, dt, control_method_index)
			u_list[:, n_iter, n_rob] = u_mod
			execution_time[n_rob, n_iter] = step_execution_time

			sol = solve_ivp(update, [0, dt], x_list[:, n_iter, n_rob], args=(u_list[:, n_iter, n_rob]))
			x_list[:, n_iter+1, n_rob] = shift_theta(sol.y[:, -1])

			xy = x_list[:2, n_iter, n_rob].reshape(1,2)
			_h, *_ = om.h_grad_standard(xy, onM=6)

			### For visualization
			if _h < -0.035:
				color = 'r'
				print("collision")
				print("h",_h)
				isSafe[n_rob] = 0
			else:
				color = "#135e08" # basically green
			if prob_status == "infeasible":
				color = 'b'
				infeasible_num[n_rob] += 1
			colormap = abs(u_list[0, :n_iter, n_rob])
			c_bar = ax.scatter(
				x_list[0, : n_iter, n_rob],
				x_list[1, : n_iter, n_rob],
				marker="o",
				s=15,
				c=colormap,
				cmap='jet',
				vmin=0,
				vmax=speed+0.5,
			)
			ax.plot(
				x_list[0, n_iter, n_rob],
				x_list[1, n_iter, n_rob],
				"o",
				color=color,
				markersize=30,
			)
			ax.plot(
				[x_list[0, n_iter, n_rob], x_list[0, n_iter, n_rob]+0.3*np.cos(x_list[2, n_iter, n_rob])],
				[x_list[1, n_iter, n_rob], x_list[1, n_iter, n_rob]+0.3*np.sin(x_list[2, n_iter, n_rob])],
				color='k',
			)
		else:
			pass
	ax.plot(target[0], target[1], "X", color='k', markersize=30, label='goal')
	ax.set_xlabel("$x_1$")
	ax.set_ylabel("$x_2$")
	if c_bar:
		cbar = fig.colorbar(c_bar)
		cbar.ax.tick_params(labelsize=23)

	om.update_xc(np.array([-5-n_iter*0.05, 0]), np.array([-1, 0]))


if __name__ == "__main__":
	margin = 0
	rob_num = 2
	obs_num = 1
	it_max = 300
	speed = 1.0
	x_list = np.zeros((3, it_max+1, rob_num))
	u_list = np.zeros((2, it_max, rob_num))
	execution_time =  np.zeros((rob_num, it_max))
	iterations = it_max*np.ones((rob_num,))
	isSuccess = np.zeros((rob_num,))
	isSafe = np.ones((rob_num,))
	infeasible_num = np.zeros((rob_num,))
	dt = 0.05
	target = np.array([2.0, 0.0])
	# target_map_shape = [10, 10, 10]
	# obstacle_map_shape = [30, 30, 30]
	# single = True
	# velocity_limit = None

	MCBF = True
	om = OnMan_Approx(target, False, True, radius=0.6, rho=10, hold_time=1, w=0.1)
	#----------------------------------------------------------------------------------------------------------------
	#animation
	for n in range(rob_num):
		x_list[:, 0, n] = [0.0, 0.0, n*2*np.pi/rob_num]

	writervideo = animation.FFMpegWriter(fps=60)
	fig, ax = plt.subplots(figsize=[12.7,10])

	anim = animation.FuncAnimation(fig, animate, frames=it_max, interval=dt*1000, repeat=False)
	fig.canvas.draw()
	anim.event_source.stop()
	anim.save("test.gif", writer=writervideo)
	print(execution_time)
	# df = pd.DataFrame()
	# for k in range(rob_num):
	# 	df['execution_time'+str(k)]=pd.Series(execution_time[k])
	# 	df['isSuccess'+str(k)]=pd.Series(isSuccess[k])
	# 	df['isSafe'+str(k)]=pd.Series(isSafe[k])
	# 	df['iterations'+str(k)]=pd.Series(iterations[k])
	# 	df['infeasible'+str(k)]=pd.Series(infeasible_num[k])
	# df.to_csv("../tro_results/simulation_MCBF_s1_short.csv")
