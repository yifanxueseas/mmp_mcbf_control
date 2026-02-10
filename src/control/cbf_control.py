import os
import pathlib
import time
from math import cos, sin
from typing import Optional, TypedDict, List, Union
from copy import deepcopy

import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp # type: ignore
import cv2 # type: ignore
opencv_version = cv2.__version__.split('.')[0]

from basic_boundary_function.gpdf_w_rh import p_dis_grad, ITER, ITER1, ITER2, ITER_C, p_dis_grad_t # type: ignore
from basic_boundary_function.onMan_approximation import OnMan_Approx
from basic_boundary_function.env import Env

from configs import CircularRobotSpecification


yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', 'spec_robot.yaml')
robot_config = CircularRobotSpecification.from_yaml(yaml_path)


class ControllerStatus(TypedDict):
	isSuccess: bool
	isSafe: bool
	isInfeasible: bool

class DebugInfo(TypedDict):
	active_gpdf_idx: int
	e_vec: Optional[List[List[float]]]
	current_margin: float
	grad_value: List[float]
	om_idx: int
	beta: Optional[float]


class OnManCBFController:
	def __init__(self,threeD_controller:bool=False, autotune:bool=True, dynamics:str="differential") -> None:
		self.dynamics = dynamics
		self.threeD_controller = threeD_controller
		self._target : Optional[np.ndarray] = None
		self._nominal_speed : Optional[float] = None
		self._dt : Optional[float] = None
		self.autotune = autotune

		if self.dynamics == "differential":
			self._state = np.array([np.inf, 0, 0])
		else:
			self._state = np.array([np.inf, 0])

		self._u_prev = np.zeros(2)

		self.init_done = False
		self.controller_status = ControllerStatus(
			isSuccess=False,
			isSafe=True,
			isInfeasible=False,
		)
		self.e_prev = None # XXX Add new cost term for seletcing the d_Theta direction
		self.margin_levels:List[float] = []

		self.debug_info = DebugInfo(
			active_gpdf_idx=-1,
			e_vec=[],
			current_margin=0.0,
			grad_value=[],
			om_idx = -1,
			beta = None
		)

		self.hom_history = [np.inf for _ in range(5)]
		self.target_range:Optional[float] = None
		self.on_boundary = False

	@property
	def target(self) -> Optional[np.ndarray]:
		return self._target
	
	@property
	def nominal_speed(self) -> Optional[float]:
		return self._nominal_speed
	
	@property
	def dt(self) -> Optional[float]:
		return self._dt
	
	@property
	def margin(self) -> float:
		return self._margin

	@property
	def state(self) -> np.ndarray:
		return self._state

	@property
	def u_prev(self) -> np.ndarray:
		return self._u_prev

	@staticmethod
	def process_shifted(x, a = 0):
		"""Post processing for robot states in differential drive."""
		x_shited = np.zeros(3,)
		x_shited[0] = x[0] + a*np.cos(x[2])
		x_shited[1] = x[1] + a*np.sin(x[2])
		x_shited[2] = x[2]
		return x_shited
	
	def func_f(self,x):
		if self.dynamics == "differential":
			return np.zeros((3, 1))
		elif self.dynamics == "omni-directional":
			return np.zeros((2, 1))
		else:
			raise RuntimeError('Undefined robot dynamics.')

	
	def func_g(self, x, a=0):
		if self.dynamics == "differential":
			g = np.zeros((3,2))
			g[0][0] = cos(x[2])
			g[0][1] = -a*sin(x[2])
			g[1][0] = sin(x[2])
			g[1][1] = a*cos(x[2])
			g[2][1] = 1
			return g
		
		elif self.dynamics == "omni-directional":
			return np.eye(2)
		else:
			raise RuntimeError('Undefined robot dynamics.')

	
	def func_alpha(self, x, env="vicon"): # larger alpha means larger safety boundary and more dramatic reaction, could lead to bounding back and stop sometimes
		if env=="vicon" or env=="vicon_fork":
			if self.threeD_controller:
				if self.autotune:
					return 0.75*x
				else:
					return 0.5*x
			else:
				return 1.5*x
		elif env=="hospital":
			if self.threeD_controller:
				return x
			else:
				return 0.7*x
		elif env=="mmp":
			return x
		else:
			return x


	def set_params(
			self, 
			target:np.ndarray,
			sampling_time:float,
			nominal_speed:float=1.0,
			base_margin:float=0.0,
			target_range:float=1.0,
			ve: float=0.4,
			MCBF: bool=True,
			MMP: bool = False,
			om_range:float=3.0,
			beta_coef: Optional[float]=1.0,
			om_expand_threshold:Optional[float]=0.0,
			dir_num: Optional[int]=2,
			on_boundary: bool=True,
		):
		if target is not None:
			self._target = target
		if nominal_speed is not None:
			self._nominal_speed = nominal_speed
		if sampling_time is not None:
			self._dt = sampling_time
		self._margin = base_margin
		self.debug_info['current_margin'] = base_margin
		if self.target_range is None:
			self.target_range = target_range
		if not self.margin_levels:
			self.margin_levels = [base_margin]
		self.ve = ve
		self.beta_coef = beta_coef
		self.om_expand_threshold = om_expand_threshold
		self.om_range = om_range
		self.MCBF = MCBF
		self.dir_num = dir_num
		self.MMP = MMP
		self.on_boundary = on_boundary

	def set_dynamic_margin(self, margin_levels: List[float]):
		"""Set dynamic margin levels."""
		self.margin_levels = sorted(margin_levels, reverse=True) # from large/positive to small/negative

	def set_state(self, current_state: np.ndarray):
		# """Used to calibrate the state of the robot."""
		self._state = np.array(current_state)

	def set_init_data(self, init_state: np.ndarray, max_iter: int, num_action=2):
		num_state = len(self._state)
		init_state = init_state[:num_state]
		self.execution_times = np.zeros((max_iter))
		self.init_done = True
		self._state = init_state


	def get_nominal_ctrl(self, k_p=1.0):
		"""Generate nominal control signal.

		Args:
			k_p: The proportional gain. Default to 1.

		Returns:
			nomial_action: The nominal action, [v, omega].
		"""
		assert self.target is not None, "Target position is not set."
		assert self.nominal_speed is not None, "Nominal speed is not set."
		assert self.dt is not None, "Sampling time is not set."
		assert self.dynamics is not None, "Robot dynamics is not set."
			
		v_xy = k_p*(self.target-self.state[:2])
		speed = float(np.linalg.norm(v_xy))
		if speed > self.nominal_speed:
			v_xy = self.nominal_speed/speed * v_xy
			speed = self.nominal_speed
		if self.dynamics == "differential":
			theta_current = (self.state[2]+np.pi) % (2*np.pi) - np.pi
			theta_goal = np.arctan2(v_xy[1], v_xy[0])
			delta_theta = theta_goal - theta_current
			if abs(delta_theta) > np.pi:
				delta_theta = -np.sign(delta_theta)*(2*np.pi-abs(delta_theta))
			nominal_action = [speed, delta_theta/self.dt]
		elif self.dynamics == "omni-directional":
			nominal_action = v_xy
		else:
			raise RuntimeError('Undefined robot dynamics.')

		return nominal_action
	
	def get_backup_nominal_ctrl(self, v_xy):
		"""Generate nominal control signal.

		Args:
			v_xy: Desired linear velocity vector [vx, vy].

		Returns:
			nomial_action: The nominal action, [v, omega].
		"""
		assert self.target is not None, "Target position is not set."
		assert self.nominal_speed is not None, "Nominal speed is not set."
		assert self.dt is not None, "Sampling time is not set."

		speed = float(np.linalg.norm(v_xy))
		if speed > self.nominal_speed:
			v_xy = self.nominal_speed/speed * v_xy
			speed = self.nominal_speed
		theta_current = (self.state[2]+np.pi) % (2*np.pi) - np.pi
		theta_goal = np.arctan2(v_xy[1], v_xy[0])
		delta_theta_pos = theta_goal - theta_current
		if abs(delta_theta_pos) > np.pi:
			delta_theta_pos = -np.sign(delta_theta_pos)*(2*np.pi-abs(delta_theta_pos))

		if abs(delta_theta_pos)<=np.pi/2:
			nominal_action = [speed, delta_theta_pos/self.dt]
			return nominal_action

		theta_goal = np.arctan2(-v_xy[1], -v_xy[0])
		delta_theta_neg = theta_goal - theta_current
		if abs(delta_theta_neg) > np.pi:
			delta_theta_pos = -np.sign(delta_theta_neg)*(2*np.pi-abs(delta_theta_neg))
		nominal_action = [-speed, delta_theta_neg/self.dt]
		return nominal_action

	def check_termination_condition(self, terminal_distance:float=0.4):
		assert self.target is not None, "Target position is not set."
		return np.linalg.norm(self.state[:2]-self.target[:2]) < terminal_distance
	
	@staticmethod
	def on_line_segment(start:np.ndarray, end:np.ndarray, x:np.ndarray):
		# start: 1 x dim (target location in our case)
		# end: 1 x dim (robot location in our case)
		# x: n x dim (obstacle center in our case)
		s2e = (end - start).squeeze()
		x2s = (x - start).squeeze()
		innerProduct = x2s@s2e
		return np.where(innerProduct<=s2e@s2e)[0]


	def get_polygon_from_gpdf(self, env: Env, h_om:float=0.0, extra_margin:float=0.0, resolution:int=200,
						   	  obstacle_idx:list[int]=[-1], circle=False):
		"""Get the boundary points of a GPDF as a polygon.

		Args:
			env: For calling the gradient calculation. # XXX This is not a good design.
			h_om: The threshold for distance level to be regarded as occupied areas. Defaults to 0.0.
			extra_margin: Extra margin for searching area range. Defaults to 0.0.
			resolution: Number of points at each direction (x and y). Defaults to 100.
			obstacle_idx: The index of the obstacle, -1 for all obstacles. Defaults to -1.
			dynamic_obstacle: Whether the obstacle is dynamic. Defaults to False.

		Returns:
			boundary_points: The boundary points of the GPDF.
		"""
		
		xmin = 10e8
		xmax = -10e8
		ymin = 10e8
		ymax = -10e8

		if circle:
			xmin = min(env.xc[:,0]) - env.radius - extra_margin
			xmax = max(env.xc[:,0]) + env.radius + extra_margin
			ymin = min(env.xc[:,1]) - env.radius - extra_margin
			ymax = max(env.xc[:,1]) + env.radius + extra_margin
		else:
			for i, id in enumerate(obstacle_idx):
				points_for_gpdf = env.gpdf_set[id].pc_coords.reshape(-1, 2)
				xmin = min(points_for_gpdf[:, 0].min() - extra_margin, xmin)
				xmax = max(points_for_gpdf[:, 0].max() + extra_margin, xmax)
				ymin = min(points_for_gpdf[:, 1].min() - extra_margin, ymin)
				ymax = max(points_for_gpdf[:, 1].max() + extra_margin, ymax)

		if xmin == 10e8: # no obstacle between robot and target
			breakpoint()
			return None

		_x = np.linspace(xmin, xmax, resolution)
		_y = np.linspace(ymin, ymax, resolution)
		X, Y = np.meshgrid(_x, _y)
		dis_mat = np.zeros(X.shape)
		all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
		if circle:
			dis_mat, _ = env.h_gradc(all_xy_coords)
		else:
			dis_mat = env.h_grad_vector(all_xy_coords, obstacle_idx=obstacle_idx)
		dis_mat = dis_mat.reshape(X.shape)
		dis_mat[dis_mat < h_om] = -np.inf
		dis_mat[dis_mat >= h_om] = 1
		dis_mat[dis_mat < 1] = 0
		dis_mat[[0, -1], :] = 1
		dis_mat[:, [0, -1]] = 1
		
		edges_img = cv2.Canny(np.uint8(dis_mat), threshold1=0.5, threshold2=0.5)

		if int(opencv_version) >= 4:
			contour_img = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
		else:
			contour_img = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

		boundary_point_set = []
		for i in range(len(contour_img)):
			contour_x = _x[contour_img[i][:, 0, 0]]
			contour_y = _y[contour_img[i][:, 0, 1]]
			new_contour = np.column_stack((contour_x, contour_y))
			if len(new_contour)>10:
				boundary_point_set.append(new_contour)
		
		if len(boundary_point_set)==0:
			contour_img = max(contour_img, key=cv2.contourArea).reshape(-1, 2)
			contour_x = _x[contour_img[i][:, 0, 0]]
			contour_y = _y[contour_img[i][:, 0, 1]]
			new_contour = np.column_stack((contour_x, contour_y))
			boundary_point_set = [new_contour]

		return boundary_point_set


	@staticmethod
	def get_projection_info(coords: List[tuple], points: List[tuple]):
		"""Return the projection information of points onto the line segments defined by coords.

		Args:
			coords: A list of coordinates (n*2) that define the line segments.
			points: A list of points (m*2) to project onto the line segments.

		Returns:
			min_indices: The indices of the line segments that the points are projected onto.
			projection_points: The projection points of the points onto the line segments.
			distances_from_start: The distances from the start of the line segments to the projection points.
			total_length: The total length of the line segments.

		Notes:
			This is based on the formula:
			```
				t = dot(AP, AB) / dot(AB, AB)
			```

			Alternative Shapely implementation:
			```
				def minimal_distance_along_polygon(start: tuple, goal: tuple, poly_coords: List[tuple]):
					line = LineString(poly_coords)
					distance = abs(line.project(Point(*goal)) - line.project(Point(*start)))
					return min(distance, line.length - distance)
			```
		"""
		coords_np = np.asarray(coords)
		points_np = np.asarray(points)

		A = coords_np
		B = np.roll(coords_np, -1, axis=0)
		AB = B - A # (n-1, 2)
		AP = points_np[:, np.newaxis] - A # (m, n-1, 2)

		segment_lengths = np.linalg.norm(AB, axis=1)
		cum_segment_lengths_with_zero = np.concatenate(([0], np.cumsum(segment_lengths)))

		AB_dot_AP = np.einsum('ij,kij->ki', AB, AP) # Dot product of AB and AP, shape=(m, n-1)
		AB_dot_AB = np.einsum('ij,ij->i', AB, AB)   # Dot product of AB with itself
		t_clipped = np.clip(AB_dot_AP / AB_dot_AB.reshape(1, -1), 0, 1) # (m, n-1)

		projections = A + t_clipped[:, :, np.newaxis] * AB # (m, n-1, 2)
		distances = np.linalg.norm(projections - points_np[:, np.newaxis, :], axis=2) # (m, n-1)
		if distances.size == 0:
			breakpoint()
		min_indices:np.ndarray = np.argmin(distances, axis=1)
		rows = np.arange(len(min_indices))
		projection_points:np.ndarray = projections[rows, min_indices] # (m, 2)
		distances_from_A_to_projection = np.linalg.norm(projection_points - A[min_indices], axis=1)
		distances_from_start:np.ndarray = cum_segment_lengths_with_zero[min_indices] + distances_from_A_to_projection
		if np.isnan(distances_from_start).any():
			breakpoint()
		return min_indices, projection_points, distances_from_start, np.sum(segment_lengths)
	

	def get_opt_problem_2D(
			self, 
			u_nom: np.ndarray, 
			om: OnMan_Approx, 
			external_margin:Optional[float]=None, 
			check_mode:bool=False):
		assert self.target is not None, "Target position is not set."
		margin = self.margin if external_margin is None else external_margin
		if self.dynamics == "differential":
			a = 0.2
			x = self.process_shifted(self._state, a).reshape(1,3)
		else:
			a = 0
			x = self._state.reshape(1,-1)
		h_set, grad_set, gradt_set, xc, _ = om.env.h_grad_set(x[:,:2])
		dis2xc = np.linalg.norm(self.state[:2].reshape(1,2)-xc, axis=1)
		h_set_physical = h_set
		h_set_physical[:om.env.num_dyn_mmp,0] = dis2xc[:om.env.num_dyn_mmp]
		active_obs_idx = np.where(h_set_physical.flatten()<=robot_config.lidar_range/2)[0]
		gradt_set[om.env.mmp_active_idx] = np.clip(gradt_set[om.env.mmp_active_idx], None, -0.1)
		
		dynamic_obstacle = False
		merge_idx = None
		# breakpoint()
		if self.MMP:
			env_name = "mmp"
			if om.env.num_dyn_circle>0 and om.env.num_dyn_mmp==0:
				h_c = h_set[:len(h_set)-len(om.env.gpdf_set)]
				c_idx = np.argmin(h_c)
				if ((h_c[c_idx]-margin)<1.0 and gradt_set[c_idx]<-0.01):
					dynamic_obstacle = True
			elif om.env.num_dyn_mmp>0:
				if om.env.env_name=="social_nav_pred":
					active_gpdf_idx = np.where(h_set_physical[-len(om.env.gpdf_set):].flatten()<=robot_config.lidar_range/2)[0]
				else:
					active_gpdf_idx = np.where(h_set_physical[-len(om.env.gpdf_set):-1].flatten()<=robot_config.lidar_range/2)[0]
				if (active_gpdf_idx < om.env.num_dyn_mmp).any():
					merge_idx = active_gpdf_idx
		else:
			assert om.env.env_name is not None
			env_name = om.env.env_name

		if len(om.env.gpdf_set)>=3:
			min_h_idx_list = np.argpartition(h_set[-len(om.env.gpdf_set):-1].flatten(), 2)[:2]
			if om.env.num_dyn_mmp==0:
				min_h_idx_list =min_h_idx_list+om.env.num_dyn_circle
		else:
			min_h_idx_list =  np.array([0])

		u_mod = cp.Variable(len(u_nom))
		rho = cp.Variable()
		dx = self.func_f(x.flatten()) + self.func_g(x.flatten(),a) @ u_mod
		pi_list = None
		xi_list = None
		self.boundary_points = None
		beta = None

		if active_obs_idx[0].size == 0:
			constraints = []
		else:
			constraints = [grad_set[active_obs_idx] @ dx[:2] + self.func_alpha(h_set.flatten()[active_obs_idx]-margin, env=env_name) + gradt_set.flatten()[active_obs_idx]>=0]

		if self.dynamics == "differential":
			obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
			constraints	+= [u_mod[1]<=robot_config.ang_vel_onm_max]+[u_mod[1]>=-robot_config.ang_vel_onm_max] +[u_mod[0]>=robot_config.lin_vel_min]+[u_mod[0]<=robot_config.lin_vel_max]
		else:
			obj = cp.Minimize((u_mod[0] - u_nom[0])**2+(u_mod[1] - u_nom[1])**2)
			constraints += [u_mod[1]<=robot_config.lin_vel_max]+[u_mod[1]>=robot_config.lin_vel_min] +[u_mod[0]>=robot_config.lin_vel_min]+[u_mod[0]<=robot_config.lin_vel_max]
			
		e_vec_raw = None
		self.debug_info['grad_value'] = grad_set[min_h_idx_list[0]].tolist()
		self.debug_info['e_vec'] = None
		self.debug_info['om_idx'] = min_h_idx_list[0]
		self.boundary_point_set = None
		self.boundary_points = None
		self.rob_proj_points = None
		self.target_proj_points = None
		self.boundary_idx = None
		self.min_distance_set = None

		if self.MCBF and not dynamic_obstacle:
			e_idx = -1
			while len(min_h_idx_list)-e_idx>1 and e_idx<=0 and h_set[min_h_idx_list[e_idx+1]]<self.om_range:
				e_idx = e_idx+1
				if om.env.num_dyn_mmp==0:
					idx = min_h_idx_list[e_idx]-om.env.num_dyn_circle
				else:
					idx = min_h_idx_list[e_idx]
				if merge_idx is not None and idx in merge_idx:
					idx = merge_idx
				else:
					idx = [idx]
				h_om, grad_om = om.env.h_grad_uni(self._state[:2].reshape(1,2), idx=idx)
				if h_om > -0.2:
					if self.on_boundary:
						_h_om = 0.00
					else:
						_h_om = min(h_om, self.om_range)
					boundary_point_set = self.get_polygon_from_gpdf(om.env, h_om=max(_h_om,0), extra_margin=float(abs(_h_om))+1.0, obstacle_idx=idx)
					target_proj_set = np.zeros((len(boundary_point_set),2))
					rob_proj_set = np.zeros((len(boundary_point_set),2))
					rob_proj_dis = np.zeros((len(boundary_point_set),))
					minimal_distance_set = np.zeros((len(boundary_point_set),))
					if type(boundary_point_set) is not list:
						breakpoint()
					for i in range(len(boundary_point_set)):
						if len(boundary_point_set[i])<3:
							continue
						_, proj_point, distances_from_start, total_length = self.get_projection_info(boundary_point_set[i], [self._state[:2].tolist(), self.target[:2].tolist()])
						target_proj_set[i] = proj_point[1]
						rob_proj_set[i] = proj_point[0]
						rob_proj_dis[i] = np.linalg.norm(self._state[:2]-proj_point[0])
						minimal_distance_set[i] = min(abs(distances_from_start[0] - distances_from_start[1]), total_length - abs(distances_from_start[0] - distances_from_start[1])) # this is the alpha reference
						
					self.boundary_point_set = boundary_point_set
					self.rob_proj_points = rob_proj_set
					self.target_proj_points = target_proj_set
					self.min_distance_set = minimal_distance_set

					isOnLine = self.on_line_segment(self.target.reshape(1,2), self._state[:2].reshape(1,2),target_proj_set)
					if len(isOnLine) != 1:
						self.boundary_idx = np.argmin(rob_proj_dis)
					else:
						self.boundary_idx = isOnLine[0]

					self.boundary_points = boundary_point_set[self.boundary_idx]
					minimal_distance = minimal_distance_set[self.boundary_idx]
					target_proj = target_proj_set[self.boundary_idx]
					robot_proj = rob_proj_set[self.boundary_idx]
					
					beta = (minimal_distance) / (2*ITER) * self.beta_coef  # NOTE 100 is iter number
					if np.isnan(beta):
						breakpoint()
					# beta = 0.02

				else:
					e_idx = 3
					break
				
				assert self.target_range is not None, "Target range is not set."
				if (np.linalg.norm(target_proj-self._state[:2].flatten())>total_length/15 or 
					np.linalg.norm(self.target-self._state[:2].flatten())<self.target_range):
					break
			
			if e_idx<2 and h_set[min_h_idx_list[e_idx]]<self.om_range:
				if (type(idx) is not list) and (type(idx) is not np.ndarray):
					breakpoint()
				if self.on_boundary:
					# _,grad_om = om.env.h_grad_uni(robot_proj.reshape(1,2), idx=idx)
					e_vec_output = om.geodesic_approx_phi_2D(robot_proj.reshape(1,2), grad_om.reshape(1,2),self.target.reshape(1,2), beta, onM=idx, 
											 checking_mode=check_mode, e_prev=None)
				else:
					e_vec_output = om.geodesic_approx_phi_2D(self._state[:2].reshape(1,2), grad_om.reshape(1,2),self.target.reshape(1,2), beta, onM=idx, 
											checking_mode=check_mode, e_prev=None)
			
				if check_mode:
					e_vec_raw, pi_list, xi_list = e_vec_output
				else:
					e_vec_raw = e_vec_output
				e_vec = e_vec_raw.flatten()

				v_b_e = e_vec.flatten()@ dx[:2]
				if self.dynamics == "differential":
					# ve_coef = 0.5 if min(h_set)<0.3 else 1
					# breakpoint()
					ve_coef = 1
					constraints += [v_b_e>=self.ve*robot_config.lin_vel_max*ve_coef]
					constraints += [grad_om@dx[:2]+self.func_alpha(0.1*h_om, env=env_name)>=0]

				elif self.dynamics == "omni-directional":
					constraints += [v_b_e>=self.ve*np.linalg.norm([robot_config.lin_vel_max, robot_config.lin_vel_min])]

				self.debug_info['grad_value'] = grad_set[min_h_idx_list[e_idx]].tolist()
				self.debug_info['e_vec'] = e_vec_raw.tolist()
				self.debug_info['om_idx'] = min_h_idx_list[e_idx]
		
		self.debug_info['beta'] = beta
		self.e_prev = e_vec_raw
		# breakpoint()
		
		prob = cp.Problem(obj, constraints)
		if check_mode:
			return prob, u_mod, h_set[min_h_idx_list[0]], grad_set[min_h_idx_list[0]], pi_list, xi_list

		return prob, u_mod

	def get_opt_problem_2D_c(self, u_nom: np.ndarray, om: OnMan_Approx, external_margin:Optional[float]=None, check_mode:bool=False):
		"""Get opt problem with circular obstacles."""
		assert self.target is not None, "Target position is not set."
		
		alpha = self.func_alpha
		margin = self.margin
		pi_list = None
		xi_list = None
		a = 0.2
		x = self.process_shifted(self._state, a).reshape(1,3)
		p_c, grad_c, dtp, _, _ = om.env.h_grad_set(x[:,:2])
		h_om,grad_om = om.env.h_grad_uni_c(p_c, grad_c)

		u_mod = cp.Variable(len(u_nom))
		dx = self.func_f(x.flatten()) + self.func_g(x.flatten(),a) @ u_mod
		obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
		constraints = [grad_c @ dx[:2] + self.func_alpha(p_c.flatten()-margin, env=om.env.env_name) + dtp.flatten()>=0]
		if self.dynamics == "differential":
			obj = cp.Minimize((u_mod[0] - u_nom[0])**2+1E-4*(u_mod[1] - u_nom[1])**2)
			constraints	+= [u_mod[1]<=robot_config.ang_vel_onm_max]+[u_mod[1]>=-robot_config.ang_vel_onm_max] +[u_mod[0]>=robot_config.lin_vel_min]+[u_mod[0]<=robot_config.lin_vel_max]
		else:
			obj = cp.Minimize((u_mod[0] - u_nom[0])**2+(u_mod[1] - u_nom[1])**2)
			constraints += [u_mod[1]<=robot_config.lin_vel_max]+[u_mod[1]>=robot_config.lin_vel_min] +[u_mod[0]>=robot_config.lin_vel_min]+[u_mod[0]<=robot_config.lin_vel_max]
			
		if self.MCBF:
			if self.on_boundary:
				_h_om = 0
			else:
				_h_om = max(h_om[0], 0)
			boundary_point_set = self.get_polygon_from_gpdf(om.env, h_om=_h_om, extra_margin=_h_om +1.0, circle=True)
			target_proj_set = np.zeros((len(boundary_point_set),2))
			rob_proj_dis = np.zeros((len(boundary_point_set),))
			minimal_distance_set = np.zeros((len(boundary_point_set),))

			for i in range(len(boundary_point_set)):
				_, proj_point, distances_from_start, total_length = self.get_projection_info(boundary_point_set[i], [self._state[:2].tolist(), self.target[:2].tolist()])
				target_proj_set[i] = proj_point[1]
				rob_proj_dis[i] = np.linalg.norm(self._state[:2]-proj_point[0])
				minimal_distance_set[i] = min(abs(distances_from_start[0] - distances_from_start[1]), total_length - abs(distances_from_start[0] - distances_from_start[1])) # this is the alpha reference

			self.boundary_point_set = boundary_point_set
			self.min_distance_set = minimal_distance_set

			isOnLine = self.on_line_segment(self.target.reshape(1,2), self._state[:2].reshape(1,2),target_proj_set)

			if len(isOnLine) != 1:
				self.boundary_idx = isOnLine[np.argmin(rob_proj_dis[isOnLine])]
			else:
				self.boundary_idx = isOnLine[0]

			self.boundary_points = boundary_point_set[self.boundary_idx]
			minimal_distance = minimal_distance_set[self.boundary_idx]
			beta = (minimal_distance) / (2*ITER_C) * self.beta_coef  # NOTE 100 is iter number

			e_vec_output = om.geodesic_approx_phi_2D_c(self._state[:2].reshape(1,2), grad_om.reshape(1,2), self.target.reshape(1,2), beta, checking_mode=check_mode)
			if check_mode:
				e_vec, pi_list, xi_list = e_vec_output
			else:
				e_vec = e_vec_output
			self.debug_info["e_vec"] = e_vec
			v_b_e = e_vec.flatten()@dx[:2]
			constraints = constraints + [v_b_e>=0.4]
	
		prob = cp.Problem(obj, constraints)
		if check_mode:
			return prob, u_mod, h_om, grad_om, pi_list, xi_list
		return prob, u_mod


	def one_step(self, n_iter: int, om: OnMan_Approx, use_nominal=False):
		"""One step forward to get action for a single robot.

		Args:
			n_iter: The current iteration/step.

		Returns:
			u_mod: The modified control signal.
			prob_status: The status of the optimization problem.

		Note:
			0: safe_ctrl
			1: safe_ctrl_c
			2: safe_ctrl_onM
		"""
		if not self.init_done:
			raise ValueError(f"[{self.__class__.__name__}] Data logging is not initialized.")
		
		start_time = time.time()
		u_nom = self.get_nominal_ctrl(k_p=1.0)
		self.boundary_points = None

		if not use_nominal:
			for new_margin in self.margin_levels:
				if om.env.env_name == "social_nav":
					opt_prob_output = self.get_opt_problem_2D_c(u_nom, om, external_margin=new_margin)
				else:
					opt_prob_output = self.get_opt_problem_2D(u_nom, om, external_margin=new_margin)

				prob, u_mod = opt_prob_output
				
				try:
					prob.solve()
				except cp.SolverError:
					prob.solve(solver=cp.SCS)
				except:
					prob.solve(solver=cp.ECOS)

				if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
					break

				self.debug_info['current_margin'] = self.margin

			if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
				u_mod_value = u_mod.value
				prob_status = prob.status 
			else:
				print(f"[{self.__class__.__name__}] Infeasible!")
				u_mod_value = np.zeros(2,)
				prob_status = prob.status
		else:
			prob_status = 'nominal' # type: ignore
			u_mod_value = u_nom

		self._u_prev = u_mod_value

		execution_time = time.time()-start_time
		self.execution_times[n_iter] = execution_time

		return u_mod_value, prob_status

	def run_step(self, n_iter: int, om: OnMan_Approx, vb:bool=False):
		"""Run one step of the simulation for a single robot.

		Args:
			n_iter: The current iteration/step.
			scenario: Use which control method. Default to 2.

		Returns:
			u_mod: The modified control signal.
			prob_status: The status of the optimization problem.
			controller_status: The status of the controller.
		"""
		one_step_output = self.one_step(n_iter, om, use_nominal=False)
		u_mod, prob_status = one_step_output[:2]
		
		xy = self.state[:2].reshape(1, 2)
		h_set, *_ = om.env.h_grad_set(xy)
		_h = min(h_set)

		self.controller_status["isSafe"] = True
		self.controller_status["isInfeasible"] = False
		if _h < -0.05:
			print(f"Collision: h:{_h}")
			self.controller_status["isSafe"] = False
		elif self.check_termination_condition():
			self.controller_status["isSuccess"] = True
		if prob_status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
			self.controller_status["isInfeasible"] = True
			# exit()

		status: ControllerStatus = self.controller_status.copy()

		if vb:
			self.print_debug_info(n_iter, num_dynamic_gpdf=len(om.env.gpdf_mmp))

		return u_mod, status


	def print_debug_info(self, current_iter: int, num_dynamic_gpdf: int):
		print('-'*10, f'Debug Info | Iter: {current_iter}', '-'*10)
		print(f"Active gpdf index (total): {self.debug_info['active_gpdf_idx']} ({num_dynamic_gpdf})")
		print(f"Current margin: {self.debug_info['current_margin']}")
		print(f"Gradient value: {self.debug_info['grad_value']}")
		print(f"Geodesic vector: {self.debug_info['e_vec']}")
		print(f"Nearest obstacle: {self.debug_info['om_idx']}")
		print(f"Beta: {self.debug_info['beta']}")





