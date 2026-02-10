import os
import math
import pathlib
from timeit import default_timer as timer

import numpy as np
import cv2 # type: ignore
opencv_version = cv2.__version__.split('.')[0]

import torch
from torchvision.transforms import GaussianBlur # type: ignore

from pkg_mp_ebm.motion_predict import MotionPredictor
from pkg_mp_predefine.motion_predict import MotionPredictor as MP_Predefine
from control.cbf_control import OnManCBFController

from basic_boundary_function.env import Env
from basic_boundary_function.onMan_approximation import OnMan_Approx

from pkg_moving_object.moving_object import RobotObject, HumanObject
from configs import CircularRobotSpecification, PedestrianSpecification, EnvContrConfiguration, GPDFConfiguration
from visualizer.main_plot import PlotInLoop
from visualizer.object import CircularObjectVisualizer

from basic_map.map_tf import ScaleOffsetReverseTransform
import warnings
warnings.filterwarnings("ignore")


TIMEOUT = 800

SCENARIO = "hospital" # "vicon", "vicon_fork", or "hospital"
CASE_NUM = 1
DT = 0.05
AUTORUN = True
VB = True
SAVE_VIDEO = False

if SCENARIO == "vicon_fork":
	ROBOT_START = np.array([-2.5, -0.2, 0.0])
	ROBOT_TARGET = np.array([3.0, -0.2])

### --- Vicon Fork --- ###
if SCENARIO == "vicon_fork":
	HUMAN_STARTS  = [np.array([0.5, 1.5, 0.0])]
	HUMAN_TARGETS = [[np.array([0.5, -0.2]), np.array([-2.0, -0.2])]]

	RESAMPLE_TRAJ = 3 # resample the trajectory to reduce the number of points

	XMIN = -2
	XMAX = 3
	YMIN = -1
	YMAX = 2.8
	RES = 100 # m to px

### --- Hospital --- ###
elif SCENARIO == "hospital":
	if CASE_NUM == 0:
		ROBOT_START = np.array([-10, 0.0, 0.0])
		ROBOT_TARGET = np.array([-5, 0])
		HUMAN_STARTS  = [np.array([-4.8,  1.6,  0. ])]
		HUMAN_TARGETS = [[np.array([-4.8,  0.6]), np.array([-10.6,   0.6])]]
	elif CASE_NUM == 1:
		ROBOT_START = np.array([-15.0, -0.1, 0.0])
		ROBOT_TARGET = np.array([-5, 0])
		
		HUMAN_STARTS  = [np.array([-10.6, -5.4, 1.57])]
		HUMAN_TARGETS = [[np.array([-10.4, 0.0, 1.57]), np.array([-10.7, 0.0, 3.14])]]
		HUMAN_STARTS += [np.array([-4.8, -5.4, 1.57])]
		HUMAN_TARGETS += [[np.array([-4.8, 0.6, 1.57]), np.array([-8.7, 0.05, 3.14])]]
		HUMAN_STARTS += [np.array([-4.8, 2.6, -1.57])]
		HUMAN_TARGETS += [[np.array([-4.8, 0.6, -1.57]), np.array([-9.7, 0.12, -3.14])]]

		HUMAN_STARTS += [np.array([-4.8, -4.4, 0.0])] # Far-end NOTE
		HUMAN_TARGETS += [[np.array([-4.8, 5.6]), np.array([-14.8, 5.6])]] # Far-end NOTE
		HUMAN_STARTS += [np.array([-12.4, 5.6, 0.0])]                     # High-end
		HUMAN_TARGETS += [[np.array([-6.0, 5.6]), np.array([-6.0, 8.6])]] # High-end
		HUMAN_STARTS += [np.array([-13, 0, 0.0])]   # Doctor/Nurse 1
		HUMAN_TARGETS += [[np.array([-13, 0])]]     # Doctor/Nurse 1
		HUMAN_STARTS += [np.array([-16, -8, 0.0])] # Doctor/Nurse 2
		HUMAN_TARGETS += [[np.array([-16, -2.6]), np.array([-16, 0.0]),
						   np.array([-16, -2.6]), np.array([-16, 5.6])]] # Doctor/Nurse 2
	elif CASE_NUM == 2:
		raise NotImplementedError

	RESAMPLE_TRAJ = 4 # resample the trajectory to reduce the number of points

	XMIN = -18
	XMAX = -2
	YMIN = -8
	YMAX = 8
	RES = 25 # m to px

tf_img2real = ScaleOffsetReverseTransform(
	scale=1/RES, 
	offsetx_after=XMIN, 
	offsety_after=YMIN, 
	y_reverse=True, 
	x_max_before=(XMAX-XMIN)*RES,
	y_max_before=(YMAX-YMIN)*RES
)

def load_config(scenario_name: str):
    root_dir = os.path.join(pathlib.Path(__file__).resolve().parents[1])
    data_dir = os.path.join(root_dir, "data")
    config_dir = os.path.join(root_dir, "config")

    config_env = EnvContrConfiguration.from_yaml(os.path.join(config_dir, 'env_controller.yaml'))
    config_gpdf = GPDFConfiguration.from_yaml(os.path.join(config_dir, 'gpdf.yaml'))
    spec_human = PedestrianSpecification.from_yaml(os.path.join(config_dir, 'spec_human.yaml'))
    spec_robot = CircularRobotSpecification.from_yaml(os.path.join(config_dir, 'spec_robot.yaml'))

    if scenario_name == "vicon_fork":
        config_file_path = os.path.join(config_dir, 'vicon_1t10_poselu_enll_train.yaml')
        ref_image_path = os.path.join(data_dir, 'vicon', 'background.png')
        map_path = None
    elif scenario_name == "hospital":
        config_file_path = os.path.join(config_dir, 'hpd_1t20_poselu_enll_train.yaml')
        ref_image_path = os.path.join(data_dir, 'sim_hospital', 'background.png')
        map_path = os.path.join(data_dir, 'sim_hospital', "map.json")

    return config_env, config_gpdf, spec_robot, spec_human, config_file_path, ref_image_path, map_path

### Load configurations
config_env, config_gpdf, spec_robot, spec_human, config_file_path, ref_image_path, map_path = load_config(SCENARIO)

env = Env(load_env=True, env_name=SCENARIO,  rho=config_gpdf.rho, rho_plot= config_gpdf.rho_plot, radius=config_env.radius, num_dyn_circle=0)
om = OnMan_Approx(env, hold_time=config_env.hold_time, w=config_env.w)

### Create motion predictor
motion_predictor = MotionPredictor(config_file_path=config_file_path, model_suffix='1', ref_image_path=ref_image_path)
motion_predictor_pre = MP_Predefine(angular_noise_sigma=math.radians(30))

controller = OnManCBFController(threeD_controller=config_env.threeD_controller, autotune=config_env.autotune, dynamics=config_env.dynamics)
controller.set_params(nominal_speed=config_env.nominal_speed, 
					  sampling_time=spec_robot.ts, 
					  base_margin=config_env.base_margin, 
					  target_range=config_env.target_range,
					  ve = config_env.ve,
					  beta_coef=config_env.beta_coef,
					  om_expand_threshold=config_env.om_expand_threshold,
					  om_range=config_env.om_range,
					  target=ROBOT_TARGET,
					  MCBF=config_env.MCBF,
					  MMP = config_env.MMP,
					  dir_num=config_env.dir_num)
# controller.set_dynamic_margin(margin_levels=[1, 0.5, 0.0])
controller.set_init_data(ROBOT_START, max_iter=TIMEOUT)

robot = RobotObject.from_config(ROBOT_START, spec_robot) # we only need one robot for now
robot_vis = CircularObjectVisualizer(spec_robot.vehicle_width/2, indicate_angle=True)
humans = [HumanObject.from_config(hs, spec_human) for hs in HUMAN_STARTS] # we can have multiple humans
humans_vis = [CircularObjectVisualizer(spec_human.human_width/2, indicate_angle=True) for _ in humans]
for human, hs, ht in zip(humans, HUMAN_STARTS, HUMAN_TARGETS):
	if isinstance(ht, list):
		human.set_path([list(hs)[:2]] + [list(h)[:2] for h in ht])
	else:
		assert isinstance(ht, np.ndarray)
		human.set_path([list(hs)[:2], list(ht)[:2]])

# visualizer = ... # map and robot should have different sub-visulizers
if SAVE_VIDEO:
	video_path = f"test_{timer()}.mp4"
	main_plotter = PlotInLoop(sampling_time=DT, map_only=True, save_to_path=video_path, save_params={'skip_frame': 1, 'frame_size': (1080, 1080)})
else:
	main_plotter = PlotInLoop(sampling_time=DT, map_only=True)
main_plotter.set_env_map(env.plot_env_standard, color='k', plot_grad_dir=False, show_grad=False)
main_plotter.add_object(0, None, (ROBOT_START[0], ROBOT_START[1]), (ROBOT_TARGET[0], ROBOT_TARGET[1]), color='g')

robot_vis.plot(main_plotter.map_ax, *robot.state, object_color='g')
for human, human_vis in zip(humans, humans_vis):
	human_vis.plot(main_plotter.map_ax, *human.state, object_color='orange')

def pred_post_processing(prob_map:torch.Tensor, occ_thre:float=0.1, enable_blur:bool=True, num_pts:int=200, pred_len=20):
	if enable_blur:
		blur = GaussianBlur(kernel_size=(11, 11), sigma=(5, 5))
		prob_map = blur(prob_map)
	prob_map = prob_map[:pred_len, :, :]
	prob_sum = torch.sum(prob_map, dim=0) / prob_map.shape[1]

	occ_area = prob_map
	occ_area[occ_area > torch.amax(occ_area, dim=(1, 2), keepdim=True) * occ_thre] = 1
	occ_area[occ_area < 1] = 0
	invalid_steps = (occ_area == 1).all(dim=(1,2)) # tensor([T/F, ...])
	if invalid_steps.all():
		pass # TODO: Use CVM instead
	occ_area[invalid_steps, :, :] = 0 # Remove the whole image if all 1
	occ_sum = torch.sum(occ_area, dim=0)
	occ_sum[occ_sum > 0] = 1

	plt_map = occ_sum.detach().numpy()
	plt_map = 255 - (plt_map / plt_map.max() * 255)
	plt_map[plt_map<230] = 0
	plt_map[plt_map>=230] = 255

	edges_img = cv2.Canny(np.uint8(plt_map), threshold1=0.5, threshold2=0.5)
	kernel = np.ones((5, 5), np.uint8)
	edges_img = cv2.dilate(edges_img, kernel, iterations=1)

	if int(opencv_version) >= 4:
		contours_img = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	else:
		contours_img = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	contour_img = max(contours_img, key=cv2.contourArea).reshape(-1, 2)
	all_contours:list[np.ndarray] = []
	for cnt in contours_img:
		cnt_reshaped = cnt.reshape(-1, 2)
		all_contours.append(cnt_reshaped)

	risk_area_coords_raw = tf_img2real.cvt_coords(x=contour_img[:, 0], y=contour_img[:, 1], forward=True)
	all_contours = [tf_img2real.cvt_coords(x=cnt[:, 0], y=cnt[:, 1], forward=True) for cnt in all_contours]

	risk_area_coords_ = []

	edge_lengths = np.linalg.norm(np.diff(np.vstack((risk_area_coords_raw, risk_area_coords_raw[[0], :])), axis=0), axis=1)
	points_per_edge = np.floor((edge_lengths / np.sum(edge_lengths)) * num_pts).astype(int)
	zero_indices = np.flatnonzero(points_per_edge == 0)[:(num_pts - np.sum(points_per_edge))]
	points_per_edge[zero_indices] = 1
	point_deficit = num_pts - np.sum(points_per_edge)
	points_per_edge[np.argmax(points_per_edge)] += point_deficit

	for i in range(len(risk_area_coords_raw) - 1):
		p1, p2 = risk_area_coords_raw[i], risk_area_coords_raw[i + 1]
		segment = np.linspace(p1, p2, num=points_per_edge[i], endpoint=False)
		risk_area_coords_.append(segment)
	p1, p2 = risk_area_coords_raw[-1], risk_area_coords_raw[0]
	segment = np.linspace(p1, p2, num=points_per_edge[-1], endpoint=False)
	risk_area_coords_.append(segment)
	risk_area_coords = np.vstack(risk_area_coords_)

	return risk_area_coords, prob_sum


# try:
controller_status = None
for kt in range(TIMEOUT):
	if not VB:
		print(f"\r Time step: {kt}/{TIMEOUT}", end='    ')
	else:
		print('='*15, f'Time step: {kt}/{TIMEOUT}', '='*15)

	if controller.check_termination_condition():
		break

	prob_sum = None
	if humans:
		all_predicted_motion = []
		all_predicted_vel = []
		human_idle_set = []
		pred_time_list = []
		post_time_list = []
		for nh, human in enumerate(humans):
			input_traj = human.past_traj[::-1]
			if len(input_traj) <= RESAMPLE_TRAJ*4 or human.idle or not config_env.MMP:
				input_traj = input_traj[::-1]
				risk_area_coords = np.array([human.state[:2] + np.array([np.cos(theta), np.sin(theta)]) * 0.5 for theta in np.linspace(0, 2*np.pi, 201)[:200]])
			else:
				input_traj = input_traj[::RESAMPLE_TRAJ][::-1]
				if len(input_traj) > 5:
					input_traj = input_traj[-5:]
				input_traj_NN = [tf_img2real(list(x), forward=False) for x in input_traj]

				### Predict
				start_time = timer()
				prob_map = motion_predictor.get_network_output(input_traj=input_traj_NN) # type: ignore
				pred_time_list.append(timer()-start_time)
				start_time = timer()
				risk_area_coords, prob_sum = pred_post_processing(prob_map, occ_thre=0.1, enable_blur=True, num_pts=200)
				post_time_list.append(timer()-start_time)
				### End predict
			_, pred_vel_abs = motion_predictor_pre.mp_cvm.get_motion_prediction_for_cbf(human.past_traj, pred_len=20)

			all_predicted_motion.append(risk_area_coords)
			all_predicted_vel.append(pred_vel_abs)
			human_idle_set.append(human.idle)

		if VB and pred_time_list:
			print(f"Prediction time (mean/total): {round(np.mean(pred_time_list), 4)}/{round(sum(pred_time_list), 4)} ({round(np.min(pred_time_list), 4)}-{round(np.max(pred_time_list), 4)}) s")
			print(f"Post-processing time (mean/total): {round(np.mean(post_time_list), 4)}/{round(sum(post_time_list), 4)} ({round(np.min(post_time_list), 4)}-{round(np.max(post_time_list), 4)}) s")


		if all_predicted_motion:
			env.reset_mmp_gpdf(coords_list=all_predicted_motion, offset=0.1, human_idle=human_idle_set)
			env.update_xc(xc=np.vstack(all_predicted_vel)[:, 2:], dxc=np.vstack(all_predicted_vel)[:, :2]/DT)

	### Get the control signal from the controller
	controller.set_state(robot.state)
	start_time = timer()
	if controller_status is not None and controller_status['isInfeasible']:
		breakpoint()
	run_step_output = controller.run_step(kt, om, vb=VB)
	u_mod, controller_status = run_step_output[:2]
	
	if VB:
		print(f"Controller solve time: {round(timer()-start_time, 4)} s")

	### Apply the control signal to the robot
	robot.one_step(u_mod)
	if controller_status['isInfeasible']:
		robot_color = 'b'
	elif not controller_status['isSafe']:
		robot_color = 'r'
	else:
		robot_color = '#135e08' # basically green
	robot_vis.update(*robot.state, color=robot_color)

	### Update the motion of the humans
	for human in humans:
		human.run_step(social_force=None)
		human.set_idle(robot.position, 10) #change 6 to 20 to restore old settings

	ctr, ctrf = env.plot_env_standard(main_plotter.map_ax, dynamic_obstacle=True, show_grad=False, plot_grad_dir=False)
	try:
		e_vec_list = controller.debug_info['e_vec']
		if e_vec_list:
			e_vec = e_vec_list[0]
			e_vec_viz = main_plotter.map_ax.quiver(*robot.state[:2], e_vec[0], e_vec[1], color='r', scale=1, scale_units='xy', angles='xy')
		else:
			e_vec_viz = None
	except:
		e_vec_viz = None

	color_list = ['g-', 'y-', 'c-', 'm-', 'tab:pink','tab:gray','tab:orange','tab:olive','tab:cyan', 'tab:brown','tab:purple']
	boundary_viz_set = []
	if controller.boundary_point_set is not None:
		for i in range(len(controller.boundary_point_set)):
			boundary_points_to_plot = controller.boundary_point_set[i]
			boundary_points_to_plot = np.vstack((boundary_points_to_plot, boundary_points_to_plot[0]))
			if i == controller.boundary_idx:
				boundary_viz = main_plotter.map_ax.plot(*zip(*boundary_points_to_plot), 'r')[0]
			elif i<len(color_list):
				boundary_viz = main_plotter.map_ax.plot(*zip(*boundary_points_to_plot), color_list[i])[0]
			else:
				boundary_viz = main_plotter.map_ax.plot(*zip(*boundary_points_to_plot), 'b')[0]
			boundary_viz_set.append(boundary_viz)
	
	rp_viz_set = []
	if controller.rob_proj_points is not None:
		for i in range(len(controller.rob_proj_points)):
			if i == controller.boundary_idx:
				rp_viz = main_plotter.map_ax.plot(controller.rob_proj_points[i][0], controller.rob_proj_points[i][1], 'r', marker='o')[0]
			elif i<len(color_list):
				rp_viz = main_plotter.map_ax.plot(controller.rob_proj_points[i][0], controller.rob_proj_points[i][1], color_list[i], marker='o')[0]
			else:
				rp_viz = main_plotter.map_ax.plot(controller.rob_proj_points[i][0], controller.rob_proj_points[i][1], 'b', marker='o')[0]
			rp_viz_set.append(rp_viz)

	tp_viz_set = []
	if controller.rob_proj_points is not None:
		for i in range(len(controller.rob_proj_points)):
			assert controller.target_proj_points is not None
			if i == controller.boundary_idx:
				tp_viz = main_plotter.map_ax.plot(controller.target_proj_points[i][0], controller.target_proj_points[i][1], 'r', marker='x')[0]
			elif i< len(color_list):
				tp_viz = main_plotter.map_ax.plot(controller.target_proj_points[i][0], controller.target_proj_points[i][1], color_list[i], marker='x')[0]
			else:
				tp_viz = main_plotter.map_ax.plot(controller.target_proj_points[i][0], controller.target_proj_points[i][1], 'b', marker='x')[0]
			tp_viz_set.append(tp_viz)
			
	main_plotter.plot_in_loop(
		mask=motion_predictor.ref_image.numpy(),
		mask_extent=[XMIN, XMAX, YMIN, YMAX],
		polygonal_dyn_obstacle_list=all_predicted_motion if humans else None,
		other_plt_objects=[ctr, ctrf, e_vec_viz] + boundary_viz_set + rp_viz_set + tp_viz_set,
		time=kt*DT, autorun=AUTORUN, zoom_in=[-18,-4, -7, 7], auto_release=False,
	)
	main_plotter.update_object(0, kt, u_mod, robot.state, controller.debug_info['current_margin'], None, None)
	for human, human_vis in zip(humans, humans_vis):
		human_vis.update(*human.state, color=None)
	main_plotter.plot_in_loop_release(autorun=AUTORUN)


input("\nPress Enter to cleanup...")


if main_plotter.is_active:
	main_plotter.show()
	main_plotter.close()


	
