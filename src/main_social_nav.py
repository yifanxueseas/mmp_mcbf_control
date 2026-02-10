import os
import math
import pathlib
from timeit import default_timer as timer

import numpy as np

from pkg_moving_object.moving_object import RobotObject
from pkg_moving_object.moving_object import HumanObject
from pkg_moving_object.human_trajs import x_c, dx_c

from pkg_mp_predefine.motion_predict import MotionPredictor as MP_Predefine

from control.cbf_control import OnManCBFController

from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker as MPCController

from configs import MpcConfiguration
from configs import CircularRobotSpecification, PedestrianSpecification 
from configs import EnvContrConfiguration, GPDFConfiguration

from basic_motion_model.motion_model import UnicycleModel
from basic_boundary_function.env import Env
from basic_boundary_function.onMan_approximation import OnMan_Approx

from visualizer.main_plot import PlotInLoop
from visualizer.object import CircularObjectVisualizer

import warnings
warnings.filterwarnings("ignore")


def load_config():
    config_dir = os.path.join(pathlib.Path(__file__).resolve().parents[1], "config")

    config_env = EnvContrConfiguration.from_yaml(os.path.join(config_dir, 'env_controller.yaml'))
    config_gpdf = GPDFConfiguration.from_yaml(os.path.join(config_dir, 'gpdf.yaml'))
    config_mpc = MpcConfiguration.from_yaml(os.path.join(config_dir, "mpc_fast.yaml"))
    spec_robot = CircularRobotSpecification.from_yaml(os.path.join(config_dir, 'spec_robot.yaml'))
    spec_human = PedestrianSpecification.from_yaml(os.path.join(config_dir, 'spec_human.yaml'))
    return config_env, config_gpdf, config_mpc, spec_robot, spec_human


def main(time_out:int=300, pred_len:int=100, run_mpc:bool=False,
         auto_run:bool=True, verbose:bool=False):
    
    traj = np.zeros((time_out, 3))
    start_heading = 0*2*np.pi/10
    goal_heading = 0*2*np.pi/10

    config_env, config_gpdf, config_mpc, spec_robot, spec_human = load_config()

    robot_init_state = np.array([config_env.init_state[0], config_env.init_state[1], start_heading])
    robot_goal_state = np.array([config_env.target[0], config_env.target[1], goal_heading])

    scenario_name = "social_nav_pred" if pred_len > 0 else "social_nav"

    DT = 0.05
    if run_mpc:
        DT = 0.2
        pred_len = int(pred_len * spec_robot.ts / DT)
        spec_robot.ts = DT

    if pred_len > 0:
        env = Env(load_env=True, env_name=scenario_name, rho=config_gpdf.rho, radius=config_env.radius)
    else:
        env = Env(load_env=True, env_name=scenario_name, rho=config_gpdf.rho, radius=config_env.radius, num_dyn_circle=config_env.num_dyn_circle)
    om = OnMan_Approx(env, hold_time=config_env.hold_time, w=config_env.w)

    ### Prepare predictor
    motion_predictor_pre = MP_Predefine(angular_noise_sigma=math.radians(30))

    ### Prepare controller
    if run_mpc:
        scheduled_path_coords = [tuple(robot_init_state[:2].tolist()), tuple(robot_goal_state[:2].tolist())]
        planner = LocalTrajPlanner(spec_robot.ts, config_mpc.N_hor, spec_robot.lin_vel_max, verbose=False)
        planner.load_map([], [])
        planner.load_path(scheduled_path_coords, None, nomial_speed=spec_robot.lin_vel_max, method="linear")
        controllerMPC = MPCController(config_mpc, spec_robot, robot_id=0, verbose=False)
        controllerMPC.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
        controllerMPC.load_init_states(robot_init_state, robot_goal_state)
    else:
        controller = OnManCBFController(threeD_controller=config_env.threeD_controller, 
                                        autotune=config_env.autotune, 
                                        dynamics=config_env.dynamics)
        controller.set_params(nominal_speed=config_env.nominal_speed, 
                            sampling_time=spec_robot.ts, 
                            base_margin=config_env.base_margin, 
                            target_range=config_env.target_range,
                            ve = config_env.ve,
                            beta_coef=config_env.beta_coef,
                            om_expand_threshold=config_env.om_expand_threshold,
                            om_range=config_env.om_range,
                            target=config_env.target,
                            MCBF=config_env.MCBF,
                            dir_num=config_env.dir_num,
                            on_boundary=True,
                            MMP=True)
        controller.set_init_data(robot_init_state, max_iter=time_out)


    if config_env.num_dyn_circle:
        human_starts = x_c(0, steps=30)
        if pred_len < 1:
            om.env.update_xc(x_c(0, steps=30)[:,:2], dx_c(0,steps=30)[:,:2])
        humans = [HumanObject.from_config(hs, spec_human) for hs in human_starts]
    else:
        human_starts = []
        humans = []
    robot = RobotObject.from_config(robot_init_state, spec_robot) # we only need one robot for now
    robot_vis = CircularObjectVisualizer(spec_robot.vehicle_width/2, indicate_angle=True if config_env.dynamics=="differential" else False)
    if om.env.num_dyn_circle:
        humans_vis = [CircularObjectVisualizer(config_env.radius, indicate_angle=True) for _ in range(len(human_starts))]
    else:
        humans_vis = [CircularObjectVisualizer(spec_human.human_width/2, indicate_angle=True) for _ in range(len(human_starts))]


    main_plotter = PlotInLoop(sampling_time=spec_robot.ts, map_only=True)#, save_to_path='./mpc_sn_pred.mp4', save_params={'skip_frame': 0})
    main_plotter.set_env_map(env.plot_env_standard, color='k', plot_grad_dir=False, show_grad=False)
    main_plotter.map_ax.set_xlim(-5,5)
    main_plotter.map_ax.set_ylim(-5,5)
    main_plotter.add_object(0, None, 
                            (robot_init_state[0], robot_init_state[1]), 
                            (robot_goal_state[0], robot_goal_state[1]), color='g')


    robot_vis.plot(main_plotter.map_ax, *robot.state, object_color='g')
    for  state, human_vis in zip(human_starts, humans_vis):
        human_vis.plot(main_plotter.map_ax, *state, object_color='orange')


    for kt in range(time_out):
        if not verbose:
            print(f"\r Time step: {kt}/{time_out}", end='    ')
        else:
            print('='*30)

        all_predicted_motion = []
        all_mpc_predicted_motion = []
        all_predicted_vel = []
        human_idle_set = []
        if humans and pred_len>0:
            for nh, human in enumerate(humans):
                past_traj = human.past_traj
                if len(past_traj) > 5:
                    past_traj = past_traj[-5:]
                    risk_area_coords, pred_vel_abs, risk_zone = motion_predictor_pre.get_motion_prediction_for_cbf(past_traj, pred_len=pred_len, enable_mmp=False)
                else:
                    risk_area_coords = np.array([human.state[:2] + np.array([np.cos(theta), np.sin(theta)]) * 0.5 for theta in np.linspace(0, 2*np.pi, 201)[:200]])
                    _, pred_vel_abs, risk_zone = motion_predictor_pre.get_motion_prediction_for_cbf(past_traj, pred_len=pred_len, enable_mmp=False)
                    # continue
                all_predicted_motion.append(risk_area_coords)
                all_predicted_vel.append(pred_vel_abs)
                all_mpc_predicted_motion.append(risk_zone)
                human_idle_set.append(False)
                
            if all_predicted_motion:
                om.env.reset_mmp_gpdf(coords_list=all_predicted_motion, human_idle=human_idle_set,offset=0.3)
                om.env.update_xc(
                    x_c(kt, steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)[:,:2], 
                    dx_c(kt,steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)[:,:2]
                )

        if run_mpc:
            if controllerMPC.check_termination_condition():
                break
        else:  
            if controller.check_termination_condition():
                break

        if run_mpc:
            ref_states, ref_speed, *_ = planner.get_local_ref(kt*spec_robot.ts, (float(robot.state[0]), float(robot.state[1])) )
            controllerMPC.set_ref_states(ref_states, ref_speed=ref_speed)
            controllerMPC.set_current_state(np.array(robot.state))
        else:
            controller.set_state(robot.state)
        traj[kt] = robot.state
        start_time = timer()

        pred_states = None
        if run_mpc:
            actions, pred_states, current_refs, debug_info = controllerMPC.run_step(
                static_obstacles=all_mpc_predicted_motion,
                full_dyn_obstacle_list=None)
            u_mod = actions[0]
        else:
            run_step_output = controller.run_step(kt, om, vb=verbose)
            u_mod, controller_status = run_step_output[:2]
        if verbose:
            print(f"Controller solve time: {timer()-start_time} s")

        # apply the control signal to the robot
        robot.one_step(u_mod)
        if run_mpc:
            humans_pos = x_c(kt, steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)[:,:2]
            isCollision = controllerMPC.check_collision(
                circles=np.concatenate(
                    (humans_pos, np.ones(humans_pos.shape[0]).reshape(-1,1)*(spec_human.human_width/2+spec_robot.vehicle_width/2)), axis=1
                )
            )
            robot_color = 'r' if isCollision else '#135e08'
            robot_vis.update(*robot.state, color=robot_color)
        else:
            if controller_status['isInfeasible']:
                robot_color = 'b'
            elif not controller_status['isSafe']:
                robot_color = 'r'
            else:
                robot_color = '#135e08' # basically green
            robot_vis.update(*robot.state, color=robot_color)
        # update the motion of the humans
        if om.env.num_dyn_circle:
            actions = dx_c(kt,steps=30)
        else:
            actions= None

        human_state = x_c(kt, steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)
        for i, human_vis in zip(range(om.env.num_dyn_mmp), humans_vis):
            humans[i].past_traj.append(human_state[i])
            human_vis.update(*human_state[i], color=None)
            
        if pred_len < 1:
            om.env.update_xc(
                x_c(kt, steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)[:,:2], 
                dx_c(kt,steps=30/(spec_robot.ts/0.05),dt=spec_robot.ts)[:,:2]
            )

        ctr, ctrf = env.plot_env_standard(main_plotter.map_ax, dynamic_obstacle=True, show_grad=False, plot_grad_dir=False)
        current_refs_viz = None
        pred_states_viz = None
        if run_mpc:
            boundary_viz = None
            pred_states_viz = main_plotter.map_ax.plot(np.array(pred_states)[:, 0], np.array(pred_states)[:, 1], 'mx')[0]
        elif controller.boundary_points is not None:
            plt_b_pts = np.concatenate((controller.boundary_points, controller.boundary_points[[0], :]), axis=0)
            boundary_viz = main_plotter.map_ax.plot(*zip(*plt_b_pts), 'b-')[0]
        else:
            boundary_viz = None
        try:
            e_vec_list = controller.debug_info['e_vec']

            if e_vec_list is not None:
                e_vec = e_vec_list[0]
                e_vec_viz = main_plotter.map_ax.quiver(*robot.state[:2], e_vec[0], e_vec[1], color='r', scale=1, scale_units='xy', angles='xy')
            else:
                e_vec_viz = None
        except:
            e_vec_viz = None
        
        main_plotter.update_object(0, kt, u_mod, robot.state, None, None, None)

        
        main_plotter.plot_in_loop(
            polygonal_dyn_obstacle_list=all_predicted_motion if humans else None,
            other_plt_objects=[ctr, ctrf, boundary_viz, e_vec_viz, current_refs_viz, pred_states_viz],
            time=kt*spec_robot.ts, autorun=auto_run, zoom_in=None,
        )

    input("\nPress Enter to continue...")


    if main_plotter.is_active:
        main_plotter.show()
        main_plotter.close()

    print("Done!")


if __name__ == "__main__":

    # main(pred_len=100, run_mpc=False, auto_run=True, verbose=False)

    import argparse
    parser = argparse.ArgumentParser(description='Social Navigation with MPC')
    parser.add_argument('--time_out', type=int, default=300, help='Time out for the simulation')
    parser.add_argument('--pred_len', type=int, default=100, help='Prediction length for the social navigation')
    parser.add_argument('--run_mpc', type=bool, default=False, help='Run MPC or not')
    parser.add_argument('--auto_run', type=bool, default=True, help='Auto run the simulation')
    parser.add_argument('--verbose', action='store_false', help='Verbose mode')
    args = parser.parse_args()

    main(args.time_out, args.pred_len, args.run_mpc, args.auto_run, args.verbose)
    