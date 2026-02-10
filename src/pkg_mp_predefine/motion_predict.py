import math
from typing import Tuple, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN # type: ignore
from scipy.spatial import ConvexHull # type: ignore

from pkg_mp_cvm.motion_predict import MotionPredictor as MP_CVM


PathNode = Tuple[float, float]


class MotionPredictor:
    def __init__(self, angular_noise_sigma=math.radians(10)) -> None:
        print(f"[{self.__class__.__name__}] Initializing predefined motion predictor.")

        self.sigma = angular_noise_sigma
        self.mp_cvm = MP_CVM(angular_noise_sigma, verbose=False)

        self.turned = False
        self.multi_goal = False
        self.hold_time_cnt = 0

        self.weighted_step_speed = 0.0

    def load_agent_info(self, turn: PathNode, possible_goals: List[PathNode], actual_goal: PathNode) -> None:
        self.turn = turn
        self.possible_goals = possible_goals
        self.actual_goal = actual_goal

    def get_motion_prediction_for_cbf(self, input_traj: List[PathNode], num_samples_for_cvm=100, pred_len=20, num_points_for_polygon=100, smooth_speed=False, smooth_velocity=False, enable_mmp=True):
        """
        """
        current_pos = input_traj[-1][:2]

        polygon_coords_cvm, predicted_velocity_abs, risk_zone_cvm = self.mp_cvm.get_motion_prediction_for_cbf(
            input_traj, 
            num_samples=num_samples_for_cvm, 
            pred_len=pred_len, 
            num_points_for_polygon=num_points_for_polygon, 
            smooth_speed=smooth_speed,
            smooth_velocity=smooth_velocity,
            output_risk_zone=True, 
        )
        self.weighted_step_speed = self.mp_cvm.weighted_step_speed
        if not enable_mmp:
            return polygon_coords_cvm, predicted_velocity_abs, risk_zone_cvm

        if smooth_velocity:
            heading = math.atan2(current_pos[1] - input_traj[0][1], current_pos[0] - input_traj[0][0])
        else:
            heading = math.atan2(current_pos[1] - input_traj[-2][1], current_pos[0] - input_traj[-2][0])
        dist_to_turn = math.hypot(self.turn[0]-current_pos[0], self.turn[1]-current_pos[1])

        if abs(heading)<math.pi/9 or heading>8/9*math.pi or heading<-8/9*math.pi or self.turned:
            self.hold_time_cnt += 1
            if self.hold_time_cnt > 10:
                self.turned = True
                self.multi_goal = False
            polygon_coords = polygon_coords_cvm
        else: # not turned yet
            self.hold_time_cnt = 0
            if dist_to_turn >= self.weighted_step_speed*pred_len:
                self.multi_goal = False
                polygon_coords = polygon_coords_cvm
            else: # multi-goal
                self.multi_goal = True
                future_path_1 = self.interpolate_path(current_pos, self.turn, self.weighted_step_speed)
                future_paths_2 = [self.interpolate_path(self.turn, goal, self.weighted_step_speed) for goal in self.possible_goals]
                for ipath, path in enumerate(future_paths_2):
                    if len(path) < pred_len - len(future_path_1):
                        future_paths_2[ipath] += [path[-1]] * (pred_len - len(path) - len(future_path_1))
                    else:
                        future_paths_2[ipath] = path[:max(0, pred_len-len(future_path_1))]
                # print('Len:', len(future_path_1), "Lens:", [len(asas) for asas in future_paths_2]) # XXX
                
                if len(future_paths_2[0]) > 0:
                    risk_zone_multi = [current_pos]
                    for path in future_paths_2:
                        risk_zone_multi += [path[-1]]
                    risk_zone_multi = list(set(tuple(x) for x in risk_zone_multi)) # type: ignore
                    if len(risk_zone_multi) < 3:
                        risk_zone = risk_zone_cvm
                    else:
                        risk_zone_all = np.array(risk_zone_multi+risk_zone_cvm)
                        hull = ConvexHull(risk_zone_all)
                        risk_zone = risk_zone_all[hull.vertices]
                    polygon_coords = []
                    for i in range(len(risk_zone)-1):
                        samples = np.linspace(risk_zone[i], risk_zone[i+1], num_points_for_polygon//4, endpoint=False)
                        polygon_coords.extend(samples)
                    samples = np.linspace(risk_zone[-1], risk_zone[0], num_points_for_polygon//4, endpoint=False)
                    polygon_coords.extend(samples)
                else:
                    polygon_coords = polygon_coords_cvm

        # print("Turned:", self.turned, "Multigoal:", self.multi_goal) # XXX
        return polygon_coords, predicted_velocity_abs
        
       
    @staticmethod
    def interpolate_path(A, B, step: float) -> List[PathNode]:
        """Interpolate points between two points A and B.

        Returns:
            A list of interpolated points.
        """
        distance = np.linalg.norm(np.array(B) - np.array(A))
        num_points = int(distance / step) + 1
        return [((1 - t) * np.array(A) + t * np.array(B)).tolist() for t in np.linspace(0, 1, num_points)]
    







