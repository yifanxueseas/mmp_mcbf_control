import math
from typing import Optional, Tuple, List

import numpy as np
from sklearn.cluster import DBSCAN # type: ignore

from basic_obstacle.geometry_plain import PlainPolygon


PathNode = Tuple[float, float]


class MotionPredictor:
    def __init__(self, angular_noise_sigma=math.radians(10), verbose=True) -> None:
        """
        Args:
            angular_noise_sigma: The standard deviation (in rad) of the angular noise. Defaults to 10 degrees.
        
        Notes:
            The angular noise is assumed to be Gaussian distributed as in the reference paper.

        References:
            Paper - What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction
        """
        if verbose:
            print(f"[{self.__class__.__name__}] Initializing CVM with angular noise sigma: {angular_noise_sigma}")
        self.sigma = angular_noise_sigma

        self.weighted_step_speed = 0.0

    def get_motion_prediction_samples(self, input_traj: List[PathNode], num_samples=100, pred_len=20, smooth_speed=False, smooth_velocity=False) -> np.ndarray:
        """Get motion prediction (samples from the CVM with angular noise)

        Args:
            input_traj: List of tuples. Each tuple is a 2D point (x, y).
            num_samples: Number of samples to generate. Defaults to 100.
            pred_len: Prediction length. Defaults to 20.

        Returns:
            prediction_samples: [T * num_samples * 2]

        Notes:
            The sampling time is obtained from the input trajectory.
        """
        # fixed_speed = 0.5 # m/s

        if not (isinstance(input_traj, list) and len(input_traj) > 0):
            raise ValueError('The input trajectory should be a list of tuples.')

        if len(input_traj) < 2:
            prediction_samples = np.tile(input_traj[0][:2], (num_samples, 1))
            prediction_samples = np.tile(prediction_samples, (pred_len, 1, 1))
            return prediction_samples

        last_pos = np.array(input_traj[-1][:2])
        llast_pos = np.array(input_traj[-2][:2])

        ### Rescale the direction according to the weighted speed
        if smooth_velocity:
            pred_dir = (np.array(input_traj[-1][:2]) - np.array(input_traj[0][:2])) / (len(input_traj)-1)
            weighted_step_speed = float(np.linalg.norm(pred_dir))
        elif smooth_speed:
            pred_dir = (last_pos - llast_pos)[:2]
            prev_step_speeds = [np.linalg.norm(np.array(input_traj[i+1][:2]) - np.array(input_traj[i][:2])) for i in range(len(input_traj)-1)]
            cur_step_speed = np.linalg.norm(pred_dir)
            weighted_step_speed = float((1 * cur_step_speed + sum(prev_step_speeds)) / (1 + len(prev_step_speeds)))
            if cur_step_speed < 1e-6:
                pred_dir = np.array([0.0, 0.0])
            else:
                pred_dir = pred_dir / np.linalg.norm(pred_dir) * weighted_step_speed
        else:
            pred_dir = (last_pos - llast_pos)[:2]
            weighted_step_speed = float(np.linalg.norm(pred_dir))
        self.weighted_step_speed = weighted_step_speed

        multiplier = np.arange(1, pred_len+1) # for time steps
        pred_sample_pure = np.vstack((last_pos[0] + pred_dir[0]*multiplier,
                                      last_pos[1] + pred_dir[1]*multiplier)).T
        pred_samples = np.zeros((pred_len, num_samples, 2))
        pred_samples[:, 0, :] = pred_sample_pure
        
        noise = np.random.normal(0, self.sigma, num_samples-1).reshape(-1, 1, 1)
        rotation_matrix = np.concatenate((np.cos(noise), -np.sin(noise), np.sin(noise), np.cos(noise)), axis=2).reshape(-1, 2, 2)
        pred_dir_noisy:np.ndarray = np.dot(rotation_matrix, pred_dir) # (n, 2)
        pred_dir_noisy = pred_dir_noisy[:, :, np.newaxis] * multiplier[np.newaxis, np.newaxis, :]
        pred_samples_noisy = last_pos[np.newaxis, np.newaxis, :] + pred_dir_noisy.transpose(2, 0, 1)
        pred_samples[:, 1:, :] = pred_samples_noisy

        return pred_samples
    
    def clustering_and_fitting_from_samples(self, traj_samples: np.ndarray, eps=10, min_sample=5, enlarge=1.0, extra_margin=0.0):
        """Inference the network and then do clustering.

        Args:
            traj_samples: numpy array [T * (x, y)], meaning (x, y) at time T
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 10.
            min_sample: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.
            enlarge: The factor to enlarge the standard deviation. Defaults to 1.0.
            extra_margin: The extra margin to add to the standard deviation. Defaults to 0.0.

        Raises:
            ValueError: If the input probability maps are not [CxHxW].

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
        """
        if len(traj_samples.shape) != 3:
            raise ValueError('The input trajectory samples should be [T*(x,y)].')

        clusters_list = []
        mu_list_list = []
        std_list_list = []
        conf_list_list = []
        for i in range(traj_samples.shape[0]):
            clusters, _ = self.fit_DBSCAN(traj_samples[i,:], eps=eps, min_sample=min_sample)
            clusters_list.append(clusters)
            mu_list, std_list = self.fit_cluster2gaussian(clusters, enlarge, extra_margin)

            conf_list = []
            for cluster in clusters:
                conf_list.append(float(cluster.shape[0]))
            conf_list = [round(x/sum(conf_list), 2) for x in conf_list]

            mu_list_list.append(mu_list)
            std_list_list.append(std_list)
            conf_list_list.append(conf_list)
        return clusters_list, mu_list_list, std_list_list, conf_list_list
        
    def get_motion_prediction_for_cbf(self, input_traj: List[PathNode], num_samples=100, pred_len=20, num_points_for_polygon=100, smooth_speed=False, smooth_velocity=False, output_risk_zone=False):
        """
        Args:
            input_traj: List of tuples. Each tuple is a 2D point (x, y).
            num_samples: Number of samples to generate. Defaults to 100.
            pred_len: Prediction length (time step). Defaults to 20.
            num_points_for_polygon: Number of points to represent the polygon. Defaults to 100.

        Returns:
            polygon_coords: The coordinates of the predicted risk zone.
            predicted_velocity_abs: The predicted velocity of the last time step (vx, vy, x, y).
        """
        idx_mid = int(pred_len//2) # original sample time is 0.1s
        idx_end = pred_len-1 # =int(15*2.5) # original sample time is 0.1s
        if len(input_traj)==0:
            return None, (0, 0, 99, 99)
        samples = self.get_motion_prediction_samples(input_traj, num_samples, pred_len, smooth_speed=smooth_speed, smooth_velocity=smooth_velocity)
        _, mu_list_list, std_list_list, _ = self.clustering_and_fitting_from_samples(samples)
        predicted_velocity = mu_list_list[-1][0] - mu_list_list[-2][0]
            
        predicted_velocity_abs = (self.weighted_step_speed, predicted_velocity[1], mu_list_list[idx_mid][0][0], mu_list_list[idx_mid][0][1])
        if np.linalg.norm(mu_list_list[0][0]-mu_list_list[idx_end][0]) < 1e-6:
            polygon_coords = [
                mu_list_list[0][0], 
                mu_list_list[0][0] + np.array([0.1, 0]),
                mu_list_list[0][0] + np.array([0.1, 0.1]),
                mu_list_list[0][0] + np.array([0, 0.1]),
            ]
            risk_zone = polygon_coords
        else:
            middle_pt_1, middle_pt_2 = self.find_perpendicular_points(
                mu_list_list[0][0], mu_list_list[idx_mid][0], mu_list_list[idx_end][0], max(std_list_list[idx_mid][0]))
            risk_zone = [
                mu_list_list[0][0], 
                # input_traj[0][:2],
                middle_pt_1,
                mu_list_list[idx_end][0],
                middle_pt_2,
            ]
            inflated_obs = PlainPolygon.from_list_of_tuples(risk_zone).inflate(0.5)
            risk_zone = inflated_obs()
            polygon_coords = []
            for i in range(len(risk_zone)-1):
                samples = np.linspace(risk_zone[i], risk_zone[i+1], num_points_for_polygon//4, endpoint=False)
                polygon_coords.extend(samples)
            samples = np.linspace(risk_zone[-1], risk_zone[0], num_points_for_polygon//4, endpoint=False)
            polygon_coords.extend(samples)
        if output_risk_zone:
            return np.array(polygon_coords), predicted_velocity_abs, risk_zone
        return np.array(polygon_coords), predicted_velocity_abs
        

    @staticmethod
    def fit_DBSCAN(data, eps: float, min_sample: int) -> Tuple[List[np.ndarray], List[int]]:
        """Generate clusters using DBSCAN.

        Args:
            data: Should be a 2D array, each row is a sample.
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_sample: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

        Returns:
            clusters: A list of clusters, each cluster is a 2D array.
            labels: A list of labels for each sample (row).
        """
        clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(data)
        nclusters = len(list(set(clustering.labels_)))
        if -1 in clustering.labels_:
            nclusters -= 1
        clusters = []
        for i in range(nclusters):
            cluster = data[clustering.labels_==i, :]
            clusters.append(cluster)
        return clusters, list(clustering.labels_)
    
    @staticmethod
    def fit_cluster2gaussian(clusters: List[np.ndarray], enlarge=1.0, extra_margin=0.0) -> Tuple[list, list]:
        """Generate Gaussian distributions from clusters.

        Args:
            clusters: A list of clusters, each cluster is a 2D array.

        Returns:
            mu_list: A list of means, each mean is a pair of coordinates.
            std_list: A list of standard deviations, each std is a pair of numbers.
        """
        mu_list  = []
        std_list = []
        for cluster in clusters:
            mu_list.append(np.mean(cluster, axis=0))
            std_list.append(np.std(cluster, axis=0)*enlarge+extra_margin)
        return mu_list, std_list
    
    @staticmethod
    def find_perpendicular_points(A, B, C, d):
        x_A, y_A = A
        x_B, y_B = B
        x_C, y_C = C

        v_x = x_C - x_A
        v_y = y_C - y_A
        v_perp_x = -v_y
        v_perp_y = v_x

        norm = np.sqrt(v_perp_x**2 + v_perp_y**2)
        v_perp_x_unit = v_perp_x / norm
        v_perp_y_unit = v_perp_y / norm

        B1_x = x_B + d * v_perp_x_unit
        B1_y = y_B + d * v_perp_y_unit
        B2_x = x_B - d * v_perp_x_unit
        B2_y = y_B - d * v_perp_y_unit
        B1 = (B1_x, B1_y)
        B2 = (B2_x, B2_y)

        return B1, B2



