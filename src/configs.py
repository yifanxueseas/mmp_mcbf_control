import os
import pathlib
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union
import yaml # type: ignore
import warnings

# import torch # type: ignore

### Base Configurator
class Configurator:
    FIRST_LOAD = False

    def __init__(self, config_fp: str, config_type:str='yaml', with_partition=False) -> None:
        """Configuration file loader.

        Args:
            config_type: Type of configuration file, either 'yaml' or 'torch'. Defaults to 'yaml'.
            with_partition: If configuration file is a partitioned yaml file. Defaults to False.
        """
        if Configurator.FIRST_LOAD:
            print(f'{self.__class__.__name__} Loading configuration from "{config_fp}".')
            Configurator.FIRST_LOAD = False

        if config_type == 'yaml':
            if with_partition:
                yaml_load = self.from_yaml_all(config_fp)
            else:
                yaml_load = self.from_yaml(config_fp)
            for key in yaml_load:
                setattr(self, key, yaml_load[key])
                # getattr(self, key).__set_name__(self, key)
        # elif config_type == 'torch':
        #     torch_load = self.from_torch_checkpoint(config_fp)
        #     for key in torch_load:
        #         setattr(self, key, torch_load[key])

    def set_extra_attr(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def from_yaml(load_path) -> Union[dict, Any]:
        with open(load_path, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml
    
    @staticmethod
    def from_yaml_all(load_path) -> Union[dict, Any]:
        parsed_yaml = {}
        with open(load_path, 'r') as stream:
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml.update(data)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    # @staticmethod
    # def from_torch_checkpoint(load_path) -> Union[dict, Any]:
    #     checkpoint = torch.load(load_path)
    #     return checkpoint

class _Configuration(ABC):
    """Base class for configuration/specification classes."""
    def __init__(self, config: Configurator, manual_load=False) -> None:
        self._config = config
        if not manual_load:
            self._load_config()

    @abstractmethod
    def _load_config(self):
        pass

    def manual_load(self):
        self._load_config()

    @classmethod
    def from_yaml(cls, yaml_fp: str, with_partition=False):
        config = Configurator(yaml_fp, with_partition=with_partition)
        return cls(config)
    
    # @classmethod
    # def from_torch(cls, torch_fp: str):
    #     config = Configurator(torch_fp, config_type='torch')
    #     return cls(config)

### Simulation Configurations
class WarehouseSimConfiguration(_Configuration):
    """Configuration for warehouse simulation."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.scene = config.scene
        self.map_dir = config.map_dir
        self.map_file = config.map_file
        self.graph_file = config.graph_file

        self.sim_width = config.sim_width
        self.sim_height = config.sim_height

        self.scale2nn = config.scale2nn
        self.scale2real = config.scale2real

        self.image_axis = config.image_axis
        self.corner_coords = config.corner_coords

### Agent Specifications
class CircularRobotSpecification(_Configuration):
    """Specification class for circular robots."""
    def __init__(self, config: Configurator):
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts     # sampling time

        self.vehicle_width = config.vehicle_width
        self.vehicle_margin = config.vehicle_margin
        self.social_margin = config.social_margin
        self.lin_vel_min = config.lin_vel_min
        self.lin_vel_max = config.lin_vel_max
        self.lin_acc_min = config.lin_acc_min
        self.lin_acc_max = config.lin_acc_max
        self.ang_vel_max = config.ang_vel_max
        self.ang_acc_max = config.ang_acc_max
        self.ang_vel_onm_max = config.ang_vel_onm_max
        self.lidar_range = config.lidar_range

class PedestrianSpecification(_Configuration):
    """Specification class for pedestrians."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts     # sampling time

        self.human_width = config.human_width
        self.human_vel_max = config.human_vel_max
        self.human_stagger = config.human_stagger

### Controller/Planner Configurations
class MpcConfiguration(_Configuration):
    """Configuration class for MPC Trajectory Tracker Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time

        self.N_hor = config.N_hor  # control/pred horizon
        self.action_steps = config.action_steps # number of action steps (normally 1)

        self.ns = config.ns        # number of states
        self.nu = config.nu        # number of inputs
        self.nq = config.nq        # number of penalties
        self.Nother = config.Nother   # number of other robots
        self.nstcobs = config.nstcobs # dimension of a static-obstacle description
        self.Nstcobs = config.Nstcobs # number of static obstacles
        self.ndynobs = config.ndynobs # dimension of a dynamic-obstacle description
        self.Ndynobs = config.Ndynobs # number of dynamic obstacles

        self.solver_type = config.solver_type           # Determines which solver to use ([P]ANOC or [C]asadi)

        self.max_solver_time = config.max_solver_time   # [P] maximum time for the solver to run
        self.build_directory = config.build_directory   # [P] directory to store the generated solver
        self.build_type = config.build_type             # [P] type of the generated solver
        self.bad_exit_codes = config.bad_exit_codes     # [P] bad exit codes of the solver
        self.optimizer_name = config.optimizer_name     # [P] name of the generated solver

        self.lin_vel_penalty = config.lin_vel_penalty   # Cost for linear velocity control action
        self.lin_acc_penalty = config.lin_acc_penalty   # Cost for linear acceleration
        self.ang_vel_penalty = config.ang_vel_penalty   # Cost for angular velocity control action
        self.ang_acc_penalty = config.ang_acc_penalty   # Cost for angular acceleration
        self.qrpd = config.qrpd                         # Cost for reference path deviation
        self.qpos = config.qpos                         # Cost for position deviation each time step to the reference
        self.qvel = config.qvel                         # Cost for speed    deviation each time step to the reference
        self.qtheta = config.qtheta                     # Cost for heading  deviation each time step to the reference
        self.qstcobs = config.qstcobs                   # Cost for static obstacle avoidance
        self.qdynobs = config.qdynobs                   # Cost for dynamic obstacle avoidance
        self.qpN = config.qpN                           # Terminal cost; error relative to final reference position       
        self.qthetaN = config.qthetaN                   # Terminal cost; error relative to final reference heading

class GPDFConfiguration(_Configuration):
    """Configuration class for GPDF Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.d_offset_0 = config.D_OFFSET_0 # offset distance from the original boundary
        self.L = config.L               # interpolation value to determine the closeness of points
        self.rho = config.RHO
        self.rho_plot = config.RHO_PLOT
        self.uni_thres = config.UNI_THRES

class EnvContrConfiguration(_Configuration):
    """Configuration class for GPDF Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.dynamics = config.dynamics
        self.env_name = config.env_name
        self.init_state = config.init_state
        self.target = np.array(config.target)
        self.num_dyn_circle = config.num_dyn_circle
        self.radius = config.radius 
        self.threeD_controller = config.threeD_controller
        self.MCBF = config.MCBF
        self.autotune = config.autotune
        self.nominal_speed = config.nominal_speed
        self.MMP = config.MMP

        condition1 = self.threeD_controller and self.env_name != "vicon" and self.env_name != "hospital"
        condition2 = not self.threeD_controller and not self.autotune and self.env_name != "vicon" and self.env_name != "hospital"
        # self.load_default = False
        if condition1 or condition2:
            warnings.warn('Cannot find file autotune_3D_'+self.env_name+'.yaml or file handtune_2D_'+self.env_name+'.yaml, default settings loaded instead.', RuntimeWarning)
            # self.env_name = "hospital"
            self.autotune = True

        if self.autotune:
            if self.threeD_controller:
                contr_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'autotune_3D_'+self.env_name+'.yaml')
            else:
                contr_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', 'autotune_2D.yaml')

        else:
            if (self.target == [-5., 0.]).all() and self.env_name == "hospital":
                scenario = 1
            elif (self.target == [-10., 0.]).all() and self.env_name == "hospital":
                scenario = 2
            elif (self.target == [-14., 0.]).all() and self.env_name == "hospital":
                scenario = 3
            elif (self.target == [-10., -5.5]).all() and self.env_name == "hospital":
                scenario = 4
            elif (self.target == [-6.,5.5]).all() and self.env_name == "hospital":
                scenario = 5
            elif (self.target == [3., 0.]).all() and self.env_name == "vicon":
                scenario = 1
            else:
                scenario = 1
                warnings.warn("This target is not supported in handtune mode.", RuntimeWarning)

            if self.threeD_controller:
                contr_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config','handtune','3D_controller', self.env_name+'_3Dcontroller_scenario'+str(scenario)+'.yaml')
            else:
                contr_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config','handtune','2D_controller', self.env_name+'_2Dcontroller_scenario'+str(scenario)+'.yaml')
        
        contr = ContrConfiguration.from_yaml(contr_path)
        self.w = contr.w             
        self.iter = contr.iter
        self.iter_g = contr.iter_g
        self.iter_c = contr.iter_c
        self.hold_time = contr.hold_time
        self.base_margin = contr.base_margin
        self.target_range = contr.target_range
        self.om_range = contr.om_range
        self.ve = contr.ve
        self.beta_coef = contr.beta_coef
        self.om_expand_threshold = contr.om_expand_threshold
        self.dir_num = contr.dir_num
        self.custom_iter1 = contr.custom_iter1
        self.custom_iter2 = contr.custom_iter2

class ContrConfiguration(_Configuration):
    """Configuration class for GPDF Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.w = config.w             
        self.iter = config.iter
        self.iter_g = config.iter_g
        self.iter_c = config.iter_c
        self.hold_time = config.hold_time
        self.base_margin = config.base_margin
        self.target_range = config.target_range
        self.om_range = config.om_range
        self.ve = config.ve
        self.beta_coef = config.beta_coef
        self.om_expand_threshold = config.om_expand_threshold
        self.dir_num = config.dir_num
        self.custom_iter1 = config.custom_iter1
        self.custom_iter2 = config.custom_iter2


class TebConfiguration(_Configuration):
    """Configuration class for TEB Trajectory Planner Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time

        self.N_hor = config.N_hor
        self.N_obs = config.N_obs

class RPPConfiguration(_Configuration):
    """Configuration class for RPP Trajectory Planner Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time
        
        self.lt = config.lt   #lookahead gain
        self.Tk = config.Tk   #minimum threshold
        self.r_min = config.r_min #minimum radius
        self.d_prox = config.d_prox #the proximity distance to obstacles
        self.vt = config.vt #initial velocity

        self.N_hor = config.N_hor  # control/pred horizon

### Predictor Configurations
class SGANConfig(_Configuration):
    """Configuration class for SGAN."""
    def __init__(self, config: Configurator):
        super().__init__(config, manual_load=True)
        self._extra_init()

    def _extra_init(self):
        for key in self._config.args:
            self._config.set_extra_attr(key, self._config.args[key])
        self.manual_load()

    def _load_config(self):
        config = self._config

        self.dataset_name = config.dataset_name
        self.skip = config.skip
        self.delim = config.delim
        self.batch_size = config.batch_size
        self.loader_num_workers = config.loader_num_workers
        
        self.obs_len = config.obs_len
        self.pred_len = config.pred_len
        self.embedding_dim = config.embedding_dim
        self.encoder_h_dim = config.encoder_h_dim_g
        self.decoder_h_dim = config.decoder_h_dim_g
        self.mlp_dim = config.mlp_dim
        self.num_layers = config.num_layers
        self.noise_dim = tuple(config.noise_dim)
        self.noise_type = config.noise_type
        self.noise_mix_type = config.noise_mix_type
        self.pooling_type = config.pooling_type
        self.pool_every_timestep = config.pool_every_timestep
        self.dropout = config.dropout
        self.bottleneck_dim = config.bottleneck_dim
        self.neighborhood_size = config.neighborhood_size
        self.grid_size = config.grid_size
        self.batch_norm = config.batch_norm


