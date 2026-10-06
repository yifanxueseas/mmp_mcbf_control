from math import cos, sin
from typing import Optional, TypedDict, List, Union
import time

import numpy as np
from scipy.spatial.transform import Rotation as R # type: ignore

from .env import Env
from .gpdf_w_rh import receding_horizon_2D, receding_horizon_2D_c, receding_horizon_2D_grad, ITER, ITER_G, ITER_C, ITER1, ITER2 # type: ignore
from .gpdf_w_rh import receding_horizon_3D_c, receding_horizon_3D_custom1, receding_horizon_3D_custom2, receding_horizon_3D_p, receding_horizon_3D_target# type: ignore
from .gpdf_w_rh import receding_horizon_2D_uni, receding_horizon_2D_goal, MAX_DIS 

class OnMan_Approx:
    def __init__(self, env: Env, w:Optional[float]=0.1, hold_time=1) -> None:
        self.env = env
        self.min_indices = np.zeros(len(self.env.gpdf_set),).astype('int')
        self.counter = hold_time*np.ones(len(self.env.gpdf_set),)
        self.hold_time = hold_time*np.ones(len(self.env.gpdf_set),)
        self.w = w

    def get_basis_direction(self, n, e_num):
        #n ->Nxdim
        #e_directions -> e_numxNxdim

        dim = n.shape[1]
        e_directions = np.zeros((e_num, n.shape[0],dim))
        temp = np.concatenate((n[:,1].reshape(n.shape[0],1), -n[:,0].reshape(n.shape[0],1)), axis=1)
        
        if dim == 2:
            e_directions[0,:,:] = temp
            e_directions[1,:,:] = -temp
            return e_directions

        e_directions[0,:,:] = np.concatenate((temp,  np.zeros((n.shape[0],1))),axis =1)
        e0 = e_directions[0].reshape((n.shape[0],dim,1))
        for i in range(e_num-1):
            r = R.from_rotvec(-2*(i+1)*np.pi/e_num*n)
            Rot = r.as_matrix()
            e_directions[i+1] = (Rot@e0).reshape((n.shape[0],dim))
        return e_directions
    
    def theta_filter(self, e, rob_theta, extra_filter=True, uni_dir = False):
        e = e.flatten()
        if abs(e[2])>0.9 or abs(e[2])<0.05:
            return False
        
        if uni_dir and e[2]<0:
            return False
        if not extra_filter:
            return True
        e_xy = e[:2]
        e_xy_dir = np.arctan2(e_xy[1], e_xy[0])
        delta_theta = e_xy_dir - rob_theta
        if abs(delta_theta) > np.pi:
            delta_theta = -np.sign(delta_theta)*(2*np.pi-abs(delta_theta))
        if delta_theta*e[2]>=-0.015:
            return True
        else:
            return False

    def geodesic_approx_phi_3D(self, x, n, e_num, target, beta, uni_dir=True, onM=1, checking_mode=False, even=True,  e_prev=None, extra_filter=False):
        # x: robot x, y position  (dim -> N x dim)
        # n: gradient of the obstacle to be geodesic approximated (dim -> N x dim)
        # onM: index of the obstacle (int)
        # on_boundary: whether the intial locations of geodesic approximation should be on obstacle boundaries
        # checking_mode: return x_i and p_i from geodesic approximation
        # e_prev: e_selected from the previous iteration. 
        # return e_selected: e vector corresponding to the smallest pi cost (dim -> N x dim)

        e_directions = self.get_basis_direction(n,e_num)

        _gpdf = self.env.gpdf_set[onM]

        if self.counter[onM]<self.hold_time[onM]:
            self.counter[onM] = self.counter[onM]+1
            min_indices = self.min_indices
            pi_list = None
            xi_list = None
        else:
            pi_list = 10000*np.ones((e_num,x.shape[0]))
            if ITER != "NA":
                xi_list = np.zeros((e_num,int(2*ITER),1,3))
            else:
                xi_list = np.zeros((e_num,ITER1+ITER2,1,3))
            for i in range(e_num):
                if self.theta_filter(e_directions[i],x[:,2], extra_filter, uni_dir=uni_dir):
                    if ITER != "NA":
                        carry, stack= receding_horizon_3D_p(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x, e_directions[i])
                        _,_, _,_, x_temp, pi_list[i],u = carry
                        _,_,_,_,xi_list[i,:ITER,:,:],_,_ = stack
                        carry, stack= receding_horizon_3D_target(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x_temp, u)
                        _,_, _,_,_, pi,_ = carry
                        _,_,_,_,xi_list[i,ITER:int(2*ITER),:,:],_,_ = stack
                        pi_list[i] = 6*beta*pi_list[i] + pi
                    else:
                        carry, stack= receding_horizon_3D_custom1(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x, e_directions[i])
                        _,_, _,_, x_temp, _,u = carry
                        _,_,_,_,xi_list[i,:ITER1,:,:],_,_ = stack
                        carry, stack= receding_horizon_3D_custom2(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x_temp, u)
                        _,_, _,_,_, pi_list[i],_ = carry
                        _,_,_,_,xi_list[i,ITER1:ITER1+ITER2,:,:],_,_ = stack
            
            if not np.any(e_prev == None):
                p_mean = np.mean(pi_list[pi_list != 10000])
                pi_list = pi_list + np.abs(e_prev[None,:,2]-e_directions[:,:,2]) # XXX Add new cost term for seletcing the d_Theta direction
            min_indices = np.argmin(pi_list, axis=0)
            self.counter[onM] = 1
            self.min_indices = min_indices

        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0],1)]

        if checking_mode:
            return e_selected.reshape((x.shape[0],3)), pi_list, xi_list

        return e_selected.reshape((x.shape[0],3))

    

    def geodesic_approx_phi_3D_c(self, x, n, e_num, target, beta, uni_dir=True, checking_mode=False):
        # x->Nxdim
        # n ->Nxdim
        # return -> Nxdim
        e_directions = self.get_basis_direction(n,e_num, uni_dir=uni_dir)

        if self.counter[0]<self.hold_time[0]:
            self.counter[0] = self.counter[0]+1
            min_indices = self.min_indices

        else:
            pi_list = 10000*np.ones((e_num,x.shape[0]))
            xi_list = np.zeros((e_num,30,1,3))
            for i in range(e_num):
                if abs(e_directions[i,:,2])>0.06:
                    carry, stack= receding_horizon_3D_c(self.env.xc, self.env.dxc, target, beta,x, e_directions[i])
                    xc_pred,_, _,_, x_temp, _,u = carry
                    _,_,_,_,xi_list[i,:15,:,:],_,_ = stack
                    carry, stack= receding_horizon_3D_c(xc_pred, self.env.dxc, target, beta, x_temp, u)
                    _,_, _,_,_, pi_list[i],_ = carry
                    _,_,_,_,xi_list[i,15:30,:,:],_,_ = stack
            
            min_indices = np.argmin(pi_list, axis=0)
            self.counter[0] = 1
            self.min_indices = min_indices

        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0],1)]

        if checking_mode:
            return pi_list, xi_list

        return e_selected.reshape((x.shape[0],3))

    def geodesic_approx_phi_2D_uni(self, x, n, target, beta, onM, checking_mode=False, on_boundary=False, e_prev=None, target_range=3.0):
        # x: robot x, y position  (dim -> N x dim)
        # n: gradient of the obstacle to be geodesic approximated (dim -> N x dim)
        # onM: list/range of obstacle indices to merge via soft-min (h_grad_uni), instead of
        #     committing to a single nearest obstacle as geodesic_approx_phi_2D does.
        # on_boundary: whether the intial locations of geodesic approximation should be on obstacle boundaries
        # checking_mode: return x_i and p_i from geodesic approximation
        # e_prev: e_selected from the previous iteration.
        # target_range: within this distance of the target, fall back to a straight-line-to-target cost
        #     instead of the geodesic rollout cost.
        # return e_selected: e vector corresponding to the smallest pi cost (dim -> N x dim)

        n = n.reshape(1, 2) / np.linalg.norm(n)
        e_directions = self.get_basis_direction(n, 2)
        pi_list = np.zeros((2, x.shape[0]))
        if on_boundary:
            ITER_g = ITER_G
        else:
            ITER_g = 0
        xi_list = np.zeros((ITER_g + ITER, 2, 3))

        gpdf_model = [self.env.gpdf_set[i].gpdf_model for i in onM]
        pc_coords = [self.env.gpdf_set[i].pc_coords for i in onM]
        all_offsets = np.concatenate((self.env.mmp_offset, self.env.env_offset), axis=0)
        offset = [all_offsets[i] for i in onM]
        x_init = np.repeat(x, 2, axis=0)

        if on_boundary:
            carry, stack = receding_horizon_2D_goal(gpdf_model, pc_coords, offset, target, beta, x_init)
            *_, x_onB, _ = carry
            *_, xi_list[:ITER_g, :, :2], _ = stack
        else:
            x_onB = x_init

        h, grad_onB = self.env.h_grad_uni(x_onB[0].reshape(1, 2), idx=range(len(self.env.gpdf_set)))

        if h <= MAX_DIS:
            dot = (e_directions * grad_onB).sum(axis=-1)
            if (abs(dot) > 0.98).all():
                grad_onB = grad_onB / np.linalg.norm(grad_onB)
                e_directions = self.get_basis_direction(grad_onB, 2)

            carry, stack = receding_horizon_2D_uni(
                gpdf_model, pc_coords, offset, target, beta, x_onB, e_directions[:, 0, :])
            *_, x_temp, pi_list, u, _, _ = carry
            *_, xi_list[ITER_g:ITER_g + ITER, :, :2], _, _, _, _ = stack

        xi_list = np.expand_dims(np.transpose(xi_list, (1, 0, 2)), axis=2)
        pi_list = np.array(pi_list)

        dis2goal = np.linalg.norm(x_init[0].reshape(1, 2) - target)
        if dis2goal > target_range:
            if pi_list[0] == pi_list[1]:
                return None
        else:
            goal_dir = x - target
            goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
            pi_list = np.sum(e_directions[:, 0, :] * goal_dir, axis=1)

        if e_prev is not None:
            p_mean = np.mean(pi_list)
            pi_list[0] = pi_list[0] - max(p_mean / 20, 0.5) * e_prev @ e_directions[0].T * np.ones((x.shape[0],))
            pi_list[1] = pi_list[1] - max(p_mean / 20, 0.5) * e_prev @ e_directions[1].T * np.ones((x.shape[0],))

        min_indices = np.argmin(pi_list, axis=0)
        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0], 1)]

        if checking_mode:
            return e_selected.reshape((x.shape[0], 2)), pi_list, xi_list
        return e_selected.reshape((x.shape[0], 2))


    def geodesic_approx_phi_2D_c(self, x, n, target, beta, checking_mode=False):
        # states dim -> 2xN
        # breakpoint()
        e_directions = self.get_basis_direction(n,2)
        pi_list = np.zeros((2,x.shape[0]))
        xi_list = np.zeros((2,ITER_C*2,1,3))

        carry, stack = receding_horizon_2D_c(self.env.xc, self.env.dxc, target, beta, x, e_directions[0])
        xc_pred,_, _,_, x_temp, _,u = carry

        _,_,_,_,xi_list[0,0:ITER_C,:,:2],_,_ = stack
        carry, stack= receding_horizon_2D_c(xc_pred, self.env.dxc, target, beta, x_temp, u)
        _,_, _,_,_, pi_list[0],_ = carry
        _,_,_,_,xi_list[0,ITER_C:2*ITER_C,:,:2],_,_ = stack

        carry, stack= receding_horizon_2D_c(self.env.xc, self.env.dxc, target, beta, x, e_directions[1])
        xc_pred,_, _,_, x_temp, _,u = carry
        _,_,_,_,xi_list[1,0:ITER_C,:,:2],_,_ = stack
        carry, stack= receding_horizon_2D_c(xc_pred, self.env.dxc, target, beta, x_temp, u)
        _,_, _,_,_, pi_list[1],_ = carry
        _,_,_,_,xi_list[1,ITER_C:2*ITER_C,:,:2],_,_ = stack

        min_indices = np.argmin(pi_list, axis=0)
        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0],1)]
        
        if checking_mode:
            return e_selected.reshape((x.shape[0],2)), pi_list, xi_list
        return e_selected.reshape((x.shape[0],2))
    
    
    def p_dis_grad_c(self, x):
        # compute high order control barrier function h_HO for 2D circular obstacles
        # x: robot state (x, y, theta) -> Nxdim
        # return distance -> Nx1, gradient -> Nx2, and H_HO, h_HO's time derivate -> Nx1
        
        dis, grad = self.env.h_gradc(x[:,:2])
        hes = self.env.hes_c(x[:,:2])
        dir = np.concatenate((np.cos(x[:,2,None]),np.sin(x[:,2,None])),axis=1)
        p = dis + self.w*np.sum(dir*grad, axis=1, keepdims=True)
        p_grad_x = grad+ self.w*(dir[:,None,:]@hes)[:,0,:]
        p_grad_theta = -self.w*grad[:,0]*np.sin(x[:,2, None])+self.w*grad[:,1]*np.cos(x[:,2,None])

        dht = np.array(self.env.h_grad_t(grad,self.env.dxc)).reshape(-1,1)
        dgradt = self.w*(dir[:,None,:]@hes@self.env.dxc[:,:,None]).reshape(-1,1)
        return p-1, np.concatenate((p_grad_x, p_grad_theta.T), axis=1), dht+dgradt
    