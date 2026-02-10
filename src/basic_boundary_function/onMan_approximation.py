from math import cos, sin
from typing import Optional, TypedDict, List, Union
import time

import numpy as np
from scipy.spatial.transform import Rotation as R # type: ignore

from .env import Env
from .gpdf_w_rh import receding_horizon_2D, receding_horizon_2D_c, receding_horizon_2D_grad, ITER, ITER_G, ITER_C, ITER1, ITER2 # type: ignore
from .gpdf_w_rh import receding_horizon_3D_c, receding_horizon_3D_custom1, receding_horizon_3D_custom2, receding_horizon_3D_p, receding_horizon_3D_target# type: ignore


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

    
    def geodesic_approx_phi_3D_handtuned(self, x, n, e_num, target, beta, uni_dir=True, onM=1, checking_mode=False, e_prev=None):
        # x->Nxdim
        # n ->Nxdim
        # return -> Nxdim

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
                if abs(e_directions[i,0,2])>0.9 or abs(e_directions[i,0,2])<=0.06 or (uni_dir and e_directions[i,0,2]<0):
                    continue
                if ITER != "NA":
                    carry, stack= receding_horizon_3D_target(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x, e_directions[i])
                    _,_, _,_, x_temp, _,u = carry
                    _,_,_,_,xi_list[i,:ITER,:,:],_,_ = stack
                    carry, stack= receding_horizon_3D_target(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x_temp, u)
                    _,_, _,_,_, pi_list[i],_ = carry
                    _,_,_,_,xi_list[i,ITER:int(2*ITER),:,:],_,_ = stack
                else:
                    carry, stack= receding_horizon_3D_custom1(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x, e_directions[i])
                    _,_, _,_, x_temp, _,u = carry
                    _,_,_,_,xi_list[i,:ITER1,:,:],_,_ = stack
                    carry, stack= receding_horizon_3D_custom2(_gpdf.gpdf_model, _gpdf.pc_coords, target, beta, x_temp, u)
                    _,_, _,_,_, pi_list[i],_ = carry
                    _,_,_,_,xi_list[i,ITER1:ITER1+ITER2,:,:],_,_ = stack

            if not np.any(e_prev == None):
                if self.env.env_name=="vicon":
                    pi_list = pi_list + np.abs(np.sign(e_prev[None,:,2])-np.sign(e_directions[:,:,2])) # XXX Add new cost term for seletcing the d_Theta direction
                else:
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

    def geodesic_approx_phi_2D(self, x, n, target, beta, onM, checking_mode=False, on_boundary=False, e_prev = None):
        # x: robot x, y position  (dim -> N x dim)
        # n: gradient of the obstacle to be geodesic approximated (dim -> N x dim)
        # onM: index of the obstacle (int)
        # on_boundary: whether the intial locations of geodesic approximation should be on obstacle boundaries
        # checking_mode: return x_i and p_i from geodesic approximation
        # e_prev: e_selected from the previous iteration. 
        # return e_selected: e vector corresponding to the smallest pi cost (dim -> N x dim)

        # _, n = self.env.h_grad_uni(x, idx=onM)
        n = n.reshape(1,2)/np.linalg.norm(n)
        e_directions = self.get_basis_direction(n,2)
        pi_list = np.zeros((2,x.shape[0]))
        if on_boundary:
            ITER_g = ITER_G
        else:
            ITER_g = 0
        xi_list = np.zeros((2,ITER_g+ITER*2,1,3))


        if on_boundary and len(onM)==1:
            offset = self.env.mmp_offset[onM[0]] if onM[0] < self.env.num_dyn_mmp else self.env.env_offset[onM[0]-self.env.num_dyn_mmp]
            carry, stack= receding_horizon_2D_grad(self.env.gpdf_set[onM[0]].gpdf_model, self.env.gpdf_set[onM[0]].pc_coords, x, offset)
            _,_, x_onB, _,_ = carry
            _,_,xi_list[0,:ITER_g,:,:2],_,_ = stack
            _,_,xi_list[1,:ITER_g,:,:2],_,_ = stack
        else:
            x_onB = x

        gpdf_model = [self.env.gpdf_set[i].gpdf_model for i in onM]
        pc_coords = [self.env.gpdf_set[i].pc_coords for i in onM]
        all_offsets = np.concatenate((self.env.mmp_offset, self.env.env_offset), axis=0)
        # breakpoint()
        offset = [all_offsets[i] for i in onM]

        start_time = time.time()
        carry, stack= receding_horizon_2D(gpdf_model, pc_coords, offset, target, beta, x_onB, e_directions[0])
        *_, x_temp, _,u = carry
        *_,xi_list[0,ITER_g:ITER_g+ITER,:,:2],_,_ = stack
        carry, stack= receding_horizon_2D(gpdf_model, pc_coords, offset, target, beta, x_temp, u)
        *_, pi,_ = carry
        pi_list[0] = pi_list[0]+pi
        *_,xi_list[0,ITER_g+ITER:ITER_g+2*ITER,:,:2],_,_ = stack

        carry, stack= receding_horizon_2D(gpdf_model, pc_coords, offset, target, beta, x_onB, e_directions[1])
        *_, x_temp, _,u = carry
        *_,xi_list[1,ITER_g:ITER_g+ITER,:,:2],_,_ = stack
        carry, stack= receding_horizon_2D(gpdf_model, pc_coords, offset, target, beta, x_temp, u)
        *_, pi,_ = carry
        pi_list[1] = pi_list[1] + pi
        *_,xi_list[1,ITER_g+ITER:ITER_g+2*ITER,:,:2],_,_ = stack
        # # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",time.time()-start_time)


        if e_prev is not None:
            #penalize the e_direaction candidates proportionally to their angle difference with e_prev
            p_mean = np.mean(pi_list)
            pi_list[0] = pi_list[0]-p_mean/10*e_prev @e_directions[0].T*np.ones((x.shape[0],))
            pi_list[1] = pi_list[1]-p_mean/10*e_prev @e_directions[1].T*np.ones((x.shape[0],))


        if (xi_list[0,:,:,0]<-16.5).any() or (xi_list[0,:,:,0]>-4.5).any() or (abs(xi_list[0,:,:,1])>6).any():
            pi_list[0] = pi_list[0]+100
        
        if (xi_list[1,:,:,0]<-16.5).any() or (xi_list[1,:,:,0]>-4.5).any() or (abs(xi_list[1,:,:,1])>6).any():
            pi_list[1] = pi_list[1]+100
        min_indices = np.argmin(pi_list, axis=0)
        e_directions = np.moveaxis(e_directions, 1, 0)
        e_selected = e_directions[np.arange(x.shape[0])[:, None], min_indices.reshape(x.shape[0],1)]


        if checking_mode:
            return e_selected.reshape((x.shape[0],2)), pi_list, xi_list
        return e_selected.reshape((x.shape[0],2))

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
    