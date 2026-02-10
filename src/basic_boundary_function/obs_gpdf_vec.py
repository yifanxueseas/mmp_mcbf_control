import sys
import math
from typing import Optional

import numpy as np
from PIL import Image # type: ignore

from .gpdf_w_hes import train_gpdf
from .gpdf_w_hes import infer_gpdf_dis, infer_gpdf_hes, infer_gpdf, infer_gpdf_grad

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


# offset = 0.7

def load_image_as_target(image_path):
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        target = np.array(img_gray, dtype=float)
        target = np.flipud(target)
        return target
    
class Obstacle:
    def __init__(self, center_x) -> None:
        self.center_x = center_x

    def dis_func(self, x):
        raise NotImplementedError

    def basis_func(self, x, norm=True):
        raise NotImplementedError

    def get_plot_blob(self):
        raise NotImplementedError
    
    
class Circle(Obstacle):
    def __init__(self, center_x, radius) -> None:
        super().__init__(center_x)
        self.radius = radius

    def dis_func(self, x):
        return np.linalg.norm(x-self.center_x,axis=1) - self.radius + 1.0
    
    def basis_func(self, x, norm=True):
        xdiff = x-self.center_x
        dist = np.linalg.norm(x-self.center_x,axis=1)
        normal = xdiff.T/dist

        R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]])
        tangent = R @ normal

        if norm:
            normal = normal / np.linalg.norm(normal, axis=0)
            tangent = tangent / np.linalg.norm(tangent, axis=0)

        return normal, tangent  #flatten vectors
    
class Star(Obstacle):
    def __init__(self, center_x, c, b) -> None:
        super().__init__(center_x)
        self.c = c
        self.b = b

    def dis_func(self, x):
        x = np.array(x)
        return np.power(((x[:,0]-self.center_x[0])**2-self.c)**2 + (x[:,1]-self.center_x[1])**4, 1/4) - np.power(self.c**2+self.b, 1/4)+1
    
    def basis_func(self, x, norm=True):
        x = np.array(x)
        dist = np.power(((x[:,0]-self.center_x[0])**2 - self.c)**2 + (x[:,1]-self.center_x[1])**4, 3/4)
        normal = np.array([0.25*(4*(x[:,0]-self.center_x[0])**3-4*self.c*(x[:,0]-self.center_x[0])), 0.25*4*(x[:,1]-self.center_x[1])**3])/dist

        R = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],[np.sin(np.pi/2), np.cos(np.pi/2)]])
        tangent = R @ normal

        if norm:
            normal = normal / np.linalg.norm(normal, axis=0)
            tangent = tangent / np.linalg.norm(tangent, axis=0)

        return normal, tangent  #flatten vectors


class GassianProcessDistanceField:
    """Gaussian Process Distance Field"""
    def __init__(self, pc_coords:Optional[np.ndarray]=None) -> None:
        if pc_coords is not None:
            self.update_gpdf(pc_coords)

    def create_obs(self, u=0, v=0, r_in=0, r_out=0, isArc=True, isLine=False, line_start=None, line_end=None, res=30):
        #Define Obstacles
        if isArc:
            arc1 = self.create_arc(u, v, r_out, int(res/3))
            arc2 = self.create_arc(u, v, r_in, int(res/3))
            arc3 = self.create_arc(u, v, 0.5*(r_in+r_out), res-int(2*int(res/3)))
            if not isLine:
                return np.hstack((arc1, arc2, arc3)).T

        line = self.create_line(line_start, line_end, res)
        if not isArc:
            return line.T
        
        pc_coords =  np.hstack((arc1, arc2, arc3, line)).T
  
        return pc_coords

    def create_arc(self, u, v, r, res, start=-0.5, end=0.5):
        """Create a point cloud of an arc.

        Args:
            u: The x-position of the center.
            v: The y-position of the center.
            r: The radius of the arc.
            res: The number of points to generate.
            start: The start angle (*pi) of the arc. Defaults to -0.5.
            end: The end angle (*pi) of the arc. Defaults to 0.5.

        Returns:
            obs: The point cloud of the arc, shape=(2, res).
        """
        t = np.linspace(start*math.pi, end*math.pi, res)
        obs = np.vstack((u+r*np.cos(t), v+r*np.sin(t)))
        return obs

    def create_line(self, l1, l2, res):
        t = np.linspace(l1, l2, int(res/l1.shape[0]))
        return t.reshape(-1,2).T
    
    def update_gpdf(self, new_pc_coords: np.ndarray):
        """The shape of new_pc_coords is (n, 2)"""
        self.pc_coords = new_pc_coords
        self.gpdf_model = train_gpdf(self.pc_coords)

    def dis_func(self, states):
        # states nxd
        # breakpoint()
        # print(self.pc_coords.shape)
        return infer_gpdf_dis(self.gpdf_model, self.pc_coords, states).flatten()+1
    
    def dis_normal_hes_func(self, states):
        # states nxd
        dis, normal, hes = infer_gpdf_hes(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal, hes

    def dis_normal_func(self, states):
        # states nxd
        dis, normal = infer_gpdf(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal

    def normal_func(self, states):
        # states nxd
        normal = infer_gpdf_grad(self.gpdf_model, self.pc_coords, states)
        return normal