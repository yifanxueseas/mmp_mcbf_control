import os
import pathlib
import time

import jax
import jax.numpy as jp
from jax import jit, vmap
from jax.lax import scan

from configs import GPDFConfiguration, EnvContrConfiguration

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'cpu')

"""
    GPDF with receding horizon
"""

yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', 'env_controller.yaml')
env_config = EnvContrConfiguration.from_yaml(yaml_path)
### Parameter settings for Matern 1/2 GPIS
RADIUS = env_config.radius
W = env_config.w
ITER = env_config.iter
ITER_G = env_config.iter_g
ITER_C = env_config.iter_c
ITER1 = env_config.custom_iter1
ITER2 = env_config.custom_iter2

yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', 'gpdf.yaml')
gpdf_config = GPDFConfiguration.from_yaml(yaml_path)

### Parameter settings for Matern 1/2 GPIS
L = gpdf_config.L
D_OFFSET = gpdf_config.d_offset_0
RHO = gpdf_config.rho
UNI_THRES = gpdf_config.uni_thres

@jit
def pair_dist(x1: jax.Array, x2: jax.Array) -> jax.Array:
    """Distance between two points"""
    return jp.linalg.norm(x2 - x1)

@jit
def cdist(x, y):
    """Distance between each pair of the two collections of row vectors"""
    return vmap(lambda x1: vmap(lambda y1: pair_dist(x1, y1))(y))(x)


def pair_diff(x1, x2):
    return x1 - x2

@jit
def cdiff(x, y):
    return vmap(lambda x1: vmap(lambda y1: pair_diff(x1, y1))(y))(x)

@jit
def cov(x1, x2):
    d = cdist(x1, x2)
    return jp.exp(-d / L) # covariance matrix

@jit
def cov_grad(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    return (-jp.exp(-d / L) / L / d).reshape((x1.shape[0], x2.shape[0], 1)) * pair_diff_values

@jit
def cov_hessian(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    h = (pair_diff_values.reshape((x1.shape[0], x2.shape[0], 2, 1)) @ pair_diff_values.reshape((x1.shape[0], x2.shape[0], 1, 2)))
    hessian_constant_1 = (jp.exp(-d / L)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_2 = (1 / (L ** 2 * d ** 2) + 1 / (L * d ** 3)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_3 = (1 / (L * d)).reshape((x1.shape[0], x2.shape[0], 1, 1)) * jp.eye(2).reshape((1, 1, 2, 2))
    return hessian_constant_1 * (hessian_constant_2 * h + hessian_constant_3)


@jit
def reverting_function(x):
    """Revert the covariance function to get the distance"""
    return -L * jp.log(x)

@jit
def reverting_function_derivative(x):
    return -L / x

@jit
def reverting_function_second_derivative(x):
    return L / (x+1e-9) ** 2 # avoid numerical unstability

@jit
def infer_gpdf_dis(model, coords, query):
    # distance inference
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)
    return mean

@jit
def infer_gpdf(model, coords, query):
    # distance inference
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)

    # gradient inference
    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)
    return mean, grad

@jit
def infer_gpdf_hes(model, coords, query):
    # distance inference
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)

    # gradient inference
    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)

    # hessian inference
    hessian = (jp.moveaxis(mu_grad, 0, 1)
               @ jp.moveaxis(mu_grad, 0, 2)
               * reverting_function_second_derivative(mu)[:, :, None])
    covariance_hessian = cov_hessian(query, coords)
    mu_hessian = ((jp.moveaxis(covariance_hessian, 1, -1) @ model)[..., 0]
                  * reverting_function_derivative(mu)[:, :, None])
    hessian += mu_hessian
    return mean, grad, hessian

@jit
def infer_gpdf_grad(model, coords, query):
    k = cov(query, coords)
    mu = k @ model
    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)
    return grad

@jit
def train_gpdf(coords):
    coords = jp.array(coords)
    K = cov(coords, coords)
    y = jp.ones((len(coords), 1)) # all boundary points are at distance 1
    model = jp.linalg.solve(K, y)
    return model

@jit
def h_grad_uni(h_set, grad_set):
    # h_set [N_obs x 1]
    # grad_set [N_obs x 2]
    grad_num = jp.sum(jp.exp(-RHO*h_set[:, None,:])*grad_set[:,None,:], axis=0)
    grad_den = jp.sum(jp.exp(-RHO*h_set),axis=0, keepdims=True)
    dis_uni = -1/RHO*jp.log(grad_den)-1
    grad_uni = grad_num/grad_den
    return dis_uni, grad_uni


def h_gradc(x: jax.Array, xc: jax.Array):
    """_summary_

    Args:
        x: _description_
        xc: _description_

    Returns:
        _description_
    """
    #x -> 1xdim
    #xc -> Nxdim
    #return -> Nxdim
    if x.shape[0] !=1:
        diff = (-xc[:, jp.newaxis] + x).reshape(-1, xc.shape[1])
        den = jp.linalg.norm(diff, axis=1)
        dis = den.reshape(xc.shape[0], x.shape[0])- RADIUS + 1
        grad = (diff/den.reshape(-1, 1)).reshape(xc.shape[0], x.shape[0], 2)
        return h_grad_uni(dis, grad)
    den = jp.linalg.norm(x-xc, axis=1).reshape(xc.shape[0], 1)
    dis = den - RADIUS + 1
    grad = (x-xc)/den
    return h_grad_uni(dis, grad)

def hes_c(x, xc):
    diff = x-xc
    dist = jp.linalg.norm(diff, axis=1).reshape(xc.shape[0], 1)
    dist32 = dist**3
    hes_00 = (1/dist-diff[:,0,None]**2/dist32)[:,:,None]
    hes_11 = (1/dist-diff[:,1,None]**2/dist32)[:,:,None]
    hes_01 = (-diff[:,0,None]*diff[:,1,None]/dist32)[:,:,None]
    hes_0 = jp.concatenate((hes_00, hes_01),axis=2)
    hes_1 = jp.concatenate((hes_01, hes_11),axis=2)
    hes = jp.concatenate((hes_0, hes_1),axis=1)
    return hes

@jit
def p_dis_grad(model, coords, query):
    """Query the high-order boundary function `p` and its gradient"""
    #input query, return grad -> Nxdim
    dis, grad, hes = infer_gpdf_hes(model, coords, query[:,:2])
    dis = dis + D_OFFSET
    dir = jp.concatenate((jp.cos(query[:, 2, None]), jp.sin(query[:, 2, None])), axis=1) # direction
    p = dis + W*jp.sum(dir*grad[:, :, 0].T, axis=1, keepdims=True) # modified potential field
    p_grad_x = grad[:, :, 0].T + W*(dir[:, None, :]@hes)[:, 0, :]
    p_grad_theta = -W*grad[0, :]*jp.sin(query[:, 2, None]) + W*grad[1, :]*jp.cos(query[:, 2, None])
    return p, jp.concatenate((p_grad_x, p_grad_theta), axis=1)

@jit
def p_dis_grad_t(model, coords, query, xc, dxc):
    """Query the high-order boundary function `p` and its gradient"""
    #input query, return grad -> Nxdim
    dis, grad, hes = infer_gpdf_hes(model, coords, query[:,:2])
    dis = dis + D_OFFSET
    dir = jp.concatenate((jp.cos(query[:, 2, None]), jp.sin(query[:, 2, None])), axis=1) # direction
    p = dis + W*jp.sum(dir*grad[:, :, 0].T, axis=1, keepdims=True) # modified potential field
    p_grad_x = grad[:, :, 0].T + W*(dir[:, None, :]@hes)[:, 0, :]
    p_grad_theta = -W*grad[0, :]*jp.sin(query[:, 2, None]) + W*grad[1, :]*jp.cos(query[:, 2, None])
    
    dht = h_grad_t(grad[:,:,0].T,dxc)
    dgradt = W*(dir[:,None,:]@hes@dxc[:,:,None]).squeeze()
    
    return p, jp.concatenate((p_grad_x, p_grad_theta), axis=1), dht+dgradt

@jit
def p_dis_grad_c(x, xc, dxc):
    #input query -> Nxdim
    #return grad -> Nxdim
    dis, grad = h_gradc(x[:,:2], xc)
    hes = hes_c(x[:,:2], xc)
    dir = jp.concatenate((jp.cos(x[:,2,None]), jp.sin(x[:,2,None])), axis=1)
    p = dis + W*jp.sum(dir*grad, axis=1, keepdims=True)
    p_grad_x = grad+ W*(dir[:,None,:]@hes)[:,0,:]
    p_grad_theta = -W*grad[:,0]*jp.sin(x[:,2, None]) + W*grad[:,1]*jp.cos(x[:,2,None])
    return p-1, jp.concatenate((p_grad_x, p_grad_theta.T), axis=1)

def h_grad_t(grad, total_dxc):
    #grad -> Nxdim or MxNxdim
    #xc -> Mxdim
    #return -> Nx1
    if len(grad.shape)>2:
        return -jp.sum(grad @ total_dxc.reshape(total_dxc.shape[0], total_dxc.shape[1], 1), axis=2)
    else:
        # print(grad.shape)
        # print(total_dxc.shape)
        return -jp.sum(grad * total_dxc, axis=1)

@jit
def get_basis(n):
    #n ->Nxdim
    #e_directions -> e_numxNx3
    dim = n.shape[1]
    n1 = n[:,0,None]
    n2 = n[:,1,None]
    n3 = n[:,2,None]
    e1 = jp.concatenate((jp.concatenate((n2, -n1), axis=1), jp.zeros((n.shape[0],1))), axis=1)[None,:,:]
    norms = jp.linalg.norm(e1, axis=2, keepdims=True)
    e1 = jp.where(norms!=0, e1, e1/jp.min(jp.abs(e1), axis=2, keepdims=True))
    e1 /= jp.linalg.norm(e1, axis=2, keepdims=True)

    temp = jp.concatenate((-n3/(n1+n2**2/n1), -n2*n3/(n1*(n1+n2**2/n1))), axis=1)
    e2 = jp.concatenate((jp.concatenate((-n3/(n1+n2**2/n1), -n2*n3/(n1*(n1+n2**2/n1))), axis=1), jp.ones((n.shape[0],1))), axis=1)[None,:,:]
    norms = jp.linalg.norm(e2, axis=2, keepdims=True)
    e2 = jp.where(norms!=0, e2, e2/jp.min(jp.abs(e2), axis=2, keepdims=True))
    e2 /= jp.linalg.norm(e2, axis=2, keepdims=True)
    return jp.concatenate((e1,e2), axis=0)

@jit
def rh_3D_fun_p(carry,t):
    model,coords,target,alpha,x,Pi, u = carry
    p,grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (model,coords, target, alpha, x, Pi -p[0],u),  (model,coords, target, alpha,x, Pi -p[0],u)


@jit
def rh_3D_fun_target(carry,t):
    model,coords,target,alpha,x,Pi, u = carry
    p,grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (model,coords, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1),u),  (model,coords, target, alpha,x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1),u)


@jit
def rh_3D_fun(carry,t):
    model,coords,target,alpha,x,Pi, u = carry
    p,grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (model,coords, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1)-10*p[0],u),  (model,coords, target, alpha,x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1)-10*p[0],u)



@jit
def rh_3D_c_fun(carry, t):
    xc, dxc, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    _, grad_x_p = p_dis_grad_c(x,xc,dxc)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    xc = xc + dxc*alpha
    return (xc, dxc, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u), (xc, dxc, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)


# @jit
def rh_2D_fun(carry, t):
    model, coords, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    grad_x_p = infer_gpdf_grad(model, coords, x)
    E_T = jp.concatenate((grad_x_p[1,:], -grad_x_p[0,:]),axis=1)[:,None,:]
    # print("ET", E_T.shape)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model, coords, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry

# @jit
def rh_2D_merge2_fun(carry, t):
    model0, coords0, model1, coords1, offset, target, alpha, x, Pi, u = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h_set = jp.vstack((h0, h1))
    # print("grad0", grad0[None, :, 0, 0].shape)
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0]))
    # print("grad_set", grad_set.shape)
    # h_set = jp.concatenate((h0, h1), axis=0)
    # grad_set = jp.concatenate((grad_x_p0[None,:,:], grad_x_p1[None, :, :]), axis=0)
    h, grad_x_p = h_grad_uni(h_set + offset, grad_set)
    # grad_x_p = modify_uni_grad(grad_x_p, grad_set, h_set, x, target)
    # print(grad_x_p.shape)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, offset, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry



# @jit
def rh_2D_merge3_fun(carry, t):
    model0, coords0, model1, coords1, model2, coords2, offset, target, alpha, x, Pi, u = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h_set = jp.vstack((h0, h1, h2))
    # print("grad0", grad0[None, :, 0, 0].shape)
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0]))
    # print("grad_set", grad_set.shape)
    # h_set = jp.concatenate((h0, h1), axis=0)
    # grad_set = jp.concatenate((grad_x_p0[None,:,:], grad_x_p1[None, :, :]), axis=0)
    h, grad_x_p = h_grad_uni(h_set + offset, grad_set)
    # grad_x_p = modify_uni_grad(grad_x_p, grad_set, h_set, x, target)
    # print(grad_x_p.shape)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, offset,target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry

# @jit
def rh_2D_merge4_fun(carry, t):
    model0, coords0, model1, coords1, model2, coords2, model3, coords3, offset, target, alpha, x, Pi, u = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h3, grad3 = infer_gpdf(model3, coords3, x)
    h_set = jp.vstack((h0, h1, h2, h3))
    # print("grad0", grad0[None, :, 0, 0].shape)
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0], grad3[None, :, 0, 0]))
    # print("grad_set", grad_set.shape)
    # h_set = jp.concatenate((h0, h1), axis=0)
    # grad_set = jp.concatenate((grad_x_p0[None,:,:], grad_x_p1[None, :, :]), axis=0)
    print("!!!!!!!!!!!!!!!!!!!!!!!",h_set.shape)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", (offset).shape)
    h, grad_x_p = h_grad_uni(h_set+offset, grad_set)
    # grad_x_p = modify_uni_grad(grad_x_p, grad_set, h_set, x, target)
    # print(grad_x_p.shape)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, model3, coords3, offset, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry

# @jit
def rh_2D_merge5_fun(carry, t):
    (model0, coords0, model1, coords1, model2, coords2, model3, coords3, model4, coords4, 
    offset, target, alpha, x, Pi, u) = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h3, grad3 = infer_gpdf(model3, coords3, x)
    h4, grad4 = infer_gpdf(model4, coords4, x)
    h_set = jp.vstack((h0, h1, h2, h3, h4))
    # print("grad0", grad0[None, :, 0, 0].shape)
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0], grad3[None, :, 0, 0], grad4[None, :, 0, 0]))
    # print("grad_set", grad_set.shape)
    # h_set = jp.concatenate((h0, h1), axis=0)
    # grad_set = jp.concatenate((grad_x_p0[None,:,:], grad_x_p1[None, :, :]), axis=0)
    h, grad_x_p = h_grad_uni(h_set+offset, grad_set)
    # grad_x_p = modify_uni_grad(grad_x_p, grad_set, h_set, x, target)
    # print(grad_x_p.shape)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, model3, coords3, model4, coords4, 
                 offset, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry


def rh_2D_merge6_fun(carry, t):
    model0, coords0, model1, coords1, model2, coords2, model3, coords3, model4, coords4, model5, coords5, offset, target, alpha, x, Pi, u = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h3, grad3 = infer_gpdf(model3, coords3, x)
    h4, grad4 = infer_gpdf(model4, coords4, x)
    h5, grad5 = infer_gpdf(model5, coords5, x)
    h_set = jp.vstack((h0, h1, h2, h3, h4, h5))
    # print("grad0", grad0[None, :, 0, 0].shape)
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0], grad3[None, :, 0, 0], grad4[None, :, 0, 0], grad5[None, :,0, 0]))
    # print("grad_set", grad_set.shape)
    # h_set = jp.concatenate((h0, h1), axis=0)
    # grad_set = jp.concatenate((grad_x_p0[None,:,:], grad_x_p1[None, :, :]), axis=0)
    h, grad_x_p = h_grad_uni(h_set+offset, grad_set)
    # grad_x_p = modify_uni_grad(grad_x_p, grad_set, h_set, x, target)
    # print(grad_x_p.shape)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, model3, coords3, model4, coords4, model5, coords5, offset, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry

def rh_2D_merge7_fun(carry, t):
    (model0, coords0, model1, coords1, model2, coords2, model3, coords3, 
     model4, coords4, model5, coords5, model6, coords6, offset,
     target, alpha, x, Pi, u) = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h3, grad3 = infer_gpdf(model3, coords3, x)
    h4, grad4 = infer_gpdf(model4, coords4, x)
    h5, grad5 = infer_gpdf(model5, coords5, x)
    h6, grad6 = infer_gpdf(model6, coords6, x)
    h_set = jp.vstack((h0, h1, h2, h3, h4, h5, h6))
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0], grad3[None, :, 0, 0], grad4[None, :, 0, 0], grad5[None, :,0, 0], grad6[None, :,0, 0]))
    h, grad_x_p = h_grad_uni(h_set+offset, grad_set)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, model3, coords3, 
                 model4, coords4, model5, coords5, model6, coords6, offset,
                 target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry

def rh_2D_merge8_fun(carry, t):
    (model0, coords0, model1, coords1, model2, coords2, model3, coords3, 
     model4, coords4, model5, coords5, model6, coords6, model7, coords7,
     offset, target, alpha, x, Pi, u) = carry
    h0, grad0 = infer_gpdf(model0, coords0, x)
    h1, grad1 = infer_gpdf(model1, coords1, x)
    h2, grad2 = infer_gpdf(model2, coords2, x)
    h3, grad3 = infer_gpdf(model3, coords3, x)
    h4, grad4 = infer_gpdf(model4, coords4, x)
    h5, grad5 = infer_gpdf(model5, coords5, x)
    h6, grad6 = infer_gpdf(model6, coords6, x)
    h7, grad7 = infer_gpdf(model7, coords7, x)
    h_set = jp.vstack((h0, h1, h2, h3, h4, h5, h6, h7))
    grad_set = jp.vstack((grad0[None, :, 0, 0], grad1[None, :, 0, 0], grad2[None, :, 0, 0], grad3[None, :, 0, 0], grad4[None, :, 0, 0], grad5[None, :,0, 0], grad6[None, :,0, 0], grad7[None, :,0, 0]))
    h, grad_x_p = h_grad_uni(h_set+offset, grad_set)

    E_T = jp.concatenate((grad_x_p[None, :, 1], -grad_x_p[None, :, 0]),axis=1)[:,None,:]
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (model0, coords0, model1, coords1, model2, coords2, model3, coords3, 
                 model4, coords4, model5, coords5, model6, coords6, model7, coords7,
                 offset, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)
    return new_carry, new_carry


def rh_2D_grad_fun(carry, t):
    model, coords, x, Pi, offset = carry
    h, u = infer_gpdf(model, coords, x)
    u = -u/jp.linalg.norm(u, axis=1, keepdims=True)
    alpha = (h+offset)/(ITER_G-Pi)
    x = x + alpha * u[:,:,0].T
    new_carry = (model, coords, x, Pi+1, offset)
    return new_carry, new_carry

@jit
def rh_2D_c_fun(carry, t):
    xc, dxc, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    _, grad_x_p = h_gradc(x[:,:2],xc)
    print("grad_x_p", grad_x_p.shape)
    grad_x_p = grad_x_p.T
    E_T = jp.concatenate((grad_x_p[1,:,None], -grad_x_p[0,:,None]),axis=1)[:,None,:]
    print(E_T.shape)
    E = jp.transpose(E_T, axes=(0,2,1))
    u_temp = E@E_T@(u[:,:,None])
    u = u_temp[:,:,0]
    print(u.shape)
    u = u/jp.linalg.norm(u, axis=1, keepdims=True)
    print(x.shape)
    x = x + alpha * u
    return (xc, dxc, target, alpha, x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u),  (xc, dxc, target, alpha,x, Pi + alpha*jp.linalg.norm(x[:,:2]-target, axis=1), u)

@jit
def receding_horizon_3D_p(model, coords, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_3D_fun_p, init, None, length=ITER)

@jit
def receding_horizon_3D_target(model, coords, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_3D_fun_target, init, None, length=ITER)

@jit
def receding_horizon_3D(model, coords, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_3D_fun, init, None, length=ITER)

@jit
def receding_horizon_3D_custom1(model, coords, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_3D_fun, init, None, length=ITER1)

@jit
def receding_horizon_3D_custom2(model, coords, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_3D_fun, init, None, length=ITER2)

@jit
def receding_horizon_3D_c(xc,dxc, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (xc, dxc, target, alpha, x, Pi, u)
    return scan(rh_3D_c_fun, init, None, length=ITER_C)

def receding_horizon_2D(model, coords, offset, target, alpha, x, u):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    offset = jp.array(offset)[:, None]
    # offset = 0
    # print("!!!!!!!!!!!!!!!!!!!!!!!",len(offset[0]))
    if len(model) == 1:
        # start_time = time.time()
        return receding_horizon_2D_merge1(model, coords, target, alpha, x, u)
    elif len(model) == 2:
        return receding_horizon_2D_merge2(model, coords, offset, target, alpha, x, u)
    elif len(model) == 3:
        return receding_horizon_2D_merge3(model, coords, offset, target, alpha, x, u)
    elif len(model) == 4:
        return receding_horizon_2D_merge4(model, coords, offset, target, alpha, x, u)
    elif len(model) == 5:
        return receding_horizon_2D_merge5(model, coords, offset, target, alpha, x, u)
        # return receding_horizon_2D_merge5(model[0], coords[0], model[1], coords[1],model[2], coords[2],model[3], coords[3],model[4], coords[4],target, alpha, x, u)
    elif len(model) == 6:
        return receding_horizon_2D_merge6(model, coords, offset, target, alpha, x, u)
    elif len(model) == 7:
        return receding_horizon_2D_merge7(model, coords, offset, target, alpha, x, u)
    elif len(model) == 8:
        return receding_horizon_2D_merge8(model, coords, offset, target, alpha, x, u)
    else:
        raise RuntimeError("Current Jax Implementation cannot handle"+str(len(model))+" obstacles. Define rh_2D_merge"+str((len(model)))+"fun.")

# def receding_horizon_2D(model, coords, target, alpha, x, u):
#     # states dim -> 2xN
#     Pi = jp.zeros(x.shape[0])
#     if len(model) == 1:
#         init = (model[0], coords[0], target, alpha, x, Pi, u)
#         return scan(rh_2D_fun, init, None, length=ITER)
#     elif len(model) == 2:
#         init = (model[0], coords[0], model[1], coords[1], target, alpha, x, Pi, u)
#         return scan(rh_2D_merge2_fun, init, None, length=ITER)
#     elif len(model) == 3:
#         init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], target, alpha, x, Pi, u)
#         return scan(rh_2D_merge3_fun, init, None, length=ITER)
#     elif len(model) == 4:
#         init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
#                 model[3], coords[3], target, alpha, x, Pi, u)
#         return scan(rh_2D_merge4_fun, init, None, length=ITER)
#     elif len(model) == 5:
#         init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
#                 model[3], coords[3], model[4], coords[4], target, alpha, x, Pi, u)
#         return scan(rh_2D_merge5_fun, init, None, length=ITER)
#     else:
#         raise RuntimeError("Current Jax Implementation cannot handle"+str(len(model))+" obstacles. Define rh_2D_merge"+str((len(model)))+"fun.")
# @jit
# def receding_horizon_2D_merge(model_list, coords_list, target, alpha, x, u):
#     # states dim -> 2xN
#     Pi = jp.zeros(x.shape[0])
#     init = (model_list[0], coords_list[0], model_list[1], coords_list[1], target, alpha, x, Pi, u)
#     return scan(rh_2D_merge2_fun, init, None, length=ITER)


def receding_horizon_2D_grad(model, coords, x, offset):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, x, Pi, offset)
    # rh_2D_grad_fun(init,0)
    return scan(rh_2D_grad_fun, init, None, length=ITER_G)

# @jit
def receding_horizon_2D_c(xc,dxc, target, alpha, x, u):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    init = (xc, dxc, target, alpha, x, Pi, u)
    # rh_2D_c_fun(init,0)
    return scan(rh_2D_c_fun, init, None, length=ITER_C)

@jit
def receding_horizon_2D_merge1(model, coords, target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], target, alpha, x, Pi, u)
    return scan(rh_2D_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge2(model, coords, offset, target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], offset, target, alpha, x, Pi, u)
    return scan(rh_2D_merge2_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge3(model, coords, offset, target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], offset, target, alpha, x, Pi, u)
    return scan(rh_2D_merge3_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge4(model, coords, offset, target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
    model[3], coords[3], offset, target, alpha, x, Pi, u)
    return scan(rh_2D_merge4_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge5(model, coords, offset, target, alpha, x, u):
# def receding_horizon_2D_merge5(model0, coords0, model1, coords1,model2, coords2,model3, coords3,model4, coords4,target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
                model[3], coords[3], model[4], coords[4], offset, target, alpha, x, Pi, u)
    # init = (model0, coords0, model1, coords1, model2, coords2, 
    #             model3, coords3, model4, coords4, target, alpha, x, Pi, u)
    return scan(rh_2D_merge5_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge6(model, coords, offset, target, alpha, x, u):
# def receding_horizon_2D_merge5(model0, coords0, model1, coords1,model2, coords2,model3, coords3,model4, coords4,target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
                model[3], coords[3], model[4], coords[4], model[5], coords[5], offset, target, alpha, x, Pi, u)
    return scan(rh_2D_merge6_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge7(model, coords, offset, target, alpha, x, u):
# def receding_horizon_2D_merge5(model0, coords0, model1, coords1,model2, coords2,model3, coords3,model4, coords4,target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
                model[3], coords[3], model[4], coords[4], model[5], coords[5], 
                model[6], coords[6], offset, target, alpha, x, Pi, u)
    return scan(rh_2D_merge7_fun, init, None, length=ITER)

@jit
def receding_horizon_2D_merge8(model, coords, offset, target, alpha, x, u):
# def receding_horizon_2D_merge5(model0, coords0, model1, coords1,model2, coords2,model3, coords3,model4, coords4,target, alpha, x, u):
    Pi = jp.zeros(x.shape[0])
    init = (model[0], coords[0], model[1], coords[1], model[2], coords[2], 
                model[3], coords[3], model[4], coords[4], model[5], coords[5], 
                model[6], coords[6], model[7], coords[7], offset, 
                target, alpha, x, Pi, u)
    return scan(rh_2D_merge8_fun, init, None, length=ITER)