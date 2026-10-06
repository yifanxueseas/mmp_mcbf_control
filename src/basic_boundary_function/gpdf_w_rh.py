import os
import pathlib

import jax
import jax.numpy as jp
from jax import jit, vmap
from jax.lax import scan

from configs import EnvContrConfiguration, GPDFConfiguration

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", "cpu")

"""
    GPDF with receding horizon
"""

yaml_path = os.path.join(
    pathlib.Path(__file__).resolve().parents[2], "config", "env_controller.yaml"
)
env_config = EnvContrConfiguration.from_yaml(yaml_path)
### Parameter settings for Matern 1/2 GPIS
RADIUS = env_config.radius
W = env_config.w
ITER = env_config.iter
ITER_G = env_config.iter_g
ITER_C = env_config.iter_c
ITER1 = env_config.custom_iter1
ITER2 = env_config.custom_iter2

yaml_path = os.path.join(
    pathlib.Path(__file__).resolve().parents[2], "config", "gpdf.yaml"
)
gpdf_config = GPDFConfiguration.from_yaml(yaml_path)

### Parameter settings for Matern 1/2 GPIS
L = gpdf_config.L
D_OFFSET = gpdf_config.d_offset_0
RHO = gpdf_config.rho

MAX_OBS = 8


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
    return jp.exp(-d / L)  # covariance matrix


@jit
def cov_grad(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    return (-jp.exp(-d / L) / L / d).reshape(
        (x1.shape[0], x2.shape[0], 1)
    ) * pair_diff_values


@jit
def cov_hessian(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    h = pair_diff_values.reshape(
        (x1.shape[0], x2.shape[0], 2, 1)
    ) @ pair_diff_values.reshape((x1.shape[0], x2.shape[0], 1, 2))
    hessian_constant_1 = (jp.exp(-d / L)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_2 = (1 / (L**2 * d**2) + 1 / (L * d**3)).reshape(
        (x1.shape[0], x2.shape[0], 1, 1)
    )
    hessian_constant_3 = (1 / (L * d)).reshape(
        (x1.shape[0], x2.shape[0], 1, 1)
    ) * jp.eye(2).reshape((1, 1, 2, 2))
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
    return L / (x + 1e-9) ** 2  # avoid numerical unstability


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
    hessian = (
        jp.moveaxis(mu_grad, 0, 1)
        @ jp.moveaxis(mu_grad, 0, 2)
        * reverting_function_second_derivative(mu)[:, :, None]
    )
    covariance_hessian = cov_hessian(query, coords)
    mu_hessian = (jp.moveaxis(covariance_hessian, 1, -1) @ model)[
        ..., 0
    ] * reverting_function_derivative(mu)[:, :, None]
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
    y = jp.ones((len(coords), 1))  # all boundary points are at distance 1
    model = jp.linalg.solve(K, y)
    return model


def h_gradc(x: jax.Array, xc: jax.Array):
    """_summary_

    Args:
        x: _description_
        xc: _description_

    Returns:
        _description_
    """
    # x -> 1xdim
    # xc -> Nxdim
    # return -> Nxdim
    if x.shape[0] != 1:
        diff = (-xc[:, jp.newaxis] + x).reshape(-1, xc.shape[1])
        den = jp.linalg.norm(diff, axis=1)
        dis = den.reshape(xc.shape[0], x.shape[0]) - RADIUS + 1
        grad = (diff / den.reshape(-1, 1)).reshape(xc.shape[0], x.shape[0], 2)
        return h_grad_uni(dis, grad)
    den = jp.linalg.norm(x - xc, axis=1).reshape(xc.shape[0], 1)
    dis = den - RADIUS + 1
    grad = (x - xc) / den
    return h_grad_uni(dis, grad)


def h_grad_uni(h_set, grad_set):
    # h_set:    (M,N,1)
    # grad_set: (M,N,D)
    weights = jp.exp(-RHO * h_set)  # (M,N,1)
    grad_num = jp.sum(weights * grad_set, axis=0)  # (M,N,D)  # (N,D)
    grad_den = jp.sum(weights, axis=0)  # (N,1)
    dis_uni = -1 / RHO * jp.log(grad_den) - 1  # (N,1)
    grad_uni = grad_num / grad_den  # (N,D)
    return dis_uni.squeeze(-1), grad_uni


def hes_c(x, xc):
    diff = x - xc
    dist = jp.linalg.norm(diff, axis=1).reshape(xc.shape[0], 1)
    dist32 = dist**3
    hes_00 = (1 / dist - diff[:, 0, None] ** 2 / dist32)[:, :, None]
    hes_11 = (1 / dist - diff[:, 1, None] ** 2 / dist32)[:, :, None]
    hes_01 = (-diff[:, 0, None] * diff[:, 1, None] / dist32)[:, :, None]
    hes_0 = jp.concatenate((hes_00, hes_01), axis=2)
    hes_1 = jp.concatenate((hes_01, hes_11), axis=2)
    hes = jp.concatenate((hes_0, hes_1), axis=1)
    return hes


@jit
def p_dis_grad(model, coords, query):
    """Query the high-order boundary function `p` and its gradient"""
    # input query, return grad -> Nxdim
    dis, grad, hes = infer_gpdf_hes(model, coords, query[:, :2])
    dis = dis + D_OFFSET
    dir = jp.concatenate(
        (jp.cos(query[:, 2, None]), jp.sin(query[:, 2, None])), axis=1
    )  # direction
    p = dis + W * jp.sum(
        dir * grad[:, :, 0].T, axis=1, keepdims=True
    )  # modified potential field
    p_grad_x = grad[:, :, 0].T + W * (dir[:, None, :] @ hes)[:, 0, :]
    p_grad_theta = -W * grad[0, :] * jp.sin(query[:, 2, None]) + W * grad[
        1, :
    ] * jp.cos(query[:, 2, None])
    return p, jp.concatenate((p_grad_x, p_grad_theta), axis=1)


@jit
def p_dis_grad_t(model, coords, query, xc, dxc):
    """Query the high-order boundary function `p` and its gradient"""
    # input query, return grad -> Nxdim
    dis, grad, hes = infer_gpdf_hes(model, coords, query[:, :2])
    dis = dis + D_OFFSET
    dir = jp.concatenate(
        (jp.cos(query[:, 2, None]), jp.sin(query[:, 2, None])), axis=1
    )  # direction
    p = dis + W * jp.sum(
        dir * grad[:, :, 0].T, axis=1, keepdims=True
    )  # modified potential field
    p_grad_x = grad[:, :, 0].T + W * (dir[:, None, :] @ hes)[:, 0, :]
    p_grad_theta = -W * grad[0, :] * jp.sin(query[:, 2, None]) + W * grad[
        1, :
    ] * jp.cos(query[:, 2, None])

    dht = h_grad_t(grad[:, :, 0].T, dxc)
    dgradt = W * (dir[:, None, :] @ hes @ dxc[:, :, None]).squeeze()

    return p, jp.concatenate((p_grad_x, p_grad_theta), axis=1), dht + dgradt


@jit
def p_dis_grad_c(x, xc, dxc):
    # input query -> Nxdim
    # return grad -> Nxdim
    dis, grad = h_gradc(x[:, :2], xc)
    hes = hes_c(x[:, :2], xc)
    dir = jp.concatenate((jp.cos(x[:, 2, None]), jp.sin(x[:, 2, None])), axis=1)
    p = dis + W * jp.sum(dir * grad, axis=1, keepdims=True)
    p_grad_x = grad + W * (dir[:, None, :] @ hes)[:, 0, :]
    p_grad_theta = -W * grad[:, 0] * jp.sin(x[:, 2, None]) + W * grad[:, 1] * jp.cos(
        x[:, 2, None]
    )
    return p - 1, jp.concatenate((p_grad_x, p_grad_theta.T), axis=1)


def h_grad_t(grad, total_dxc):
    # grad -> Nxdim or MxNxdim
    # xc -> Mxdim
    # return -> Nx1
    if len(grad.shape) > 2:
        return -jp.sum(
            grad @ total_dxc.reshape(total_dxc.shape[0], total_dxc.shape[1], 1), axis=2
        )
    else:
        # print(grad.shape)
        # print(total_dxc.shape)
        return -jp.sum(grad * total_dxc, axis=1)


@jit
def get_basis(n):
    # n ->Nxdim
    # e_directions -> e_numxNx3
    dim = n.shape[1]
    n1 = n[:, 0, None]
    n2 = n[:, 1, None]
    n3 = n[:, 2, None]
    e1 = jp.concatenate(
        (jp.concatenate((n2, -n1), axis=1), jp.zeros((n.shape[0], 1))), axis=1
    )[None, :, :]
    norms = jp.linalg.norm(e1, axis=2, keepdims=True)
    e1 = jp.where(norms != 0, e1, e1 / jp.min(jp.abs(e1), axis=2, keepdims=True))
    e1 /= jp.linalg.norm(e1, axis=2, keepdims=True)

    temp = jp.concatenate(
        (-n3 / (n1 + n2**2 / n1), -n2 * n3 / (n1 * (n1 + n2**2 / n1))), axis=1
    )
    e2 = jp.concatenate(
        (
            jp.concatenate(
                (-n3 / (n1 + n2**2 / n1), -n2 * n3 / (n1 * (n1 + n2**2 / n1))), axis=1
            ),
            jp.ones((n.shape[0], 1)),
        ),
        axis=1,
    )[None, :, :]
    norms = jp.linalg.norm(e2, axis=2, keepdims=True)
    e2 = jp.where(norms != 0, e2, e2 / jp.min(jp.abs(e2), axis=2, keepdims=True))
    e2 /= jp.linalg.norm(e2, axis=2, keepdims=True)
    return jp.concatenate((e1, e2), axis=0)


@jit
def rh_3D_fun_p(carry, t):
    model, coords, target, alpha, x, Pi, u = carry
    p, grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (model, coords, target, alpha, x, Pi - p[0], u), (
        model,
        coords,
        target,
        alpha,
        x,
        Pi - p[0],
        u,
    )


@jit
def rh_3D_fun_target(carry, t):
    model, coords, target, alpha, x, Pi, u = carry
    p, grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (
        model,
        coords,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    ), (
        model,
        coords,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    )


@jit
def rh_3D_fun(carry, t):
    model, coords, target, alpha, x, Pi, u = carry
    p, grad_x_p = p_dis_grad(model, coords, x)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    # u = u/jp.linalg.norm(u[:,:2], axis=1, keepdims=True)

    x = x + alpha * u
    return (
        model,
        coords,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1) - 10 * p[0],
        u,
    ), (
        model,
        coords,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1) - 10 * p[0],
        u,
    )


@jit
def rh_3D_c_fun(carry, t):
    xc, dxc, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    _, grad_x_p = p_dis_grad_c(x, xc, dxc)
    E_T = get_basis(grad_x_p)
    E_T = jp.moveaxis(E_T, 1, 0)
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    xc = xc + dxc * alpha
    return (
        xc,
        dxc,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    ), (
        xc,
        dxc,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    )


@jit
def rh_2D_fun(carry, t):
    model, coords, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    grad_x_p = infer_gpdf_grad(model, coords, x)
    E_T = jp.concatenate((grad_x_p[1, :], -grad_x_p[0, :]), axis=1)[:, None, :]
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    new_carry = (
        model,
        coords,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    )
    return new_carry, new_carry


def rh_2D_grad_fun(carry, t):
    model, coords, x, Pi, offset = carry
    h, u = infer_gpdf(model, coords, x)
    u = -u / jp.linalg.norm(u, axis=1, keepdims=True)
    alpha = (h + offset) / (ITER_G - Pi)
    x = x + alpha * u[:, :, 0].T
    new_carry = (model, coords, x, Pi + 1, offset)
    return new_carry, new_carry


@jit
def rh_2D_c_fun(carry, t):
    xc, dxc, target, alpha, x, Pi, u = carry
    N, dim = x.shape[0], x.shape[1]
    _, grad_x_p = h_gradc(x[:, :2], xc)
    grad_x_p = grad_x_p.T
    E_T = jp.concatenate((grad_x_p[1, :, None], -grad_x_p[0, :, None]), axis=1)[
        :, None, :
    ]
    E = jp.transpose(E_T, axes=(0, 2, 1))
    u_temp = E @ E_T @ (u[:, :, None])
    u = u_temp[:, :, 0]
    u = u / jp.linalg.norm(u, axis=1, keepdims=True)
    x = x + alpha * u
    xc = xc + dxc * alpha
    return (
        xc,
        dxc,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    ), (
        xc,
        dxc,
        target,
        alpha,
        x,
        Pi + alpha * jp.linalg.norm(x[:, :2] - target, axis=1),
        u,
    )


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
def receding_horizon_3D_c(xc, dxc, target, alpha, x, u):
    # x dim -> N x dim
    # u dim -> N x dim
    # target dim -> 1 x dim
    Pi = jp.zeros(x.shape[0])
    init = (xc, dxc, target, alpha, x, Pi, u)
    return scan(rh_3D_c_fun, init, None, length=ITER_C)


@jit
def receding_horizon_2D(model, coords, target, alpha, x, u):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, target, alpha, x, Pi, u)
    return scan(rh_2D_fun, init, None, length=ITER)


def receding_horizon_2D_grad(model, coords, x, offset):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    init = (model, coords, x, Pi, offset)
    # rh_2D_grad_fun(init,0)
    return scan(rh_2D_grad_fun, init, None, length=ITER_G)


# @jit
def receding_horizon_2D_c(xc, dxc, target, alpha, x, u):
    # states dim -> 2xN
    Pi = jp.zeros(x.shape[0])
    init = (xc, dxc, target, alpha, x, Pi, u)
    # rh_2D_c_fun(init,0)
    return scan(rh_2D_c_fun, init, None, length=ITER_C)


def pad_obstacles(models, coords, offset):

    offset = jp.asarray(offset)

    n_obs = len(models)

    if n_obs > MAX_OBS:
        raise ValueError(f"Too many obstacles: {n_obs}")

    dummy_model = models[0]
    dummy_coord = coords[0]

    models_pad = list(models)
    coords_pad = list(coords)

    for _ in range(MAX_OBS - n_obs):
        models_pad.append(dummy_model)
        coords_pad.append(dummy_coord)

    # pad offset
    offset_pad = jp.zeros((MAX_OBS,), dtype=offset.dtype)
    offset_pad = offset_pad.at[:n_obs].set(offset)
    obs_mask = jp.zeros(MAX_OBS, dtype=bool)
    obs_mask = obs_mask.at[:n_obs].set(True)

    return (
        tuple(models_pad),
        tuple(coords_pad),
        offset_pad[:, None, None],
        obs_mask[:, None, None],
    )


@jit
def receding_horizon_2D_uni(
    models,
    coords,
    offset,
    target,
    alpha,
    x,
    u,
):

    models, coords, offset, obs_mask = pad_obstacles(
        models,
        coords,
        offset,
    )

    Pi = jp.zeros(x.shape[0])

    done = jp.zeros(x.shape[0], dtype=bool)
    use_tangent = jp.zeros(x.shape[0], dtype=bool)

    init = (
        models,
        coords,
        obs_mask,
        offset,
        target,
        alpha,
        x,
        Pi,
        u,
        done,
        use_tangent,
    )

    return scan(rh_2D_merge_fun, init, None, length=ITER)


@jit
def receding_horizon_2D_goal(
    models,
    coords,
    offset,
    target,
    alpha,
    x,
):

    models, coords, offset, obs_mask = pad_obstacles(
        models,
        coords,
        offset,
    )

    Pi = jp.zeros(x.shape[0])
    init = (models, coords, obs_mask, offset, target, alpha, x, Pi)

    return scan(rh_2D_goal_fun, init, None, length=ITER_G)


MAX_DIS = 0.2


def rh_2D_merge_fun(carry, t):
    # jax.debug.print("ENTER rh_2D_merge_fun")
    models, coords, obs_mask, offset, target, alpha, x, Pi, u, done, use_tangent = carry

    h_list = []
    grad_list = []

    for model_i, coord_i in zip(models, coords):
        h_i, grad_i = infer_gpdf(model_i, coord_i, x)
        h_list.append(h_i + 1)
        grad_list.append(jp.swapaxes(grad_i[:, :, 0], 0, 1))

    h_set = jp.stack(h_list, axis=0)  # (M,1)
    grad_set = jp.stack(grad_list, axis=0)  # (M,2)

    h_set = jp.where(obs_mask, h_set, 1e6)
    grad_set = jp.where(obs_mask, grad_set, 0.0)
    h, grad_x_p = h_grad_uni(h_set + offset, grad_set)

    fg = target - x
    cos_theta = jp.sum(grad_x_p * fg, axis=1) / (
        jp.linalg.norm(grad_x_p, axis=1) * jp.linalg.norm(fg, axis=1) + 1e-8
    )
    # done = jp.logical_or(done, cos_theta >= 0.5)

    grad_norm = grad_x_p / (jp.linalg.norm(grad_x_p, axis=1, keepdims=True) + 1e-8)

    u_new = u - jp.sum(u * grad_norm, axis=1, keepdims=True) * grad_norm
    u_new = u_new / (jp.linalg.norm(u_new, axis=1, keepdims=True) + 1e-8)

    use_tangent = jp.logical_or(use_tangent, h <= MAX_DIS)
    x_new = x + alpha * u_new
    Pi_new = Pi + alpha * jp.linalg.norm(x_new - target, axis=1)

    x = jp.where(done[:, None], x, x_new)
    u = jp.where(done[:, None], u, u_new)
    Pi = jp.where(done, Pi, Pi_new)

    new_carry = (
        models,
        coords,
        obs_mask,
        offset,
        target,
        alpha,
        x,
        Pi,
        u,
        done,
        use_tangent,
    )

    return new_carry, new_carry


def rh_2D_goal_fun(carry, t):
    models, coords, obs_mask, offset, target, alpha, x, Pi = carry

    h_list = []

    for model_i, coord_i in zip(models, coords):
        h_i = infer_gpdf_dis(model_i, coord_i, x)
        h_list.append(h_i)

    h_set = jp.stack(h_list, axis=0)
    h_set = jp.where(obs_mask, h_set, jp.inf) + offset

    h_min = jp.min(h_set, axis=0)

    u = target - x
    u = u / (jp.linalg.norm(u, axis=1, keepdims=True) + 1e-8)

    # Switch based on h
    u = jp.where(
        h_min > MAX_DIS,  # (n,1,1)
        u,  # (n,1,2)
        jp.zeros_like(u),
    )
    x = x + alpha * u

    Pi = Pi + alpha * jp.linalg.norm(x - target, axis=1)

    new_carry = (models, coords, obs_mask, offset, target, alpha, x, Pi)

    return new_carry, new_carry