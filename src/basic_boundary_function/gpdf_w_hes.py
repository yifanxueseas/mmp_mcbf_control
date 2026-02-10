import os
import pathlib

import jax
import jax.numpy as jp
from jax import vmap, jit

from configs import GPDFConfiguration

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'cpu')


yaml_path = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'config', 'gpdf.yaml')
gpdf_config = GPDFConfiguration.from_yaml(yaml_path)

### Parameter settings for Matern 1/2 GPIS
L = gpdf_config.L
D_OFFSET = gpdf_config.d_offset_0


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
    return L / x ** 2

@jit
def infer_gpdf_dis(model, coords, query):
    # distance inference
    # print("hes",coords.shape)
    # print("hes",query.shape)
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)
    return mean-D_OFFSET

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
    return mean-D_OFFSET, grad

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
    return mean-D_OFFSET, grad, hessian

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

if __name__ == "__main__":
    import math

    from matplotlib import cm # type: ignore
    import matplotlib.pyplot as plt # type: ignore

    u= 20.     #x-position of the center
    v= 50.    #y-position of the center
    a= 15.     #radius on the x-axis
    b= 15.    #radius on the y-axis

    t = jp.linspace(0, 2*math.pi, 100)
    coords = jp.vstack((u+a*jp.cos(t), v+b*jp.sin(t))).T
    
    x = jp.linspace(0, 96, 96)
    y = jp.linspace(0, 96, 96)
    X, Y = jp.meshgrid(x, y)
    query = jp.vstack([X.ravel(), Y.ravel()]).T

    K = cov(coords, coords)
    k = cov(query, coords)
    y = jp.ones((coords.shape[0], 1))
    model = jp.linalg.solve(K, y)

    # distance inference
    mu = k @ model
    mean = reverting_function(mu)

    # gradient inference
    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad

    # hessian inference
    hessian = (jp.moveaxis(mu_grad, 0, 1)
               @ jp.moveaxis(mu_grad, 0, 2)
               * reverting_function_second_derivative(mu)[:, :, None])
    covariance_hessian = cov_hessian(query, coords)
    mu_hessian = ((jp.moveaxis(covariance_hessian, 1, -1) @ model)[..., 0]
                  * reverting_function_derivative(mu)[:, :, None])
    hessian += mu_hessian

    # gradient normalization
    grad_orig = jp.copy(grad)
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, mean.reshape(X.shape), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    ax.set_title('Distance field')
    plt.show()