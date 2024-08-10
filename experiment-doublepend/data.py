# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski
# Extension to double pendulum | 2024
# Neil Anthony

import jax
import jax.numpy as jnp
import numpy as np
from utils import to_pickle, from_pickle
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

@jax.jit
def hamiltonian_fn(coords, m1=1, m2=1, l1=1, l2=1, g=3):
    q, p = jnp.split(coords, 2)
    theta1, theta2 = q
    p_theta1, p_theta2 = p

    denominator = 2 * m2 * l1**2 * l2**2 * (m1 + m2 * jnp.sin(theta1 - theta2)**2)
    kinetic_energy = (
        m2 * l1**2 * p_theta1**2 +
        (m1 + m2) * l1**2 * l2**2 * p_theta2**2 -
        2 * m2 * l1 * l2 * p_theta1 * p_theta2 * jnp.cos(theta1 - theta2)
    ) / denominator
    
    potential_energy = (
        -(m1 + m2) * g * l1 * jnp.cos(theta1) -
        m2 * g * l2 * jnp.cos(theta2)
    )
    
    H = kinetic_energy + potential_energy
    return H

def dynamics_fn(t, coords):
    dcoords = jax.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = jnp.split(dcoords, 2)
    S = jnp.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(t_span=[0, 3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = jnp.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
    # pop key from kwargs
    subkey1, subkey2 = jax.random.split(kwargs.pop('key'))
    
    y0 = [jnp.pi / 4, jnp.pi / 4, 0.0, 0.0]

    doublepend_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = doublepend_ivp['y'][0], doublepend_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in doublepend_ivp['y'].T]
    dydt = jnp.stack(dydt).T
    dqdt, dpdt = jnp.split(dydt, 2)
    
    # add noise
    q += jax.random.normal(subkey1, shape=(*q.shape,)) * noise_std
    p += jax.random.normal(subkey2, shape=(*p.shape,)) * noise_std
    return q, p, dqdt, dpdt, t_eval

def make_pend_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    xs, dxs = [], []
    for s in range(samples):
        kwargs['key'] = jax.random.PRNGKey(s)
        x, y, dx, dy, t = get_trajectory(**kwargs)
        xs.append(jnp.stack([x, y]).T)
        dxs.append(jnp.stack([dx, dy]).T)
        
    data['x'] = jnp.concatenate(xs)
    data['dx'] = jnp.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x', 'dx']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = jnp.meshgrid(jnp.linspace(xmin, xmax, gridsize), jnp.linspace(ymin, ymax, gridsize))
    ys = jnp.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = jnp.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field


##### LOAD OR SAVE THE DATASET #####
def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-doublepend-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_pend_dataset(**kwargs)
        to_pickle(data, path)

    return data