import cProfile
import pstats
import jax
import jax.numpy as jnp
from jax import grad, jit

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from data import make_pend_dataset

cProfile.run('make_pend_dataset()', 'profile_stats')

# Print the profiling results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)