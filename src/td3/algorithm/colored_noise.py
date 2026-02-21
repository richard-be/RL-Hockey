# NOTE: taken from Richard
import colorednoise as cn
import numpy as np

def reset_noise(i, noise, beta, samples, action_shape):
    """
    Docstring for reset_noise
    
    :param i: Description
    :param noise: Description
    :param beta: Description
    :param samples: Description
    :param action_shape: Description
    """
    noise[i] = np.array([cn.powerlaw_psd_gaussian(beta, samples) for _ in range(action_shape)])
    return noise