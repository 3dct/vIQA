"""Module to provide kernels for image processing."""

# Authors
# -------
# Author: Lukas Behammer
# Research Center Wels
# University of Applied Sciences Upper Austria, 2023
# CT Research Group
#
# Modifications
# -------------
# Original code, 2024, Lukas Behammer
#
# License
# -------
# BSD-3-Clause License

import numpy as np


def gsm_kernel_z():
    z = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0],
                [0, -3, 0, 3, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, -1, 0, 1, 0],
                [0, -3, 0, 3, 0],
                [0, -8, 0, 8, 0],
                [0, -3, 0, 3, 0],
                [0, -1, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0],
                [0, -3, 0, 3, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )
    return z


def gsm_kernel_y():
    y = np.rot90(gsm_kernel_z(), 1, (2, 1))
    return y


def gsm_kernel_x():
    x = np.rot90(gsm_kernel_z(), 1, (2, 0))
    return x


def gsm_kernel_yz1():
    yz1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -3, -1, 0, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 1, 3, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, -1, 0, 0],
                [0, -8, -3, 0, 0],
                [-1, -3, 0, 3, 1],
                [0, 0, 3, 8, 0],
                [0, 0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -3, -1, 0, 0],
                [0, -1, 0, 1, 0],
                [0, 0, 1, 3, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )
    return yz1


def gsm_kernel_yz2():
    yz2 = np.rot90(gsm_kernel_yz1(), 1, (2, 1))
    return yz2


def gsm_kernel_xz1():
    xz1 = np.rot90(gsm_kernel_yz1(), 1, (1, 0))
    return xz1


def gsm_kernel_xz2():
    xz2 = np.rot90(gsm_kernel_yz1(), 1, (0, 1))
    return xz2


def gsm_kernel_xy1():
    xy1 = np.rot90(gsm_kernel_yz1(), 1, (2, 0))
    return xy1


def gsm_kernel_xy2():
    xy2 = np.rot90(gsm_kernel_yz1(), 1, (0, 2))
    return xy2


def gsm_kernel_xyz1():
    xyz1 = np.array(
        [
            [
                [0, -1, 0, 0, 0],
                [1, 0, 0, -1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, -1],
                [0, 0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 3, 0, -3, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, -1, 0, 0],
                [0, 0, 0, -8, 0],
                [1, 0, 0, 0, -1],
                [0, 8, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, -3, 0, 0],
                [0, 3, 0, -3, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, -1, 0, 0, 0],
                [1, 0, 0, -1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, -1],
                [0, 0, 0, 1, 0],
            ],
        ]
    )
    return xyz1


def gsm_kernel_xyz2():
    xyz2 = np.rot90(gsm_kernel_xyz1(), 1, (1, 2))
    return xyz2


def gsm_kernel_xyz3():
    xyz3 = np.rot90(gsm_kernel_xyz1(), 1, (0, 1))
    return xyz3


def gsm_kernel_xyz4():
    xyz4 = np.rot90(gsm_kernel_xyz1(), 1, (0, 2))
    return xyz4


def sobel_kernel_z():
    z = np.array(
        [
            [
                [0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0],
            ],
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            [
                [0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0],
            ],
        ]
    )
    return z


def sobel_kernel_y():
    y = np.rot90(sobel_kernel_z(), 1, (2, 1))
    return y


def sobel_kernel_x():
    x = np.rot90(sobel_kernel_z(), 1, (2, 0))
    return x


def prewitt_kernel_z():
    z = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ]
    )
    return z


def prewitt_kernel_y():
    y = np.rot90(prewitt_kernel_z(), 1, (2, 1))
    return y


def prewitt_kernel_x():
    x = np.rot90(prewitt_kernel_z(), 1, (2, 0))
    return x


def scharr_kernel_z():
    z = np.array(
        [
            [
                [0, 0, 0],
                [-47, 0, 47],
                [0, 0, 0],
            ],
            [
                [-47, 0, 47],
                [-162, 0, 162],
                [-47, 0, 47],
            ],
            [
                [0, 0, 0],
                [-47, 0, 47],
                [0, 0, 0],
            ],
        ]
    )
    return z


def scharr_kernel_y():
    y = np.rot90(scharr_kernel_z(), 1, (2, 1))
    return y


def scharr_kernel_x():
    x = np.rot90(scharr_kernel_z(), 1, (2, 0))
    return x


# 2D kernels
#####################################################################
def prewitt_kernel_2d_x():
    x = np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]
    )
    return x


def prewitt_kernel_2d_y():
    y = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ]
    )
    return y


def sobel_kernel_2d_x():
    x = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]
    )
    return x


def sobel_kernel_2d_y():
    y = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]
    )
    return y


def scharr_kernel_2d_x():
    x = np.array(
        [
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3],
        ]
    )
    return x


def scharr_kernel_2d_y():
    y = np.array(
        [
            [3, 10, 3],
            [0, 0, 0],
            [-3, -10, -3],
        ]
    )
    return y


def gsm_kernel_2d_y():
    y = np.array(
        [
            [0, 0, 0, 0, 0],
            [-1, -3, -8, -3, -1],
            [0, 0, 0, 0, 0],
            [1, 3, 8, 3, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    return y


def gsm_kernel_2d_x():
    x = np.array(
        [
            [0, -1, 0, 1, 0],
            [0, -3, 0, 3, 0],
            [0, -8, 0, 8, 0],
            [0, -3, 0, 3, 0],
            [0, -1, 0, 1, 0],
        ]
    )
    return x


def gsm_kernel_2d_xy():
    xy = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, -3, -8, 0],
            [-1, -3, 0, 3, 1],
            [0, 8, 3, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    return xy


def gsm_kernel_2d_yx():
    yx = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -8, -3, 0, 0],
            [1, 3, 0, -3, -1],
            [0, 0, 3, 8, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    return yx
