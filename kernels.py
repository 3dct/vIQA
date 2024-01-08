import numpy as np


def gsm_kernel_z():
    z = np.array([[[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0],
                   [0, -3, 0, 3, 0],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0]],

                  [[0, -1, 0, 1, 0],
                   [0, -3, 0, 3, 0],
                   [0, -8, 0, 8, 0],
                   [0, -3, 0, 3, 0],
                   [0, -1, 0, 1, 0]],

                  [[0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0],
                   [0, -3, 0, 3, 0],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0]],

                  [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 0, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]])
    return z


def gsm_kernel_y():
    y = np.rot90(gsm_kernel_z(), 1, (2, 1))
    return y


def gsm_kernel_x():
    x = np.rot90(gsm_kernel_z(), 1, (2, 0))
    return x


def gsm_kernel_yz1():
    yz1 = np.array([[[0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0],
                     [0, -3, -1, 0, 0],
                     [0, -1, 0, 1, 0],
                     [0, 0, 1, 3, 0],
                     [0, 0, 0, 0, 0]],

                    [[0, 0, -1, 0, 0],
                     [0, -8, -3, 0, 0],
                     [-1, -3, 0, 3, 1],
                     [0, 0, 3, 8, 0],
                     [0, 0, 1, 0, 0]],

                    [[0, 0, 0, 0, 0],
                     [0, -3, -1, 0, 0],
                     [0, -1, 0, 1, 0],
                     [0, 0, 1, 3, 0],
                     [0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0],
                     [0, -1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0]]])
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


def sobel_kernel_z():
    z = np.array([[[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]],

                  [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]],

                  [[0, 0, 0],
                   [-1, 0, 1],
                   [0, 0, 0]]])
    return z


def sobel_kernel_y():
    y = np.rot90(sobel_kernel_z(), 1, (2, 1))
    return y


def sobel_kernel_x():
    x = np.rot90(sobel_kernel_z(), 1, (2, 0))
    return x


def prewitt_kernel_z():
    z = np.array([[[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]],

                  [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]],

                  [[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]])
    return z


def prewitt_kernel_y():
    y = np.rot90(prewitt_kernel_z(), 1, (2, 1))
    return y


def prewitt_kernel_x():
    x = np.rot90(prewitt_kernel_z(), 1, (2, 0))
    return x


def scharr_kernel_z():
    z = np.array([[[0, 0, 0],
                   [-47, 0, 47],
                   [0, 0, 0]],

                  [[-47, 0, 47],
                   [-162, 0, 162],
                   [-47, 0, 47]],

                  [[0, 0, 0],
                   [-47, 0, 47],
                   [0, 0, 0]]])
    return z


def scharr_kernel_y():
    y = np.rot90(scharr_kernel_z(), 1, (2, 1))
    return y


def scharr_kernel_x():
    x = np.rot90(scharr_kernel_z(), 1, (2, 0))
    return x
