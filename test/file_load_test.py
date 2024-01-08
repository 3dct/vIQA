from utils import load_data, _load_data_from_disk
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch


file_dir = '../../samples'
# file_name = 'Catec_Two_PlateIQI_20um_810proj_220kV_Rayscan-SimCT_800x800x1000_16bit.raw'
file_name = 'AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.raw'
# file_name = 'AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated.raw'
# file_name = 'AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976_watershed_bin.raw'
# file_name = 'Catec_Two_PlateIQI_20um_810proj_220kV_Rayscan-SimCT_800x800x1000_8bit.raw'
# file_name = 'Catec_Two_PlateIQI_20um_810proj_220kV_Rayscan-SimCT_800x800x1000.raw'
# file_name = 'Catec_Two_PlateIQI_20um_810proj_220kV_Rayscan-SimCT.raw'

# data_file_ext = ".raw"
# header_file_ext = ".mhd"

file_name_split = os.path.splitext(file_name)
file_name_head = file_name_split[0]
file_ext = file_name_split[-1]
file_path = os.path.join(file_dir, file_name)
file_dir = os.path.dirname(file_path)
file_name = os.path.basename(file_path)

dim_search_result = re.search("(\d+(x|_)\d+(x|_)\d+)", file_name_head)
if dim_search_result is not None:
    dim = dim_search_result.group(1)
else:
    raise Exception("No dimension found")
bit_depth_search_result = re.search("(\d{1,2}bit)", file_name_head)
if bit_depth_search_result is not None:
    bit_depth = bit_depth_search_result.group(1)
else:
    raise Exception("No bit depth found")

if bit_depth == '16bit':
    type = np.ushort
    print(type)
elif bit_depth == '8bit':
    type = np.ubyte
    print(type)
else:
    print("error")

dim_size = re.split("x|_", dim)
print(dim_size)

# print(dim_size)

# print(type(file_path) is str)
# img = load_data_from_disk(file_dir, file_name)

# img_arr = load_data(file_path)

# print(img_arr.shape)

# plt.imshow(img_arr[500], cmap="gray")
# plt.show()
pass
