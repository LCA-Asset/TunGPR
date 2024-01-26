# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'aeroplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1
    }
elif cfgs.DATASET_NAME == 'hyperbola':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'left_hyperbola': 1,
        'right_hyperbola': 2,
        }
elif cfgs.DATASET_NAME == 'aligned_hyperbola':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'hyperbola': 1,
        }
elif cfgs.DATASET_NAME == 'r_aligned_hyperbola':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'hyperbola': 1,
        }
elif cfgs.DATASET_NAME == 'gprmax':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'rebar': 1,
        'grout_surface': 2,
        'rock_surface': 3,
        'separation': 4,
        'water_void': 5,
        'air_void': 6
        }
elif cfgs.DATASET_NAME == 'newgprmax1':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'separation': 1,
        'water_void': 2,
        'air_void': 3
        }
elif cfgs.DATASET_NAME == 'newgprmax2':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'separation': 1,
        'water_void': 2,
        'air_void': 3,
        'rebar': 4
        }
elif cfgs.DATASET_NAME == 'newgprmax3': #total=376
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'separation': 1,
        'water_void': 2,
        'air_void': 3,
        }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()