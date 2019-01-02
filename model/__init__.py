"""Set up paths for dataset.py"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add dataset.py to PYTHONPATH
dataset_path = osp.join(this_dir, '..')
print('add path: %s' % dataset_path)
add_path(dataset_path)
