from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import vot
import sys
import os
import argparse
import os.path as osp
import importlib
import numpy as np
print ("0")

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


del os.environ['MKL_NUM_THREADS']

from pytracking.evaluation import Tracker
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker

class VOT_SIAM_Wrapper(object):

    def __init__(self, tracker, imagefile, selection):

        #selection = Rectangle(selection.x, selection.y, selection.width, selection.height)
        
        self._tracker = tracker
        image = self._tracker._read_image(imagefile)
        init_state = selection
        self._tracker.initialize(image, init_state)

    def track(self, image, i):
        image = self._tracker._read_image(imagefile)
        res_rect = self._tracker.track(image)
        print("res_rect",res_rect)
        tracked_bb = np.array(res_rect).astype(int)
        print ("tracked_bb ",tracked_bb )
        return vot.Rectangle(res_rect[0], res_rect[1], res_rect[2], res_rect[3])

#CURRENT_DIR = osp.dirname(__file__)
#sys.path.append(osp.join(CURRENT_DIR))
#checkpoint = '/project/RDS-FEI-PFZ-RW/YUFEI/SiamFC-TensorFlow-renew/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-scratch'

print("1")
handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
print(imagefile)
if not imagefile:
    sys.exit(0)


tracker_param="default_vot_2"
tracker_name="atom"

tracker_ = Tracker(tracker_name, tracker_param, 0)
tracker=tracker_.runvot()

print("2")

#init
trk = VOT_SIAM_Wrapper(tracker, imagefile, selection)
i = 0

while True:
    i = i + 1
    imagefile = handle.frame()
    if not imagefile:
        break
    region = trk.track(imagefile, i)
    handle.report(region)

sess.close()
