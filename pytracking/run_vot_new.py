from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import vot
import sys
from inference import inference_wrapper_renew as inference_wrapper
from inference.new_tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import load_cfgs
import tensorflow as tf
import os
from utils.misc_utils import auto_select_gpu, mkdir_p, sort_nicely, load_cfgs
import os.path as osp


class VOT_SIAM_Wrapper(object):

    def __init__(self, model, sess, image, selection, model_config, track_config):

        selection = Rectangle(selection.x, selection.y, selection.width, selection.height)

        self.sess = sess
        self._tracker = Tracker(model, model_config=model_config, track_config=track_config)
        self._tracker.track_init(sess, selection, image)

    def track(self, image, logdir, i):

        res_rect = self._tracker.track(self.sess, image, logdir, i)
        return vot.Rectangle(res_rect.x, res_rect.y, res_rect.width, res_rect.height)

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR))
checkpoint = '/project/RDS-FEI-PFZ-RW/YUFEI/SiamFC-TensorFlow-renew/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-scratch'
model_config, _, track_config = load_cfgs(checkpoint)
track_config['log_level'] = 0

g = tf.Graph()
with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
g.finalize()

"""
gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(graph=g, config=sess_config)
"""
sess = tf.Session(graph=g)
restore_fn(sess)

handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

trk = VOT_SIAM_Wrapper(model, sess, imagefile, selection, model_config, track_config)
i = 0
path = osp.dirname(imagefile)
path = osp.join(path, 'record')
if not osp.isdir(path):
    mkdir_p(path)
logdir = path
while True:
    i = i + 1
    imagefile = handle.frame()
    if not imagefile:
        break
    region = trk.track(imagefile, logdir, i)
    handle.report(region)

sess.close()
