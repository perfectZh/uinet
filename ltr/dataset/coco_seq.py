import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
env_path = os.path.join(env_path, '..')
if env_path not in sys.path:
    sys.path.append(env_path)
#print (sys.path)
#from .base_dataset import BaseDataset
from ltr.dataset.base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import torch
from pycocotools.coco import COCO
from collections import OrderedDict
from ltr.admin.environment import env_settings
import numpy as np
import scipy.ndimage
import torchvision.transforms.functional as tvF
class MSCOCOSeq(BaseDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
            - images
                - train2014

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=default_image_loader):
        root = env_settings().coco_dir if root is None else root
        super().__init__(root, image_loader)

        self.img_pth = os.path.join(root, 'train2014/')
        self.anno_path = os.path.join(root, 'annotations/instances_train2014.json')

        # Load the COCO set.
        self.coco_set = COCO(self.anno_path)
        
        self.cats = self.coco_set.cats
        self.num_cats = len(self.cats.keys())+1
        self.map_id_cat = {cat_id: i+1 for i, cat_id in enumerate(list(self.coco_set.cats.keys()))}  #背景层
        self.sequence_list = self._get_sequence_list()
        #self.map_id_cat = {cat_id: i+1 for i, cat_id in enumerate(list(self.coco.cats.keys()))}

    def _get_sequence_list(self):
        ann_list = list(self.coco_set.anns.keys())
        seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]

        return seq_list

    def is_video_sequence(self):
        return False

    def get_name(self):
        return 'coco'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        return anno, torch.Tensor([1])

    def _get_anno(self, seq_id):
        #print (self.coco_set.anns[self.sequence_list[seq_id]])
        anno = self.coco_set.anns[self.sequence_list[seq_id]]['bbox']
        return torch.Tensor(anno).view(1, 4)

    def _get_frames(self, seq_id):
        path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
            object_meta = OrderedDict({'object_class': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self._get_anno(seq_id)
        
        anno_frames = [anno.clone()[0, :] for _ in frame_ids]
        
        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta
    
    def get_frames_mask(self, seq_id=None, frame_ids=None, anno=None,mask=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        print("seq_id",seq_id)
        imgIds=self.sequence_list[seq_id]
        #ann_ids = self.coco_set.getAnnIds(imgIds=imgIds, iscrowd=None)
        #anns = self.coco_set.loadAnns(ann_ids)
        frame = self._get_frames(seq_id)

        target_shape=(frame.shape[0],frame.shape[1],self.num_cats)
        frame_list = [frame.copy() for _ in frame_ids]
        
        if anno is None:
            anno = self._get_anno(seq_id)
        
        anno_frames = [anno.clone()[0, :] for _ in frame_ids]

        if mask is None:
            mask = self._get_mask(seq_id)
        if mask is None:
            print(imgIds)
        mask_frames = [mask.clone() for _ in frame_ids]
        
        
        object_meta = self.get_meta_info(seq_id)
        
        return frame_list, anno_frames, mask_frames,object_meta
    
    def resize_mask(mask, scale, padding, crop=None):
        """Resizes a mask using the given scale and padding.
        Typically, you get the scale and padding from resize_image() to
        ensure both, the image and the mask, are resized consistently.
        scale: mask scaling factor
        padding: Padding to add to the mask in the form
                [(top, bottom), (left, right), (0, 0)]
        """
        # Suppress warning from scipy 0.13.0, the output shape of zoom() is
        # calculated with round() instead of int()
       
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        if crop is not None:
            y, x, h, w = crop
            mask = mask[y:y + h, x:x + w]
        else:
            mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask

    def _get_mask(self, seq_id):

        #mask_one_hot = torch.zeros(target_shape, dtype=np.uint8)
        anno = self.coco_set.anns[self.sequence_list[seq_id]]
        mask_partial = self.coco_set.annToMask(anno)
        #rG=tvF.resize(mask_partial,(1,288,288))
        return torch.Tensor(mask_partial)

import random
if __name__ == '__main__':
    coco_train = MSCOCOSeq()
    dataset=coco_train
    seq_id=1
    train_frame_ids=[1]
    anno, visible = dataset.get_sequence_info(seq_id)
    train_frames, train_anno,mask_frames, _ = coco_train.get_frames_mask(seq_id, train_frame_ids, anno)
    print(mask_frames[0].shape)