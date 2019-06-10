import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.evaluation.nfsdataset import NFSDataset
from pytracking.evaluation.uavdataset import UAVDataset
from pytracking.evaluation.tpldataset import TPLDataset
from pytracking.evaluation.votdataset import VOTDataset
from pytracking.evaluation.lasotdataset import LaSOTDataset
from pytracking.evaluation.trackingnetdataset import TrackingNetDataset
from pytracking.evaluation.got10kdataset import GOT10KDatasetTest, GOT10KDatasetVal, GOT10KDatasetLTRVal
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker
from os.path import join, isdir, isfile
import glob
import cv2
import numpy as np

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0):
    """Run tracker on sequence or dataset."""
    
    dataset = LaSOTDataset()
    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id)]

    run_dataset(dataset, trackers, dataset_name,debug, threads)

def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])  # 0-index


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', default='atom',type=str, help='Name of tracking method.')
    parser.add_argument('--tracker_param',default='default', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    #parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    #parser.add_argument('--base_path', default='/home/peter/Downloads/305938513_bdwkanfly007/votcode&paper/固定摄像头/M2U02600', help='datasets')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    #parser.add_argument('--base_path', default='/home/peter/PycharmProjects/vot-toolkit/vot2018/sequences/fish3/color', help='datasets')
    #parser.add_argument('--base_path', default='/home/peter/Downloads/305938513_bdwkanfly007/votcode&paper/固定摄像头/M2U02600', help='datasets')
    parser.add_argument('--base_path', default='/home/peter/Downloads/305938513_bdwkanfly007/basketball', help='datasets')

    args = parser.parse_args()

    #run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset, args.sequence, args.debug, args.threads)
    tracker = Tracker(args.tracker_name, args.tracker_param, args.runid)
    print("loading uinet","*"*20)
    uinet =tracker.get_tracker()
    print("load sucess","*"*20)
    #seq = 

    #cfg = load_config(args)
    
    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    #for imf in img_files :
    #    print(imf)
    print("load image","*"*20)
    #ims = [cv2.resize(cv2.imread(imf),(224,224)) for imf in img_files]
    ims = [cv2.imread(imf) for imf in img_files]
    print("image loaded","*"*20)
    # Select ROI
    cv2.namedWindow("UINet", cv2.WND_PROP_FULLSCREEN)
    
    try:
        init_rect = cv2.selectROI('UINet', ims[0], False, False)
        x, y, w, h = init_rect
        print(init_rect)
        #input()
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        print(f)
        tic = cv2.getTickCount()
        if f == 0:  # init
            state = np.array((x,y,w,h))
            print("init state",state)
            state = uinet.initialize(im, state)  # init tracker
        elif f > 0:  # tracking
            print("start tracking","*"*30)
            state = uinet.track(im)  # track
            s = []
            for k in state:
                s.append(int(k))
            #state = int(state[0])
            print("f",f)
            #print("state:",state)
            print("s",s)
            state=s
            location = cxy_wh_2_rect((s[0],s[1]), (s[2],s[3]))
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
            location = rbox_in_img.flatten()
            # #mask = state['mask'] > state['p'].seg_thr
            cv2.rectangle(im, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)
            # #im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            #cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('UINet', im)
            key = cv2.waitKey(50)
            if key > 0:
                #os.system("pause")
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('UINet Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    tracker.free()


   # tracked_bb, exec_times = tracker.run(seq,debug=debug)
     
if __name__ == '__main__':
    main()