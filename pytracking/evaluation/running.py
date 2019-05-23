import numpy as np
import multiprocessing
import os
from itertools import product
from pytracking.evaluation import Sequence, Tracker


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")
		

def run_sequence(seq: Sequence, tracker: Tracker,dataset_name, debug=False):
    """Runs a tracker on a sequence."""
    print(dataset_name)
    if (dataset_name=="vot"):
        file_dir= os.path.join(tracker.results_dir,"vot",seq.name)
        mkdir(file_dir)
    else:
        file_dir=tracker.results_dir
    base_results_path = '{}/{}'.format(file_dir, seq.name)
    results_path = '{}_001.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)
    print(results_path)
    print(os.path.isfile(results_path))
    if os.path.isfile(results_path) and not debug:
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        tracked_bb, exec_times = tracker.run(seq,debug=debug)
    else:
        try:
            tracked_bb, exec_times = tracker.run(seq,debug=debug)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(int)
    exec_times = np.array(exec_times).astype(float)
   
    print('FPS: {}'.format(len(exec_times) / exec_times.sum()))
    #input()
    
    if not debug:
        if (dataset_name=="vot"):
            np.savetxt(results_path, tracked_bb, delimiter=',', fmt='%d')
            np.savetxt(times_path, exec_times, delimiter=',', fmt='%f')
        else:
            np.savetxt(results_path, tracked_bb, delimiter='\t', fmt='%d')
            np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')


def run_dataset(dataset, trackers,dataset_name, debug=False, threads=0):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq,tracker_info,dataset_name, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info,dataset_name, debug) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
