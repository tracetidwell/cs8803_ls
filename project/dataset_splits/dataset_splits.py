import sys
import os
import os.path as osp
from os import walk

import pickle


def process_pickle_bysplit(pickle_filepath, split):
    file = open(pickle_filepath, 'rb')
    data = pickle.load(file)
    file.close()

    classes = data.keys()
    splitspecific_data = {}
    for c in classes:
        splitspecific_data[c] = data[c][split]

    return splitspecific_data


def find_split_paths(current_path, split, dataset='ModelNet10'):
    
    file_paths = []
    
    pickle_filename = dataset + '_TrainingSplits_New.pickle'
    pickle_filepath = osp.join(current_path, pickle_filename)
    splitspecific_data = process_pickle_bysplit(pickle_filepath, split)
    classes = splitspecific_data.keys()
    filenames_for_split = [x for c in classes for x in splitspecific_data[c]]
    
    if dataset=='ModelNet10' or dataset == 'ModelNet40':
        MN_path = 'data/' + dataset + '/raw'
        data_path = osp.join(current_path, MN_path)
        
        for c in classes:
            class_path = osp.join(data_path, str(c) + '/train')
            
            for (dirpath, dirnames, filenames) in walk(class_path):
                for file in filenames:
                    if file in filenames_for_split:
                        file_paths.append(osp.join(dirpath, file))
    
    elif dataset == 'ShapeNet16':
        data_path = osp.join(current_path, 'data/ShapeNet16/raw/train_data')
        
        for c in classes:
            class_path = osp.join(data_path, str(c))
            
            for (dirpath, dirnames, filenames) in walk(class_path):
                for file in filenames:
                    if file in filenames_for_split:
                        file_paths.append(osp.join(dirpath, file))
    
#     print(file_paths)
    return file_paths

if __name__ == "__main__":
    current_path = os.getcwd()
    print(current_path)
    training_10split_paths = find_split_paths(current_path, 10, dataset='ShapeNet16')
    print(training_10split_paths)