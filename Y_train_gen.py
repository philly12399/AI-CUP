
import os
import json
import pickle
import numpy as np
import sys
if __name__ == '__main__':
    time = 0
    THE_FOLDER = "./"
    Y_train = np.zeros((3868315))
    j = 0
    for the_dir in os.listdir(THE_FOLDER):
        
        if not os.path.isdir(the_dir):
            continue
        vc_path = THE_FOLDER+ the_dir+ f"/{the_dir}_vocal.json"
        gt_path = THE_FOLDER + "/" + the_dir + "/" + the_dir + "_groundtruth.txt"
        vc_file = open(vc_path, 'r')
        gt_file = open(gt_path, 'r')
        vc = json.loads(vc_file.read())
        vc = [int(x[0] * 1000000)  for x in vc]
        boundary = []
        for line in gt_file:
            onset, offset, _ = list(map(float, line.split()))
            boundary.append(int(onset*1000000))
            boundary.append(int(offset*1000000))
        y_train = []
        length =len(vc) 
        vi=0
        for i in boundary:
            if vc[vi]-int(16000) > i: 
                continue
            while not ((vc[vi] + int(16000) > i) and (vc[vi]-int(16000) <= i)):
                y_train.append(0)
                vi += 1
            y_train.append(1)
            vi += 1
        d = length-vi
        y_train = y_train + [0]*d        
        #print(y_train)
        for i in range(length):
            #print(np.array([x[i:i+5] for x in data]))
            Y_train[i] = y_train[i]
            #print("XJ")
            #print(X_train[j])
            j += 1
        time += 1
    with open("Y_train.pickle", "wb") as pkfile:
        pickle.dump(Y_train, pkfile)

