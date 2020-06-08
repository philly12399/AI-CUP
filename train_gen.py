
import os
import json
import pickle
import numpy as np
import random as rd
import sys
if __name__ == '__main__':
    o=0
    z=0
    j = 0
    THE_FOLDER = "./"
    Y_train = []
    Y_in = []
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
        length = len(y_train)     
        for i in range(length):
            #print(np.array([x[i:i+5] for x in data]))
            if y_train[i] == 0:
                if rd.random()<(1/12):
                    Y_in.append(1)
                    Y_train.append(0)
                    z+=1
                else:
                    Y_in.append(0)
            else:
                Y_in.append(1)
                Y_train.append(1)
                o+=1
    Y_in = np.array(Y_in)
    Y_train = np.array(Y_train)
    print(Y_in.shape,Y_train.shape)
    print(z,o)

    with open("Y_train.pickle", "wb") as pkfile:
        pickle.dump(Y_train, pkfile)
    with open("Y_in.pickle", "wb") as pkfile:
        pickle.dump(Y_in, pkfile)


    
