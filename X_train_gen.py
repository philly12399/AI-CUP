import os
import json
import pickle
import sys
import numpy as np
if __name__ == '__main__':
    pickle_in = open("Y_train.pickle", "rb")
    Y = pickle.load(pickle_in)
    pickle_in = open("Y_in.pickle", "rb")
    Y_in = pickle.load(pickle_in)
    ys = Y.shape[0]

    y_idx = 0
    time = 0
    THE_FOLDER = "./"
    X_train = np.zeros((ys, 23, 5))
    j = 0
    for the_dir in os.listdir(THE_FOLDER):
        
        if not os.path.isdir(the_dir):
            continue
        json_path = THE_FOLDER + "/" + the_dir + f"/{the_dir}_feature.json"

        js_file = open(json_path, 'r')
        temp = json.loads(js_file.read())
        data = []
        for key, value in temp.items():
            mx = max(value)
            value = [x/mx for x in value]
            data.append([0]*2 + value + [0]*2)

        length = len(data[0])
        for i in range(length - 4):
            if Y_in[y_idx] == 1:
                X_train[j] = np.array([x[i:i+5] for x in data])
                j += 1
            y_idx += 1   
    X_train = X_train.reshape(ys, 23, 5, 1)
    with open("X_train.pickle", "wb") as pkfile:
        pickle.dump(X_train, pkfile)
