
import os
import json
import pickle
import sys
if __name__ == '__main__':
    time = 0
    THE_FOLDER = "./"
    X_train = []
    for the_dir in os.listdir(THE_FOLDER):
        
        if not os.path.isdir(the_dir):
            continue
        json_path = THE_FOLDER + "/" + the_dir + f"/{the_dir}_feature.json"

        js_file = open(json_path, 'r')
        temp = json.loads(js_file.read())
        data = []
        for key, value in temp.items():
            data.append(value)
        X_train.append(data)

    with open("X_train.pickle", "wb") as pkfile:
        pickle.dump(X_train, pkfile)
