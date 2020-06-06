import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
import json
import pickle
import sys
import statistics as st
def mid(l,r,feat):
    if r-l+1 % 2 ==0: 
        return st.median(feat[l:r+1] + [int(500)])
    return st.median(feat[l:r+1])
sid="song_id"
THE_FOLDER = "./"
ans_path = THE_FOLDER + "ans.json"
ans_file = open(ans_path, 'w')
ans_dict = {}
new_model = tf.keras.models.load_model('model_66-2.model')
for the_dir in os.listdir(THE_FOLDER):
    print(the_dir)
    prediction = []
    l_ans = []
    if not os.path.isdir(the_dir):
        continue
    try:
        int(the_dir)
    except:
        print(the_dir)
        continue
    json_path = THE_FOLDER + "/" + the_dir + f"/{the_dir}_feature.json"
    js_file = open(json_path, 'r')
    temp = json.loads(js_file.read())
    data = []
    npp = []
    feat = temp["vocal_pitch"]
    for key, value in temp.items():
        mx = max(value)
        value = [x/mx for x in value]
        data.append([0]*2 + value + [0]*2)
    length = len(data[0])
    for i in range(length - 4):
        npp.append([x[i:i+5] for x in data]) 
        #        np.array([x[i:i+5] for x in data]).reshape(1, 23, 5, 1)
    npp = np.array(npp).reshape(-1, 23, 5, 1)
    predict = new_model.predict([npp])
    len_p = len(predict)
    for i in range(len_p):
        if predict[i] > 0.8:
            prediction.append(i)
    p_len=len(prediction)
    for i in range(p_len-1):
        mdn = mid(prediction[i],prediction[i+1],feat)
        mdn = round(mdn)
        if mdn < 10:
            continue
        else:
            l_ans.append([0.016+0.032*prediction[i],0.016+0.032*prediction[i+1], mdn])
    ans_dict[sid+the_dir] = l_ans
json.dump(ans_dict, ans_file)
ans_file.close()
