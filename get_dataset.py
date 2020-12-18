import os
import json
import numpy as np

#Action lables
LABELS = [
 'back_cross_step',
 'clap',
 'cross_heel',
 'downup',
 'knee_up',
 'lunge',
 'openclose',
 'shoulder',
 'side',
 'stand_clap',
 'sting',
 'wave'
]
WINDOW  = 10
n_classes = len(LABELS)

def getDataset(data_path = "train/", file="train", window_size=WINDOW, dirs=os.listdir("train/"), overlap_size=8):
    lst = []
    for a in dirs:
        currPose = []
        path = data_path + a + "/"
        print("[folder]", path)
        for video in os.listdir(path):
            currVideo = []
            for filename in os.listdir(path+video):
                print(filename)
                data = []
                with open(path + video + "/" + filename) as json_data:
                    d = json.load(json_data)
                    try:
                        data = d['people'][0]['pose_keypoints_2d']
                    except:
                        print ("Failed at" + filename)
                        continue
                    json_data.close
                #separate X and Y coord
                npdata = np.asarray(data)
                Xdata = data[::3]
                Ydata = data[1::3]
                stk = np.dstack((Xdata, Ydata)) #stack vertically
                currVideo.append(stk)
            currPose.append(currVideo)
        lst.append(currPose)
        #break
    # frame overlap
    cases = []
    labels = []
    for i, pose in enumerate(lst):
        for vid in pose:
            for start in range(0, len(vid)-window_size, window_size - overlap_size):
                currCase = np.empty([window_size,25,2])
                for index in range(0,window_size):
                    currCase[index] = vid[start+index]
                cases.append(currCase)
                labels.append(i)
    #convert to numpy array and save
    arr = np.empty([len(cases), window_size, 25, 2])
    for i, ele in enumerate(cases):
        arr[i] = ele
    np.save(file + "X", arr)
    np.save(file + "Y", np.asarray(labels))
    return dirs

print("[START training dataset]")
LABELS = getDataset("train/", "train", dirs=LABELS)
print("[END training dataset]")
print("[START val dataset]")
getDataset("val/", "val", dirs=LABELS)
print("[END test dataset]")