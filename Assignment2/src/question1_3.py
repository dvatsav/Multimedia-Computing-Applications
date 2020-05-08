"""
* Deepak Srivatsav
"""

import numpy as np
from word2number import w2n
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import os
import sys
from tqdm import tqdm
import pickle

def get_data(mode, train_feature_dir, val_feature_dir):
    
    def get_samples(feature_dir, mode):
        X_data = []
        y_data = []
        classes = os.listdir(feature_dir)
        for c in tqdm(classes):
            data_dir = os.path.join(feature_dir, c)
            data_samples = os.listdir(data_dir)
            for sample in tqdm(data_samples):
                sample_path = os.path.join(data_dir, sample)
                data = np.zeros(3564)
                
                if mode == "spectrogram":
                    data = np.zeros(21879)
                    
                data_r = np.load(sample_path).ravel()[:data.shape[0]] 
                data[:data_r.shape[0]] = data_r

                X_data.append(data)
                y_data.append(w2n.word_to_num(c))

        return np.array(X_data), np.array(y_data)


    X_train, y_train = get_samples(train_feature_dir, mode)
    X_val, y_val = get_samples(val_feature_dir, mode)



    return X_train, y_train, X_val, y_val

def classify(mode, train_feature_dir, val_feature_dir):
    X_train, y_train, X_val, y_val = get_data(mode, train_feature_dir, val_feature_dir)
    print ("\n[*] Loaded Data")
    clf = SVC(kernel='linear', C=0.5)

    clf.fit(X_train, y_train)
    print ("[*] Fit Classifier, Saving")
    model_name = ""
    if "aug" in train_feature_dir:
        model_name = mode+"_aug.pkl"
    else:
        model_name = mode+".pkl"
    pickle.dump(clf, open(model_name, 'wb'))
    y_pred = clf.predict(X_val)
    print (classification_report(y_val, y_pred))
    print (accuracy_score(y_val, y_pred))

def main():
    cargs = len(sys.argv)
    if cargs != 4:
        print ("Usage: python question1_3.py [spectrogram/mfcc] [train feature dir] [val feature dir]")
        sys.exit()
    modes = ['spectrogram', 'mfcc']
    mode = sys.argv[1].lower()
    if mode not in modes:
        print ("Invalid Mode\nUsage: python question1_3.py [spectrogram/mfcc] [train feature dir] [val feature dir]")
        sys.exit()
    
    train_feature_dir = sys.argv[2]
    if not os.path.exists(train_feature_dir):
        print ("Invalid Directory\nUsage: python question1_3.py [spectrogram/mfcc] [train feature dir] [val feature dir]")
        sys.exit()

    val_feature_dir = sys.argv[3]
    if not os.path.exists(val_feature_dir):
        print ("Invalid Directory\nUsage: python question1_3.py [spectrogram/mfcc] [train feature dir] [val feature dir]")
        sys.exit()

    classify(mode, train_feature_dir, val_feature_dir)
    
if __name__ == '__main__':
    main()