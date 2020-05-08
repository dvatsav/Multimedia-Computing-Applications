import numpy as np
from word2number import w2n
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import os
import sys
from tqdm import tqdm
import pickle

def get_data(mode, test_dir):
    
    def get_samples(test_dir, mode):
        X_data = []
        y_data = []
        classes = os.listdir(test_dir)
        for c in tqdm(classes):
            data_dir = os.path.join(test_dir, c)
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

    X_test, y_test = get_samples(test_dir, mode)

    return X_test, y_test

def classify(mode, model, test_dir):
    X_test, y_test = get_data(mode, test_dir)
    print ("\n[*] Loaded Data")
    clf = pickle.load(open(model, 'rb'))
    print ("[*] Loaded Model")
    y_pred = clf.predict(X_test)
    print (classification_report(y_test, y_pred))
    print (accuracy_score(y_test, y_pred))

def main():
    cargs = len(sys.argv)
    if cargs != 4:
        print ("Usage: python test.py [mode] [model] [test_dir]")
        sys.exit()
    
    modes = ['spectrogram', 'mfcc']
    mode = sys.argv[1].lower()
    if mode not in modes:
        print ("Invalid Mode\nUsage: python test.py [mode] [model] [test_dir]")
        sys.exit()

    model = sys.argv[2]
    if not os.path.exists(model):
        print ("Invalid Model\nUsage: python test.py [mode] [model] [test_dir]")
        sys.exit()

    test_dir = sys.argv[3]
    if not os.path.exists(test_dir):
        print ("Invalid Directory\nUsage: python test.py [mode] [model] [test_dir]")
        sys.exit()

    classify(mode, model, test_dir)

if __name__ == '__main__':
    main()
