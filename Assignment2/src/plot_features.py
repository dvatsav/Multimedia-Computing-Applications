import matplotlib.pyplot as plt 
import numpy as np 
import sys
import os

def plot_feat(feat, ylabel):
    time, freq = feat.shape
    plt.imshow(feat.T, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.xlim([0, time-1])
    plt.ylim([0, freq])
    plt.show()

def main():
    cargs = len(sys.argv)
    if cargs != 5:
        print ("Usage: python plot_features.py [spectrogram/mfcc] [feature dir] [num images] [num_classes]")
        sys.exit()

    modes = ['spectrogram', 'mfcc']
    mode = sys.argv[1].lower()
    if mode not in modes:
        print ("Invalid Mode\nUsage: python plot_features.py [spectrogram/mfcc] [feature dir] [num images] [num_classes]")
        sys.exit()

    feature_dir = sys.argv[2]
    if not os.path.exists(feature_dir):
        print ("Invalid Mode\nUsage: python plot_features.py [spectrogram/mfcc] [feature dir] [num images] [num_classes]")
        sys.exit()

    num_imgs = int(sys.argv[3])

    num_classes = int(sys.argv[4])

    classes = os.listdir(feature_dir)

    for i, cls in enumerate(classes):
        if i == num_classes:
            break
        path = os.path.join(feature_dir, cls)
        audio_files = os.listdir(path)
        for idx, audio in enumerate(audio_files):
            if idx == num_imgs:
                break
            audio_path = os.path.join(path, audio)
            feat = np.load(audio_path)
            if mode == "mfcc":
                plot_feat(feat, "mfcc coefficients")
            else:
                plot_feat(feat, "Frequency (Scaled to num windows)")

        

if __name__ == '__main__':
    main()