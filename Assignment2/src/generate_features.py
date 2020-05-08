from question1_1 import get_spectrogram
from question1_2 import get_mfcc
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm

def generate_features(mode, data_dir, save_dir):
    classes = os.listdir(data_dir)

    for cls in tqdm(classes):
        path = os.path.join(data_dir, cls)
        audio_files = os.listdir(path)
        for audio in tqdm(audio_files):
            audio_path = os.path.join(path, audio)
            samples, sampling_rate = librosa.load(audio_path)
            audio_duration = len(samples) / sampling_rate
            feat = 0
            if mode == "mfcc":
                feat = get_mfcc(samples, sampling_rate, window_time = 25*1e-3, overlap = 0.5)
            else:
                feat = get_spectrogram(samples, sampling_rate, window_time = 25*1e-3, overlap = 0.5)
               
            save_cls_dir = os.path.join(save_dir, cls)
            if not os.path.exists(save_cls_dir):
                os.mkdir(save_cls_dir)
           
            save_path = os.path.join(save_cls_dir, audio[:-4] + ".npy")
            np.save(save_path, feat)

def main():
    cargs = len(sys.argv)
    if cargs != 4:
        print ("Usage: python generate_features.py [spectrogram/mfcc] [data dir] [save dir]")
        sys.exit()

    modes = ['spectrogram', 'mfcc']
    mode = sys.argv[1].lower()
    if mode not in modes:
        print ("Invalid mode\nUsage: python generate_features.py [spectrogram/mfcc] [data dir] [save dir]")
        sys.exit()

    data_dir = sys.argv[2]
    if not os.path.exists(data_dir):
        print ("Invalid Directory\nUsage: python generate_features.py [spectrogram/mfcc] [data dir] [save dir]")
        sys.exit()

    save_dir = sys.argv[3]
    if not os.path.exists(save_dir):

        print ("Invalid Directory\nUsage: python generate_features.py [spectrogram/mfcc] [data dir] [save dir]")
        sys.exit()

    generate_features(mode, data_dir, save_dir)


if __name__ == "__main__":
    main()