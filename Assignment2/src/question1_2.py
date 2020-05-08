"""
* Deepak Srivatsav
* Reference - http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from tqdm import tqdm
import os
from random import random, choice, seed
seed(42)

def augment_data(augment, data, sampling_rate, noise_dir, augment_prob=0.04):
    
    if not augment:
        return [data]


    def add_bg_noise(noise_factor):
        noises = os.listdir(noise_dir)
        bg_noise = choice(noises)
        bg_path = os.path.join(noise_dir, bg_noise)
        bg, bg_sr = librosa.load(bg_path)
        bg = bg[:data.shape[0]]
        aug_data = data + bg*noise_factor
        return aug_data

    def manipulate_pitch(pitch_factor):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def manipulate_speed(speed_factor):
        return librosa.effects.time_stretch(data, speed_factor)

    
    samples = []
    samples.append(data)
    noise_factor = choice(np.linspace(0.1, 2, 15))
    pitch_factor = choice(np.linspace(0.1, 2, 15))
    speed_factor = choice(np.linspace(0.1, 2, 15))
    if random() < augment_prob:
        samples.append(add_bg_noise(noise_factor))
    if random() < augment_prob:
        samples.append(manipulate_pitch(pitch_factor))
    if random() < augment_prob:
        samples.append(manipulate_speed(speed_factor))

    return samples

def get_mel_frequencies(upper_hertz, num_filters):
    lower_mel = 0
    upper_mel = 1125*np.log(1+upper_hertz/700)
    frequencies = np.linspace(lower_mel, upper_mel, num_filters+2)

    frequencies = 700*(np.exp(frequencies/1125)-1)
    return frequencies

def get_delta_delta(delta):
    windows, ceps = delta.shape
    scale = 2 * sum([i**2 for i in range(2)])
    ddeltas = np.empty_like(delta)
    padded_deltas = np.pad(delta, ((2, 2), (0, 0)), mode='edge')
    for i in range(windows):
        ddeltas[i] = np.dot(np.arange(-2, 3), padded_deltas[i:i+5]) / scale
    return ddeltas

def get_delta(mfcc):
    windows, ceps = mfcc.shape
    scale = 2 * sum([i**2 for i in range(2)])
    deltas = np.empty_like(mfcc)
    padded_feats = np.pad(mfcc, ((2, 2), (0, 0)), mode='edge')
    for i in range(windows):
        deltas[i] = np.dot(np.arange(-2, 3), padded_feats[i:i+5]) / scale
    return deltas

def get_mfcc(samples, sampling_rate, window_time = 25*1e-3, overlap = 0.5):
    num_filters = 26
    num_ceps = 12
    window_size = int(sampling_rate * window_time)
    stride = int(sampling_rate * window_time * overlap)
    num_windows = (len(samples) - window_size) // stride + 1
    nfft = 512
    samples = samples[:window_size + stride*(num_windows - 1)]
    new_shape = (num_windows, window_size)
    new_strides = (samples.strides[0] * stride, samples.strides[0])
    windows = np.lib.stride_tricks.as_strided(samples, shape=new_shape, strides=new_strides)
    weight = np.hanning(window_size)
    fft = np.fft.rfft(windows*weight, nfft)
    power_spectral_estimate = np.abs(fft) ** 2 / nfft
    
    frequencies = get_mel_frequencies(sampling_rate/2, num_filters)
    bins = np.floor((nfft+1)*frequencies/sampling_rate)
    fb = np.zeros((num_filters, int(np.floor(nfft/2+1))))

    for m in range(1, num_filters+1):
        for k in range(int(bins[m-1]), int(bins[m+1])):
            if k == bins[m]:
                fb[m-1][k] = 1
            elif bins[m-1] <= k < bins[m]:
                fb[m-1][k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
            elif bins[m] < k <= bins[m+1]:
                fb[m-1][k] = (bins[m+1] - k) / (bins[m+1] - bins[m])

    filter_bank = np.dot(power_spectral_estimate, fb.T)
   
    filter_bank[np.where(filter_bank == 0)] = np.finfo(float).eps
    filter_bank = np.log(filter_bank)

    mfcc = fftpack.dct(filter_bank, type=2, axis=1, norm='ortho')[:, :num_ceps]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (22 / 2) * np.sin(np.pi * n / 22)
    mfcc *= lift
    

    deltas = get_delta(mfcc)
    ddeltas = get_delta_delta(deltas)
    mfcc = np.append(mfcc, deltas, axis=1)
    mfcc = np.append(mfcc, ddeltas, axis=1)

    return mfcc

def plot_mfcc(mfcc):
    time, freq = mfcc.shape
    plt.imshow(mfcc.T, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("mfcc coefficients")
    plt.xlim([0, time-1])
    plt.ylim([0, freq])
    plt.show()    

def get_features(save_folder, train_folder, plot=False, augment=False, noise_dir="_background_noise_"):
    classes = os.listdir(train_folder)

    for cls in tqdm(classes):
        path = os.path.join(train_folder, cls)
        audio_files = os.listdir(path)
        for audio in tqdm(audio_files):
            audio_path = os.path.join(path, audio)
            samples, sampling_rate = librosa.load(audio_path)
            audio_duration = len(samples) / sampling_rate
            samples = augment_data(augment, samples, sampling_rate, noise_dir)
            for aug_id, sample in enumerate(samples):
                mfcc = get_mfcc(sample, sampling_rate)
                if plot:
                    plot_mfcc(mfcc)
                save_cls_dir = os.path.join(save_folder, cls)
                if not os.path.exists(save_cls_dir):
                    os.mkdir(save_cls_dir)
                if aug_id > 0:
                    save_path = os.path.join(save_cls_dir, audio[:-4] + "_augment_" + str(aug_id) +".npy")
                else:
                    save_path = os.path.join(save_cls_dir, audio[:-4] + ".npy")
                np.save(save_path, mfcc)

def main():
    save_folder = 'mfcc_train_aug'
    train_folder = 'training'
    noise_folder = '_background_noise_'
    get_features(save_folder, train_folder, noise_dir=noise_folder, augment=True)
    print ("[*] Generated Training features with augmentations")
    save_folder = 'mfcc_train'
    get_features(save_folder, train_folder, noise_dir=noise_folder)
    print ("[*] Generated Training features")
    val_folder = 'validation'
    save_folder = 'mfcc_val'
    get_features(save_folder, val_folder)
    print ("[*] Generated Validation features")

if __name__ == '__main__':
    main()