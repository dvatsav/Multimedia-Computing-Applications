"""
* Deepak Srivatsav
* https://en.wikipedia.org/wiki/Short-time_Fourier_transform
* normalization - https://dsp.stackexchange.com/questions/47304/stft-computation
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import os
from random import random, choice, seed
seed(42)

def stft(windows):
    num_inps = windows.shape[0]
    n = windows.shape[1]
    k = np.arange(n//2+1).repeat(n).reshape(n//2+1, n)
    t = np.arange(n).repeat(n//2+1).reshape(n, n//2+1).T
    exp = np.exp(-2j * np.pi * t * k / n)
    output = np.zeros((num_inps, n//2+1), dtype=np.complex64)
    output = np.dot(windows, exp.T)
    
    return output

def augment_data(augment, data, sampling_rate, noise_dir, augment_prob=0.3):
    
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


def get_spectrogram(data, sampling_rate, window_time = 25*1e-3, overlap = 0.5):
    
    window_size = int(sampling_rate * window_time)
    stride = int(sampling_rate * window_time * overlap)
    num_windows = (len(data) - window_size) // stride + 1
    data = data[:window_size + stride*(num_windows - 1)]
    new_shape = (num_windows, window_size)
    new_strides = (data.strides[0] * stride, data.strides[0])
    windows = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=new_strides)
    weight = np.hanning(window_size)
    fft = stft(windows*weight)
    scale = 1/np.sum(weight)**2
    scale = np.sqrt(scale)
    fft *= scale

    fft = np.absolute(fft)
    fft = fft**2
    fft[np.where(fft == 0)] = np.finfo(float).eps
    spectrogram = np.log10(fft)
    
    return spectrogram

def plot_spectrogram(spectrogram):
    time, freq = spectrogram.shape
    plt.imshow(spectrogram.T, cmap='jet', origin='lower', aspect='auto')
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Scaled to num windows)")
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
                spectrogram = get_spectrogram(sample, sampling_rate)
                if plot:
                    plot_spectrogram(spectrogram)
                save_cls_dir = os.path.join(save_folder, cls)
                if not os.path.exists(save_cls_dir):
                    os.mkdir(save_cls_dir)
                if aug_id > 0:
                    save_path = os.path.join(save_cls_dir, audio[:-4] + "_augment_" + str(aug_id) +".npy")
                else:
                    save_path = os.path.join(save_cls_dir, audio[:-4] + ".npy")
                np.save(save_path, spectrogram)
                

def main():
    save_folder = 'spectrogram_train_aug'
    train_folder = 'training'
    noise_folder = '_background_noise_'
    get_features(save_folder, train_folder, noise_dir=noise_folder, augment=True)
    print ("[*] Generated Training features with augmentations")
    save_folder = 'spectrogram_train'
    get_features(save_folder, train_folder, noise_dir=noise_folder)
    print ("[*] Generated Training features")
    val_folder = 'validation'
    save_folder = 'spectrogram_val'
    get_features(save_folder, val_folder)
    print ("[*] Generated Validation features")

if __name__ == '__main__':
    main()