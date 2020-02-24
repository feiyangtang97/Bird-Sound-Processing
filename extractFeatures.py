
# coding: utf-8

# In[24]:


"""
Spectrogram is one of the most popular way to visualize the sound waveforms.
Sound waveform is a collection of sample points in time series.
First of all, we need to read the raw .wav file into computer.
There are normally three ways to do so: 1. python initial wave lib 2.scikits.audiolab 3.librosa
The following code is based on 1 and 3.
You can refer to the following links to get some insights of these libs.
http://librosa.github.io/librosa/index.html#
https://docs.python.org/3/library/wave.html

"""
import wave
import numpy as np
import librosa
import os
import glob
import pandas as pd
import platform

import matplotlib.pyplot as plt
from sklearn import preprocessing


def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)


def load_audio(filename, mono=True, fs=None):
    """Load audio file into numpy array
    Supports 24-bit wav-format

    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    file_id = file_base.split('/')[-1]
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()
        # self._nframes = chunk.chunksize // self._framesize
        # _framesize -- size of one frame in the file

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.frombuffer(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            # audio_data = np.mean(audio_data, axis=0)
            # only use the first channel
            audio_data = audio_data[0]

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate and fs is not None:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return [file_id, audio_data, sample_rate, sample_width, number_of_channels, number_of_frames]
    return None, None


def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        if not __class_labels.has_key(words[-1]):
            raise ValueError('The event {} is not in the dictionary'.format(words[-1]))
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[-3]), float(words[-2]), __class_labels[words[-1]]])
    return _desc_dict


def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=(int)(_nfft / 2), power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))


# ###################################################################
#              Main script starts here
# ###################################################################

# set up path

_data_path = 'dataset/Birds/'
wav_folder = _data_path + 'wav/'
label_file = _data_path + 'meta.csv'

#print(wav_folder)
#print(label_file)


# Output path
feat_folder = _data_path + 'asset/feature/'
#print(feat_folder)

create_folder(feat_folder)

# -----------------------------------------------------------------------
# read wave file
# -----------------------------------------------------------------------
#print(wav_folder)
wav_files = glob.glob(wav_folder + '*.wav')
#print(wav_files)
if(platform.system()=='Windows'):
    wav_name = [f.split('/')[-1] for f in wav_files]
    for index in range(len(wav_name)):
        wav_name[index] = wav_name[index][4:]
    
else:
    wav_name = [f.split('/')[-1] for f in wav_files]

#print(wav_name)

waves = pd.DataFrame(
    columns=['id', 'audio_data', 'sample_rate', 'sample_width', 'number_of_channels', 'number_of_frames'])


is_mono = True
i = 0


######

print(waves)
for w in wav_name:
    print(w)
    print(wav_folder)
    waves.loc[i] = load_audio(wav_folder + w, is_mono)
    print("{}/{} - {}".format(i, len(wav_name)-1, w))
    i += 1

# save the waves
# waves.to_pickle(_data_path + 'waves.pkl')

# read all labels
labels = pd.read_table(label_file, dtype={'id': np.str})  # sep:str, defaults to ',' for read_csv(),  \t for read_table()
raw_labels = labels['label']
print(raw_labels.describe())

label_freq = pd.crosstab(raw_labels, columns='freq')
# map raw labels to numbers
__class_labels = pd.DataFrame(label_freq.index.get_values(), columns=['label'])
__class_labels['num_label'] = [i for i in range(len(__class_labels))]

# update labels with num_label
num_labels = pd.merge(labels, __class_labels, on='label')
# num_labels.to_csv(_data_path+'num_labels.csv', index=False)
# select id and num_label
_n_labels = num_labels[['id', 'num_label']]
# update waves with num_label
waves_labels = pd.merge(waves, num_labels, on='id')
# waves_labels.to_pickle(_data_path+'waves_numlabels.pkl', index=False)


# User set parameters
nfft = 512
win_len = nfft
hop_len = win_len / 2
nb_mel_bands = 40
sr = 48000

# extract mel band energy feature
train_data = pd.DataFrame(columns=['id', 'mbe', 'label'])
for index, w in waves_labels.iterrows():
    print("Extracting {} -- {}/7002".format(w['id'], index))
#get some errors here 

    feat = extract_mbe(w["audio_data"], sr, nfft, nb_mel_bands)
    train_data.loc[index] = [w['id'], feat, w['num_label']]

def get_data():
    return train_data
def get_wav():
    return waves

"""
For other features, you can refer to librosa feature extraction section
https://librosa.github.io/librosa/feature.html#  
"""


# In[25]:
