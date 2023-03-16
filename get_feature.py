import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def add_noise1(x, w=0.004):
    # w：噪声因子
    wn = np.random.randn(len(x))
    output = x + w * wn
    return output

def dB_gain(wav, dB=20):
    """
    :param wav: 语音
    :param dB: 音量
    :return:返回以指定dB增益后的语音
    """
    wav_p = np.mean(wav ** 2)  # 平均功率
    scalar = np.sqrt(10 ** (dB / 10) / (wav_p + np.finfo(np.float32).eps))
    ans = wav * scalar
    return ans


def add_noise2(x, snr=50):
    """
    :param x:纯净语音
    :param snr: 信噪比
    :return: 生成执行信噪比的带噪语音
    """
    P_signal = np.mean(x**2)    # 信号功率
    k = np.sqrt(P_signal / 10 ** (snr / 10.0))  # 噪声系数 k
    return x + np.random.randn(len(x)) * k

def data_roll(x):
# Shifting the sound
    datar = np.roll(x, 1600)
    return datar


def get_x_test_features(directory):
    # Creating an empty list to store all file names
    files = []
    srs = []
    labels = []
    chroma_mean = []
    chroma_var = []
    rms_mean = []
    rms_var = []
    tonal_mean = []
    tempos = []
    hamony_mean = []
    hamony_var = []
    perc_mean = []
    perc_var = []
    zcr_mean = []
    zcr_var = []
    spec_centroid_mean = []
    spec_centroid_var = []
    spec_rolloff_mean = []
    spec_rolloff_var = []
    spec_bw_mean = []
    spec_bw_var = []
    mfcc_1_mean = []
    mfcc_2_mean = []
    mfcc_3_mean = []
    mfcc_4_mean = []
    mfcc_5_mean = []
    mfcc_6_mean = []
    mfcc_7_mean = []
    mfcc_8_mean = []
    mfcc_9_mean = []
    mfcc_10_mean = []
    mfcc_11_mean = []
    mfcc_12_mean = []
    mfcc_13_mean = []
    mfcc_14_mean = []
    mfcc_15_mean = []
    mfcc_16_mean = []
    mfcc_17_mean = []
    mfcc_18_mean = []
    mfcc_19_mean = []
    mfcc_20_mean = []
    mfcc_1_var = []
    mfcc_2_var = []
    mfcc_3_var = []
    mfcc_4_var = []
    mfcc_5_var = []
    mfcc_6_var = []
    mfcc_7_var = []
    mfcc_8_var = []
    mfcc_9_var = []
    mfcc_10_var = []
    mfcc_11_var = []
    mfcc_12_var = []
    mfcc_13_var = []
    mfcc_14_var = []
    mfcc_15_var = []
    mfcc_16_var = []
    mfcc_17_var = []
    mfcc_18_var = []
    mfcc_19_var = []
    mfcc_20_var = []

    ct = 0

    # Looping through each file in the directory
    for file in os.scandir(directory):
        # Loading in the audio file
        '''s=str(file)
        if(s!="<DirEntry \'jazz.00054.wav\'>"):
            continue
        else:
            print(file)'''
        y, sr = librosa.core.load(file)
        srs.append(len(y))
        ct += 1
        print(ct)
        # print(file)
        # Adding the file to our list of files
        files.append(file)

        # Adding the label to our list of labels
        label = str(file).split('.')[0]
        labels.append(label)

        # Calculating chromagram
        hop_length = 5000
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_mean.append(np.mean(chroma))
        chroma_var.append(np.var(chroma))
        # print(chroma.shape)

        # Calculating rms
        rms = librosa.feature.rms(y=y)
        # print(y)
        rms_mean.append(np.mean(rms))
        rms_var.append(np.var(rms))
        # print(rms.shape)

        # Calculalting tempos
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempos.append(tempo)

        # Harmonics and Perceptrual
        y_harm, y_perc = librosa.effects.hpss(y=y)
        hamony_mean.append(np.mean(y_harm))
        hamony_var.append(np.var(y_harm))
        perc_mean.append(np.mean(y_perc))
        perc_var.append(np.var(y_perc))

        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean.append(np.mean(zcr))
        zcr_var.append(np.var(zcr))
        # print(zcr)

        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean.append(np.mean(spec_centroid))
        spec_centroid_var.append((np.var(spec_centroid)))
        # print(spec_centroid.shape)

        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_rolloff_mean.append(np.mean(spec_rolloff))
        spec_rolloff_var.append(np.var(spec_rolloff))
        # print(spec_rolloff.shape)

        # Calculating spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_mean.append(np.mean(spec_bw))
        spec_bw_var.append(np.var(spec_bw))
        # print(spec_bw.shape)

        # Calculating the first 20 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_1_mean.append(mfcc_scaled[0])
        mfcc_2_mean.append(mfcc_scaled[1])
        mfcc_3_mean.append(mfcc_scaled[2])
        mfcc_4_mean.append(mfcc_scaled[3])
        mfcc_5_mean.append(mfcc_scaled[4])
        mfcc_6_mean.append(mfcc_scaled[5])
        mfcc_7_mean.append(mfcc_scaled[6])
        mfcc_8_mean.append(mfcc_scaled[7])
        mfcc_9_mean.append(mfcc_scaled[8])
        mfcc_10_mean.append(mfcc_scaled[9])
        mfcc_11_mean.append(mfcc_scaled[10])
        mfcc_12_mean.append(mfcc_scaled[11])
        mfcc_13_mean.append(mfcc_scaled[12])
        mfcc_14_mean.append(mfcc_scaled[13])
        mfcc_15_mean.append(mfcc_scaled[14])
        mfcc_16_mean.append(mfcc_scaled[15])
        mfcc_17_mean.append(mfcc_scaled[16])
        mfcc_18_mean.append(mfcc_scaled[17])
        mfcc_19_mean.append(mfcc_scaled[18])
        mfcc_20_mean.append(mfcc_scaled[19])

        mfcc_1_var.append(mfcc_scaled[0])
        mfcc_2_var.append(mfcc_scaled[1])
        mfcc_3_var.append(mfcc_scaled[2])
        mfcc_4_var.append(mfcc_scaled[3])
        mfcc_5_var.append(mfcc_scaled[4])
        mfcc_6_var.append(mfcc_scaled[5])
        mfcc_7_var.append(mfcc_scaled[6])
        mfcc_8_var.append(mfcc_scaled[7])
        mfcc_9_var.append(mfcc_scaled[8])
        mfcc_10_var.append(mfcc_scaled[9])
        mfcc_11_var.append(mfcc_scaled[10])
        mfcc_12_var.append(mfcc_scaled[11])
        mfcc_13_var.append(mfcc_scaled[12])
        mfcc_14_var.append(mfcc_scaled[13])
        mfcc_15_var.append(mfcc_scaled[14])
        mfcc_16_var.append(mfcc_scaled[15])
        mfcc_17_var.append(mfcc_scaled[16])
        mfcc_18_var.append(mfcc_scaled[17])
        mfcc_19_var.append(mfcc_scaled[18])
        mfcc_20_var.append(mfcc_scaled[19])

    # Creating a data frame with the values we collected

    df = pd.DataFrame({
        'filename': files,
        'length': srs,
        'chroma_mean': chroma_mean,
        'chroma_val': chroma_var,
        'rms_mean': rms_mean,
        'rms_var': rms_var,
        'spectral_centroid_mean': spec_centroid_mean,
        'spectral_centroid_var': spec_centroid_var,
        'spectral_bandwidth_mean': spec_bw_mean,
        'spectral_bandwidth_var': spec_bw_var,
        'spectral_rolloff_mean': spec_rolloff_mean,
        'spectral_rolloff_var': spec_rolloff_var,
        'zero_crossing_rate_mean': zcr_mean,
        'zero_crossing_rate_var': zcr_var,
        'harmony_mean': hamony_mean,
        'harmony_var': hamony_var,
        'perceptr_mean': perc_mean,
        'perceptr_var': perc_var,
        'tempo': tempos,
        'mfcc_1_mean': mfcc_1_mean,
        'mfcc_2_mean': mfcc_2_mean,
        'mfcc_3_mean': mfcc_3_mean,
        'mfcc_4_mean': mfcc_4_mean,
        'mfcc_5_mean': mfcc_5_mean,
        'mfcc_6_mean': mfcc_6_mean,
        'mfcc_7_mean': mfcc_7_mean,
        'mfcc_8_mean': mfcc_8_mean,
        'mfcc_9_mean': mfcc_9_mean,
        'mfcc_10_mean': mfcc_10_mean,
        'mfcc_11_mean': mfcc_11_mean,
        'mfcc_12_mean': mfcc_12_mean,
        'mfcc_13_mean': mfcc_13_mean,
        'mfcc_14_mean': mfcc_14_mean,
        'mfcc_15_mean': mfcc_15_mean,
        'mfcc_16_mean': mfcc_16_mean,
        'mfcc_17_mean': mfcc_17_mean,
        'mfcc_18_mean': mfcc_18_mean,
        'mfcc_19_mean': mfcc_19_mean,
        'mfcc_20_mean': mfcc_20_mean,
        'mfcc_1_var': mfcc_1_var,
        'mfcc_2_var': mfcc_2_var,
        'mfcc_3_var': mfcc_3_var,
        'mfcc_4_var': mfcc_4_var,
        'mfcc_5_var': mfcc_5_var,
        'mfcc_6_var': mfcc_6_var,
        'mfcc_7_var': mfcc_7_var,
        'mfcc_8_var': mfcc_8_var,
        'mfcc_9_var': mfcc_9_var,
        'mfcc_10_var': mfcc_10_var,
        'mfcc_11_var': mfcc_11_var,
        'mfcc_12_var': mfcc_12_var,
        'mfcc_13_var': mfcc_13_var,
        'mfcc_14_var': mfcc_14_var,
        'mfcc_15_var': mfcc_15_var,
        'mfcc_16_var': mfcc_16_var,
        'mfcc_17_var': mfcc_17_var,
        'mfcc_18_var': mfcc_18_var,
        'mfcc_19_var': mfcc_19_var,
        'mfcc_20_var': mfcc_20_var,
        'label': labels
    })

    # Returning the data frame
    return df

def extract_audio_features(directory):
    '''
    This function takes in a directory of .wav files and returns a
    DataFrame that includes several numeric features of the audio file
    as well as the corresponding genre labels.

    The numeric features incuded are the first 13 mfccs, zero-crossing rate,
    spectral centroid, and spectral rolloff.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a table of audio files that includes several numeric features
    and genre labels.
    '''

    # Creating an empty list to store all file names
    files = []
    srs=[]
    labels = []
    chroma_mean=[]
    chroma_var=[]
    rms_mean=[]
    rms_var=[]
    tonal_mean=[]
    tempos=[]
    hamony_mean=[]
    hamony_var=[]
    perc_mean=[]
    perc_var=[]
    zcr_mean = []
    zcr_var = []
    spec_centroid_mean = []
    spec_centroid_var = []
    spec_rolloff_mean = []
    spec_rolloff_var = []
    spec_bw_mean = []
    spec_bw_var = []
    mfcc_1_mean = []
    mfcc_2_mean = []
    mfcc_3_mean = []
    mfcc_4_mean = []
    mfcc_5_mean = []
    mfcc_6_mean = []
    mfcc_7_mean = []
    mfcc_8_mean = []
    mfcc_9_mean = []
    mfcc_10_mean = []
    mfcc_11_mean = []
    mfcc_12_mean = []
    mfcc_13_mean = []
    mfcc_14_mean = []
    mfcc_15_mean = []
    mfcc_16_mean = []
    mfcc_17_mean = []
    mfcc_18_mean = []
    mfcc_19_mean = []
    mfcc_20_mean = []
    mfcc_1_var = []
    mfcc_2_var = []
    mfcc_3_var = []
    mfcc_4_var = []
    mfcc_5_var = []
    mfcc_6_var = []
    mfcc_7_var = []
    mfcc_8_var = []
    mfcc_9_var = []
    mfcc_10_var = []
    mfcc_11_var = []
    mfcc_12_var = []
    mfcc_13_var = []
    mfcc_14_var = []
    mfcc_15_var = []
    mfcc_16_var = []
    mfcc_17_var = []
    mfcc_18_var = []
    mfcc_19_var = []
    mfcc_20_var = []




    ct=0

    # Looping through each file in the directory
    for file in os.scandir(directory):
        # Loading in the audio file
        '''s=str(file)
        if(s!="<DirEntry \'jazz.00054.wav\'>"):
            continue
        else:
            print(file)'''
        ys, sr = librosa.core.load(file)
        lens=len(ys)
        lens=lens//10*10
        for i in range(0,lens,lens//10):
            y=ys[i:i+lens//10]

            srs.append(len(y))
            ct+=1
            print(ct)
            #print(file)
            # Adding the file to our list of files
            files.append(file)

            # Adding the label to our list of labels
            label = str(file).split('.')[0]
            labels.append(label)

            # Calculating chromagram
            hop_length=5000
            chroma=librosa.feature.chroma_stft(y=y,sr=sr,hop_length=hop_length)
            chroma_mean.append(np.mean(chroma))
            chroma_var.append(np.var(chroma))
            #print(chroma.shape)

            # Calculating rms
            rms=librosa.feature.rms(y=y)
            #print(y)
            rms_mean.append(np.mean(rms))
            rms_var.append(np.var(rms))
            #print(rms.shape)

            #Calculalting tempos
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempos.append(tempo)

            #Harmonics and Perceptrual
            y_harm, y_perc = librosa.effects.hpss(y=y)
            hamony_mean.append(np.mean(y_harm))
            hamony_var.append(np.var(y_harm))
            perc_mean.append(np.mean(y_perc))
            perc_var.append(np.var(y_perc))


            # Calculating zero-crossing rates
            zcr = librosa.feature.zero_crossing_rate(y=y)
            zcr_mean.append(np.mean(zcr))
            zcr_var.append(np.var(zcr))
            #print(zcr)

            # Calculating the spectral centroids
            spec_centroid = librosa.feature.spectral_centroid(y=y,sr=sr)
            spec_centroid_mean.append(np.mean(spec_centroid))
            spec_centroid_var.append((np.var(spec_centroid)))
            #print(spec_centroid.shape)

            # Calculating the spectral rolloffs
            spec_rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr)
            spec_rolloff_mean.append(np.mean(spec_rolloff))
            spec_rolloff_var.append(np.var(spec_rolloff))
            #print(spec_rolloff.shape)

            # Calculating spectral bandwidth
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_bw_mean.append(np.mean(spec_bw))
            spec_bw_var.append(np.var(spec_bw))
            #print(spec_bw.shape)



            # Calculating the first 20 mfcc coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            mfcc_1_mean.append(mfcc_scaled[0])
            mfcc_2_mean.append(mfcc_scaled[1])
            mfcc_3_mean.append(mfcc_scaled[2])
            mfcc_4_mean.append(mfcc_scaled[3])
            mfcc_5_mean.append(mfcc_scaled[4])
            mfcc_6_mean.append(mfcc_scaled[5])
            mfcc_7_mean.append(mfcc_scaled[6])
            mfcc_8_mean.append(mfcc_scaled[7])
            mfcc_9_mean.append(mfcc_scaled[8])
            mfcc_10_mean.append(mfcc_scaled[9])
            mfcc_11_mean.append(mfcc_scaled[10])
            mfcc_12_mean.append(mfcc_scaled[11])
            mfcc_13_mean.append(mfcc_scaled[12])
            mfcc_14_mean.append(mfcc_scaled[13])
            mfcc_15_mean.append(mfcc_scaled[14])
            mfcc_16_mean.append(mfcc_scaled[15])
            mfcc_17_mean.append(mfcc_scaled[16])
            mfcc_18_mean.append(mfcc_scaled[17])
            mfcc_19_mean.append(mfcc_scaled[18])
            mfcc_20_mean.append(mfcc_scaled[19])

            mfcc_1_var.append(mfcc_scaled[0])
            mfcc_2_var.append(mfcc_scaled[1])
            mfcc_3_var.append(mfcc_scaled[2])
            mfcc_4_var.append(mfcc_scaled[3])
            mfcc_5_var.append(mfcc_scaled[4])
            mfcc_6_var.append(mfcc_scaled[5])
            mfcc_7_var.append(mfcc_scaled[6])
            mfcc_8_var.append(mfcc_scaled[7])
            mfcc_9_var.append(mfcc_scaled[8])
            mfcc_10_var.append(mfcc_scaled[9])
            mfcc_11_var.append(mfcc_scaled[10])
            mfcc_12_var.append(mfcc_scaled[11])
            mfcc_13_var.append(mfcc_scaled[12])
            mfcc_14_var.append(mfcc_scaled[13])
            mfcc_15_var.append(mfcc_scaled[14])
            mfcc_16_var.append(mfcc_scaled[15])
            mfcc_17_var.append(mfcc_scaled[16])
            mfcc_18_var.append(mfcc_scaled[17])
            mfcc_19_var.append(mfcc_scaled[18])
            mfcc_20_var.append(mfcc_scaled[19])

    # Creating a data frame with the values we collected

    df = pd.DataFrame({
        'filename': files,
        'length':srs,
        'chroma_mean':chroma_mean,
        'chroma_val':chroma_var,
        'rms_mean':rms_mean,
        'rms_var':rms_var,
        'spectral_centroid_mean': spec_centroid_mean,
        'spectral_centroid_var': spec_centroid_var,
        'spectral_bandwidth_mean':spec_bw_mean,
        'spectral_bandwidth_var':spec_bw_var,
        'spectral_rolloff_mean': spec_rolloff_mean,
        'spectral_rolloff_var': spec_rolloff_var,
        'zero_crossing_rate_mean': zcr_mean,
        'zero_crossing_rate_var': zcr_var,
        'harmony_mean':hamony_mean,
        'harmony_var':hamony_var,
        'perceptr_mean':perc_mean,
        'perceptr_var':perc_var,
        'tempo':tempos,
        'mfcc_1_mean': mfcc_1_mean,
        'mfcc_2_mean': mfcc_2_mean,
        'mfcc_3_mean': mfcc_3_mean,
        'mfcc_4_mean': mfcc_4_mean,
        'mfcc_5_mean': mfcc_5_mean,
        'mfcc_6_mean': mfcc_6_mean,
        'mfcc_7_mean': mfcc_7_mean,
        'mfcc_8_mean': mfcc_8_mean,
        'mfcc_9_mean': mfcc_9_mean,
        'mfcc_10_mean': mfcc_10_mean,
        'mfcc_11_mean': mfcc_11_mean,
        'mfcc_12_mean': mfcc_12_mean,
        'mfcc_13_mean': mfcc_13_mean,
        'mfcc_14_mean': mfcc_14_mean,
        'mfcc_15_mean': mfcc_15_mean,
        'mfcc_16_mean': mfcc_16_mean,
        'mfcc_17_mean': mfcc_17_mean,
        'mfcc_18_mean': mfcc_18_mean,
        'mfcc_19_mean': mfcc_19_mean,
        'mfcc_20_mean': mfcc_20_mean,
        'mfcc_1_var': mfcc_1_var,
        'mfcc_2_var': mfcc_2_var,
        'mfcc_3_var': mfcc_3_var,
        'mfcc_4_var': mfcc_4_var,
        'mfcc_5_var': mfcc_5_var,
        'mfcc_6_var': mfcc_6_var,
        'mfcc_7_var': mfcc_7_var,
        'mfcc_8_var': mfcc_8_var,
        'mfcc_9_var': mfcc_9_var,
        'mfcc_10_var': mfcc_10_var,
        'mfcc_11_var': mfcc_11_var,
        'mfcc_12_var': mfcc_12_var,
        'mfcc_13_var': mfcc_13_var,
        'mfcc_14_var': mfcc_14_var,
        'mfcc_15_var': mfcc_15_var,
        'mfcc_16_var': mfcc_16_var,
        'mfcc_17_var': mfcc_17_var,
        'mfcc_18_var': mfcc_18_var,
        'mfcc_19_var': mfcc_19_var,
        'mfcc_20_var': mfcc_20_var,
        'label': labels
    })

    # Returning the data frame
    return df

'''df = extract_audio_features('./Data/data')
df.to_csv('./Data/genre.csv', index=False)'''