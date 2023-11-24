'''
The goal of the program it to manipulate the training data for a model.
This program will create noisy data, by overlaying keywords "yes" and "no" with noise.
This program can can then filter the noisy data, with HPF, LPF, and Bandpass filters.
The resilting data is noisy and partial data that will be used to train a model.

We can draw conclusions on the imporance of certain frequencies present in signal processing
for training machine learning models. The test data can either be unfiltered noisy or non-noisy data.
'''

from scipy.signal import butter, lfilter
from pydub import AudioSegment
import os
import wave
import numpy as np
from scipy.io.wavfile import write

'''
Param: 
Input Audio paths (x2)

Output:
Write overlayed audio to path.
'''
def overlay_audio(input_file1, input_file2):
    # Load the audio files
    audio1 = AudioSegment.from_wav(input_file1)
    audio2 = AudioSegment.from_wav(input_file2)

    # Ensure both audio files have the same length
    if len(audio1) > len(audio2):
        audio2 = audio2 + AudioSegment.silent(duration=len(audio1) - len(audio2))
    else:
        audio1 = audio1 + AudioSegment.silent(duration=len(audio2) - len(audio1))

    # Overlay the audio
    output_audio = audio1.overlay(audio2)

    return output_audio

def file_list(path):
    return os.listdir(path)

def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff_freq, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

def save_wav(file_path, audio_data, sampling_rate):
    # Ensure the audio data is in the correct data type (16-bit PCM)
    audio_data = audio_data.astype(np.int16)

    # Save the audio data as a WAV file
    write(file_path, sampling_rate, audio_data)

# Folder paths
yes_folder_path = "./keywords/yes"
no_folder_path = "./keywords/no"
noise_folder_path = "./keywords/noise"
output_path = "./audio"

# File List
yes_list = file_list(yes_folder_path)
no_list = file_list(no_folder_path)
noise_list = file_list(noise_folder_path)

# Constants
LPF_cutoff_freq = 5000 # Hz     //Change
HPF_cutoff_freq = 1000 # Hz     //Change

yes_stop_train_index = int(len(yes_list) * 0.8)     # First 80% for training data
no_stop_train_index = int(len(no_list) * 0.8)       # First 80% for training data


def process_wav_files(stop_train_index, file_list, wav_folder_path, keyword):
    # Processing for wav files in given folder
    for index, wav_file in enumerate(file_list[:stop_train_index]):
        wav_file_path = wav_folder_path + "/" + wav_file
        noise_file_path = noise_folder_path + "/" + noise_list[index]

        # Load audio from local file repo
        audio = wave.open(wav_file_path)

        # Get length of data and sampling rate in audio
        audio_nframes = audio.getnframes()
        sampling_rate = audio.getframerate()

        # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
        signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
        signal = signal.astype(float)

        # Overlay audio with noise and convert Audio Segment to np array
        noisy_signal = overlay_audio(wav_file_path, noise_file_path)
        noisy_signal = np.array(noisy_signal.get_array_of_samples())

        # Write Unfiltered Noisy Signal
        output_file_path = f"{output_path}/{keyword}_Noisy/{keyword}_Noisy_{index}.wav"
        save_wav(output_file_path, noisy_signal, sampling_rate)

        # LPF Filter the Signal
        filtered_signal_LPF = butter_lowpass_filter(noisy_signal, LPF_cutoff_freq, sampling_rate)
        # Write Noisy LPF audio to folder
        output_file_path = f"{output_path}/{keyword}_LPF/{keyword}_LPF_{index}.wav"
        save_wav(output_file_path, filtered_signal_LPF, sampling_rate)

        # HPF Filter the Signal
        filtered_signal_HPF = butter_highpass_filter(noisy_signal, HPF_cutoff_freq, sampling_rate)
        # Write Noisy HPF audio to folder
        output_file_path = f"{output_path}/{keyword}_HPF/{keyword}_HPF_{index}.wav"
        save_wav(output_file_path, filtered_signal_HPF, sampling_rate)


        # Delete, just so not much is processed right now.
        if index > 3:
            break

process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes")
process_wav_files(no_stop_train_index, no_list, no_folder_path, "No")



# # Example usage
# input_file1 = "keywords/noise/noise.doing_the_dishes.wav.0.wav"
# input_file2 = "C:/Users/Xabi0/OneDrive/Documents/College/5th Semester/AI Hardware/AI_HW_Project/keywords/yes/yes.0a7c2a8d_nohash_0.wav"
# output_file = "mixed_audio.wav"





# # Load song from local file repo
# audio = wave.open("C:/Users/Xabi0/OneDrive/Documents/College/5th Semester/AI Hardware/AI_HW_Project/keywords/yes/yes.0a7c2a8d_nohash_0.wav")

# # Get length of data in song
# audio_nframes = audio.getnframes()

# # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
# signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
# signal = signal.astype(float) # Convert signal to floating point to make our lives easier
# cutoff_freq = 5000 # Hz

# mixed_audio = overlay_audio(input_file1, input_file2, output_file)
# mixed_audio_converted = np.array(mixed_audio.get_array_of_samples())

# sampling_rate = audio.getframerate()
# print(f"Sampling rate {sampling_rate}")






# filtered_data = butter_lowpass_filter(mixed_audio_converted, cutoff_freq, sampling_rate)

# save_wav("filtered_output.wav", filtered_data, sampling_rate)

# # song_module = Audio(data = filtered_data, rate = sampling_rate)
# # display(song_module)







# # # Get length of data in song
# # test_song_nframes = test_song_file.getnframes()



# # print("--- Test Song Loaded ---")
# # print("Length of song in samples:",len(test_song_signal))

# # # Also get the samplerate of the song for future use
# # test_song_samplerate = test_song_file.getframerate()
# # print('Sample rate of song:', test_song_samplerate, 'Hz')




# # LPF Filter
# def LPF(input_signal, sampling_rate, corner_freq, order):
#     # Calculate the Nyquist frequency
#     nyquist_freq = sampling_rate / 2

#     # Normalize the cutoff frequency to the Nyquist frequency
#     normalized_cutoff = corner_freq / nyquist_freq

#     # Design a lowpass Butterworth filter
#     b, a = sp.signal.butter(order, normalized_cutoff, btype='low')

#     # Apply the filter to the input signal
#     filtered_signal = sp.signal.lfilter(b, a, input_signal)

#     return filtered_signal


# # HPF Filter
# def HPF(input_signal, sampling_rate, corner_freq, order):
#     nyquist_freq = sampling_rate / 2

#     normalized_cutoff = corner_freq / nyquist_freq

#     b, a = sp.signal.butter(order, normalized_cutoff, btype='high')

#     filtered_signal = sp.signal.lfilter(b, a, input_signal)

#     return filtered_signal


# _________________________________________________________________
# Processing for Yes wav
    # for index, yes_file in enumerate(file_list[:stop_train_index]):
    #     yes_file_path = yes_folder_path + "/" + yes_file
    #     noise_file_path = noise_folder_path + "/" + noise_list[index]

    #     # Load audio from local file repo
    #     audio = wave.open(yes_file_path)

    #     # Get length of data and sampling rate in audio
    #     audio_nframes = audio.getnframes()
    #     sampling_rate = audio.getframerate()

    #     # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
    #     signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
    #     signal = signal.astype(float)

    #     # Overlay audio with noise and convert Audio Segment to np array
    #     noisy_signal = overlay_audio(yes_file_path, noise_file_path)
    #     noisy_signal = np.array(noisy_signal.get_array_of_samples())

    #     # Write Unfiltered Noisy Signal
    #     output_file_path = f"{output_path}/Yes_Noisy/Yes_Noisy_{index}.wav"
    #     save_wav(output_file_path, noisy_signal, sampling_rate)

    #     # LPF Filter the Signal
    #     filtered_signal_LPF = butter_lowpass_filter(noisy_signal, LPF_cutoff_freq, sampling_rate)
    #     # Write Noisy LPF audio to folder
    #     output_file_path = f"{output_path}/Yes_LPF/Yes_LPF_{index}.wav"
    #     save_wav(output_file_path, filtered_signal_LPF, sampling_rate)

    #     # HPF Filter the Signal
    #     filtered_signal_HPF = butter_highpass_filter(noisy_signal, LPF_cutoff_freq, sampling_rate)
    #     # Write Noisy HPF audio to folder
    #     output_file_path = f"{output_path}/Yes_HPF/Yes_HPF_{index}.wav"
    #     save_wav(output_file_path, filtered_signal_HPF, sampling_rate)


    #     # Delete, just so not much is processed right now.
    #     if index > 10:
    #         break