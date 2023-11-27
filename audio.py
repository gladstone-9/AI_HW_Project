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
import librosa
import scipy
import soundfile as sf

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


def process_wav_files(stop_train_index, file_list, wav_folder_path, keyword, technique):
    # Processing for wav files in given folder
    for index, wav_file in enumerate(file_list[:stop_train_index]):
        wav_file_path = wav_folder_path + "/" + wav_file
        noise_file_path = noise_folder_path + "/" + noise_list[index]

        if technique == 'create_noisy_data':
            # Load audio from local file repo
            audio = wave.open(wav_file_path)

            # Get sampling rate in audio
            sampling_rate = audio.getframerate()

            # Overlay audio with noise and convert Audio Segment to np array
            noisy_signal = overlay_audio(wav_file_path, noise_file_path)
            noisy_signal = np.array(noisy_signal.get_array_of_samples())

            # Write Unfiltered Noisy Signal
            output_file_path = f"{output_path}/{keyword}_Noisy/{keyword}_Noisy_{index}.wav"
            save_wav(output_file_path, noisy_signal, sampling_rate)
        
        # Adapted from https://github.com/shun60s/spectral-subtraction/blob/master/ss1.py
        if technique == 'spectral_substraction':
            # edit following wav file name
            infile= f"./audio/{keyword}_Noisy/{keyword}_Noisy_{index}.wav"
            outfile= f"./audio/{keyword}_Subtraction/{keyword}_Subtraction_{index}.wav"
            noisefile=f"./keywords/noise/{noise_list[index]}"

            # load input file, and stft (Short-time Fourier transform)
            # print ('load wav', infile)
            w, sr = librosa.load( infile, sr=None, mono=True) # keep native sr (sampling rate) and trans into mono
            s= librosa.stft(w)    # Short-time Fourier transform
            ss= np.abs(s)         # get magnitude
            angle= np.angle(s)    # get phase
            b=np.exp(1.0j* angle) # use this phase information when Inverse Transform

            # load noise only file, stft, and get mean
            # print ('load wav', noisefile)
            nw, nsr = librosa.load( noisefile, sr=None, mono=True)
            ns= librosa.stft(nw) 
            nss= np.abs(ns)
            mns= np.mean(nss, axis=1) # get mean

            # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
            sa= ss - mns.reshape((mns.shape[0],1))  # reshape for broadcast to subtract
            sa0= sa * b  # apply phase information
            y= librosa.istft(sa0) # back to time domain signal

            # save as a wav file
            scipy.io.wavfile.write(outfile, sr, (y * 32768).astype(np.int16)) # save signed 16-bit WAV format
            #librosa.output.write_wav(outfile, y , sr)  # save 32-bit floating-point WAV format, due to y is float 
            # print ('write wav', outfile)

        # Adapted from https://ankurdhuriya.medium.com/audio-enhancement-and-denoising-methods-3644f0cad85b
        if technique == 'wiener_filter':
            noisy_signal_file_path = f"./audio/{keyword}_Noisy/{keyword}_Noisy_{index}.wav"
            if keyword == 'Yes':
                clean_signal_file_path = f"keywords/{keyword}/{yes_list[index]}"
            else:
                clean_signal_file_path = f"keywords/{keyword}/{no_list[index]}"

            noise_audio, sr = librosa.load(noisy_signal_file_path)
            clean_audio, sr = librosa.load(clean_signal_file_path)

            noise_stft = np.abs(librosa.stft(noise_audio))
            clean_stft = np.abs(librosa.stft(clean_audio))

            wiener_filter = clean_stft**2 / (clean_stft**2 + noise_stft**2)

            enhanced_stft = wiener_filter * noise_stft

            enhanced_audio = librosa.istft(enhanced_stft)

            output_file_path = f"./audio/{keyword}_Wiener/{keyword}_Wiener_{index}.wav"
            sf.write(output_file_path, enhanced_audio, sr, 'PCM_16')

        if technique == 'butterworth_filtering':
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

        if technique == 'bandpass_filter':
            low_pass_cutoff = 1000  # Hz
            high_pass_cutoff = 100  # Hz

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

            # LPF Filter the Signal
            filtered_signal_LPF = butter_lowpass_filter(noisy_signal, low_pass_cutoff, sampling_rate)

            # HPF Filter the Signal
            filtered_signal_bandpass = butter_highpass_filter(filtered_signal_LPF, high_pass_cutoff, sampling_rate)
            # Write Noisy HPF audio to folder
            output_file_path = f"{output_path}/{keyword}_Bandpass/{keyword}_Bandpass_{index}.wav"
            save_wav(output_file_path, filtered_signal_bandpass, sampling_rate)

        # Delete, just so not much is processed right now.
        if index > 3:
            break

# Create the Noisy Data
process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "create_noisy_data")
process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "create_noisy_data")
print("Data Created")

# Process Audio with Butterworth
process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "butterworth_filtering")
process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "butterworth_filtering")
print("Butterworth Filtering Finished")

# Process Audio with Butterworth
process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "bandpass_filter")
process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "bandpass_filter")
print("Bandpass Filtering Finished")

yes_noisy_signal_folder_path = "./audio/Yes_Noisy"
yes_noisy_signal_list = file_list(yes_noisy_signal_folder_path)
no_noisy_signal_folder_path = "./audio/No_Noisy"
no_noisy_signal_list = file_list(no_noisy_signal_folder_path)

# Process Audio with Spectral Substraction 
process_wav_files(yes_stop_train_index, yes_noisy_signal_list, yes_folder_path, "Yes", "spectral_substraction")
process_wav_files(no_stop_train_index, no_noisy_signal_list, no_folder_path, "No", "spectral_substraction")
print("Spectral Substraction Finished")

# Process Audio with Wiener Filtering
process_wav_files(yes_stop_train_index, yes_noisy_signal_list, yes_folder_path, "Yes", "wiener_filter")
process_wav_files(no_stop_train_index, no_noisy_signal_list, no_folder_path, "No", "wiener_filter")
print("Wiener Filtering Finished")







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