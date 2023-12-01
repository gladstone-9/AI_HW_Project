'''
The goal of the program it to manipulate the training data for a model.
This program will create noisy data, by overlaying keywords "yes" and "no" with noise.
This program can can then filter the noisy data, with HPF, LPF, and Bandpass filters.
The resilting data is noisy and partial data that will be used to train a model.

We can draw conclusions on the importance of certain frequencies present in signal processing
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
import matplotlib.pyplot as plt
from IPython.display import Audio, display

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
LPF_cutoff_freq = 600 # Hz     //Change
HPF_cutoff_freq = 100 # Hz     //Change

yes_stop_train_index = int(len(yes_list) * 0.8)     # First 80% for training data
no_stop_train_index = int(len(no_list) * 0.8)       # First 80% for training data

def write_clean_files_to_folder(stop_train_index, file_list, wav_folder_path, keyword):
    for index, wav_file in enumerate(file_list[stop_train_index:]):
        wav_file_path = wav_folder_path + "/" + wav_file
        
        # Load audio from local file repo
        audio = wave.open(wav_file_path)

        # Get length of data and sampling rate in audio
        audio_nframes = audio.getnframes()
        sampling_rate = audio.getframerate()

        # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
        signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
        signal = signal.astype(float)

        # Write Clean Signal
        output_file_path = f"{output_path}/{keyword}_Validation/{keyword}_Validation_{index}.wav"
        save_wav(output_file_path, signal, sampling_rate)

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
            filtered_signal_LPF = butter_lowpass_filter(noisy_signal, LPF_cutoff_freq, sampling_rate, order=10)
            # Write Noisy LPF audio to folder
            output_file_path = f"{output_path}/{keyword}_LPF/{keyword}_LPF_{index}.wav"
            save_wav(output_file_path, filtered_signal_LPF, sampling_rate)

            # HPF Filter the Signal
            filtered_signal_HPF = butter_highpass_filter(noisy_signal, HPF_cutoff_freq, sampling_rate, order=10)
            # Write Noisy HPF audio to folder
            output_file_path = f"{output_path}/{keyword}_HPF/{keyword}_HPF_{index}.wav"
            save_wav(output_file_path, filtered_signal_HPF, sampling_rate)

        if technique == 'bandpass_filter':
            low_pass_cutoff = 1200  # Hz
            high_pass_cutoff = 130  # Hz

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
            filtered_signal_LPF = butter_lowpass_filter(noisy_signal, low_pass_cutoff, sampling_rate, order=10)

            # HPF Filter the Signal
            filtered_signal_bandpass = butter_highpass_filter(filtered_signal_LPF, high_pass_cutoff, sampling_rate, order=10)
            # Write Noisy HPF audio to folder
            output_file_path = f"{output_path}/{keyword}_Bandpass/{keyword}_Bandpass_{index}.wav"
            save_wav(output_file_path, filtered_signal_bandpass, sampling_rate)

        # Delete, just so not much is processed right now.
        if index >= 1500:
            break

# Move Validation Data to Folder
write_clean_files_to_folder(yes_stop_train_index, yes_list, yes_folder_path, "Yes")
write_clean_files_to_folder(no_stop_train_index, no_list, no_folder_path, "No")
print("Validation Data Created")

# # Create the Noisy Data
# process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "create_noisy_data")
# process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "create_noisy_data")
# print("Data Created")

# # # Process Audio with Butterworth
# # process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "butterworth_filtering")
# # process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "butterworth_filtering")
# # print("Butterworth Filtering Finished")

# # Process Audio with Bandpass
# process_wav_files(yes_stop_train_index, yes_list, yes_folder_path, "Yes", "bandpass_filter")
# process_wav_files(no_stop_train_index, no_list, no_folder_path, "No", "bandpass_filter")
# print("Bandpass Filtering Finished")

# yes_noisy_signal_folder_path = "./audio/Yes_Noisy"
# yes_noisy_signal_list = file_list(yes_noisy_signal_folder_path)
# no_noisy_signal_folder_path = "./audio/No_Noisy"
# no_noisy_signal_list = file_list(no_noisy_signal_folder_path)

# # Process Audio with Spectral Substraction 
# process_wav_files(yes_stop_train_index, yes_noisy_signal_list, yes_folder_path, "Yes", "spectral_substraction")
# process_wav_files(no_stop_train_index, no_noisy_signal_list, no_folder_path, "No", "spectral_substraction")
# print("Spectral Substraction Finished")

# # Process Audio with Wiener Filtering
# process_wav_files(yes_stop_train_index, yes_noisy_signal_list, yes_folder_path, "Yes", "wiener_filter")
# process_wav_files(no_stop_train_index, no_noisy_signal_list, no_folder_path, "No", "wiener_filter")
# print("Wiener Filtering Finished")


def plot_waveform(wav_file_path, title):
    # Load audio from local file repo
    audio = wave.open(wav_file_path)

    # Get length of data and sampling rate in audio
    audio_nframes = audio.getnframes()
    sample_rate = audio.getframerate()

    # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
    signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
    signal = signal.astype(float)

    # Plot the entire song waveform
    # plt.figure().set_figwidth(12)
    plt.plot(signal, label=f'{title}')
    plt.xlabel('Samples')
    plt.ylabel('Signal')
    # plt.show()

def plot_fft_waveform(wav_file_path, title):
    # FFT of Full Audio Signal
    # Load audio from local file repo
    audio = wave.open(wav_file_path)

    # Get length of data and sampling rate in audio
    audio_nframes = audio.getnframes()
    sample_rate = audio.getframerate()

    # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
    signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
    signal = signal.astype(float)

    # Compute the FFT of the signal
    song_bins = np.fft.rfft(signal)

    # Take the absolute value of the amplitudes and rescale the data from 0 to 1 using the max bin value
    song_bins_vals = abs(song_bins) * 1/np.max(abs(song_bins))

    # Compute the frequency present in each bin
    song_bin_freqs = np.arange(0,len(song_bins)) / (len(signal)*(1/sample_rate))

    # Select an upper bin to restrict the view of the graph
    select_freq = 1500 # Hz # EDIT THIS
    closest_bin_index = np.argmin(np.abs(song_bin_freqs - select_freq))

    # plt.figure().set_figwidth(12)
    plt.plot(song_bin_freqs[:closest_bin_index], song_bins_vals[:closest_bin_index], label=f'{title}')
    plt.xlabel('Hz')
    plt.ylabel('|X[k]|')
    # plt.show()

def display_audio(wav_file_path):               # Didn't really work
    # Load audio from local file repo
    audio = wave.open(wav_file_path)

    # Get length of data and sampling rate in audio
    audio_nframes = audio.getnframes()
    sample_rate = audio.getframerate()

    # Copy samples from loaded file to a NumPy buffer, loaded as signed 16bit PCM
    signal = np.frombuffer(audio.readframes(audio_nframes), dtype = np.int16)
    signal = signal.astype(float)

    song_module = Audio(data = signal, rate = sample_rate)
    display(song_module) # Display module

# Plot Waveforms
'''
Types:
plot_waveform("keywords\yes\yes.0a7c2a8d_nohash_0.wav","Yes Waveform")
plot_waveform("keywords\\noise\\noise.doing_the_dishes.wav.0.wav","Noisy Waveform")
plot_waveform("audio\Yes_Noisy\Yes_Noisy_0.wav","Yes Noisy Waveform")
plot_waveform("audio\Yes_Wiener\Yes_Wiener_0.wav","Yes Wiener Waveform")
plot_waveform("audio\Yes_Subtraction\Yes_Subtraction_0.wav","Yes Subtraction Waveform")
'''
# plt.figure().set_figwidth(12)
# plot_waveform(f"audio\Yes_Noisy\Yes_Noisy_2.wav","Yes Noisy")
# plot_waveform(f"keywords\yes\{yes_list[2]}","Yes")
# plt.title('Clean and Noisy Keyword Waveform Comparison')
# plt.legend()
# plt.show()

# plt.figure().set_figwidth(12)
# plot_waveform(f"keywords\yes\{yes_list[2]}","Yes")
# plot_waveform(f"audio\Yes_Bandpass\Yes_Bandpass_2.wav","Yes Bandpass")
# plot_waveform(f"audio\Yes_Subtraction\Yes_Subtraction_2.wav","Yes Subtraction")
# plot_waveform(f"audio\Yes_Wiener\Yes_Wiener_2.wav","Yes Wiener")
# plt.title('Noisy Keyword Waveform Filter Comparison')
# plt.legend()
# plt.show()


# Plot FFTs
'''
Types:
plot_fft_waveform("audio\Yes_HPF\Yes_HPF_2.wav","Yes HPF")
plot_fft_waveform("audio\Yes_LPF\Yes_LPF_2.wav","Yes LPF")
plot_fft_waveform("keywords\yes\yes.0a7c2a8d_nohash_0.wav","Yes")
plot_fft_waveform("keywords\\noise\\noise.doing_the_dishes.wav.0.wav","Noisy")
plot_fft_waveform("audio\Yes_Noisy\Yes_Noisy_0.wav","Yes Noisy")
plot_fft_waveform("audio\Yes_Bandpass\Yes_Bandpass_0.wav","Yes Bandpass")
plot_fft_waveform("audio\Yes_Wiener\Yes_Wiener_0.wav","Yes Wiener")
plot_fft_waveform("audio\Yes_Subtraction\Yes_Subtraction_0.wav","Yes Subtraction")
'''

# plt.figure().set_figwidth(12)
# plot_fft_waveform("audio\Yes_Noisy\Yes_Noisy_2.wav","Yes Noisy")
# plot_fft_waveform(f"keywords\yes\{yes_list[2]}","Yes")
# plt.title('Clean and Noisy Keyword FFT Comparison')
# plt.legend()
# plt.show()

# plt.figure().set_figwidth(12)
# plot_fft_waveform("audio\Yes_Bandpass\Yes_Bandpass_2.wav","Yes Bandpass")
# plot_fft_waveform("audio\Yes_Subtraction\Yes_Subtraction_2.wav","Yes Subtraction")
# plot_fft_waveform("audio\Yes_Wiener\Yes_Wiener_2.wav","Yes Wiener")
# plot_fft_waveform(f"keywords\yes\{yes_list[2]}","Yes")
# plt.title('Noisy Keyword FFT Filter Comparison')
# plt.legend()
# plt.show()

# print(noise_list[2])
# print(len(yes_list))
# print(len(no_list))
# print(len(noise_list))

# print(yes_list[2])

# display_audio('audio\Yes_Bandpass\Yes_Bandpass_0.wav')













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