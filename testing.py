import os

# Specify the path to the folder containing the .wav files
folder_path = "./keywords/yes"

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter only the .wav files
wav_files = [file for file in files]

print(wav_files[0])

# # Print the names of the .wav files
# for wav_file in files:
#     print(wav_file)
