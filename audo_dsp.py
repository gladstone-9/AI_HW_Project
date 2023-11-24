import tensorflow as tf
import tensorflow_datasets as tfds
from pydub import AudioSegment
import os

# Specify the dataset name
dataset_name = "speech_commands"

# Load the dataset
ds = tfds.load(name=dataset_name, split="train")

# Output directory for saving audio files
output_dir = "speech_commands_audio"
os.makedirs(output_dir, exist_ok=True)

# Function to convert and save audio
def save_audio(example):
    audio_data = example["audio"]
    label = example["label"].numpy().decode("utf-8")
    file_name = f"{label}_{example['file_name'].numpy().decode('utf-8')}.wav"
    file_path = os.path.join(output_dir, file_name)

    # Convert to PCM format
    pcm_data = tf.audio.encode_wav(audio_data, sample_rate=16000)

    # Write to WAV file
    with tf.io.gfile.GFile(file_path, "wb") as file:
        file.write(pcm_data.numpy())

# Iterate through the dataset and save audio files
for example in ds.take(5):  # Change the number to save more or fewer files
    save_audio(example)

print(f"Audio files saved to {output_dir}")
