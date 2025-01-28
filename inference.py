import torch
import sys
import os
import soundfile as sf
import sounddevice as sd

# Get the absolute path to GPT_SoVITS directory
base_path = os.path.abspath("C:/Users/spava/OneDrive/Desktop/GPT-SoVITS-Training/GPT_SoVITS")
sys.path.append(base_path)  # Add GPT_SoVITS to Python path

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

# Define paths to your trained models
t2s_checkpoint = "C:/Users/spava/OneDrive/Desktop/GPT-SoVITS-Training/GPT_SoVITS/pretrained_models/HuTaoClone-e15.ckpt"  # Replace with actual path
vits_checkpoint = "C:/Users/spava/OneDrive/Desktop/GPT-SoVITS-Training/GPT_SoVITS/pretrained_models/HuTaoClone_e8_s184.pth"  # Replace with actual path
ref_audio = "audio4.wav"  # A reference audio file for cloning voice

# Create a config dictionary
config = {
    "custom":{
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "t2s_weights_path": t2s_checkpoint,
        "vits_weights_path": vits_checkpoint,
    }

    # For default use
    # "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Initialize the TTS model
tts = TTS(TTS_Config(config))

# Set reference audio (used for voice cloning)
tts.set_ref_audio(ref_audio)

# Define the text input
text_input = {
    # "text": "Hello, how do you do? My name is Hu Tao. How can I assist you? Would you perhaps like a cup of tea? Anyway, what have you been up to as of lately?",
    # "text": "Two households both alike in dignity, In fair Verona, where we lay our scene, From ancient grudge break to new mutiny, Where civil blood makes civil hands unclean.",
    "text":"Oh how joyous! How wonderful would it be if I could travel across the world. It has always been a dream of mine.",
    "text_lang": "en",
    # "prompt_text": "When the sun's out, bathe in sunlight. But when the moon's out, bathe in moonlight.",
    # "prompt_lang": "en",
    "ref_audio_path": ref_audio,
    "temperature": 1,
    "batch_size": 1,
    "stream_output": True,
    "max_chunk_size": 10, #Ideally keep low to reduce perceived delay, but not too low. Default 10 is good for most cases.
}

import time
from queue import Queue, Empty
from threading import Thread
import numpy as np

print('\n')
def audio_gen():
    generator = tts.run(text_input)
    receive_time = time.time()
    while True:
        try:
            audio_chunk = next(generator)
            audio_queue.put(audio_chunk)
            print(f"Received new audio {time.time() - receive_time}!")
            # np.set_printoptions(threshold=sys.maxsize)
            # with open("stream_audio.txt", "a") as f:
            #     f.write(f"\n\n\n\n\n\n\n\n\n\n\n{str(audio_chunk)}")
        except StopIteration:
            print("Stream Queue Done.")
            break

with sd.OutputStream(samplerate=tts.configs.sampling_rate, channels=1, dtype="float32") as stream:
    audio_queue = Queue()
    audio_gen_thread = Thread(target=audio_gen)
    audio_gen_thread.start()
    i = 1
    while True:
        try:
            new_audio_data = audio_queue.get(timeout=5)
            stream.write(new_audio_data)
            if new_audio_data is not None:
                # sf.write(f"C:/Users/spava/OneDrive/Desktop/GPT-SoVITS-Training/output/new_output_full.wav", new_audio_data, tts.configs.sampling_rate)
                # sf.write(f"C:/Users/spava/OneDrive/Desktop/GPT-SoVITS-Training/output/stream_output{i}.wav", new_audio_data, tts.configs.sampling_rate)
                i += 1
        except (KeyboardInterrupt, Empty):
            print("Exiting!")
            break

# CURRENT SEED
# seed = 2390348277