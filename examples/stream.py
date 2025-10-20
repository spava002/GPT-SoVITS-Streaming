import os, sys, time
import numpy as np
import soundfile as sf

cwd = os.getcwd()
sys.path.append(cwd)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

config = {
    "custom": {
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": "cuda",
        "is_half": False,
        "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "version": "v2ProPlus",
        "vits_weights_path": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    }
}

tts_config = TTS_Config(config) # or you can use and edit "GPT_SoVITS/configs/tts_infer.yaml" 
tts_pipeline = TTS(tts_config)

inputs = {
    "text": "", # your text to be synthesized
    "text_lang": "en", # your text's language
    "ref_audio_path": "", # your audio file path
    "prompt_text": "", # your audio file's transcription
    "prompt_lang": "", # your audio file's language
    "return_fragment": True, # streaming enabled
    "parallel_infer": False,
    "initial_chunk_size": 10, # the amount of tokens to begin yielding audio for
    "chunk_increase_rate": 5, # the amount of tokens to increase the chunk size by
    "max_chunk_size": 20, # the maximum amount of tokens to be yielded per chunk
}

# Warmup the model for faster inference
warmup_inputs = inputs.copy()
warmup_inputs["text"] = "Hello, world!"
next(tts_pipeline.run(warmup_inputs))

start_time = time.time()
chunks = []
for i, (sample_rate, audio_chunk) in enumerate(tts_pipeline.run(inputs)):
    print(f"Received chunk {len(audio_chunk) / sample_rate:.3f}s at {time.time() - start_time:.3f}s")
    sf.write(f"examples/output/stream_output_{i}.wav", audio_chunk, sample_rate)
    chunks.append(audio_chunk)

full_audio = np.concatenate(chunks)

print(f"Time taken: {time.time() - start_time:.3f}s")
print(f"Audio length: {len(full_audio) / sample_rate:.3f}s")

sf.write("examples/output/stream_output_full.wav", full_audio, sample_rate)