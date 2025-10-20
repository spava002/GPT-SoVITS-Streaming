import os, sys, time
import soundfile as sf

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

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
    "text_lang": "", # your text's language
    "ref_audio_path": "", # your audio file path
    "prompt_text": "", # your audio file's transcription
    "prompt_lang": "", # your audio file's language
    "return_fragment": False, # streaming disabled
    "parallel_infer": True
}

# Warmup the model for faster inference
warmup_inputs = inputs.copy()
warmup_inputs["text"] = "Hello, world!"
next(tts_pipeline.run(warmup_inputs))

start_time = time.time()
sample_rate, audio_chunk = next(tts_pipeline.run(inputs))

print(f"Time taken: {time.time() - start_time:.3f}s")
print(f"Audio length: {len(audio_chunk) / sample_rate:.3f}s")

sf.write("examples/output/run_output_full.wav", audio_chunk, sample_rate)