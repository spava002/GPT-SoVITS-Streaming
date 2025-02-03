import torch
import sounddevice as sd
import time
from queue import Queue
from threading import Thread

class TTS:
    def __init__(self):
        # Replace with your checkpoints and reference audio here
        # Note: Using a venv may require updating the default paths provided here
        self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        self.vits_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        self.bert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.cnhuhbert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.ref_audio = "audio4.wav"

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        self.config = {
            "custom": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "t2s_weights_path": self.t2s_checkpoint,
                "vits_weights_path": self.vits_checkpoint,
                "bert_base_path": self.bert_checkpoint,
                "cnhuhbert_base_path": self.cnhuhbert_checkpoint,
            }
        }
        
        self.tts = TTS(TTS_Config(self.config))
        
        self.audio_queue = Queue()
        self.generating_audio = False
    
    def audio_stream(self, start_time):
        with sd.OutputStream(samplerate=self.tts.configs.sampling_rate, channels=1, dtype="float32") as stream:
            while True:
                new_audio_data = self.audio_queue.get()
                if new_audio_data is None:
                    print(f"Stream Thread Done: {time.time() - start_time}")
                    break
                stream.write(new_audio_data)
    
    def synthesize(self, text, start_time, generating_text=False):
        if not self.generating_audio:
            audio_stream_thread = Thread(target=self.audio_stream, args=(start_time,))
            audio_stream_thread.start()
            self.generating_audio = True

        args = {
            "text": text,
            "text_lang": "en",
            "ref_audio_path": self.ref_audio,
            "temperature": 1,
            "batch_size": 1,
            "stream_output": True,
            "max_chunk_size": 10,
        }
        
        if text:
            print(f"Synthesis Start: {time.time() - start_time}")
            generator = self.tts.run(args)
            while True:
                try:
                    audio_chunk = next(generator)
                    self.audio_queue.put(audio_chunk)
                except StopIteration:
                    break

        if not generating_text:
            self.audio_queue.put(None)
            self.generating_audio = False
        
        print(f"Synthesis End: {time.time() - start_time}")

# Usage
tts = TTS()
"""
Time is only for debugging purposes. If not needed, feel free to remove.
Since this TTS model was built to be paired with LLM text streaming, we use a generating_text bool
this bool signifies if we are receiving the last chunk of streamed text.
"""
tts.synthesize("One day, a fierce storm rolled in, bringing heavy rain and strong winds that threatened to destroy the wheat crops.", time.time(), False)