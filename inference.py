import os
import time
import torch
from queue import Queue
import sounddevice as sd
from threading import Thread
from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITS_TTS, TTS_Config

class TTS:
    """A simple class that defines an entry point for text to speech tasks."""
    def __init__(self):
        # Base Paths
        self.bert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.cnhuhbert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        
        # Custom v2ProPlus
        self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka_EN_v2ProPlus-e15.ckpt"
        self.vits_checkpoint = "GPT_SoVITS/pretrained_models/v2Pro/ayaka/Ayaka_EN_v2ProPlus_e8_s2440.pth"
        
        # Base Paths for Each v2 Model If No Custom Model Available
        
        # Base v2ProPlus
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        
        # Base v2Pro
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
        
        # Base v2
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        
        self.ref_audio = "audio/ayaka/ref_audio/ref1.wav"
        
        self.config = {
            "custom": {
                "bert_base_path": self.bert_checkpoint,
                "cnhuhbert_base_path": self.cnhuhbert_checkpoint,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": False,
                "t2s_weights_path": self.t2s_checkpoint,
                "vits_weights_path": self.vits_checkpoint,
            }
        }
        
        self.tts = GPTSoVITS_TTS(TTS_Config(self.config))
        
        # Reference Audios (Optional For Better Tuning To Target Voice)
        aux_ref_audios_path = "audio/ayaka/aux_ref_audio"
        self.aux_ref_audios = [f"{aux_ref_audios_path}/{file_name}" for file_name in os.listdir(aux_ref_audios_path)]
        
        # Audio Streaming Setup
        self.stream_thread: Thread = None
        self.stream = sd.OutputStream(samplerate=32000, channels=1, dtype="float32")
        self.audio_queue = Queue()
        self.streaming_audio = False

        # Runs a quick warmup to get everything setup for fast inference in the following calls to synthesize
        self.synthesize("Hello world.", is_warmup=True)
    
    
    def audio_stream(self):
        """Handles audio playback from synthesized data."""
        self.stream.start()
        self.streaming_audio = True
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                self.stream.stop()
                self.streaming_audio = False
                return
            self.stream.write(audio_data)
    
    
    def synthesize(self, text: str, text_lang: str = "en", speed_factor: float = 1, is_warmup: bool = False):
        """Entry point to synthesizing text into speech.

        Args:
            text (str, required): The text to synthesize into speech.
            text_lang (str, optional): The language of the text to synthesize into speech. Defaults to english ("en").
            speed_factor (float, optional): The speed of the synthesized audio. Usually left alone unless the speech needs to be sped up/slowed down. Defaults to 1.
            is_warmup (bool, optional): Marks whether this call to synthesize is for warming up the model. Usually only called when initializing the model for inference. Defaults to False.
        """

        args = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self.ref_audio,
            "aux_ref_audio_paths": self.aux_ref_audios,
            "prompt_text": "Don't worry. Now that I've experienced the event once already, I won't be easily frightened. I'll see you later. Have a lovely chat with your friend.",
            "prompt_lang": "en",
            "batch_size": 1,
            "temperature": 1,
            "top_k": 50,
            "top_p": 0.9,
            "speed_factor": speed_factor,
            "fragment_interval": 0.01, # This doesnt do anything with v2
            "seed": 42,
            "streaming": True,
            # Helps get the first chunk of audio out faster and then adjusts to a more reasonable chunk size over time
            "initial_chunk_size": 10,
            "chunk_increase_rate": 5,
            "max_chunk_size": 20,
        }
        
        start_time = time.time()
        print(f"Synthesis Start ({time.time() - start_time:.2f}s)")
        if text:
            for _, audio_data in self.tts.run(args):
                if not is_warmup:   
                    if not self.streaming_audio:
                        self.stream_thread = Thread(target=self.audio_stream)
                        self.stream_thread.start()
                    self.audio_queue.put(audio_data)
                    print(f"Queueing Audio Data ({time.time() - start_time:.2f}s) Of Length {audio_data.shape[0] / 32000}s")

        if is_warmup:
            print(f"TTS Warmup Completed ({time.time() - start_time:.2f}s)")
        else:
            self.audio_queue.put(None)
            print(f"Last Chunk. Inserting Sentinel Value.")
            self.stream_thread.join()
            print(f"Stream Thread Complete ({time.time() - start_time:.2f}s)")

# Usage
tts = TTS()
text = "Earth is the third planet from the Sun and the only known astronomical object to harbor life, characterized by its dynamic systems including oceans, atmosphere, and tectonic plates that continuously reshape its surface."
tts.synthesize(text, text_lang="en", speed_factor=1)