import os, time, torch
import numpy as np
from queue import Queue
import sounddevice as sd
from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITS_TTS, TTS_Config

class TTS:
    """An entry point for TTS using GPTSoVITS."""
    
    # How often the callback for the output stream is called
    # After experimentation, this should not fall below 0.25 to avoid audio glitches from python GIL releasing resources when TTS finishes
    # and should be at most 0.75 to ensure the first chunk gets played as soon as its received
    STREAM_BLOCK_SIZE = 0.25
    
    def __init__(self):
        # Base Paths
        self.bert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.cnhuhbert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        
        # Custom v2ProPlus
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka_EN_v2ProPlus-e15.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka_EN_v2ProPlus_e8_s2440.pth"
        
        # Base Paths for Each v2 Model If No Custom Model Available
        
        # Base v2ProPlus
        self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        self.vits_checkpoint = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        
        # Base v2Pro
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
        
        # Base v2
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        
        self.ref_audio = "audio/ayaka/ref_audio/ref1.wav"
        
        config = {
            "custom": {
                "bert_base_path": self.bert_checkpoint,
                "cnhuhbert_base_path": self.cnhuhbert_checkpoint,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": False,
                "version": "v2ProPlus",
                "t2s_weights_path": self.t2s_checkpoint,
                "vits_weights_path": self.vits_checkpoint,
            }
        }
        
        self.tts = GPTSoVITS_TTS(TTS_Config(config))
        
        # Reference Audios (Optional For Better Tuning To Target Voice)
        aux_ref_audios_path = "audio/ayaka/aux_ref_audio"
        self.aux_ref_audios = [f"{aux_ref_audios_path}/{file_name}" for file_name in os.listdir(aux_ref_audios_path)]
        
        # Audio Streaming Setup
        self.stream = sd.OutputStream(samplerate=32000, channels=1, dtype="float32", callback=self.callback, blocksize=int(32000 * self.STREAM_BLOCK_SIZE))
        self.audio_queue: Queue[np.ndarray] = Queue()
        
        # Chunk playback state
        self.current_chunk = None

        # Runs a quick warmup to get everything setup for fast inference in the following calls to synthesize
        self.synthesize("Hello world.", is_warmup=True)
 

    def callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):        
        # Ensure we have enough data to fill the output buffer
        while self.current_chunk is None or len(self.current_chunk) < frames:
            new_chunk = self.audio_queue.get()
            
            # Check if we've received the sentinel value
            if new_chunk is None:
                print("Sentinel Value Reached, Stopping Output Stream!")
                remaining_frames = frames - len(self.current_chunk)
                outdata[:] = np.concatenate((self.current_chunk, np.zeros((remaining_frames, 1), dtype=np.float32)))
                raise sd.CallbackStop()
            
            # Ensure chunk is 2D (frames, channels)
            if new_chunk.ndim == 1:
                new_chunk = new_chunk[:, np.newaxis]
            
            if self.current_chunk is None:
                self.current_chunk = new_chunk
            else:
                self.current_chunk = np.concatenate((self.current_chunk, new_chunk))
                
        # Copy the data to the output buffer and save any remaining data for the next callback
        outdata[:] = self.current_chunk[:frames]
        self.current_chunk = self.current_chunk[frames:]


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
            "seed": 42,
            "streaming": True,
            "split_bucket": False,
            # Helps get the first chunk of audio out faster and then adjusts to a more reasonable chunk size over time
            "initial_chunk_size": 5,
            "chunk_increase_rate": 10,
            "max_chunk_size": 100,
        }
        
        start_time = time.time()
        print(f"Synthesis Start ({time.time() - start_time:.2f}s)")
        # i = 0
        if text:
            for sample_rate, audio_chunk in self.tts.run(args):
                if is_warmup:
                    continue

                if self.stream and not self.stream.active:
                    self.stream.start()
                    
                self.audio_queue.put(audio_chunk)
                print(f"Queueing Audio Data ({time.time() - start_time:.2f}s) Of Length {audio_chunk.shape[0] / sample_rate}s")

        if is_warmup:
            print(f"TTS Warmup Completed ({time.time() - start_time:.2f}s)")
        else:
            print(f"Synthesis Complete. Inserting Sentinel Value.")
            self.audio_queue.put(None)
            
            # Wait for stream to finish playback
            while self.stream.active:
                time.sleep(0.1)
            
            print(f"Stream Thread Complete ({time.time() - start_time:.2f}s)")


# Usage
tts = TTS()
text = "Earth is the third planet from the Sun and the only known astronomical object to harbor life, characterized by its dynamic systems including oceans, atmosphere, and tectonic plates that continuously reshape its surface. Its unique position in the habitable zone of our solar system, along with its protective magnetic field and diverse ecosystems, has allowed for the evolution of millions of species over approximately 4.5 billion years. Despite covering only a fraction of the universe, Earth remains our irreplaceable home—a remarkable blue marble suspended in the vastness of space that continues to reveal its secrets through scientific discovery."
# text = "Romeo and Juliet fall in love with each other and they are going to get married. But their families are not going to let them get married. They are going to fight with each other and they are going to kill each other. But Romeo and Juliet are not going to let their families kill them. They are going to get married and they are going to live happily ever after."
tts.synthesize(text, text_lang="en", speed_factor=1)