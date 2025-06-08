import sys,os,torch
sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")
sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
import GPT_SoVITS.eres2net.kaldi as Kaldi
class SV:
    def __init__(self,device,is_half):
        self.sv_path = self.get_venv_site_packages_path(sv_path)
        pretrained_state = torch.load(sv_path, map_location='cpu', weights_only=False)
        embedding_model = ERes2NetV2(baseWidth=24,scale=4,expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model=embedding_model
        if is_half == False:
            self.embedding_model=self.embedding_model.to(device)
        else:
            self.embedding_model=self.embedding_model.half().to(device)
        self.is_half=is_half

    def compute_embedding3(self,wav):
        with torch.no_grad():
            if self.is_half==True:wav=wav.half()
            feat = torch.stack([Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav])
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb

    def get_venv_site_packages_path(self, sv_path):
        root_dir = os.getcwd()

        local_sovits_path = os.path.join(root_dir, sv_path)
        # Check if the current sovits path is valid
        if os.path.isfile(local_sovits_path):
            sv_path = local_sovits_path
        else:
            # If the current sovits path is not valid, fallback to venv site-packages path
            site_packages_path = next(p for p in sys.path if 'site-packages' in p)
            sv_path = os.path.join(site_packages_path, sv_path)
        return sv_path