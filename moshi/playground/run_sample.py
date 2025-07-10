from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from moshi.models import LMGen, loaders

# 1. Download weights from Hugging Face
mimi_ckpt = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
moshi_ckpt = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)

# 2. Initialize Mimi (audio codec) and Moshi LM
device = "cuda:0" if torch.cuda.is_available() else "cpu"
mimi = loaders.get_mimi(mimi_ckpt, device=device)
mimi.set_num_codebooks(8)  # up to 32 for full Mimi; 8 is default for Moshi

moshi_lm = loaders.get_moshi_lm(moshi_ckpt, device=device)
lm_gen = LMGen(moshi_lm, temp=0.8, temp_text=0.7)

# 3. Load and (optionally) resample your input audio
audio_path = Path(__file__).parent / "input.wav"
wav, sr = torchaudio.load(audio_path)  # [1, T], any sr
if sr != 24000:
    wav = torchaudio.functional.resample(wav, sr, 24000)
wav = wav.unsqueeze(0)  # [B=1, C=1, T]
wav = wav.to(device)

# 4. Encode entire signal into Mimi codes (non-streaming demo)
with torch.no_grad():
    full_codes = mimi.encode(wav)  # [1, K=8, T_frames]


# 5. Stream-based inference: feed Mimi codes into Moshi, decode on the fly
frame_size = mimi.frame_size
out_chunks = []

with torch.no_grad(), lm_gen.streaming(1), mimi.streaming(1):
    # First, get the raw Mimi codes broken into frames:
    for offset in range(0, full_codes.shape[-1], 1):  # one code-frame at a time
        code_frame = full_codes[:, :, offset : offset + 1].to(device)
        # Moshi generates audio+text tokens; here we step one code-frame
        tokens = lm_gen.step(code_frame)
        if tokens is not None:
            # tokens[:, 1:] are Mimi audio codes (first slot is text)
            audio_codes = tokens[:, 1:]
            chunk = mimi.decode(audio_codes)
            out_chunks.append(chunk.cpu())

# 6. Concatenate output chunks and save
output_wav = torch.cat(out_chunks, dim=-1)  # [1, 1, T_out]
torchaudio.save("moshi_output.wav", output_wav.squeeze(0), 24000)
print("Saved generated audio to moshi_output.wav")
