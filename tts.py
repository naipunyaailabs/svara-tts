# tts.py
import torch
import numpy as np
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
from gpu_config import device

# -------------------------------------------------
# Device Selection handled by gpu_config.py
# -------------------------------------------------

print(f"Using device: {device}")

# -------------------------------------------------
# Load Models
# -------------------------------------------------
print("Loading SNAC model...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

print("Loading Svara-TTS model...")
model_name = "kenpath/svara-tts-v1"
tts_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("✓ Models loaded successfully")

# -------------------------------------------------
# Main TTS Function
# -------------------------------------------------
def generate_svara_tts(text: str, language: str, gender: str, outfile="output.wav"):
    """
    Generate TTS audio using Svara-TTS + SNAC decoder.
    """
    # -------------------------
    # Step 1 — Build Prompt
    # -------------------------
    voice_tag = f"{language} ({gender})"
    formatted_text = f"<|audio|> {voice_tag}: {text}<|eot_id|>"
    prompt = f"<custom_token_3>{formatted_text}<custom_token_4><custom_token_5>"

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Add special tokens
    start_token = torch.tensor([[128259]])
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]])

    modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)

    # -------------------------
    # Step 2 — Generate SNAC Tokens
    # -------------------------
    with torch.no_grad():
        output = tts_model.generate(
            input_ids=modified_ids,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=128258,
        )

    row = output[0]

    START = 128257
    END = 128258
    BASE = 128266

    token_indices = (row == START).nonzero(as_tuple=True)[0]

    if len(token_indices) == 0:
        raise ValueError("No speech tokens found")

    start_idx = token_indices[-1].item() + 1
    audio_tokens = row[start_idx:]
    audio_tokens = audio_tokens[audio_tokens != END]
    audio_tokens = audio_tokens[audio_tokens != 128263]

    # Valid SNAC range
    valid_mask = (audio_tokens >= BASE)
    audio_tokens = audio_tokens[valid_mask]
    snac_tokens = (audio_tokens - BASE).tolist()

    # Trim to multiples of 7
    snac_tokens = snac_tokens[: len(snac_tokens)//7 * 7]

    # -------------------------
    # Step 3 — Rebuild Hierarchical SNAC Levels
    # -------------------------
    offsets = [i * 4096 for i in range(7)]
    lvls = [[] for _ in range(3)]

    for i in range(0, len(snac_tokens), 7):
        lvls[0].append(snac_tokens[i] - offsets[0])
        lvls[1].append(snac_tokens[i+1] - offsets[1])
        lvls[1].append(snac_tokens[i+4] - offsets[4])
        lvls[2].append(snac_tokens[i+2] - offsets[2])
        lvls[2].append(snac_tokens[i+3] - offsets[3])
        lvls[2].append(snac_tokens[i+5] - offsets[5])
        lvls[2].append(snac_tokens[i+6] - offsets[6])

    hierarchical = [
        torch.tensor(lvl, dtype=torch.long).unsqueeze(0).to(device)
        for lvl in lvls
    ]

    # -------------------------
    # Step 4 — Decode to Audio
    # -------------------------
    with torch.no_grad():
        audio = snac_model.decode(hierarchical)

    audio_np = audio.squeeze().cpu().numpy()

    # Save to WAV
    sf.write(outfile, audio_np, 24000)

    print(f"✔ Audio generated: {outfile}")
    print(f"⏱ Duration: {len(audio_np) / 24000:.2f} seconds")

    return outfile
