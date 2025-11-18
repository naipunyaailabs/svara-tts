import streamlit as st
import torch
import numpy as np
import soundfile as sf
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_config import device

# ---------------------------------------------------------
# DEVICE handled by gpu_config.py
# ---------------------------------------------------------

# ---------------------------------------------------------
# LOAD MODELS (cached by HF)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    tts_model = AutoModelForCausalLM.from_pretrained("kenpath/svara-tts-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("kenpath/svara-tts-v1")
    return snac_model, tts_model, tokenizer


snac_model, tts_model, tokenizer = load_models()

# ---------------------------------------------------------
# TTS FUNCTION
# ---------------------------------------------------------
def generate_svara_tts(text: str, language: str, gender: str):
    voice_tag = f"{language} ({gender})"
    formatted_text = f"<|audio|> {voice_tag}: {text}<|eot_id|>"
    prompt = f"<custom_token_3>{formatted_text}<custom_token_4><custom_token_5>"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Special tokens
    start_token = torch.tensor([[128259]])
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]])

    modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)

    # Generate SNAC Tokens
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

    valid_mask = (audio_tokens >= BASE)
    audio_tokens = audio_tokens[valid_mask]
    snac_tokens = (audio_tokens - BASE).tolist()

    snac_tokens = snac_tokens[: len(snac_tokens)//7 * 7]

    # Reconstruct hierarchical codes
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
        torch.tensor(level, dtype=torch.long).unsqueeze(0).to(device)
        for level in lvls
    ]

    with torch.no_grad():
        audio = snac_model.decode(hierarchical)

    audio_np = audio.squeeze().cpu().numpy()
    return audio_np


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Svara TTS", layout="centered")

st.title("🎤 Svara TTS (Open Source Voice Generator)")
st.write("Generate high-quality multilingual speech using **Svara-TTS + SNAC**")

text = st.text_area("Enter Text", "नमस्ते, आप कैसे हैं?")
col1, col2 = st.columns(2)

with col1:
    language = st.selectbox(
        "Language",
        ['Hindi', 'English', 'Bengali', 'Marathi', 'Tamil', 'Telugu', 'Kannada',
         'Gujarati', 'Punjabi', 'Malayalam', 'Assamese', 'Bodo', 'Dogri',
         'Maithili', 'Bhojpuri', 'Chhattisgarhi', 'Nepali', 'Sanskrit']
    )

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])

generate_btn = st.button("🎙️ Generate Speech")

if generate_btn:
    with st.spinner("Generating audio... Please wait..."):
        try:
            audio = generate_svara_tts(text, language, gender)
            sf.write("output.wav", audio, 24000)

            st.success("Audio generated successfully!")
            st.audio("output.wav", format="audio/wav")

            st.download_button(
                label="⬇️ Download Audio",
                data=open("output.wav", "rb").read(),
                file_name="output.wav",
                mime="audio/wav"
            )

        except Exception as e:
            st.error(f"Error: {str(e)}")
