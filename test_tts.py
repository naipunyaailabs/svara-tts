from tts import generate_svara_tts

# Example usage
text = "नमस्ते, आप कैसे हैं? यह आवाज Svara-TTS द्वारा उत्पन्न की गई है।"
language = "Hindi"
gender = "Female"

generate_svara_tts(text, language, gender, outfile="demo.wav")
