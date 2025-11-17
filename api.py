from fastapi import FastAPI
from pydantic import BaseModel
from tts import generate_svara_tts

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    language: str
    gender: str

@app.post("/tts")
async def tts(req: TTSRequest):
    filename = "output_api.wav"
    generate_svara_tts(req.text, req.language, req.gender, filename)
    return {"status": "ok", "file": filename}
