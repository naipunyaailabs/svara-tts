# Svara TTS - Configurable GPU/CPU Execution

This is a text-to-speech (TTS) service that can be configured to run on GPU or fall back to CPU/MPS.

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch with CUDA support
- See [requirements.txt](requirements.txt) for full dependencies

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### API Server
```bash
uvicorn api:app --reload
```

### Streamlit Interface
```bash
streamlit run app.py
```

## GPU Configuration

The application can be configured to run on GPU only or automatically select the best available device (GPU > Apple Silicon > CPU).

To enforce GPU usage, set `ENFORCE_GPU = True` in [gpu_config.py](gpu_config.py).
To allow flexible device selection, set `ENFORCE_GPU = False`.

## Testing GPU Enforcement

Run the test script to verify GPU enforcement:
```bash
python test_gpu_enforcement.py
```