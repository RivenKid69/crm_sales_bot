#!/bin/bash
# Voice Bot Setup Script

echo "=================================="
echo "🚀 Voice Bot Setup"
echo "=================================="

# Check Python
echo "📦 Python version:"
python3 --version

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system dependencies for audio (Ubuntu/Debian)
echo ""
echo "📦 Installing system audio dependencies..."
echo "   You may need to run: sudo apt install portaudio19-dev python3-pyaudio"

# Install Python packages
echo ""
echo "📦 Installing Python packages..."
pip install faster-whisper ollama sounddevice soundfile numpy scipy rich torch

# TTS has heavy dependencies, install separately
echo ""
echo "📦 Installing Coqui TTS (this may take a while)..."
pip install TTS

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo "=================================="
echo ""
echo "To test components:"
echo "  1. STT:      python test_stt.py"
echo "  2. LLM:      python test_llm.py"
echo "  3. TTS:      python test_tts.py"
echo "  4. Full:     python voice_pipeline.py"
