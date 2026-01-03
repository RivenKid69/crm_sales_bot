"""
Test Speech-to-Text with faster-whisper
Records audio from microphone and transcribes it
"""
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

# Check if faster-whisper is installed
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Installing faster-whisper...")
    import subprocess
    subprocess.run(["pip", "install", "faster-whisper"])
    from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)


def record_audio(duration: float = 5.0) -> np.ndarray:
    """Record audio from microphone"""
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32
    )
    sd.wait()
    print("✅ Recording complete!")
    return audio.flatten()


def save_audio(audio: np.ndarray, filename: str) -> Path:
    """Save audio to file"""
    filepath = AUDIO_DIR / filename
    sf.write(filepath, audio, SAMPLE_RATE)
    return filepath


def transcribe_audio(model: WhisperModel, audio_path: Path) -> tuple[str, float]:
    """Transcribe audio file and return text with timing"""
    start_time = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language="ru",  # Russian language
        beam_size=5,
        vad_filter=True,  # Voice activity detection
    )

    text = " ".join([segment.text for segment in segments])
    elapsed = time.time() - start_time

    return text.strip(), elapsed


def test_stt():
    """Main test function"""
    print("=" * 50)
    print("🎙️  STT Test (faster-whisper)")
    print("=" * 50)

    # Load model
    print("\n📥 Loading Whisper model (base)...")
    model_start = time.time()

    # Use 'base' for balance of speed/quality, 'small' or 'medium' for better quality
    # compute_type: int8, float16, float32
    model = WhisperModel(
        "base",
        device="cpu",  # or "cuda" for GPU
        compute_type="int8"  # Faster on CPU
    )

    print(f"✅ Model loaded in {time.time() - model_start:.2f}s")

    # Record audio
    audio = record_audio(duration=5.0)
    audio_path = save_audio(audio, "test_recording.wav")
    print(f"💾 Audio saved to: {audio_path}")

    # Transcribe
    print("\n🔄 Transcribing...")
    text, transcribe_time = transcribe_audio(model, audio_path)

    # Results
    print("\n" + "=" * 50)
    print("📊 Results:")
    print("=" * 50)
    print(f"📝 Text: {text}")
    print(f"⏱️  Transcription time: {transcribe_time:.2f}s")
    print(f"📈 Audio duration: 5.0s")
    print(f"🚀 Real-time factor: {transcribe_time / 5.0:.2f}x")

    return text, transcribe_time


if __name__ == "__main__":
    test_stt()
