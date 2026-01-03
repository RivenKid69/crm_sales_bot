"""
Test Text-to-Speech with F5-TTS
Flow Matching based TTS - fast and high quality
"""
import time
import torch
import sounddevice as sd
import soundfile as sf
from pathlib import Path

from f5_tts.api import F5TTS

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Reference audio for voice cloning (will be created if not exists)
REFERENCE_AUDIO = AUDIO_DIR / "reference.wav"


class F5TTSWrapper:
    """F5-TTS wrapper for Russian TTS"""

    def __init__(self):
        print("📥 Loading F5-TTS model...")
        start = time.time()

        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")

        # Initialize F5-TTS
        self.model = F5TTS(device=self.device)

        print(f"✅ Model loaded in {time.time() - start:.2f}s")

    def synthesize(
        self,
        text: str,
        ref_audio: str = None,
        ref_text: str = None,
        output_path: Path = None
    ) -> tuple:
        """Generate speech from text"""
        start = time.time()

        # Generate audio
        audio, sample_rate, _ = self.model.infer(
            ref_file=ref_audio,
            ref_text=ref_text or "",
            gen_text=text,
            file_wave=str(output_path) if output_path else None,
            seed=-1,  # Random seed
        )

        elapsed = time.time() - start

        return audio, sample_rate, elapsed


def play_audio(audio, sample_rate: int):
    """Play audio"""
    print("🔊 Playing...")
    sd.play(audio, sample_rate)
    sd.wait()


def create_reference_audio():
    """Create a simple reference audio using built-in voice"""
    if REFERENCE_AUDIO.exists():
        return str(REFERENCE_AUDIO)

    print("📝 No reference audio found, using default voice")
    return None


def test_tts():
    """Main test function"""
    print("=" * 50)
    print("🔊 TTS Test (F5-TTS)")
    print("=" * 50)

    # Initialize
    tts = F5TTSWrapper()

    # Reference audio (optional - for voice cloning)
    ref_audio = create_reference_audio()

    # Test texts
    texts = [
        "Привет! Я голосовой ассистент.",
        "Искусственный интеллект помогает решать сложные задачи.",
        "Сегодня хорошая погода для прогулки.",
    ]

    results = []

    for i, text in enumerate(texts):
        print(f"\n📝 Text {i+1}: {text}")
        print("🔄 Synthesizing...")

        output_path = AUDIO_DIR / f"f5tts_test_{i+1}.wav"
        audio, sr, elapsed = tts.synthesize(
            text,
            ref_audio=ref_audio,
            output_path=output_path
        )

        duration = len(audio) / sr

        print(f"✅ Generated in {elapsed:.2f}s")
        print(f"📈 Audio duration: {duration:.2f}s")
        print(f"🚀 Real-time factor: {elapsed / duration:.2f}x")

        results.append({
            "text": text,
            "synthesis_time": elapsed,
            "audio_duration": duration,
            "rtf": elapsed / duration
        })

        # Play audio
        play_audio(audio, sr)

    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print("=" * 50)

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    print(f"🚀 Average RTF: {avg_rtf:.2f}x")

    if avg_rtf < 1.0:
        print("✅ TTS is faster than real-time!")
    else:
        print("⚠️  TTS is slower than real-time (consider GPU)")

    return results


if __name__ == "__main__":
    test_tts()
