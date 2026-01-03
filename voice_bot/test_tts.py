"""
Test Text-to-Speech with Piper TTS
Generates speech from text using local Russian voice model
"""
import time
import wave
import subprocess
import sounddevice as sd
import soundfile as sf
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "ru_RU-irina-medium.onnx"


def synthesize_speech(text: str, output_file: str = "output.wav") -> tuple[Path, float]:
    """Generate speech from text using Piper"""
    output_path = AUDIO_DIR / output_file

    start_time = time.time()

    # Run piper via subprocess
    cmd = [
        "piper",
        "--model", str(MODEL_PATH),
        "--output_file", str(output_path)
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=text.encode("utf-8"))

    elapsed = time.time() - start_time

    if process.returncode != 0:
        print(f"Error: {stderr.decode()}")
        return None, elapsed

    return output_path, elapsed


def play_audio(audio_path: Path):
    """Play audio file"""
    print(f"🔊 Playing: {audio_path.name}")
    data, samplerate = sf.read(audio_path)
    sd.play(data, samplerate)
    sd.wait()


def test_tts():
    """Main test function"""
    print("=" * 50)
    print("🔊 TTS Test (Piper)")
    print("=" * 50)

    # Check model
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run: ./setup.sh to download the model")
        return

    print(f"✅ Model: {MODEL_PATH.name}")

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

        output_path, elapsed = synthesize_speech(text, f"tts_test_{i+1}.wav")

        if output_path is None:
            print("❌ Synthesis failed")
            continue

        # Get audio duration
        data, sr = sf.read(output_path)
        duration = len(data) / sr

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
        play_audio(output_path)

    if not results:
        return

    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print("=" * 50)

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    print(f"🚀 Average RTF: {avg_rtf:.2f}x")

    if avg_rtf < 1.0:
        print("✅ TTS is faster than real-time!")
    else:
        print("⚠️  TTS is slower than real-time")

    return results


if __name__ == "__main__":
    test_tts()
