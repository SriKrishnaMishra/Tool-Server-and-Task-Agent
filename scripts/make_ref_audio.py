"""Create a Hindi reference voice clip with the existing Bodhan SPEAK client."""

from __future__ import annotations

import asyncio
import math
import shutil
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

from bodhan_mcp.client import BodhanClient  # noqa: E402

TEXT = "नमस्ते, मेरा नाम कृष्णा है। मैं एक छात्र हूँ। आज मौसम बहुत अच्छा है।"
OUTPUT = ROOT / "demo" / "inputs" / "voice-note.wav"
MAX_COST_INR = 0.20


def _rms_peak_db(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as recording:
        if recording.getsampwidth() != 2:
            raise RuntimeError("Converted reference audio is not 16-bit PCM.")
        frames = recording.readframes(recording.getnframes())
        samples = array("h")
        samples.frombytes(frames)
        if not samples:
            raise RuntimeError("Converted reference audio contains no samples.")

    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak_db = 20 * math.log10(peak / 32768) if peak else float("-inf")
    rms_db = 20 * math.log10(rms / 32768) if rms else float("-inf")
    return rms_db, peak_db


async def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("Preflight: FAIL")
        print("ffmpeg is not available on PATH.")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bodhan-ref-audio-") as temp_name:
        temp_dir = Path(temp_name)
        async with BodhanClient(
            cache_dir=temp_dir / "cache",
            max_retries=0,
        ) as client:
            result = await client.speak(TEXT, language="hi", voice="Suhani")

        if not result.ok:
            print("Synthesize: FAIL")
            print(f"  code={result.error.code if result.error else 'synthesis_failed'}")
            message = result.error.message if result.error else "Unknown synthesis failure"
            print(f"  message={message}")
            print(f"  cost_inr={result.cost_inr:.6f}")
            return 1

        source = result.data.get("audio_path") or result.data.get("output_file")
        if not isinstance(source, str) or not Path(source).is_file():
            print("Synthesize: FAIL")
            print("  response did not contain a local audio path")
            return 1

        normalized = temp_dir / "voice-note.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-1.5",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(normalized),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print("Convert: FAIL")
            print(completed.stderr[-2000:])
            return 1

        rms_db, peak_db = _rms_peak_db(normalized)
        with wave.open(str(normalized), "rb") as recording:
            duration = recording.getnframes() / recording.getframerate()

        if duration <= 0:
            print("Verify: FAIL")
            print("  duration=0")
            return 1
        if rms_db <= -30:
            print("Verify: FAIL")
            print(f"  duration={duration:.3f}s")
            print(f"  rms_db={rms_db:.2f}")
            print(f"  peak_db={peak_db:.2f}")
            print("  RMS level must be above -30 dB.")
            return 1
        if result.cost_inr >= MAX_COST_INR:
            print("Verify: FAIL")
            print(f"  cost_inr={result.cost_inr:.6f}")
            print(f"  limit_inr={MAX_COST_INR:.2f}")
            return 1

        OUTPUT.write_bytes(normalized.read_bytes())
        print(f"Synthesize: PASS | voice=Suhani | cost_inr={result.cost_inr:.6f}")
        print(f"duration={duration:.3f}s")
        print(f"rms_db={rms_db:.2f}")
        print(f"peak_db={peak_db:.2f}")
        print(f"output={OUTPUT}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
