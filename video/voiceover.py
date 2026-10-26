"""Generate the English voice-over, one WAV per scene, with Kokoro (open-source neural TTS).

Model files (download once into video/tts/):
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

Writes video/build/vo_<scene>.wav and video/build/timeline.json (scene start times and durations).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
TTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tts"
LEAD_IN = 0.6   # silence before each line, seconds
TAIL = 0.9      # breathing room after each line, seconds


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    script = json.loads((HERE / "script.json").read_text())
    kokoro = Kokoro(str(TTS / "kokoro-v1.0.onnx"), str(TTS / "voices-v1.0.bin"))
    timeline, t = [], 0.0
    for scene in script["scenes"]:
        audio, rate = kokoro.create(scene["text"], voice=script["voice"], speed=script["speed"], lang="en-us")
        audio = np.concatenate([np.zeros(int(LEAD_IN * rate)), audio, np.zeros(int(TAIL * rate))]).astype(np.float32)
        sf.write(BUILD / f"vo_{scene['id']}.wav", audio, rate)
        duration = len(audio) / rate
        timeline.append({"id": scene["id"], "start": round(t, 3), "duration": round(duration, 3),
                         "speech_start": LEAD_IN, "text": scene["text"]})
        print(f"{scene['id']:10s} {duration:5.1f}s")
        t += duration
    (BUILD / "timeline.json").write_text(json.dumps({"total": round(t, 3), "scenes": timeline}, indent=2))
    print(f"total {t:.1f}s")


if __name__ == "__main__":
    main()
