#!/usr/bin/env bash
# Build the Baseline AI film: voice-over, music, frames, final MP4 + poster + captions.
# Needs: python (kokoro-onnx, soundfile), node + playwright, ffmpeg. Kokoro model files in video/tts/ (see voiceover.py).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
python voiceover.py "${TTS_DIR:-tts}"
python prepare_data.py
python music.py
python captions.py
python -m http.server 8766 >/dev/null 2>&1 & SERVER=$!
trap 'kill $SERVER' EXIT
sleep 1
node render.js http://localhost:8766/film.html build/frames.mp4 30
# voice-over track in scene order
ls build/vo_*.wav >/dev/null
python - <<'PY'
import json, soundfile as sf, numpy as np
tl = json.load(open("build/timeline.json"))
parts = [sf.read(f"build/vo_{s['id']}.wav", dtype="float32")[0] for s in tl["scenes"]]
sf.write("build/voice.wav", np.concatenate(parts), 24000)
PY
# music ducks under the voice, then both are mixed and normalised for web playback
ffmpeg -y -loglevel error -i build/voice.wav -i build/music.wav -filter_complex \
  "[0:a]aresample=44100,asplit=2[v][key];[1:a]volume=0.32[m];[m][key]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400[duck];[v][duck]amix=inputs=2:weights=1 0.9:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]" \
  -map "[a]" -c:a aac -b:a 160k build/audio.m4a
ffmpeg -y -loglevel error -i build/frames.mp4 -i build/audio.m4a -c:v libx264 -preset slow -crf 26 -tune animation -pix_fmt yuv420p -c:a copy -shortest -movflags +faststart ../docs/media/baseline-ai-film.mp4
ffmpeg -y -loglevel error -ss 9.6 -i build/frames.mp4 -frames:v 1 -q:v 3 ../docs/media/baseline-ai-film-poster.jpg
echo "Film: docs/media/baseline-ai-film.mp4"
