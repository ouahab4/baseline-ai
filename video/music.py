"""Synthesised background score: a warm pad with a gentle arpeggio that builds and resolves.

Everything is generated here (no samples), so the film carries no third-party music rights.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
RATE = 44100
BPM = 92
BEAT = 60 / BPM


def note(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * cutoff / RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):  # one-pole filter, applied twice for a softer slope
        acc = (1 - a) * v + a * acc
        y[i] = acc
    acc = 0.0
    for i, v in enumerate(y):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def pad_chord(midis, seconds, t0_phase=0.0):
    t = np.arange(int(seconds * RATE)) / RATE
    out = np.zeros_like(t)
    for m in midis:
        for detune in (-0.07, 0.0, 0.06):
            f = note(m + detune)
            out += np.sin(2 * np.pi * f * t + t0_phase) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    env = np.minimum(1, t / 1.6) * np.minimum(1, (seconds - t) / 1.6)
    return out * env / (len(midis) * 3)


def main() -> None:
    total = json.loads((HERE / "build" / "timeline.json").read_text())["total"] + 1.0
    n = int(total * RATE)
    music = np.zeros((n, 2))
    # A minor -> F -> C -> G, uplifting loop; bass an octave down
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 55, 60, 64], [55, 59, 62, 67]]
    bar = 4 * BEAT * 2  # two bars per chord
    t, i = 0.0, 0
    while t < total:
        seg = min(bar + 1.6, total - t)
        chunk = pad_chord(chords[i % 4], seg)
        bass = pad_chord([chords[i % 4][0] - 12], seg) * 0.8
        start = int(t * RATE)
        end = min(n, start + len(chunk))
        music[start:end, 0] += (chunk + bass)[: end - start] * 0.9
        music[start:end, 1] += (chunk * 0.95 + bass)[: end - start] * 0.9
        t += bar
        i += 1
    # arpeggio enters after the intro and builds, then thins for the ending
    arp_t = np.arange(int(0.6 * RATE)) / RATE
    pluck_env = np.exp(-arp_t * 6.0)
    step = BEAT / 2
    k = 0
    t = 26.0
    while t < total - 9:
        chord = chords[int(t // bar) % 4]
        m = chord[k % 4] + 12
        tone = (np.sin(2 * np.pi * note(m) * arp_t) + 0.3 * np.sin(2 * np.pi * 2 * note(m) * arp_t)) * pluck_env
        level = 0.10 + 0.10 * min(1, (t - 26) / 30)
        s = int(t * RATE)
        e = min(n, s + len(tone))
        pan = 0.5 + 0.35 * np.sin(k * 0.7)
        music[s:e, 0] += tone[: e - s] * level * (1 - pan) * 2
        music[s:e, 1] += tone[: e - s] * level * pan * 2
        t += step
        k += 1
    for ch in range(2):
        music[:, ch] = lowpass(music[:, ch], 2400)
    fade = np.ones(n)
    fade[: int(2.5 * RATE)] = np.linspace(0, 1, int(2.5 * RATE))
    fade[-int(4 * RATE):] = np.linspace(1, 0, int(4 * RATE))
    music *= fade[:, None]
    music /= np.abs(music).max()
    sf.write(HERE / "build" / "music.wav", (music * 0.9).astype(np.float32), RATE)
    print(f"music {total:.1f}s")


if __name__ == "__main__":
    main()
