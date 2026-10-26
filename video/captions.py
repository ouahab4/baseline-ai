"""English captions (WebVTT) for the film, timed from the voice-over timeline."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "media" / "baseline-ai-film.vtt"
MAX_CHARS = 64
TAIL = 0.9  # silence after each scene's line (see voiceover.py)


def stamp(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def chunks(text: str) -> list[str]:
    """Split into caption-sized pieces at sentence ends, then at commas."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(sentence) <= MAX_CHARS:
            out.append(sentence)
            continue
        part = ""
        for piece in re.split(r"(?<=,)\s+", sentence):
            if part and len(part) + len(piece) + 1 > MAX_CHARS:
                out.append(part)
                part = piece
            else:
                part = f"{part} {piece}".strip()
        out.append(part)
    return out


def main() -> None:
    tl = json.loads((HERE / "build" / "timeline.json").read_text())
    lines = ["WEBVTT", ""]
    for scene in tl["scenes"]:
        start = scene["start"] + scene["speech_start"]
        end = scene["start"] + scene["duration"] - TAIL
        pieces = chunks(scene["text"])
        total_chars = sum(len(p) for p in pieces)
        t = start
        for p in pieces:
            d = (end - start) * len(p) / total_chars
            lines += [f"{stamp(t)} --> {stamp(t + d)}", p, ""]
            t += d
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"{OUT} ({sum(1 for l in lines if '-->' in l)} cues)")


if __name__ == "__main__":
    main()
