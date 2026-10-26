"""Download the BDG2 electricity meters and weather files (about 70 MB)."""
import urllib.request

from src.config import DATA_RAW, DATA_URL, SITES, YEARS


def download(force: bool = False) -> None:
    files = [f"{s}_clean={y}.csv" for s in SITES for y in YEARS] + [f"weather_{s}.csv" for s in SITES]
    for name in files:
        target = DATA_RAW / name
        if target.exists() and not force:
            print(f"  exists  {name}")
            continue
        print(f"  fetch   {name}")
        urllib.request.urlretrieve(f"{DATA_URL}/{name}", target)


if __name__ == "__main__":
    download()
