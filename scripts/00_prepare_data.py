"""Download BBBC039 (CC0), unpack it, remove macOS junk, write plates_clean.csv."""
import shutil, urllib.request, zipfile
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://data.broadinstitute.org/bbbc/BBBC039/"

for name in ["images", "masks", "metadata"]:
    z = RAW / f"{name}.zip"
    if not z.exists():
        print("downloading", z)
        urllib.request.urlretrieve(BASE + name + ".zip", z)
    out = RAW / name
    if not (out / name).exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    shutil.rmtree(out / "__MACOSX", ignore_errors=True)

df = pd.read_csv(RAW / "metadata/metadata/filenames_and_plates.csv",
                 header=None, names=["mask_file", "plate"])
df["well"] = df["mask_file"].str.extract(r"IXMtest_([A-P]\d{2})_")
df["site"] = df["mask_file"].str.extract(r"_s(\d)_")
df.to_csv(RAW / "plates_clean.csv", index=False)
print("ok:", len(df), "rows,", df["plate"].nunique(), "plates")
