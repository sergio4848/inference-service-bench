"""Download the reference vision model (MobileNetV2, ONNX opset 12) and the ImageNet synset labels
from the ONNX Model Zoo into ./models. Run once before starting the service or the benchmark:

    python scripts/get_model.py

Sources (ONNX Model Zoo, Apache-2.0):
  https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    "mobilenetv2-12.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    ),
    "synset.txt": "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/synset.txt",
}


def main() -> int:
    target = Path("models")
    target.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = target / name
        if dest.exists():
            print(f"exists  {dest} ({dest.stat().st_size:,} bytes)")
            continue
        print(f"getting {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:  # noqa: S310
            data = resp.read()
            fh.write(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        print(f"saved   {dest} ({len(data):,} bytes, sha256 {digest}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
