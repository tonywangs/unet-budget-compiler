"""Create the fixed, licensed real-photo subset from the upstream v1.0 archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image

URL = 'https://github.com/pynicolas/fairscan-dataset/releases/download/v1.0/document-segmentation-dataset-v1.0.zip'
ARCHIVE_SHA = '4352b37938dddd32e7d1110cffdf89c3c38875dd1931c73db4f51c1db7c88e5d'
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(archive, output):
    data = archive.read_bytes()
    if sha(data) != ARCHIVE_SHA:
        raise ValueError('archive does not match the frozen v1.0 SHA-256')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        (output / 'LICENSE.txt').write_bytes(z.read('document-segmentation-dataset/LICENSE.txt'))
        for split, count in [('train', 24), ('val', 8)]:
            names = sorted(n for n in z.namelist() if f'/{split}/images/' in n and n.endswith('.jpg'))[:count]
            assert len(names) == count
            for name in names:
                mask_name = name.replace('/images/', '/masks/').replace('.jpg', '.png')
                record = dict(split=split, id=Path(name).stem, sources=[], files=[])
                for source, kind in [(name, 'image'), (mask_name, 'mask')]:
                    raw = z.read(source)
                    with Image.open(io.BytesIO(raw)) as original:
                        record['sources'].append(dict(path=source, sha256=sha(raw), size=list(original.size)))
                        im = original.convert('RGB' if kind == 'image' else 'L')
                        im = im.resize((161, 129), Image.Resampling.BILINEAR if kind == 'image' else Image.Resampling.NEAREST)
                    path = output / f'{split}-{Path(name).stem}-{kind}.png'
                    im.save(path)
                    record['files'].append(dict(path=path.name, sha256=sha(path.read_bytes())))
                records.append(record)
    manifest = dict(dataset='FairScan v1.0', creator='pynicolas / FairScan contributors',
                    source_url=URL, archive_sha256=ARCHIVE_SHA, license='CC BY-NC-SA 4.0',
                    selection='First 24 train and first 8 val JPEG paths in lexical order; upstream splits retained.',
                    preprocessing='RGB bilinear resize to 161x129; mask grayscale nearest resize; no EXIF transpose; float32 RGB / 255; mask > 0.',
                    records=records)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.archive, a.output)
