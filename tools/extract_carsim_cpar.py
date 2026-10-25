"""Extract the Parsfiles in a CarSim CPAR archive into a local working folder.

This prepares source datasets for inspection. CarSim's Browser normally imports
the archive and expands the linked Parsfiles into a solver-ready *_all.par.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path, PureWindowsPath


FILE_MARKER = re.compile(rb"(?m)^/F ([^\r\n]+)\r?$", re.ASCII)


def safe_target(root: Path, archive_name: str) -> Path:
    parts = PureWindowsPath(archive_name.replace("/", "\\")).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe CPAR entry: {archive_name!r}")
    if PureWindowsPath(archive_name).is_absolute() or ":" in parts[0]:
        raise ValueError(f"Absolute CPAR entry: {archive_name!r}")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"CPAR entry escapes output folder: {archive_name!r}")
    return target


def extract(archive_path: Path, output: Path) -> list[Path]:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        data = archive.read("temp.par")
        markers = list(FILE_MARKER.finditer(data))
        if not markers:
            raise ValueError("CPAR has no /F Parsfile records")
        for index, marker in enumerate(markers):
            header = marker.group(1).decode("latin-1")
            relative = header.split("`", 1)[0]
            target = safe_target(root, relative)
            if target.suffix.lower() != ".par":
                continue
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(data)
            body = data[start:end].lstrip(b"\r\n")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = extract(args.archive, args.output)
    print(f"Extracted {len(files)} Parsfiles to {args.output.resolve()}")
    for file in files:
        if file.parent.name.lower() == "runs":
            print(file)


if __name__ == "__main__":
    main()
