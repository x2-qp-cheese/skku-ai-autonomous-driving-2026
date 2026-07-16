#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class CocoSource:
    name: str
    archive_path: Path
    annotations: Dict[str, str]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        sources = [open_source(value) for value in args.source]
        output = args.output.expanduser().resolve()
        merge_sources(sources, output, args.overwrite)
        if args.zip:
            archive = shutil.make_archive(
                str(output),
                "zip",
                root_dir=output.parent,
                base_dir=output.name,
            )
            print("zip:", archive)
        return 0
    except Exception as exc:
        print("error:", exc, file=sys.stderr)
        return 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Roboflow COCO-segmentation ZIP files by category name"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=ZIP",
        help="dataset name and ZIP path; repeat for every source",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true", help="also create OUTPUT.zip")
    return parser.parse_args(argv)


def open_source(value: str) -> CocoSource:
    if "=" not in value:
        raise ValueError("--source must use NAME=ZIP format: %s" % value)
    name, raw_path = value.split("=", 1)
    name = safe_prefix(name)
    archive_path = Path(raw_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        annotations = annotation_members(archive.namelist())
    missing = [split for split in SPLITS if split not in annotations]
    if missing:
        raise ValueError("%s is missing COCO splits: %s" % (archive_path, missing))
    return CocoSource(name, archive_path, annotations)


def annotation_members(members: Iterable[str]) -> Dict[str, str]:
    result = {}
    for member in members:
        path = PurePosixPath(member)
        if path.name != "_annotations.coco.json":
            continue
        for split in SPLITS:
            if split in path.parts:
                if split in result:
                    raise ValueError("multiple annotation files for split %s" % split)
                result[split] = member
    return result


def merge_sources(
    sources: List[CocoSource],
    output: Path,
    overwrite: bool,
) -> None:
    output = output.expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError("output exists; pass --overwrite: %s" % output)
        shutil.rmtree(output)
    output.mkdir(parents=True)

    category_names = collect_used_category_names(sources)
    categories = [
        {"id": index, "name": name, "supercategory": "object"}
        for index, name in enumerate(category_names)
    ]
    category_id = {item["name"]: item["id"] for item in categories}

    hashes: Dict[str, Tuple[str, str]] = {}
    report = {
        "sources": [source.name for source in sources],
        "categories": category_names,
        "splits": {},
    }
    for split in SPLITS:
        split_dir = output / split
        split_dir.mkdir()
        merged = {
            "info": {"description": "Merged COCO segmentation dataset"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": [],
        }
        split_counts = Counter()
        source_counts = Counter()
        duplicate_annotation_counts = Counter()
        next_image_id = 1
        next_annotation_id = 1

        for source in sources:
            with zipfile.ZipFile(source.archive_path) as archive:
                annotation_member = source.annotations[split]
                coco = json.loads(archive.read(annotation_member))
                archive_members = set(archive.namelist())
                old_category_names = {
                    item["id"]: item["name"] for item in coco["categories"]
                }
                annotations_by_image = defaultdict(list)
                for annotation in coco["annotations"]:
                    annotations_by_image[annotation["image_id"]].append(annotation)

                annotation_dir = PurePosixPath(annotation_member).parent
                for image in sorted(coco["images"], key=lambda item: item["id"]):
                    image_member = str(annotation_dir / image["file_name"])
                    if image_member not in archive_members:
                        raise FileNotFoundError(
                            "%s:%s" % (source.archive_path, image_member)
                        )
                    image_bytes = archive.read(image_member)
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    if digest in hashes:
                        previous = hashes[digest]
                        raise ValueError(
                            "duplicate image crosses merged inputs: %s and %s/%s"
                            % (previous, source.name, image_member)
                        )

                    suffix = PurePosixPath(image["file_name"]).suffix.lower()
                    if suffix not in IMAGE_SUFFIXES:
                        raise ValueError("unsupported image type: %s" % image["file_name"])
                    file_name = "%s__%s" % (
                        source.name,
                        PurePosixPath(image["file_name"]).name,
                    )
                    (split_dir / file_name).write_bytes(image_bytes)
                    hashes[digest] = (split, file_name)

                    new_image = dict(image)
                    new_image.update(
                        {
                            "id": next_image_id,
                            "file_name": file_name,
                            "source_dataset": source.name,
                        }
                    )
                    merged["images"].append(new_image)
                    source_counts[source.name] += 1

                    seen_annotations = set()
                    for annotation in annotations_by_image.get(image["id"], []):
                        name = old_category_names[annotation["category_id"]]
                        if name not in category_id:
                            continue
                        signature = (
                            name,
                            json.dumps(
                                annotation.get("segmentation"),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                        if signature in seen_annotations:
                            duplicate_annotation_counts[name] += 1
                            continue
                        seen_annotations.add(signature)
                        new_annotation = dict(annotation)
                        new_annotation.update(
                            {
                                "id": next_annotation_id,
                                "image_id": next_image_id,
                                "category_id": category_id[name],
                            }
                        )
                        merged["annotations"].append(new_annotation)
                        split_counts[name] += 1
                        next_annotation_id += 1
                    next_image_id += 1

        validate_coco(merged, split_dir)
        annotation_path = split_dir / "_annotations.coco.json"
        annotation_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["splits"][split] = {
            "images": len(merged["images"]),
            "annotations": len(merged["annotations"]),
            "images_by_source": dict(sorted(source_counts.items())),
            "annotations_by_class": dict(sorted(split_counts.items())),
            "duplicate_annotations_skipped": dict(
                sorted(duplicate_annotation_counts.items())
            ),
        }

    (output / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(output, report)


def collect_used_category_names(sources: List[CocoSource]) -> List[str]:
    names = set()
    for source in sources:
        with zipfile.ZipFile(source.archive_path) as archive:
            for split in SPLITS:
                coco = json.loads(archive.read(source.annotations[split]))
                id_to_name = {
                    item["id"]: item["name"] for item in coco["categories"]
                }
                names.update(
                    id_to_name[item["category_id"]] for item in coco["annotations"]
                )
    return sorted(names)


def validate_coco(coco: dict, split_dir: Path) -> None:
    image_ids = [item["id"] for item in coco["images"]]
    annotation_ids = [item["id"] for item in coco["annotations"]]
    image_id_set = set(image_ids)
    category_ids = {item["id"] for item in coco["categories"]}
    if len(image_ids) != len(image_id_set):
        raise ValueError("duplicate image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("duplicate annotation IDs")
    if any(item["image_id"] not in image_id_set for item in coco["annotations"]):
        raise ValueError("annotation references a missing image")
    if any(item["category_id"] not in category_ids for item in coco["annotations"]):
        raise ValueError("annotation references a missing category")
    if any(not (split_dir / item["file_name"]).is_file() for item in coco["images"]):
        raise ValueError("COCO JSON references a missing image file")


def safe_prefix(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("source name cannot be empty")
    return cleaned


def print_summary(output: Path, report: dict) -> None:
    print("output:", output)
    print("categories:", ", ".join(report["categories"]))
    for split in SPLITS:
        values = report["splits"][split]
        print(
            "%s: %d images, %d annotations"
            % (split, values["images"], values["annotations"])
        )


if __name__ == "__main__":
    raise SystemExit(main())
