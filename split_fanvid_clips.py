#!/usr/bin/env python3
"""
Split FANVID videos into shorter clips and remap metadata/annotations.

The script reads:
  - dataset_lp.csv: clip-level metadata with absolute source-video times/frames
  - license_plate_annotations_HR.csv: frame-level boxes and IdentityOrText

It writes:
  - clips/*.mp4
  - dataset_lp_chunks.csv
  - license_plate_annotations_HR_chunks.csv
  - clip_texts_chunks.csv

By default, each row in dataset_lp.csv is treated as a parent interval inside the
full source video. The output annotation CSV stores local frame numbers in
FrameNo and keeps OriginalFrameNo for provenance.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split FANVID video intervals and remap frame annotations."
    )
    parser.add_argument("--dataset-csv", required=True, type=Path)
    parser.add_argument("--annotations-csv", required=True, type=Path)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--clip-length",
        required=True,
        type=float,
        help="Maximum chunk length in seconds, for example 2.0 or 5.",
    )
    parser.add_argument(
        "--video-timebase",
        choices=("full", "parent"),
        default="full",
        help=(
            "'full' means source video timestamps match dataset absolute times. "
            "'parent' means each source file is already the parent dataset clip, "
            "so cuts start at 0 inside that file."
        ),
    )
    parser.add_argument(
        "--source-name-template",
        default="{video_id}",
        help=(
            "Video filename stem template. Available fields: video_id, url_video_id, "
            "clip_id, name. Extensions are searched automatically."
        ),
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help=(
            "Only process rows for this Video ID. Can be passed multiple times. "
            "Useful when splitting one source video."
        ),
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help=(
            "Only process rows for this original Clip ID. Can be passed multiple "
            "times."
        ),
    )
    parser.add_argument(
        "--output-video-template",
        default="{new_clip_id}.mp4",
        help=(
            "Output filename template. Available fields include new_clip_id, "
            "parent_clip_id, video_id, name, chunk_index."
        ),
    )
    parser.add_argument(
        "--recursive-video-search",
        action="store_true",
        help="Search videos-dir recursively when resolving source files.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only write remapped CSVs; do not call ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--stream-copy",
        action="store_true",
        help=(
            "Use ffmpeg stream copy instead of re-encoding. Faster, but cuts may "
            "land on nearby keyframes instead of exact timestamps."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="Path to ffmpeg executable. Defaults to ffmpeg on PATH.",
    )
    parser.add_argument(
        "--skip-missing-video",
        action="store_true",
        help="Keep writing CSV rows when a source video cannot be found.",
    )
    parser.add_argument(
        "--min-last-chunk-seconds",
        type=float,
        default=0.0,
        help=(
            "If the final chunk is shorter than this, merge it into the previous "
            "chunk. Default 0 keeps every remainder."
        ),
    )
    parser.add_argument(
        "--include-end-frame-annotations",
        action="store_true",
        help=(
            "Also assign annotations whose FrameNo equals a parent row's End frame "
            "to that parent's final chunk when the annotation Clip ID matches. "
            "By default End frame is treated as exclusive, matching Duration/FPS."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def get_required(row: dict[str, str], column: str) -> str:
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"Missing required column value: {column}")
    return value


def as_float(row: dict[str, str], column: str) -> float:
    return float(get_required(row, column))


def as_int(row: dict[str, str], column: str) -> int:
    return int(float(get_required(row, column)))


def format_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def youtube_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return parsed.path.strip("/")
    query_id = parse_qs(parsed.query).get("v", [""])[0]
    return query_id.strip()


def safe_token(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^\w.-]+", "_", text.strip())
    return text.strip("._") or "blank"


def append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def candidate_stems(row: dict[str, str], template: str) -> list[str]:
    video_id = row.get("Video ID", "")
    url_video_id = youtube_id_from_url(row.get("You_Tube_URL", ""))
    fields = {
        "video_id": safe_token(video_id),
        "url_video_id": safe_token(url_video_id),
        "clip_id": safe_token(row.get("Clip ID", "")),
        "name": safe_token(row.get("Name", "")),
    }
    stems: list[str] = []
    for raw in (
        template.format(**fields),
        video_id,
        url_video_id,
        row.get("Clip ID", ""),
        f"{video_id}_{row.get('Clip ID', '')}",
        f"{url_video_id}_{row.get('Clip ID', '')}",
    ):
        append_unique(stems, safe_token(raw))
    return stems


def find_video(row: dict[str, str], videos_dir: Path, template: str, recursive: bool) -> Path | None:
    if videos_dir.is_file():
        video_id = row.get("Video ID", "")
        url_video_id = youtube_id_from_url(row.get("You_Tube_URL", ""))
        file_stem = videos_dir.stem
        if file_stem in {video_id, url_video_id} or video_id in file_stem or url_video_id in file_stem:
            return videos_dir
        return None

    glob = "**/*" if recursive else "*"
    for stem in candidate_stems(row, template):
        for ext in VIDEO_EXTENSIONS:
            direct = videos_dir / f"{stem}{ext}"
            if direct.exists():
                return direct
        matches = []
        for ext in VIDEO_EXTENSIONS:
            matches.extend(videos_dir.glob(f"{glob}{stem}{ext}"))
            matches.extend(videos_dir.glob(f"{glob}{stem}.*{ext}"))
        if matches:
            return sorted(matches, key=lambda path: (len(str(path)), str(path)))[0]
    return None


def chunk_bounds(
    start_frame: int,
    end_frame: int,
    fps: float,
    clip_length_seconds: float,
    min_last_chunk_seconds: float,
) -> list[tuple[int, int]]:
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if clip_length_seconds <= 0:
        raise ValueError("--clip-length must be positive")
    frames_per_chunk = max(1, int(round(clip_length_seconds * fps)))
    bounds = []
    current = start_frame
    while current < end_frame:
        nxt = min(end_frame, current + frames_per_chunk)
        bounds.append((current, nxt))
        current = nxt
    if (
        len(bounds) > 1
        and min_last_chunk_seconds > 0
        and (bounds[-1][1] - bounds[-1][0]) / fps < min_last_chunk_seconds
    ):
        previous_start, _ = bounds[-2]
        _, last_end = bounds[-1]
        bounds[-2:] = [(previous_start, last_end)]
    return bounds


def output_video_path(
    output_dir: Path,
    template: str,
    fields: dict[str, object],
) -> Path:
    safe_fields = {key: safe_token(value) for key, value in fields.items()}
    relative = Path(template.format(**safe_fields))
    if relative.suffix == "":
        relative = relative.with_suffix(".mp4")
    return output_dir / "clips" / relative


def run_ffmpeg(
    ffmpeg_bin: str,
    source: Path,
    output: Path,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool,
    stream_copy: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        format_seconds(start_seconds),
        "-i",
        str(source),
        "-t",
        format_seconds(duration_seconds),
    ]
    if stream_copy:
        command.extend(["-c", "copy"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
            ]
        )
    command.append(str(output))
    subprocess.run(command, check=True)


def build_annotation_index(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        index[row.get("VideoID", "")].append(row)
    return index


def annotation_matches(
    annotation: dict[str, str],
    dataset_row: dict[str, str],
    chunk_start_frame: int,
    chunk_end_frame: int,
    parent_end_frame: int,
    include_end_frame_annotations: bool,
) -> bool:
    if annotation.get("VideoID") and dataset_row.get("Video ID"):
        if annotation["VideoID"] != dataset_row["Video ID"]:
            return False
    frame_no = int(float(annotation.get("FrameNo", "-1")))
    if chunk_start_frame <= frame_no < chunk_end_frame:
        return True
    return (
        include_end_frame_annotations
        and frame_no == parent_end_frame
        and chunk_end_frame == parent_end_frame
        and annotation.get("Clip ID") == dataset_row.get("Clip ID")
    )


def extended_headers(original: list[str], additions: list[str]) -> list[str]:
    result = list(original)
    for column in additions:
        if column not in result:
            result.append(column)
    return result


def main() -> int:
    args = parse_args()
    dataset_headers, dataset_rows = read_csv(args.dataset_csv)
    annotation_headers, annotation_rows = read_csv(args.annotations_csv)

    if args.video_id:
        wanted_video_ids = set(args.video_id)
        dataset_rows = [row for row in dataset_rows if row.get("Video ID") in wanted_video_ids]
    if args.clip_id:
        wanted_clip_ids = set(args.clip_id)
        dataset_rows = [row for row in dataset_rows if row.get("Clip ID") in wanted_clip_ids]
    if not dataset_rows:
        raise ValueError("No dataset rows matched the requested filters.")

    annotation_index = build_annotation_index(annotation_rows)
    chunk_dataset_rows: list[dict[str, object]] = []
    chunk_annotation_rows: list[dict[str, object]] = []
    chunk_text_rows: list[dict[str, object]] = []
    assigned_annotation_ids: set[int] = set()

    next_clip_id = 0
    missing_videos: list[str] = []

    for dataset_row in dataset_rows:
        parent_clip_id = get_required(dataset_row, "Clip ID")
        fps = as_float(dataset_row, "FPS")
        parent_start_time = as_float(dataset_row, "Start time (s)")
        parent_start_frame = as_int(dataset_row, "Start frame")
        parent_end_frame = as_int(dataset_row, "End frame")
        parent_annotations = annotation_index.get(dataset_row.get("Video ID", ""), [])
        bounds = chunk_bounds(
            parent_start_frame,
            parent_end_frame,
            fps,
            args.clip_length,
            args.min_last_chunk_seconds,
        )

        source_video = None
        if not args.no_video:
            source_video = find_video(
                dataset_row,
                args.videos_dir,
                args.source_name_template,
                args.recursive_video_search,
            )
            if source_video is None:
                message = (
                    f"Missing source video for Clip ID {parent_clip_id}, "
                    f"Video ID {dataset_row.get('Video ID', '')}"
                )
                if args.skip_missing_video:
                    missing_videos.append(message)
                else:
                    raise FileNotFoundError(message)

        for chunk_index, (chunk_start_frame, chunk_end_frame) in enumerate(bounds):
            new_clip_id = str(next_clip_id)
            next_clip_id += 1

            chunk_start_offset = (chunk_start_frame - parent_start_frame) / fps
            chunk_duration = (chunk_end_frame - chunk_start_frame) / fps
            chunk_start_time = parent_start_time + chunk_start_offset
            chunk_end_time = chunk_start_time + chunk_duration
            source_cut_start = (
                chunk_start_time if args.video_timebase == "full" else chunk_start_offset
            )

            common_fields = {
                "new_clip_id": new_clip_id,
                "parent_clip_id": parent_clip_id,
                "video_id": dataset_row.get("Video ID", ""),
                "name": dataset_row.get("Name", ""),
                "chunk_index": chunk_index,
            }
            output_video = output_video_path(
                args.output_dir,
                args.output_video_template,
                common_fields,
            )

            if source_video is not None:
                run_ffmpeg(
                    args.ffmpeg_bin,
                    source_video,
                    output_video,
                    source_cut_start,
                    chunk_duration,
                    args.overwrite,
                    args.stream_copy,
                )

            metadata = dict(dataset_row)
            metadata.update(
                {
                    "Clip ID": new_clip_id,
                    "Start time (s)": format_seconds(chunk_start_time),
                    "End time (s)": format_seconds(chunk_end_time),
                    "Start frame": chunk_start_frame,
                    "End frame": chunk_end_frame,
                    "Duration (sec)": format_seconds(chunk_duration),
                    "Parent Clip ID": parent_clip_id,
                    "Parent Chunk Index": chunk_index,
                    "Parent Chunk Count": len(bounds),
                    "Parent Start time (s)": dataset_row.get("Start time (s)", ""),
                    "Parent End time (s)": dataset_row.get("End time (s)", ""),
                    "Parent Start frame": parent_start_frame,
                    "Parent End frame": parent_end_frame,
                    "Local Start frame": 0,
                    "Local End frame": chunk_end_frame - chunk_start_frame,
                    "Source Video Path": "" if source_video is None else str(source_video),
                    "Output Video Path": str(output_video),
                }
            )
            chunk_dataset_rows.append(metadata)

            identities: list[str] = []
            annotation_count = 0
            for annotation in parent_annotations:
                if not annotation_matches(
                    annotation,
                    dataset_row,
                    chunk_start_frame,
                    chunk_end_frame,
                    parent_end_frame,
                    args.include_end_frame_annotations,
                ):
                    continue
                original_frame_no = int(float(annotation["FrameNo"]))
                local_frame_no = original_frame_no - chunk_start_frame
                remapped = dict(annotation)
                remapped.update(
                    {
                        "Clip ID": new_clip_id,
                        "FrameNo": local_frame_no,
                        "New Clip ID": new_clip_id,
                        "Parent Clip ID": parent_clip_id,
                        "OriginalFrameNo": original_frame_no,
                        "ClipStartFrame": chunk_start_frame,
                        "ClipEndFrame": chunk_end_frame,
                        "ClipStartTimeSec": format_seconds(chunk_start_time),
                        "Output Video Path": str(output_video),
                    }
                )
                chunk_annotation_rows.append(remapped)
                assigned_annotation_ids.add(id(annotation))
                annotation_count += 1
                append_unique(identities, annotation.get("IdentityOrText", ""))

            chunk_text_rows.append(
                {
                    "New Clip ID": new_clip_id,
                    "Parent Clip ID": parent_clip_id,
                    "VideoID": dataset_row.get("Video ID", ""),
                    "Name": dataset_row.get("Name", ""),
                    "IdentityOrText": "|".join(identities),
                    "Annotation Count": annotation_count,
                    "StartFrame": chunk_start_frame,
                    "EndFrame": chunk_end_frame,
                    "StartTimeSec": format_seconds(chunk_start_time),
                    "EndTimeSec": format_seconds(chunk_end_time),
                    "Output Video Path": str(output_video),
                }
            )

    dataset_output = args.output_dir / "dataset_lp_chunks.csv"
    annotations_output = args.output_dir / "license_plate_annotations_HR_chunks.csv"
    texts_output = args.output_dir / "clip_texts_chunks.csv"
    unmatched_output = args.output_dir / "unmatched_license_plate_annotations_HR.csv"

    dataset_extra = [
        "Parent Clip ID",
        "Parent Chunk Index",
        "Parent Chunk Count",
        "Parent Start time (s)",
        "Parent End time (s)",
        "Parent Start frame",
        "Parent End frame",
        "Local Start frame",
        "Local End frame",
        "Source Video Path",
        "Output Video Path",
    ]
    annotation_extra = [
        "New Clip ID",
        "Parent Clip ID",
        "OriginalFrameNo",
        "ClipStartFrame",
        "ClipEndFrame",
        "ClipStartTimeSec",
        "Output Video Path",
    ]
    text_headers = [
        "New Clip ID",
        "Parent Clip ID",
        "VideoID",
        "Name",
        "IdentityOrText",
        "Annotation Count",
        "StartFrame",
        "EndFrame",
        "StartTimeSec",
        "EndTimeSec",
        "Output Video Path",
    ]

    write_csv(dataset_output, extended_headers(dataset_headers, dataset_extra), chunk_dataset_rows)
    write_csv(
        annotations_output,
        extended_headers(annotation_headers, annotation_extra),
        chunk_annotation_rows,
    )
    write_csv(texts_output, text_headers, chunk_text_rows)
    unmatched_rows = []
    for annotation in annotation_rows:
        if id(annotation) in assigned_annotation_ids:
            continue
        unmatched = dict(annotation)
        unmatched["Reason"] = "No matching dataset_lp interval for VideoID/FrameNo"
        unmatched_rows.append(unmatched)
    write_csv(unmatched_output, extended_headers(annotation_headers, ["Reason"]), unmatched_rows)

    print(f"Wrote {len(chunk_dataset_rows)} metadata rows: {dataset_output}")
    print(f"Wrote {len(chunk_annotation_rows)} annotation rows: {annotations_output}")
    print(f"Wrote {len(chunk_text_rows)} text rows: {texts_output}")
    print(f"Wrote {len(unmatched_rows)} unmatched annotation rows: {unmatched_output}")
    if missing_videos:
        print("Missing videos:")
        for message in missing_videos:
            print(f"  - {message}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise
