from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# SOURCE_DATASET = "visual-memory/Synthetic-Persona-Chat-Mapping_10"
# SOURCE_DATASET = "visual-memory/PersonaChat-Mapping_10"
SOURCE_DATASET = "visual-memory/ConvAI2-Mapping_10"
TARGET_DATASET = SOURCE_DATASET + "-ERNIE-enhanced"
# MODEL_NAME = "Qwen/Qwen-Image-2512"
# MODEL_NAME = "black-forest-labs/FLUX.2-dev"
MODEL_NAME = "baidu/ERNIE-Image"
DEFAULT_SPLIT = "train"
PERSONA_ID_COLUMN = "persona-id"
PROMPT_COLUMN = "enhanced_persona_revised"
IMAGE_COLUMN = "persona-image"
GENERATION_PARAMS = {"width": 1024, "height": 1024}

BASE_OUTPUT_DIR = Path("/raid/aluno_paulosantana/generate_images_outputs")
DEFAULT_OUTPUT_DIR = Path(
    BASE_OUTPUT_DIR / TARGET_DATASET / "persona-images-shards"
)
DEFAULT_FINAL_OUTPUT_DIR = Path(
    BASE_OUTPUT_DIR / TARGET_DATASET / "persona-images"
)
DEFAULT_CHECKPOINT_NAME = "persona_images.jsonl"


@dataclass(frozen=True)
class ImageCheckpointRow:
    row_index: int
    persona_id: str
    image_path: Path


def parse_gpu_ids(gpus: str) -> list[str]:
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def shard_indices(total_rows: int, num_shards: int, shard_index: int) -> list[int]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be between 0 and num_shards - 1")
    return [
        row_index
        for row_index in range(total_rows)
        if row_index % num_shards == shard_index
    ]


def shard_dir(output_dir: Path, num_shards: int, shard_index: int) -> Path:
    return output_dir / f"num_shards-{num_shards}" / f"shard-{shard_index:05d}"


def image_path_for_row(images_dir: Path, row_index: int) -> Path:
    return images_dir / f"{row_index:09d}.png"


def build_prompt(prompt_text: str) -> str:
    prompt = str(prompt_text).strip()
    return f"Generate an image of this person's description: {prompt}"


def load_checkpoint(path: Path) -> dict[int, ImageCheckpointRow]:
    if not path.exists():
        return {}

    completed: dict[int, ImageCheckpointRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                row = ImageCheckpointRow(
                    row_index=int(item["row_index"]),
                    persona_id=str(item["persona_id"]),
                    image_path=Path(str(item["image_path"])),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid checkpoint line {line_number} in {path}"
                ) from exc
            if row.image_path.exists():
                completed[row.row_index] = row
    return completed


def append_checkpoint(path: Path, rows: list[ImageCheckpointRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "row_index": row.row_index,
                        "persona_id": row.persona_id,
                        "image_path": str(row.image_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class ImageGeneratorProvider:
    def __init__(self, model_name: str) -> None:
        try:
            from image_generator import Provider, get_provider
        except ImportError as exc:
            raise ImportError(
                "Could not import image_generator. Install the image-generation "
                "extra before running generation: "
                "uv sync --extra image-generation"
            ) from exc

        try:
            self.provider = get_provider(Provider.HUGGINGFACE, model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the Hugging Face image provider. Make sure "
                "the image-generation extra is installed with: "
                "uv sync --extra image-generation"
            ) from exc

    def generate(self, prompt: str, generation_params: dict[str, int]) -> bytes:
        return self.provider.generate(
            prompt=prompt,
            generation_params=generation_params,
        )


def load_source_dataset(limit: int | None):
    from datasets import load_dataset

    dataset = load_dataset(SOURCE_DATASET, split=DEFAULT_SPLIT)
    if PROMPT_COLUMN not in dataset.column_names:
        raise ValueError(
            f"Column {PROMPT_COLUMN!r} not found. Available columns: "
            f"{dataset.column_names}"
        )
    if PERSONA_ID_COLUMN not in dataset.column_names:
        raise ValueError(
            f"Column {PERSONA_ID_COLUMN!r} not found. Available columns: "
            f"{dataset.column_names}"
        )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def run_all(args: argparse.Namespace) -> None:
    gpu_ids = parse_gpu_ids(args.gpus)
    num_shards = len(gpu_ids)
    processes = []

    for shard_index, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            "-m",
            "generate_images.generate_persona_images_shards",
            "run-shard",
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.force:
            command.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TQDM_POSITION"] = str(shard_index)
        process = subprocess.Popen(command, env=env)
        processes.append((shard_index, gpu_id, process))
        logging.info("Started shard %s on CUDA_VISIBLE_DEVICES=%s", shard_index, gpu_id)

    failed = []
    for shard_index, gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((shard_index, gpu_id, return_code))

    if failed:
        raise RuntimeError(f"Shard processes failed: {failed}")

    merge_args = argparse.Namespace(
        num_shards=num_shards,
        limit=args.limit,
        output_dir=args.output_dir,
        final_output_dir=args.final_output_dir,
        target_repo_id=args.target_repo_id,
        push_pr=args.push_pr,
    )
    merge_shards(merge_args)


def run_shard(args: argparse.Namespace) -> None:
    from PIL import Image
    from tqdm.auto import tqdm

    dataset = load_source_dataset(args.limit)
    indices = shard_indices(len(dataset), args.num_shards, args.shard_index)
    output_dir = shard_dir(args.output_dir, args.num_shards, args.shard_index)
    images_dir = output_dir / "images"
    checkpoint_path = output_dir / DEFAULT_CHECKPOINT_NAME

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)

    images_dir.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(checkpoint_path)
    pending_indices = [index for index in indices if index not in completed]

    generator = ImageGeneratorProvider(MODEL_NAME)
    progress_position = int(os.environ.get("TQDM_POSITION", "0"))
    progress = tqdm(
        total=len(pending_indices),
        desc=f"Generating shard {args.shard_index}/{args.num_shards}",
        unit="image",
        position=progress_position,
        dynamic_ncols=True,
    )
    checkpoint_buffer: list[ImageCheckpointRow] = []
    for row_index in pending_indices:
        row = dataset[row_index]
        prompt = build_prompt(str(row[PROMPT_COLUMN]))
        persona_id = str(row[PERSONA_ID_COLUMN])
        image_path = image_path_for_row(images_dir, row_index)
        try:
            image_bytes = generator.generate(prompt, GENERATION_PARAMS)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(image_path)
        except Exception:
            logging.exception("Image generation failed for row %s", row_index)
            progress.update(1)
            continue

        checkpoint_buffer.append(
            ImageCheckpointRow(
                row_index=row_index,
                persona_id=persona_id,
                image_path=image_path,
            )
        )
        append_checkpoint(checkpoint_path, checkpoint_buffer)
        checkpoint_buffer.clear()
        progress.update(1)
    progress.close()

    logging.info(
        "Shard %s/%s complete: %s total rows, %s newly generated, checkpoint %s",
        args.shard_index,
        args.num_shards,
        len(indices),
        len(pending_indices),
        checkpoint_path,
    )


def build_image_dataset(dataset, checkpoint_rows: dict[int, ImageCheckpointRow]):
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

    data = {
        column_name: list(dataset[column_name])
        for column_name in dataset.column_names
    }
    data[IMAGE_COLUMN] = [
        str(checkpoint_rows[row_index].image_path)
        for row_index in range(len(dataset))
    ]

    feature_mapping = {}
    for column_name in dataset.column_names:
        feature_mapping[column_name] = dataset.features.get(
            column_name,
            Value("string"),
        )
    feature_mapping[IMAGE_COLUMN] = HFImage()

    enriched_dataset = Dataset.from_dict(data, features=Features(feature_mapping))
    return DatasetDict({DEFAULT_SPLIT: enriched_dataset})


def merge_shards(args: argparse.Namespace) -> None:
    dataset = load_source_dataset(args.limit)

    checkpoint_rows: dict[int, ImageCheckpointRow] = {}
    missing_shards = []
    for shard_index in range(args.num_shards):
        checkpoint_path = (
            shard_dir(args.output_dir, args.num_shards, shard_index)
            / DEFAULT_CHECKPOINT_NAME
        )
        if not checkpoint_path.exists():
            missing_shards.append(shard_index)
            continue
        checkpoint_rows.update(load_checkpoint(checkpoint_path))

    if missing_shards:
        raise ValueError(f"Missing checkpoint for shards: {missing_shards}")

    expected_indices = set(range(len(dataset)))
    missing_rows = sorted(expected_indices - set(checkpoint_rows))
    if missing_rows:
        preview = missing_rows[:20]
        raise ValueError(
            f"Missing {len(missing_rows)} generated images. First missing: {preview}"
        )

    enriched = build_image_dataset(dataset, checkpoint_rows)
    enriched_dataset = enriched[DEFAULT_SPLIT]

    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    local_path = args.final_output_dir / "dataset"
    parquet_path = args.final_output_dir / f"{DEFAULT_SPLIT}.parquet"
    enriched.save_to_disk(str(local_path))
    enriched_dataset.to_parquet(str(parquet_path))
    logging.info("Saved merged dataset to %s", local_path)
    logging.info("Saved merged parquet to %s", parquet_path)

    if args.push_pr:
        enriched.push_to_hub(
            args.target_repo_id,
            commit_message="Add generated persona images",
            commit_description=(
                f"Adds {IMAGE_COLUMN} generated from {PROMPT_COLUMN} with {MODEL_NAME}."
            ),
            create_pr=True,
        )
        logging.info("Opened a Hugging Face Hub pull request for %s", args.target_repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona images over dataset shards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all_parser = subparsers.add_parser("run")
    run_all_parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids, for example: 0,1,2,3",
    )
    run_all_parser.add_argument("--limit", type=int, default=None)
    run_all_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_all_parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_DIR,
    )
    run_all_parser.add_argument("--target-repo-id", default=TARGET_DATASET)
    run_all_parser.add_argument("--push-pr", action="store_true")
    run_all_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--num-shards", type=int, required=True)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--force", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--num-shards", type=int, required=True)
    merge_parser.add_argument("--limit", type=int, default=None)
    merge_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    merge_parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT_DIR,
    )
    merge_parser.add_argument("--target-repo-id", default=TARGET_DATASET)
    merge_parser.add_argument("--push-pr", action="store_true")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.command == "run":
        run_all(args)
    elif args.command == "run-shard":
        run_shard(args)
    elif args.command == "merge":
        merge_shards(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
