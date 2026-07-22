from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

from enhance_prompts.utils import GenerationConfig, append_checkpoint, build_prompt, load_checkpoint
from enhance_prompts.vendor_prompt_enhancer import Backend, VendorPromptEnhancer

DEFAULT_DATASET_ID = "visual-memory/PersonaChat-Mapping"
DEFAULT_MODEL_PATH = "/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt"
DEFAULT_TARGET_REPO_ID = DEFAULT_DATASET_ID
DEFAULT_SPLIT = "train"
DEFAULT_DESCRIPTION_COLUMN = "persona_revised"
DEFAULT_ENHANCED_COLUMN = "enhanced_persona_revised"
DEFAULT_BACKEND: Backend = "7b"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
DEFAULT_SYS_PROMPT = None
DEFAULT_OUTPUT_DIR = Path("outputs/enhanced-dataset")
DEFAULT_CHECKPOINT_NAME = "enhanced_description.jsonl"
DEFAULT_CHECKPOINT_INTERVAL = 50
DEFAULT_COMMIT_MESSAGE = "Add enhanced persona descriptions"
DEFAULT_COMMIT_DESCRIPTION = (
    "Adds enhanced_description generated from description with PromptEnhancer."
)
DEFAULT_TEMPLATE = (
    "Generate a photo of a person with the following self-description:\n"
    "{description}"
)

SHARDED_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.with_name(f"{DEFAULT_OUTPUT_DIR.name}-shards")


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


def parse_gpu_ids(gpus: str) -> list[str]:
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def run_all(args: argparse.Namespace) -> None:
    gpu_ids = parse_gpu_ids(args.gpus)
    num_shards = len(gpu_ids)
    processes = []

    for shard_index, gpu_id in enumerate(gpu_ids):
        command = [
            sys.executable,
            "-m",
            "enhance_prompts.enhance_dataset_shards",
            "run-shard",
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
            "--model-path",
            args.model_path,
            "--output-dir",
            str(args.output_dir),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.force:
            command.append("--force")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
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
    dataset = load_dataset(DEFAULT_DATASET_ID)[DEFAULT_SPLIT]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    indices = shard_indices(len(dataset), args.num_shards, args.shard_index)
    output_dir = shard_dir(args.output_dir, args.num_shards, args.shard_index)
    checkpoint_path = output_dir / DEFAULT_CHECKPOINT_NAME
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()

    completed = load_checkpoint(checkpoint_path)
    pending_indices = [index for index in indices if index not in completed]

    enhancer = VendorPromptEnhancer(
        backend=DEFAULT_BACKEND,
        model_path=args.model_path,
        device_map=DEFAULT_DEVICE_MAP,
        device=args.device,
    )
    generation_config = GenerationConfig(
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        sys_prompt=DEFAULT_SYS_PROMPT,
    )

    progress = tqdm(
        total=len(pending_indices),
        desc=f"Enhancing shard {args.shard_index}/{args.num_shards}",
        unit="row",
    )
    checkpoint_buffer = []
    for row_index in pending_indices:
        prompt = build_prompt(
            dataset[row_index][DEFAULT_DESCRIPTION_COLUMN],
            DEFAULT_TEMPLATE,
        )
        try:
            enhanced_description = enhancer.enhance_one(prompt, generation_config)
        except Exception:
            logging.exception("Enhancement failed; falling back to original prompts.")
            enhanced_description = prompt

        checkpoint_buffer.append((row_index, enhanced_description))
        if len(checkpoint_buffer) >= DEFAULT_CHECKPOINT_INTERVAL:
            append_checkpoint(checkpoint_path, checkpoint_buffer)
            checkpoint_buffer.clear()
        progress.update(1)
    progress.close()
    if checkpoint_buffer:
        append_checkpoint(checkpoint_path, checkpoint_buffer)

    logging.info(
        "Shard %s/%s complete: %s total rows, %s newly processed, checkpoint %s",
        args.shard_index,
        args.num_shards,
        len(indices),
        len(pending_indices),
        checkpoint_path,
    )


def merge_shards(args: argparse.Namespace) -> None:
    dataset = load_dataset(DEFAULT_DATASET_ID)[DEFAULT_SPLIT]
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    enhanced: dict[int, str] = {}
    missing_shards = []
    for shard_index in range(args.num_shards):
        checkpoint_path = (
            shard_dir(args.output_dir, args.num_shards, shard_index)
            / DEFAULT_CHECKPOINT_NAME
        )
        if not checkpoint_path.exists():
            missing_shards.append(shard_index)
            continue
        enhanced.update(load_checkpoint(checkpoint_path))

    if missing_shards:
        raise ValueError(f"Missing checkpoint for shards: {missing_shards}")

    expected_indices = set(range(len(dataset)))
    missing_rows = sorted(expected_indices - set(enhanced))
    if missing_rows:
        preview = missing_rows[:20]
        raise ValueError(
            f"Missing {len(missing_rows)} enhanced rows. First missing: {preview}"
        )

    enhanced_values = [enhanced[index] for index in range(len(dataset))]
    enriched_dataset = dataset.add_column(DEFAULT_ENHANCED_COLUMN, enhanced_values)
    enriched = DatasetDict({DEFAULT_SPLIT: enriched_dataset})

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
            commit_message=DEFAULT_COMMIT_MESSAGE,
            commit_description=DEFAULT_COMMIT_DESCRIPTION,
            create_pr=True,
        )
        logging.info("Opened a Hugging Face Hub pull request for %s", args.target_repo_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PromptEnhancer over dataset shards without changing vendor code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all_parser = subparsers.add_parser("run")
    run_all_parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids, for example: 0,1,2,3",
    )
    run_all_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    run_all_parser.add_argument("--limit", type=int, default=None)
    run_all_parser.add_argument("--output-dir", type=Path, default=SHARDED_OUTPUT_DIR)
    run_all_parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_all_parser.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
    run_all_parser.add_argument("--push-pr", action="store_true")
    run_all_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--num-shards", type=int, required=True)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--output-dir", type=Path, default=SHARDED_OUTPUT_DIR)
    run_parser.add_argument("--force", action="store_true")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--num-shards", type=int, required=True)
    merge_parser.add_argument("--limit", type=int, default=None)
    merge_parser.add_argument("--output-dir", type=Path, default=SHARDED_OUTPUT_DIR)
    merge_parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    merge_parser.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
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
