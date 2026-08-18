from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("persona_image_generation.yaml")
CHECKPOINT_NAME = "persona_images.jsonl"
MAX_JOB_ATTEMPTS = 3
ORIGINAL_PROMPT_PREFIX = "Generate an image of this person's description:"
VARIANTS = ("original", "enhanced")


@dataclass(frozen=True)
class GenerationJob:
    source_dataset: str
    target_dataset: str
    split: str
    persona_id_column: str
    prompt_column: str
    image_column: str
    variant: str
    model_name: str
    generation_params: dict[str, Any]
    output_dir: Path


@dataclass(frozen=True)
class CheckpointRow:
    row_index: int
    persona_id: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona image datasets with vLLM-Omni replicas."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--push-pr", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration in {path}")
    return config


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [gpu_id.strip() for gpu_id in value.split(",") if gpu_id.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpu_ids


def resolve_job(
    config: dict[str, Any],
    dataset_key: str,
    model_key: str,
    variant: str,
) -> GenerationJob:
    dataset_config = config["datasets"][dataset_key]
    model_config = config["models"][model_key]
    source_dataset = str(dataset_config["repo_id"])
    namespace, dataset_name = source_dataset.rsplit("/", maxsplit=1)
    target_dataset = (
        f"{namespace}/{dataset_name}-{model_config['dataset_suffix']}-{variant}"
    )

    return GenerationJob(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        split=str(config["split"]),
        persona_id_column=str(config["persona_id_column"]),
        prompt_column=str(dataset_config["columns"][variant]),
        image_column=str(config["image_column"]),
        variant=variant,
        model_name=str(model_config["repo_id"]),
        generation_params={
            **config["generation_params"],
            **model_config.get("generation_params", {}),
        },
        output_dir=(
            Path(config["output_root"])
            / target_dataset
            / "persona-images-vllm"
        ),
    )


def build_prompt(value: Any, variant: str) -> str:
    prompt = str(value).strip()
    if variant == "original":
        return f"{ORIGINAL_PROMPT_PREFIX} {prompt}"
    return prompt


def load_dataset_for_job(job: GenerationJob, limit: int | None):
    from datasets import load_dataset

    dataset = load_dataset(job.source_dataset, split=job.split)
    required_columns = (job.persona_id_column, job.prompt_column)
    missing_columns = [
        column for column in required_columns if column not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"Columns {missing_columns} not found in {job.source_dataset}. "
            f"Available columns: {dataset.column_names}"
        )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_checkpoint(path: Path) -> dict[int, CheckpointRow]:
    if not path.exists():
        return {}

    completed: dict[int, CheckpointRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                row = CheckpointRow(
                    row_index=int(item["row_index"]),
                    persona_id=str(item["persona_id"]),
                    image_path=Path(item["image_path"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid checkpoint line {line_number} in {path}"
                ) from exc
            if row.image_path.exists():
                completed[row.row_index] = row
    return completed


def append_checkpoint(path: Path, row: CheckpointRow) -> None:
    with path.open("a", encoding="utf-8") as handle:
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


def create_sampling_params(generation_params: dict[str, Any]):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    params = dict(generation_params)
    width = params.pop("width", 1024)
    height = params.pop("height", 1024)
    seed = params.pop("seed", None)
    num_inference_steps = params.pop("num_inference_steps", 50)
    guidance_scale = params.pop("guidance_scale", 4.0)
    true_cfg_scale = params.pop("cfg_scale", 4.0)
    params.pop("negative_prompt", None)

    return OmniDiffusionSamplingParams(
        width=width,
        height=height,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        num_outputs_per_prompt=1,
        extra_args=params,
    )


def create_omni(model_config: dict[str, Any], replica_count: int):
    import yaml
    from vllm_omni.entrypoints.omni import Omni

    vllm_params = dict(model_config.get("vllm_params", {}))
    # vLLM-Omni 0.26.0 drops stage_overrides in its single-stage diffusion fallback.
    stage_config = {
        "stage_args": [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {
                    "process": True,
                    "num_replicas": replica_count,
                    "devices": ",".join(
                        str(index) for index in range(replica_count)
                    ),
                },
                "engine_args": {
                    "model_stage": "diffusion",
                    "max_num_seqs": 1,
                    "step_execution": True,
                    "parallel_config": {"tensor_parallel_size": 1},
                },
                "engine_input_source": [],
                "default_sampling_params": {},
                "final_output": True,
                "final_output_type": "image",
            }
        ]
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        encoding="utf-8",
    ) as stage_config_file:
        yaml.safe_dump(stage_config, stage_config_file)
        stage_config_file.flush()
        omni = Omni(
            model=str(model_config["repo_id"]),
            mode="text-to-image",
            stage_configs_path=stage_config_file.name,
            **vllm_params,
        )

    actual_replicas = omni.engine.stage_pools[0].live_num_replicas
    if actual_replicas != replica_count:
        omni.close()
        raise RuntimeError(
            f"Expected {replica_count} vLLM replicas, got {actual_replicas}"
        )
    logging.info("Initialized %d vLLM replicas", actual_replicas)
    return omni


def extract_image(output: Any):
    request_output = getattr(output, "request_output", None)
    images = getattr(request_output, "images", None)
    if not images:
        raise ValueError("vLLM-Omni output did not include an image")
    return images[0]


def save_dataset(job: GenerationJob, dataset, checkpoint: dict[int, CheckpointRow]):
    from datasets import Dataset, DatasetDict, Features, Image, Value

    missing_rows = sorted(set(range(len(dataset))) - set(checkpoint))
    if missing_rows:
        raise ValueError(
            f"Missing {len(missing_rows)} generated images. "
            f"First missing: {missing_rows[:20]}"
        )

    data = {column: list(dataset[column]) for column in dataset.column_names}
    data[job.image_column] = [
        str(checkpoint[row_index].image_path) for row_index in range(len(dataset))
    ]
    features = {
        column: dataset.features.get(column, Value("string"))
        for column in dataset.column_names
    }
    features[job.image_column] = Image()

    enriched = DatasetDict(
        {
            job.split: Dataset.from_dict(
                data,
                features=Features(features),
            )
        }
    )
    dataset_path = job.output_dir / "dataset"
    parquet_path = job.output_dir / f"{job.split}.parquet"
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    parquet_path.unlink(missing_ok=True)
    enriched.save_to_disk(str(dataset_path))
    enriched[job.split].to_parquet(str(parquet_path))
    logging.info("Saved %s", dataset_path)
    logging.info("Saved %s", parquet_path)

    return enriched


def generate_job(
    omni: Any,
    job: GenerationJob,
    replica_count: int,
    limit: int | None,
    force: bool,
):
    from tqdm.auto import tqdm

    if force and job.output_dir.exists():
        shutil.rmtree(job.output_dir)
    dataset = load_dataset_for_job(job, limit)

    images_dir = job.output_dir / "images"
    checkpoint_path = job.output_dir / CHECKPOINT_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_path.touch(exist_ok=True)
    pending_indices = [index for index in range(len(dataset)) if index not in checkpoint]
    negative_prompt = job.generation_params.get("negative_prompt")

    with tqdm(
        total=len(pending_indices),
        desc=f"Generating {job.target_dataset}",
        unit="image",
        dynamic_ncols=True,
    ) as progress:
        for offset in range(0, len(pending_indices), replica_count):
            batch_indices = pending_indices[offset : offset + replica_count]
            prompts = [
                {
                    "prompt": build_prompt(
                        dataset[row_index][job.prompt_column],
                        job.variant,
                    ),
                    "negative_prompt": negative_prompt,
                }
                for row_index in batch_indices
            ]
            outputs = omni.generate(
                prompts,
                create_sampling_params(job.generation_params),
                use_tqdm=False,
            )
            if len(outputs) != len(batch_indices):
                raise ValueError(
                    f"Expected {len(batch_indices)} outputs, got {len(outputs)}"
                )

            try:
                output_positions = [
                    int(output.request_id.partition("_")[0]) for output in outputs
                ]
            except (AttributeError, ValueError) as exc:
                raise ValueError("vLLM output has an invalid request_id") from exc
            if sorted(output_positions) != list(range(len(batch_indices))):
                raise ValueError(
                    f"Unexpected vLLM request positions: {output_positions}"
                )
            ordered_outputs = [
                output
                for _, output in sorted(
                    zip(output_positions, outputs, strict=True)
                )
            ]

            for row_index, output in zip(
                batch_indices, ordered_outputs, strict=True
            ):
                image_path = images_dir / f"{row_index:09d}.png"
                extract_image(output).convert("RGB").save(image_path)
                row = CheckpointRow(
                    row_index=row_index,
                    persona_id=str(dataset[row_index][job.persona_id_column]),
                    image_path=image_path,
                )
                append_checkpoint(checkpoint_path, row)
                checkpoint[row_index] = row
                progress.update(1)

    enriched = save_dataset(job, dataset, checkpoint)
    return enriched


def run_model(
    config: dict[str, Any],
    model_key: str,
    replica_count: int,
    args: argparse.Namespace,
) -> list[str]:
    model_config = config["models"][model_key]
    jobs = [
        resolve_job(config, dataset_key, model_key, variant)
        for dataset_key in config["datasets"]
        for variant in VARIANTS
    ]
    failures: list[str] = []

    omni = None
    try:
        for job in jobs:
            enriched = None
            first_generation_attempt = True

            for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
                if omni is None:
                    try:
                        omni = create_omni(model_config, replica_count)
                    except Exception:
                        logging.exception(
                            "Could not initialize model %s (attempt %d/%d)",
                            model_config["repo_id"],
                            attempt,
                            MAX_JOB_ATTEMPTS,
                        )
                        continue

                logging.info(
                    "Starting %s (attempt %d/%d)",
                    job.target_dataset,
                    attempt,
                    MAX_JOB_ATTEMPTS,
                )
                force = args.force and first_generation_attempt
                first_generation_attempt = False

                try:
                    enriched = generate_job(
                        omni,
                        job,
                        replica_count,
                        args.limit,
                        force,
                    )
                    break
                except Exception:
                    logging.exception(
                        "Generation attempt failed for %s (attempt %d/%d)",
                        job.target_dataset,
                        attempt,
                        MAX_JOB_ATTEMPTS,
                    )
                try:
                    omni.close()
                except Exception:
                    logging.exception("Could not close model %s", job.model_name)
                omni = None

            if enriched is None:
                failures.append(job.target_dataset)
                logging.error(
                    "Generation failed for %s after %d attempts",
                    job.target_dataset,
                    MAX_JOB_ATTEMPTS,
                )
                continue

            if args.push_pr:
                try:
                    enriched.push_to_hub(
                        job.target_dataset,
                        commit_message="Add generated persona images",
                        commit_description=(
                            f"Adds {job.image_column} generated from "
                            f"{job.prompt_column} with {job.model_name}."
                        ),
                        create_pr=True,
                    )
                    logging.info(
                        "Opened a Hugging Face Hub PR for %s",
                        job.target_dataset,
                    )
                except Exception:
                    failures.append(job.target_dataset)
                    logging.exception(
                        "Could not push %s to the Hugging Face Hub",
                        job.target_dataset,
                    )
    finally:
        if omni is not None:
            try:
                omni.close()
            except Exception:
                failures.append(f"{model_config['repo_id']} (close)")
                logging.exception("Could not close model %s", model_config["repo_id"])

    return failures


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = load_config(args.config.resolve())
    gpu_ids = parse_gpu_ids(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    failures = []
    for model_key in config["models"]:
        failures.extend(run_model(config, model_key, len(gpu_ids), args))

    if failures:
        raise RuntimeError(f"Generation failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
