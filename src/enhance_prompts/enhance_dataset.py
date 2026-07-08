from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["v2", "7b"]

DEFAULT_DATASET_ID = "visual-memory/Synthetic-Persona-Chat-Mapping"
DEFAULT_MODEL_PATH = "/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt"
DEFAULT_TARGET_REPO_ID = DEFAULT_DATASET_ID
DEFAULT_SPLIT = "train"
DEFAULT_DESCRIPTION_COLUMN = "description"
DEFAULT_ENHANCED_COLUMN = "enhanced_description"
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


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    sys_prompt: str | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_prompt_enhancer_to_path() -> None:
    inference_path = repo_root() / "3rdparty" / "PromptEnhancer" / "inference"
    sys.path.insert(0, str(inference_path))


def build_prompt(description: str, template: str) -> str:
    return template.format(description=str(description).strip())


def clean_enhanced_description(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_checkpoint(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    completed: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                completed[int(item["row_index"])] = str(item["enhanced_description"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid checkpoint line {line_number} in {path}"
                ) from exc
    return completed


def append_checkpoint(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row_index, enhanced_description in rows:
            handle.write(
                json.dumps(
                    {
                        "row_index": row_index,
                        "enhanced_description": enhanced_description,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class VendorPromptEnhancer:
    def __init__(
        self,
        *,
        backend: Backend,
        model_path: str,
        device_map: str,
        device: str | None,
    ) -> None:
        add_prompt_enhancer_to_path()
        self.backend = backend
        self.device = device

        if backend == "v2":
            from prompt_enhancer_v2 import PromptEnhancerV2

            self.enhancer = PromptEnhancerV2(
                models_root_path=model_path,
                device_map=device_map,
            )
        else:
            from prompt_enhancer import HunyuanPromptEnhancer

            self.enhancer = HunyuanPromptEnhancer(
                models_root_path=model_path,
                device_map=device_map,
            )

    def enhance_many(
        self,
        prompts: list[str],
        config: GenerationConfig,
    ) -> list[str]:
        return [self.enhance_one(prompt, config) for prompt in prompts]

    def enhance_one(self, prompt: str, config: GenerationConfig) -> str:
        kwargs = {
            "prompt_cot": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
        }
        if config.sys_prompt is not None:
            kwargs["sys_prompt"] = config.sys_prompt
        if self.backend == "v2" and self.device is not None:
            kwargs["device"] = self.device
        return clean_enhanced_description(self.enhancer.predict(**kwargs))


def enhance_dataset(
    *,
    dataset_id: str,
    split: str,
    description_column: str,
    enhanced_column: str,
    template: str,
    output_dir: Path,
    checkpoint_name: str,
    checkpoint_interval: int,
    limit: int | None,
    enhancer: VendorPromptEnhancer,
    generation_config: GenerationConfig,
    force: bool,
):
    from datasets import DatasetDict, load_dataset
    from tqdm.auto import tqdm

    dataset_dict = load_dataset(dataset_id)
    dataset = dataset_dict[split]
    if description_column not in dataset.column_names:
        raise ValueError(
            f"Column {description_column!r} not found. Available columns: "
            f"{dataset.column_names}"
        )
    if enhanced_column in dataset.column_names:
        raise ValueError(f"Column {enhanced_column!r} already exists.")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    descriptions = [str(item) for item in dataset[description_column]]
    prompts = [build_prompt(description, template) for description in descriptions]

    checkpoint_path = output_dir / checkpoint_name
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()

    completed = load_checkpoint(checkpoint_path)
    enhanced: list[str | None] = [completed.get(index) for index in range(len(dataset))]

    pending_indices = [
        index for index, value in enumerate(enhanced) if value is None
    ]

    progress = tqdm(
        total=len(pending_indices),
        desc="Enhancing descriptions",
        unit="row",
    )
    checkpoint_buffer = []
    for row_index in pending_indices:
        prompt = prompts[row_index]
        try:
            enhanced_description = enhancer.enhance_one(prompt, generation_config)
        except Exception:
            logging.exception("Enhancement failed; falling back to original prompts.")
            enhanced_description = prompt

        enhanced[row_index] = enhanced_description
        checkpoint_buffer.append((row_index, enhanced_description))
        if len(checkpoint_buffer) >= checkpoint_interval:
            append_checkpoint(checkpoint_path, checkpoint_buffer)
            checkpoint_buffer.clear()
        progress.update(1)
    progress.close()
    if checkpoint_buffer:
        append_checkpoint(checkpoint_path, checkpoint_buffer)

    enhanced_values = [
        value if value is not None else prompts[index]
        for index, value in enumerate(enhanced)
    ]
    enriched_dataset = dataset.add_column(enhanced_column, enhanced_values)
    enriched = DatasetDict({split: enriched_dataset})

    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "dataset"
    enriched.save_to_disk(str(local_path))
    parquet_path = output_dir / f"{split}.parquet"
    enriched_dataset.to_parquet(str(parquet_path))
    return enriched, local_path, parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance Synthetic-Persona descriptions with PromptEnhancer."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-repo-id", default=DEFAULT_TARGET_REPO_ID)
    parser.add_argument("--push-pr", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

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

    enriched, local_path, parquet_path = enhance_dataset(
        dataset_id=DEFAULT_DATASET_ID,
        split=DEFAULT_SPLIT,
        description_column=DEFAULT_DESCRIPTION_COLUMN,
        enhanced_column=DEFAULT_ENHANCED_COLUMN,
        template=DEFAULT_TEMPLATE,
        output_dir=args.output_dir,
        checkpoint_name=DEFAULT_CHECKPOINT_NAME,
        checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
        limit=args.limit,
        enhancer=enhancer,
        generation_config=generation_config,
        force=args.force,
    )
    logging.info("Saved dataset to %s", local_path)
    logging.info("Saved parquet to %s", parquet_path)

    if args.push_pr:
        enriched.push_to_hub(
            args.target_repo_id,
            commit_message=DEFAULT_COMMIT_MESSAGE,
            commit_description=DEFAULT_COMMIT_DESCRIPTION,
            create_pr=True,
        )
        logging.info("Opened a Hugging Face Hub pull request for %s", args.target_repo_id)


if __name__ == "__main__":
    main()
