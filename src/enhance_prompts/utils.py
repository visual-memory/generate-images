from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    sys_prompt: str | None

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

def clean_enhanced_description(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def build_prompt(description: str, template: str) -> str:
    return template.format(description=str(description).strip())