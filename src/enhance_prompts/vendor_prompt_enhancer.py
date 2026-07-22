from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Literal

from enhance_prompts.utils import GenerationConfig, clean_enhanced_description

Backend = Literal["v2", "7b"]

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_prompt_enhancer_to_path() -> None:
    inference_path = repo_root() / "3rdparty" / "PromptEnhancer" / "inference"
    sys.path.insert(0, str(inference_path))

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