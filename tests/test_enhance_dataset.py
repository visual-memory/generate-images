import json

import datasets
from datasets import Dataset

from enhance_prompts.enhance_dataset import (
    DEFAULT_TEMPLATE,
    DEFAULT_TEMPERATURE,
    GenerationConfig,
    VendorPromptEnhancer,
    append_checkpoint,
    build_prompt,
    clean_enhanced_description,
    enhance_dataset,
    load_checkpoint,
)


def test_build_prompt_uses_main_template():
    description = "I work in a factory.\nI am not social."

    assert build_prompt(description, DEFAULT_TEMPLATE) == (
        "Generate a photo of a person with the following self-description:\n"
        "I work in a factory.\nI am not social."
    )


def test_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"

    append_checkpoint(checkpoint, [(0, "first"), (2, "third")])

    assert load_checkpoint(checkpoint) == {0: "first", 2: "third"}
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "row_index": 0,
        "enhanced_description": "first",
    }


def test_clean_enhanced_description_removes_leaked_tags_and_cjk():
    text = "<answer>\n<think>draft</think>\nFinal 墙上 enhanced prompt\n</answer>"

    assert clean_enhanced_description(text) == "Final enhanced prompt"


def test_vendor_wrapper_uses_predict_api():
    class FakeEnhancer:
        def __init__(self):
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return f"enhanced: {kwargs['prompt_cot']}"

    fake = FakeEnhancer()
    enhancer = VendorPromptEnhancer.__new__(VendorPromptEnhancer)
    enhancer.backend = "v2"
    enhancer.device = "cuda"
    enhancer.enhancer = fake
    config = GenerationConfig(
        temperature=DEFAULT_TEMPERATURE,
        top_p=0.9,
        max_new_tokens=256,
        sys_prompt=None,
    )

    assert enhancer.enhance_many(["one", "two"], config) == [
        "enhanced: one",
        "enhanced: two",
    ]
    assert fake.calls[0]["prompt_cot"] == "one"
    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["device"] == "cuda"


def test_enhance_dataset_flushes_checkpoint_by_interval(monkeypatch, tmp_path):
    class FakeDatasetDict(dict):
        def save_to_disk(self, path):
            pass

    class FakeEnhancer:
        def enhance_one(self, prompt, config):
            return f"enhanced: {prompt}"

    dataset = Dataset.from_dict(
        {
            "persona-id": ["0", "1", "2"],
            "description": ["first", "second", "third"],
        }
    )

    def fake_load_dataset(dataset_id):
        return {"train": dataset}

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(datasets, "DatasetDict", FakeDatasetDict)
    monkeypatch.setattr(Dataset, "to_parquet", lambda self, path: None)

    enhance_dataset(
        dataset_id="dataset",
        split="train",
        description_column="description",
        enhanced_column="enhanced_description",
        template="{description}",
        output_dir=tmp_path,
        checkpoint_name="checkpoint.jsonl",
        checkpoint_interval=2,
        limit=None,
        enhancer=FakeEnhancer(),
        generation_config=GenerationConfig(
            temperature=0,
            top_p=1,
            max_new_tokens=1,
            sys_prompt=None,
        ),
        force=False,
    )

    checkpoint = tmp_path / "checkpoint.jsonl"
    assert load_checkpoint(checkpoint) == {
        0: "enhanced: first",
        1: "enhanced: second",
        2: "enhanced: third",
    }
