import argparse
import json
from pathlib import Path

import pytest

from generate_images import generate_persona_images_shards as shards


def test_shard_indices_partition_rows():
    assert shards.shard_indices(total_rows=10, num_shards=3, shard_index=0) == [
        0,
        3,
        6,
        9,
    ]
    assert shards.shard_indices(total_rows=10, num_shards=3, shard_index=1) == [
        1,
        4,
        7,
    ]
    assert shards.shard_indices(total_rows=10, num_shards=3, shard_index=2) == [
        2,
        5,
        8,
    ]


def test_shard_indices_validate_arguments():
    with pytest.raises(ValueError):
        shards.shard_indices(total_rows=10, num_shards=0, shard_index=0)
    with pytest.raises(ValueError):
        shards.shard_indices(total_rows=10, num_shards=3, shard_index=3)


def test_parse_gpu_ids():
    assert shards.parse_gpu_ids("0,1, 2") == ["0", "1", "2"]
    with pytest.raises(ValueError):
        shards.parse_gpu_ids(" , ")


def test_checkpoint_round_trip_requires_existing_image(tmp_path):
    checkpoint = tmp_path / "persona_images.jsonl"
    present_image = tmp_path / "present.png"
    missing_image = tmp_path / "missing.png"
    present_image.write_bytes(b"not actually decoded in checkpoint tests")

    shards.append_checkpoint(
        checkpoint,
        [
            shards.ImageCheckpointRow(
                row_index=0,
                persona_id="persona-0",
                image_path=present_image,
            ),
            shards.ImageCheckpointRow(
                row_index=1,
                persona_id="persona-1",
                image_path=missing_image,
            ),
        ],
    )

    completed = shards.load_checkpoint(checkpoint)

    assert completed == {
        0: shards.ImageCheckpointRow(
            row_index=0,
            persona_id="persona-0",
            image_path=present_image,
        )
    }
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "row_index": 0,
        "persona_id": "persona-0",
        "image_path": str(present_image),
    }


def test_build_prompt_uses_enhanced_description_text():
    assert shards.build_prompt(" enhanced portrait prompt ") == (
        "Generate an image of this person's description: enhanced portrait prompt"
    )


def test_shard_dir_and_image_path_are_stable():
    base = Path("outputs/persona-images-shards")
    assert shards.shard_dir(base, 4, 2) == (
        base / "num_shards-4" / "shard-00002"
    )
    assert shards.image_path_for_row(base / "images", 42) == (
        base / "images" / "000000042.png"
    )


def test_run_all_assigns_tqdm_position_per_shard(monkeypatch, tmp_path):
    started_envs = []

    class FakeProcess:
        def wait(self):
            return 0

    def fake_popen(command, env):
        started_envs.append(env)
        return FakeProcess()

    monkeypatch.setattr(shards.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(shards, "merge_shards", lambda args: None)

    args = argparse.Namespace(
        gpus="2,2",
        limit=None,
        output_dir=tmp_path / "shards",
        final_output_dir=tmp_path / "final",
        target_repo_id="target",
        push_pr=False,
        force=False,
    )

    shards.run_all(args)

    assert [env["CUDA_VISIBLE_DEVICES"] for env in started_envs] == ["2", "2"]
    assert [env["TQDM_POSITION"] for env in started_envs] == ["0", "1"]


def test_merge_preserves_order_and_detects_missing_rows(monkeypatch, tmp_path):
    class FakeDataset:
        column_names = ["persona-id", "enhanced_description"]

        def __init__(self):
            self.rows = [
                {"persona-id": "p0", "enhanced_description": "first"},
                {"persona-id": "p1", "enhanced_description": "second"},
                {"persona-id": "p2", "enhanced_description": "third"},
            ]

        def __len__(self):
            return len(self.rows)

    class FakeDatasetDict(dict):
        def save_to_disk(self, path):
            self["saved_path"] = path

        def push_to_hub(self, *args, **kwargs):
            raise AssertionError("push_to_hub should not be called")

    recorded_rows = {}

    def fake_build_image_dataset(dataset, checkpoint_rows):
        recorded_rows.update(checkpoint_rows)
        return FakeDatasetDict({"train": FakeParquetDataset()})

    class FakeParquetDataset:
        def to_parquet(self, path):
            self.path = path

    monkeypatch.setattr(shards, "load_source_dataset", lambda limit: FakeDataset())
    monkeypatch.setattr(shards, "build_image_dataset", fake_build_image_dataset)

    output_dir = tmp_path / "shards"
    image_paths = []
    for row_index in range(3):
        shard_index = row_index % 2
        current_shard_dir = shards.shard_dir(output_dir, 2, shard_index)
        image_path = current_shard_dir / "images" / f"{row_index:09d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        image_paths.append(image_path)
        shards.append_checkpoint(
            current_shard_dir / shards.DEFAULT_CHECKPOINT_NAME,
            [
                shards.ImageCheckpointRow(
                    row_index=row_index,
                    persona_id=f"p{row_index}",
                    image_path=image_path,
                )
            ],
        )

    args = argparse.Namespace(
        num_shards=2,
        limit=None,
        output_dir=output_dir,
        final_output_dir=tmp_path / "final",
        target_repo_id="target",
        push_pr=False,
    )

    shards.merge_shards(args)

    assert list(recorded_rows) == [0, 2, 1]
    assert [recorded_rows[index].image_path for index in range(3)] == image_paths

    image_paths[1].unlink()
    with pytest.raises(ValueError, match="Missing 1 generated images"):
        shards.merge_shards(args)
