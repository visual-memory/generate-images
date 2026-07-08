from pathlib import Path

import pytest

from enhance_prompts.enhance_dataset_shards import parse_gpu_ids, shard_dir, shard_indices


def test_shard_indices_partition_rows():
    assert shard_indices(total_rows=10, num_shards=3, shard_index=0) == [0, 3, 6, 9]
    assert shard_indices(total_rows=10, num_shards=3, shard_index=1) == [1, 4, 7]
    assert shard_indices(total_rows=10, num_shards=3, shard_index=2) == [2, 5, 8]


def test_shard_indices_validate_arguments():
    with pytest.raises(ValueError):
        shard_indices(total_rows=10, num_shards=0, shard_index=0)
    with pytest.raises(ValueError):
        shard_indices(total_rows=10, num_shards=3, shard_index=3)


def test_shard_dir_is_stable():
    assert shard_dir(Path("outputs/shards"), 4, 2) == (
        Path("outputs/shards") / "num_shards-4" / "shard-00002"
    )


def test_parse_gpu_ids():
    assert parse_gpu_ids("0,1, 2") == ["0", "1", "2"]
    with pytest.raises(ValueError):
        parse_gpu_ids(" , ")
