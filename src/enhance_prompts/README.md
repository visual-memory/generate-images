# enhance_prompts

Scripts for enhancing the `description` column from
`visual-memory/Synthetic-Persona-Chat-Mapping` with the vendored
PromptEnhancer code under `3rdparty/PromptEnhancer`.

The module keeps PromptEnhancer usage intentionally thin: it imports the vendor
classes the same way `main.py` does and calls `predict(...)`. It does not modify
or reimplement the vendor model code.

## Scripts

### `enhance_dataset.py`

Single-process runner. Use this when one process/GPU is enough:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python -m enhance_prompts.enhance_dataset
```

Open a Hugging Face Hub pull request after processing:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python -m enhance_prompts.enhance_dataset --push-pr
```

Smoke test with only a few rows:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python -m enhance_prompts.enhance_dataset --limit 10 --force
```

### `enhance_dataset_shards.py`

Multi-process runner. This is the recommended path for throughput. Pass one GPU
id per process; repeated ids are allowed when the GPU has enough VRAM:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,1
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,0,0,0
```

The `run` command:

1. creates one shard per GPU id;
2. launches one subprocess per shard with `CUDA_VISIBLE_DEVICES` set;
3. waits for all shards to finish;
4. merges checkpoints back into the original row order;
5. writes the final dataset and optional PR.

Smoke test:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,1 --limit 10 --force
```

Production run with PR:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run --gpus 0,1 --push-pr
```

Manual subcommands also exist for debugging:

```bash
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards run-shard --num-shards 2 --shard-index 0
PYTHONPATH=src uv run python -m enhance_prompts.enhance_dataset_shards merge --num-shards 2
```

## Static Configuration

Most behavior is intentionally static at the top of `enhance_dataset.py`:

- dataset id: `visual-memory/Synthetic-Persona-Chat-Mapping`
- split: `train`
- input column: `description`
- output column: `enhanced_description`
- backend: `7b`
- model path: `/raid/aluno_paulosantana/models/promptenhancer-7b/reprompt`
- prompt template:

```text
Generate a photo of a person with the following self-description:
{description}
```

Generation defaults are deterministic:

- `temperature = 0.0`
- `top_p = 0.9`
- `max_new_tokens = 256`

## CLI Arguments

The single-process script exposes only operational arguments:

```text
--model-path
--device
--limit
--output-dir
--target-repo-id
--push-pr
--force
```

The sharded `run` command exposes:

```text
--gpus
--model-path
--limit
--output-dir
--final-output-dir
--target-repo-id
--push-pr
--force
```

`--force` removes the current checkpoint for the selected output path before
processing. Use it after changing generation settings or when rerunning smoke
tests from scratch.

## Checkpoints And Outputs

Checkpoints are JSONL files containing original row indices:

```json
{"row_index": 123, "enhanced_description": "..."}
```

They are flushed every 100 completed samples and once more at process exit for
the remaining buffered samples. If a process stops unexpectedly, it may lose up
to 99 completed samples from that process since the previous flush.

Single-process checkpoints are stored under:

```text
outputs/enhanced-dataset/enhanced_description.jsonl
```

Sharded checkpoints are stored under:

```text
outputs/enhanced-dataset-shards/num_shards-N/shard-XXXXX/enhanced_description.jsonl
```

The final merged dataset is written to:

```text
outputs/enhanced-dataset/dataset
outputs/enhanced-dataset/train.parquet
```

## Ordering

The final dataset preserves the original dataset order. Each shard writes the
original `row_index`; merge reconstructs:

```python
enhanced_values = [enhanced[index] for index in range(len(dataset))]
```

Merge fails if any expected row is missing.

## Testing

Run:

```bash
uv run pytest
uv run python -m compileall src tests
```

