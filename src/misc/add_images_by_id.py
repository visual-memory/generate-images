from __future__ import annotations

import argparse
import logging
from collections.abc import Hashable, Iterable
from typing import Any

from datasets import Dataset, DatasetDict, Image, concatenate_datasets, load_dataset

DEFAULT_SPLIT = "train"
DEFAULT_ID_COLUMN = "persona-id"
DEFAULT_IMAGE_COLUMN = "persona-image"
ERROR_PREVIEW_SIZE = 10


def _preview(values: Iterable[Any]) -> list[Any]:
    preview = []
    for value in values:
        preview.append(value)
        if len(preview) == ERROR_PREVIEW_SIZE:
            break
    return preview


def _validate_columns(
    dataset: Dataset,
    dataset_label: str,
    required_columns: Iterable[str],
) -> None:
    missing_columns = [
        column for column in required_columns if column not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"{dataset_label} is missing columns {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )


def _build_unique_id_index(
    dataset: Dataset,
    id_column: str,
    dataset_label: str,
) -> dict[Hashable, int]:
    id_to_index: dict[Hashable, int] = {}
    null_indices: list[int] = []
    duplicate_ids: list[Hashable] = []

    for row_index, row_id in enumerate(dataset[id_column]):
        if row_id is None:
            null_indices.append(row_index)
            continue
        if not isinstance(row_id, Hashable):
            raise ValueError(
                f"{dataset_label} has an unhashable ID at row {row_index}: "
                f"{row_id!r}"
            )
        if row_id in id_to_index:
            duplicate_ids.append(row_id)
            continue
        id_to_index[row_id] = row_index

    if null_indices:
        raise ValueError(
            f"{dataset_label} has {len(null_indices)} null IDs in column "
            f"{id_column!r}. First row indices: {_preview(null_indices)}"
        )
    if duplicate_ids:
        unique_duplicates = list(dict.fromkeys(duplicate_ids))
        raise ValueError(
            f"{dataset_label} has {len(unique_duplicates)} duplicate IDs in column "
            f"{id_column!r}. First IDs: {_preview(unique_duplicates)!r}"
        )
    return id_to_index


def add_images_by_id(
    mapping_dataset: Dataset,
    image_dataset: Dataset,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    image_column: str = DEFAULT_IMAGE_COLUMN,
) -> Dataset:
    """Add images to mapping rows by ID while retaining mapping row order."""
    _validate_columns(mapping_dataset, "Mapping dataset", [id_column])
    _validate_columns(image_dataset, "Image dataset", [id_column, image_column])
    if image_column in mapping_dataset.column_names:
        raise ValueError(
            f"Mapping dataset already contains the output column {image_column!r}"
        )

    mapping_id_to_index = _build_unique_id_index(
        mapping_dataset, id_column, "Mapping dataset"
    )
    image_id_to_index = _build_unique_id_index(
        image_dataset, id_column, "Image dataset"
    )

    mapping_ids = list(mapping_id_to_index)
    missing_ids = [row_id for row_id in mapping_ids if row_id not in image_id_to_index]
    if missing_ids:
        raise ValueError(
            f"Image dataset is missing {len(missing_ids)} IDs required by the mapping "
            f"dataset. First IDs: {_preview(missing_ids)!r}"
        )

    # Image(decode=False) keeps the underlying path/bytes representation in Arrow.
    # Selecting and concatenating columns therefore does not decode image pixels.
    encoded_images = image_dataset.cast_column(image_column, Image(decode=False))
    ordered_indices = [image_id_to_index[row_id] for row_id in mapping_ids]
    ordered_images = encoded_images.select(ordered_indices).select_columns(
        [image_column]
    )
    return concatenate_datasets([mapping_dataset, ordered_images], axis=1)


def load_split(
    dataset_id: str,
    *,
    split: str,
    revision: str | None,
    token: str | None,
) -> Dataset:
    dataset_dict = load_dataset(dataset_id, revision=revision, token=token)
    if split not in dataset_dict:
        raise ValueError(
            f"Dataset {dataset_id!r} does not contain split {split!r}. "
            f"Available splits: {list(dataset_dict)}"
        )
    return dataset_dict[split]


def run(args: argparse.Namespace) -> Any:
    logging.info("Loading mapping dataset %s", args.mapping_dataset)
    mapping_dataset = load_split(
        args.mapping_dataset,
        split=args.split,
        revision=args.mapping_revision,
        token=args.token,
    )
    logging.info("Loading image dataset %s", args.image_dataset)
    image_dataset = load_split(
        args.image_dataset,
        split=args.split,
        revision=args.image_revision,
        token=args.token,
    )

    result = add_images_by_id(
        mapping_dataset,
        image_dataset,
        id_column=args.id_column,
        image_column=args.image_column,
    )
    logging.info(
        "Matched %d/%d mapping rows; publishing columns %s",
        len(result),
        len(mapping_dataset),
        result.column_names,
    )
    dataset_dict = DatasetDict({args.split: result})
    publish_result = dataset_dict.push_to_hub(
        args.target_dataset,
        token=args.token,
        create_pr=True,
        commit_message="Add images matched by ID",
        commit_description=(
            f"Adds {args.image_column} from {args.image_dataset} to rows from "
            f"{args.mapping_dataset}, matched using {args.id_column}."
        ),
    )
    result_url = getattr(publish_result, "pr_url", None) or getattr(
        publish_result, "url", None
    )
    logging.info("Opened Hugging Face dataset pull request: %s", result_url or publish_result)
    return publish_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add an image column to mapping rows by matching a common ID."
    )
    parser.add_argument("--mapping-dataset", required=True)
    parser.add_argument("--image-dataset", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN)
    parser.add_argument("--image-column", default=DEFAULT_IMAGE_COLUMN)
    parser.add_argument("--token", default=None)
    parser.add_argument("--mapping-revision", default=None)
    parser.add_argument("--image-revision", default=None)
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()