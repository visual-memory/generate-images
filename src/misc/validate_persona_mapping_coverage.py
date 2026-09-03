#!/usr/bin/env python3
"""Valida a cobertura de IDs nos datasets de mapping."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetPair:
    name: str
    with_ids_dataset: str
    mapping_dataset: str
    persona_id_columns: tuple[str, str]


DATASET_PAIRS = (
    DatasetPair(
        name="Synthetic-Persona-Chat",
        with_ids_dataset="visual-memory/Synthetic-Persona-Chat-With-Ids_1k",
        mapping_dataset="visual-memory/Synthetic-Persona-Chat-Mapping_1k",
        persona_id_columns=("persona-1-id", "persona-2-id"),
    ),
    DatasetPair(
        name="PersonaChat",
        with_ids_dataset="visual-memory/PersonaChat-With-Ids_1k",
        mapping_dataset="visual-memory/PersonaChat-Mapping_1k",
        persona_id_columns=("your_persona_id", "partner_persona_id"),
    ),
    DatasetPair(
        name="ConvAI2",
        with_ids_dataset="visual-memory/ConvAI2-With-Ids_1k",
        mapping_dataset="visual-memory/ConvAI2-Mapping_1k",
        persona_id_columns=("your_persona_id", "partner_persona_id"),
    ),
)

MAPPING_ID_COLUMN = "persona-id"


def as_dataset_dict(dataset: Any, dataset_id: str) -> dict[str, Dataset]:
    if isinstance(dataset, DatasetDict):
        return dict(dataset.items())
    if isinstance(dataset, Dataset):
        return {"train": dataset}
    raise TypeError(
        f"{dataset_id!r} retornou um tipo inesperado: {type(dataset).__name__}"
    )


def collect_ids(
    dataset: Any,
    dataset_id: str,
    columns: tuple[str, ...],
) -> set[str]:
    ids: set[str] = set()

    for split_name, split in as_dataset_dict(dataset, dataset_id).items():
        missing_columns = [
            column for column in columns if column not in split.column_names
        ]
        if missing_columns:
            raise ValueError(
                f"{dataset_id!r}, split {split_name!r}, não contém as colunas "
                f"{missing_columns}. Colunas disponíveis: {split.column_names}"
            )

        for column in columns:
            for row_index, persona_id in enumerate(split[column]):
                if not isinstance(persona_id, str) or not persona_id:
                    raise ValueError(
                        f"ID de persona inválido em {dataset_id!r}, "
                        f"split {split_name!r}, linha {row_index}, "
                        f"coluna {column!r}: {persona_id!r}"
                    )
                ids.add(persona_id)

    return ids


def validate_pair(pair: DatasetPair, token: str | None) -> set[str]:
    print(f"\n[{pair.name}]")
    print(f"  Carregando {pair.with_ids_dataset} ...")
    with_ids = load_dataset(pair.with_ids_dataset, token=token)
    print(f"  Carregando {pair.mapping_dataset} ...")
    mapping = load_dataset(pair.mapping_dataset, token=token)

    referenced_ids = collect_ids(
        with_ids,
        pair.with_ids_dataset,
        pair.persona_id_columns,
    )
    mapping_ids = collect_ids(
        mapping,
        pair.mapping_dataset,
        (MAPPING_ID_COLUMN,),
    )
    missing_ids = referenced_ids - mapping_ids

    print(f"  Personas únicas referenciadas: {len(referenced_ids)}")
    print(f"  Personas únicas no mapping:    {len(mapping_ids)}")
    if missing_ids:
        print(f"  FALHA: {len(missing_ids)} persona(s) sem mapping.")
    else:
        print("  OK: todas as personas referenciadas estão no mapping.")

    return missing_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida se a união das duas colunas de IDs de persona de cada dataset "
            "With-Ids está contida no respectivo dataset Mapping."
        )
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Token do Hugging Face Hub, caso os datasets exijam autenticação.",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=20,
        help="Quantidade máxima de IDs ausentes exibidos por dataset (padrão: 20).",
    )
    args = parser.parse_args()
    if args.max_missing < 0:
        parser.error("--max-missing deve ser maior ou igual a zero")
    return args


def main() -> int:
    args = parse_args()
    failures: dict[str, set[str]] = {}

    for pair in DATASET_PAIRS:
        missing_ids = validate_pair(pair, args.token)
        if missing_ids:
            failures[pair.name] = missing_ids
            for persona_id in sorted(missing_ids)[: args.max_missing]:
                print(f"    - {persona_id}")
            omitted = len(missing_ids) - args.max_missing
            if omitted > 0:
                print(f"    ... e mais {omitted} ID(s)")

    print()
    if failures:
        total_missing = sum(len(ids) for ids in failures.values())
        print(
            f"VALIDAÇÃO FALHOU: {total_missing} ocorrência(s) de IDs únicos "
            f"ausentes em {len(failures)} dataset(s)."
        )
        return 1

    print("VALIDAÇÃO CONCLUÍDA: todos os mappings têm cobertura completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
