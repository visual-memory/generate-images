#!/usr/bin/env python3
"""Mescla personas redundantes nos datasets de mapping, derivados e de diálogos.

Duas personas são redundantes quando diferem apenas na ordem das frases, em
contrações, caixa ou pontuação. Cada grupo redundante é reduzido à sua primeira
ocorrência, e os IDs absorvidos passam a ser registrados em `merged-persona-ids`
e substituídos pelo ID mantido nos datasets de diálogos.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from generate_images.orchestrate_persona_images_vllm import (
    VARIANTS,
    load_config,
    resolve_job,
)
from misc.validate_persona_mapping_coverage import DATASET_PAIRS, DatasetPair

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "generate_images"
    / "configs"
    / "persona_image_generation.yaml"
)
DEFAULT_SUFFIX = "-no-redundancy"
MERGED_IDS_COLUMN = "merged-persona-ids"
ERROR_PREVIEW_SIZE = 10

STATEMENT_SPLIT = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+")
TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)

CONTRACTIONS = {
    ("i", "m"): ("i", "am"),
    ("i", "ve"): ("i", "have"),
    ("i", "ll"): ("i", "will"),
    ("i", "d"): ("i", "would"),
    ("you", "re"): ("you", "are"),
    ("you", "ve"): ("you", "have"),
    ("you", "ll"): ("you", "will"),
    ("you", "d"): ("you", "would"),
    ("we", "re"): ("we", "are"),
    ("we", "ve"): ("we", "have"),
    ("we", "ll"): ("we", "will"),
    ("we", "d"): ("we", "would"),
    ("they", "re"): ("they", "are"),
    ("they", "ve"): ("they", "have"),
    ("they", "ll"): ("they", "will"),
    ("they", "d"): ("they", "would"),
    ("he", "s"): ("he", "is"),
    ("he", "d"): ("he", "would"),
    ("she", "s"): ("she", "is"),
    ("she", "d"): ("she", "would"),
    ("it", "s"): ("it", "is"),
    ("it", "d"): ("it", "would"),
    ("that", "s"): ("that", "is"),
    ("there", "s"): ("there", "is"),
    ("isn", "t"): ("is", "not"),
    ("aren", "t"): ("are", "not"),
    ("wasn", "t"): ("was", "not"),
    ("weren", "t"): ("were", "not"),
    ("don", "t"): ("do", "not"),
    ("doesn", "t"): ("does", "not"),
    ("didn", "t"): ("did", "not"),
    ("haven", "t"): ("have", "not"),
    ("hasn", "t"): ("has", "not"),
    ("hadn", "t"): ("had", "not"),
    ("can", "t"): ("can", "not"),
    ("won", "t"): ("will", "not"),
    ("couldn", "t"): ("could", "not"),
    ("wouldn", "t"): ("would", "not"),
    ("shouldn", "t"): ("should", "not"),
    ("would", "ve"): ("would", "have"),
    ("could", "ve"): ("could", "have"),
    ("should", "ve"): ("should", "have"),
}


def normalize(value: Any) -> str:
    """Normaliza caixa, Unicode, pontuação, espaços e contrações."""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    tokens = TOKEN.findall(value)

    expanded: list[str] = []
    index = 0
    while index < len(tokens):
        replacement = CONTRACTIONS.get(tuple(tokens[index : index + 2]))
        if replacement is None:
            expanded.append(tokens[index])
            index += 1
        else:
            expanded.extend(replacement)
            index += 2

    return " ".join(expanded)


def signature(text: str) -> tuple[str, ...]:
    """Assinatura de uma persona, insensível à ordem das frases."""
    statements = [
        normalized
        for statement in STATEMENT_SPLIT.split(text.strip())
        if (normalized := normalize(statement))
    ]
    return tuple(sorted(statements))


def _preview(values: Iterable[Any]) -> list[Any]:
    preview = []
    for value in values:
        preview.append(value)
        if len(preview) == ERROR_PREVIEW_SIZE:
            break
    return preview


@dataclass(frozen=True)
class MergePlan:
    """Grupos redundantes de um dataset de mapping.

    `keep_indices` são as linhas mantidas e `merged_ids[i]` lista os IDs do grupo
    da linha `keep_indices[i]`, começando pelo ID mantido.
    """

    keep_indices: list[int]
    merged_ids: list[list[str]]

    @property
    def merged_count(self) -> int:
        return sum(len(group) - 1 for group in self.merged_ids)

    @property
    def replacement(self) -> dict[str, str]:
        """Mapeia cada ID original para o ID mantido do seu grupo."""
        return {
            persona_id: group[0]
            for group in self.merged_ids
            for persona_id in group
        }


def compute_merge_plan(
    persona_ids: Sequence[str],
    texts: Sequence[Any],
    dataset_label: str,
) -> MergePlan:
    """Agrupa personas por assinatura, mantendo a primeira ocorrência de cada grupo."""
    keep_indices: list[int] = []
    merged_ids: list[list[str]] = []
    groups: dict[tuple[str, ...], list[str]] = {}
    seen_ids: set[str] = set()

    for index, (persona_id, text) in enumerate(zip(persona_ids, texts, strict=True)):
        if not isinstance(persona_id, str) or not persona_id:
            raise ValueError(
                f"{dataset_label} tem um ID inválido na linha {index}: {persona_id!r}"
            )
        if persona_id in seen_ids:
            raise ValueError(f"{dataset_label} tem o ID duplicado {persona_id!r}")
        seen_ids.add(persona_id)

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{dataset_label} não tem texto de persona na linha {index} "
                f"(ID {persona_id!r})"
            )

        key = signature(text)
        group = groups.get(key)
        if group is None:
            group = [persona_id]
            groups[key] = group
            keep_indices.append(index)
            merged_ids.append(group)
        else:
            group.append(persona_id)

    return MergePlan(keep_indices, merged_ids)


def apply_merge_plan(dataset: Dataset, plan: MergePlan) -> Dataset:
    """Mantém apenas as linhas do plano e registra os IDs absorvidos."""
    return dataset.select(plan.keep_indices).add_column(
        MERGED_IDS_COLUMN, plan.merged_ids
    )


def validate_same_persona_ids(
    dataset: Dataset,
    dataset_label: str,
    id_column: str,
    expected_ids: Sequence[str],
) -> None:
    """Garante que um dataset derivado está alinhado linha a linha com o mapping."""
    if id_column not in dataset.column_names:
        raise ValueError(
            f"{dataset_label} não contém a coluna {id_column!r}. "
            f"Colunas disponíveis: {dataset.column_names}"
        )

    persona_ids = dataset[id_column]
    if len(persona_ids) != len(expected_ids):
        raise ValueError(
            f"{dataset_label} tem {len(persona_ids)} linhas, mas o mapping tem "
            f"{len(expected_ids)}"
        )

    divergences = [
        index
        for index, (actual, expected) in enumerate(
            zip(persona_ids, expected_ids, strict=True)
        )
        if actual != expected
    ]
    if divergences:
        raise ValueError(
            f"{dataset_label} tem {len(divergences)} linhas com {id_column!r} "
            f"diferente do mapping. Primeiros índices: {_preview(divergences)}"
        )


def remap_persona_ids(
    dataset: Dataset,
    dataset_label: str,
    id_columns: Sequence[str],
    replacement: dict[str, str],
) -> Dataset:
    """Substitui os IDs absorvidos pelos IDs mantidos, preservando todas as linhas."""
    missing_columns = [
        column for column in id_columns if column not in dataset.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"{dataset_label} não contém as colunas {missing_columns}. "
            f"Colunas disponíveis: {dataset.column_names}"
        )

    unknown_ids = {
        persona_id
        for column in id_columns
        for persona_id in dataset[column]
        if persona_id not in replacement
    }
    if unknown_ids:
        raise ValueError(
            f"{dataset_label} referencia {len(unknown_ids)} persona(s) ausente(s) "
            f"no mapping. Primeiros IDs: {_preview(sorted(unknown_ids))!r}"
        )

    return dataset.map(
        lambda batch: {
            column: [replacement[persona_id] for persona_id in batch[column]]
            for column in id_columns
        },
        batched=True,
        desc=f"Remapeando {dataset_label}",
    )


def log_remap_stats(
    dataset: Dataset,
    dataset_label: str,
    id_columns: Sequence[str],
    replacement: dict[str, str],
) -> None:
    columns = [dataset[column] for column in id_columns]
    rewritten = sum(
        replacement[persona_id] != persona_id
        for column in columns
        for persona_id in column
    )
    logging.info(
        "%s: %d referência(s) de persona reescrita(s); %d persona(s) única(s) -> %d",
        dataset_label,
        rewritten,
        len({persona_id for column in columns for persona_id in column}),
        len({replacement[persona_id] for column in columns for persona_id in column}),
    )

    if len(columns) == 2:
        collapsed = sum(
            replacement[first] == replacement[second]
            for first, second in zip(*columns, strict=True)
        )
        if collapsed:
            logging.info(
                "%s: %d diálogo(s) passam a referenciar a mesma persona nas duas colunas",
                dataset_label,
                collapsed,
            )


def log_column_divergences(
    dataset: Dataset,
    dataset_label: str,
    plan: MergePlan,
    id_column: str,
    text_column: str,
) -> None:
    """Reporta colunas cujo conteúdo diverge dentro de um grupo mesclado.

    A mesclagem preserva os valores da linha mantida; estas contagens mostram
    quanta informação das linhas absorvidas está sendo descartada.
    """
    index_by_id = {
        persona_id: index for index, persona_id in enumerate(dataset[id_column])
    }
    other_columns = [
        column
        for column in dataset.column_names
        if column not in (id_column, text_column)
        and getattr(dataset.features[column], "dtype", None) == "string"
    ]

    for column in other_columns:
        values = dataset[column]
        divergent = sum(
            len({signature(str(values[index_by_id[persona_id]])) for persona_id in group})
            > 1
            for group in plan.merged_ids
            if len(group) > 1
        )
        logging.info(
            "%s: %d grupo(s) mesclado(s) com %r divergente",
            dataset_label,
            divergent,
            column,
        )


def push(
    dataset: Dataset,
    repo_id: str,
    split: str,
    token: str | None,
    commit_description: str,
) -> None:
    logging.info("Publicando %s (%d linhas)", repo_id, len(dataset))
    DatasetDict({split: dataset}).push_to_hub(
        repo_id,
        token=token,
        commit_message="Merge redundant personas",
        commit_description=commit_description,
    )


def find_dataset_pair(mapping_dataset: str) -> DatasetPair:
    for pair in DATASET_PAIRS:
        if pair.mapping_dataset == mapping_dataset:
            return pair
    raise ValueError(
        f"Nenhum dataset de diálogos conhecido para o mapping {mapping_dataset!r}"
    )


def process_dataset(
    config: dict[str, Any],
    dataset_key: str,
    model_keys: Sequence[str],
    args: argparse.Namespace,
) -> None:
    dataset_config = config["datasets"][dataset_key]
    mapping_dataset = str(dataset_config["repo_id"])
    text_column = str(dataset_config["columns"]["original"])
    id_column = str(config["persona_id_column"])
    split = str(config["split"])
    pair = find_dataset_pair(mapping_dataset)

    logging.info("Carregando %s", mapping_dataset)
    mapping = load_dataset(mapping_dataset, split=split, token=args.token)
    plan = compute_merge_plan(
        mapping[id_column],
        mapping[text_column],
        mapping_dataset,
    )
    logging.info(
        "%s: %d linhas -> %d linhas (%d mescladas em %d grupo(s) redundante(s), "
        "assinatura por %r)",
        mapping_dataset,
        len(mapping),
        len(plan.keep_indices),
        plan.merged_count,
        sum(1 for group in plan.merged_ids if len(group) > 1),
        text_column,
    )
    if args.dry_run:
        log_column_divergences(mapping, mapping_dataset, plan, id_column, text_column)

    persona_ids = mapping[id_column]
    provenance = (
        f"Personas redundantes de {mapping_dataset} mescladas por assinatura de "
        f"{text_column!r} (ordem das frases, contrações, caixa e pontuação "
        f"ignoradas). {plan.merged_count} linha(s) absorvida(s); os IDs de cada "
        f"grupo ficam em {MERGED_IDS_COLUMN!r}."
    )

    if not args.dry_run:
        push(
            apply_merge_plan(mapping, plan),
            f"{mapping_dataset}{args.suffix}",
            split,
            args.token,
            provenance,
        )

    for model_key in model_keys:
        for variant in VARIANTS:
            derived_dataset = resolve_job(
                config, dataset_key, model_key, variant
            ).target_dataset
            if args.dry_run:
                logging.info(
                    "[dry-run] %s -> %s%s",
                    derived_dataset,
                    derived_dataset,
                    args.suffix,
                )
                continue

            logging.info("Carregando %s", derived_dataset)
            derived = load_dataset(derived_dataset, split=split, token=args.token)
            validate_same_persona_ids(
                derived, derived_dataset, id_column, persona_ids
            )
            push(
                apply_merge_plan(derived, plan),
                f"{derived_dataset}{args.suffix}",
                split,
                args.token,
                f"{provenance} As mesmas linhas são mantidas em {derived_dataset}.",
            )

    logging.info("Carregando %s", pair.with_ids_dataset)
    with_ids = load_dataset(pair.with_ids_dataset, split=split, token=args.token)
    replacement = plan.replacement
    log_remap_stats(
        with_ids, pair.with_ids_dataset, pair.persona_id_columns, replacement
    )
    if not args.dry_run:
        push(
            remap_persona_ids(
                with_ids,
                pair.with_ids_dataset,
                pair.persona_id_columns,
                replacement,
            ),
            f"{pair.with_ids_dataset}{args.suffix}",
            split,
            args.token,
            f"{provenance} As colunas {list(pair.persona_id_columns)} passam a "
            f"apontar para o ID mantido de cada grupo.",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mescla personas redundantes nos datasets de mapping, nos datasets "
            "derivados com imagens e nos datasets de diálogos, publicando o "
            "resultado no Hugging Face Hub."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subconjunto das chaves de datasets do config (padrão: todas).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subconjunto das chaves de modelos do config (padrão: todos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Apenas reporta as estatísticas de mesclagem. Não publica nada e não "
            "baixa os datasets derivados com imagens."
        ),
    )
    parser.add_argument("--token", default=None)
    return parser.parse_args(argv)


def select_keys(
    available: Iterable[str],
    requested: Sequence[str] | None,
    label: str,
) -> list[str]:
    available = list(available)
    if requested is None:
        return available
    unknown = [key for key in requested if key not in available]
    if unknown:
        raise ValueError(f"{label} desconhecido(s): {unknown}. Disponíveis: {available}")
    return [key for key in available if key in requested]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    config = load_config(args.config.resolve())
    dataset_keys = select_keys(config["datasets"], args.datasets, "Dataset(s)")
    model_keys = select_keys(config["models"], args.models, "Modelo(s)")

    for dataset_key in dataset_keys:
        process_dataset(config, dataset_key, model_keys, args)


if __name__ == "__main__":
    main()
