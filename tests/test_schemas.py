"""Tests for the centralized dataset definitions."""

from __future__ import annotations

import dataclasses

import pytest

from ql2_sixt_canada_analysis import schemas
from ql2_sixt_canada_analysis.schemas import (
    CARS_DEFINITION,
    DATASET_DEFINITIONS,
    JOBS_DEFINITION,
    DatasetDefinition,
    DatasetKey,
    get_dataset_definition,
)

DEFINITIONS = list(DATASET_DEFINITIONS.values())


def test_exactly_jobs_and_cars_are_defined() -> None:
    assert set(DATASET_DEFINITIONS) == {DatasetKey.JOBS, DatasetKey.CARS}
    assert {k.value for k in DatasetKey} == {"jobs", "cars"}
    assert DATASET_DEFINITIONS[DatasetKey.JOBS] is JOBS_DEFINITION
    assert DATASET_DEFINITIONS[DatasetKey.CARS] is CARS_DEFINITION


def test_registry_keys_match_definition_keys() -> None:
    assert all(key is definition.key for key, definition in DATASET_DEFINITIONS.items())
    assert len({d.key for d in DEFINITIONS}) == len(DEFINITIONS)


def test_jobs_and_cars_definitions_are_distinct() -> None:
    assert JOBS_DEFINITION != CARS_DEFINITION
    assert JOBS_DEFINITION.columns != CARS_DEFINITION.columns
    assert not set(JOBS_DEFINITION.filename_tokens) & set(CARS_DEFINITION.filename_tokens)


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_definition_is_frozen(definition: DatasetDefinition) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.columns = ()  # type: ignore[misc]


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
@pytest.mark.parametrize("field", ["filename_tokens", "columns"])
def test_collections_are_non_empty_immutable_unique_strings(
    definition: DatasetDefinition, field: str
) -> None:
    values = getattr(definition, field)
    assert isinstance(values, tuple) and values
    assert all(isinstance(v, str) and v.strip() for v in values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: str(d.key))
def test_filename_tokens_are_lowercase_words(definition: DatasetDefinition) -> None:
    assert all(t.isalnum() and t.islower() for t in definition.filename_tokens)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        DATASET_DEFINITIONS[DatasetKey.JOBS] = CARS_DEFINITION  # type: ignore[index]
    with pytest.raises(TypeError):
        del DATASET_DEFINITIONS[DatasetKey.JOBS]  # type: ignore[attr-defined]
    assert not hasattr(DATASET_DEFINITIONS, "clear")


def test_get_dataset_definition_accepts_key_or_value() -> None:
    assert get_dataset_definition(DatasetKey.CARS) is CARS_DEFINITION
    assert get_dataset_definition("jobs") is JOBS_DEFINITION
    with pytest.raises(KeyError):
        get_dataset_definition("unknown")


@pytest.mark.parametrize(
    ("tokens", "columns"),
    [
        ((), ("a",)),
        (("x",), ()),
        (("x",), ("a", "a")),
        (("x",), ("a", "")),
        (["x"], ("a",)),
        (("X",), ("a",)),
    ],
)
def test_invalid_definitions_are_rejected(tokens, columns) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        DatasetDefinition(key=DatasetKey.JOBS, filename_tokens=tokens, columns=columns)


def test_module_exposes_no_mutable_column_or_pattern_globals() -> None:
    for name in dir(schemas):
        if name.startswith("_"):
            continue
        assert not isinstance(getattr(schemas, name), (list, dict, set)), name
