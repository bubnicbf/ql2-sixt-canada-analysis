"""Reusable analysis and data-quality code for the QL2 Sixt Canada rate feed."""

from ql2_sixt_canada_analysis.identifiers import (
    IdentifierDtypeError,
    IdentifierTypeConflictError,
    MissingIdentifierColumnError,
    cast_identifier_fields,
    cast_identifiers_for_raw_datasets,
    validate_identifier_dtypes,
    validate_raw_dataset_identifier_dtypes,
)
from ql2_sixt_canada_analysis.ingestion import (
    RawDatasetPaths,
    RawDatasets,
    discover_raw_csvs,
    load_raw_datasets,
)
from ql2_sixt_canada_analysis.quality import (
    BlankRowResult,
    RawDatasetBlankRowResults,
    remove_blank_rows_from_raw_datasets,
    remove_completely_blank_rows,
)
from ql2_sixt_canada_analysis.schemas import (
    DATASET_DEFINITIONS,
    IDENTIFIER_DTYPE,
    SHARED_IDENTIFIER_COLUMNS,
    DatasetDefinition,
    DatasetKey,
    get_dataset_definition,
)

__all__ = [
    "DATASET_DEFINITIONS",
    "BlankRowResult",
    "DatasetDefinition",
    "DatasetKey",
    "IDENTIFIER_DTYPE",
    "IdentifierDtypeError",
    "IdentifierTypeConflictError",
    "MissingIdentifierColumnError",
    "SHARED_IDENTIFIER_COLUMNS",
    "RawDatasetBlankRowResults",
    "RawDatasetPaths",
    "RawDatasets",
    "cast_identifier_fields",
    "cast_identifiers_for_raw_datasets",
    "discover_raw_csvs",
    "get_dataset_definition",
    "load_raw_datasets",
    "remove_blank_rows_from_raw_datasets",
    "remove_completely_blank_rows",
    "validate_identifier_dtypes",
    "validate_raw_dataset_identifier_dtypes",
]
