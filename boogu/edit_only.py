"""Lightweight validation for Boogu's image-editing-only contract."""

import re
from typing import TypeAlias, TypeVar


EDIT_TASK_TYPE = "ti2i"

ReferenceSample: TypeAlias = list[object] | tuple[object, ...]
ReferenceBatch: TypeAlias = list[ReferenceSample] | tuple[ReferenceSample, ...]
ReferenceBatchT = TypeVar("ReferenceBatchT", bound=ReferenceBatch)
CorrespondingBatchT = TypeVar("CorrespondingBatchT", bound=ReferenceBatch)
CheckpointPathT = TypeVar("CheckpointPathT", bound=str)


class EditOnlyInputError(ValueError):
    """Raised when an input violates the image-editing-only contract."""


_REFERENCE_ERROR = (
    "This is an image-editing-only project; every sample requires at least "
    "one reference image."
)
_CHECKPOINT_ERROR = (
    "This standard image-editing runtime only accepts an Edit or Edit-fp8 "
    "checkpoint; Turbo checkpoints require the separate Turbo pipeline."
)
_EDIT_CHECKPOINT_NAME = re.compile(
    r"Boogu-Image(?:-\d+(?:\.\d+)*)?-Edit"
    r"(?:-fp8(?:-reuse)?)?",
    re.IGNORECASE,
)


def require_reference_batch(
    reference_batch: ReferenceBatchT | None,
    expected_batch_size: int | None = None,
) -> ReferenceBatchT:
    """Return a reference batch after requiring images for every sample."""
    if not isinstance(reference_batch, (list, tuple)) or not reference_batch:
        raise EditOnlyInputError(_REFERENCE_ERROR)

    actual_batch_size = len(reference_batch)

    if (
        expected_batch_size is not None
        and actual_batch_size != expected_batch_size
    ):
        raise EditOnlyInputError(
            f"This image-editing-only project expected batch size "
            f"{expected_batch_size}, but received {actual_batch_size}; every "
            "sample requires at least one reference image."
    )

    for sample_index, sample_references in enumerate(reference_batch):
        if not isinstance(sample_references, (list, tuple)) or not sample_references:
            raise EditOnlyInputError(
                f"{_REFERENCE_ERROR} Sample {sample_index} has no reference image."
            )

        for reference_index, reference_image in enumerate(sample_references):
            if reference_image is None or (
                isinstance(reference_image, str) and not reference_image.strip()
            ):
                raise EditOnlyInputError(
                    f"{_REFERENCE_ERROR} Sample {sample_index}, reference "
                    f"{reference_index} is empty."
                )

    return reference_batch


def require_corresponding_reference_groups(
    reference_batch: ReferenceBatchT,
    corresponding_batch: CorrespondingBatchT,
) -> tuple[ReferenceBatchT, CorrespondingBatchT]:
    """Require one-to-one reference members within every batch sample."""
    require_reference_batch(reference_batch)
    require_reference_batch(
        corresponding_batch, expected_batch_size=len(reference_batch)
    )
    for sample_index, (references, corresponding) in enumerate(
        zip(reference_batch, corresponding_batch)
    ):
        if len(references) != len(corresponding):
            raise EditOnlyInputError(
                f"At sample {sample_index}, received {len(references)} reference "
                f"image(s) but {len(corresponding)} corresponding item(s). "
                "Reference images and paths must match one-to-one per sample."
            )
    return reference_batch, corresponding_batch


def require_edit_checkpoint(
    checkpoint_path: CheckpointPathT | None,
) -> CheckpointPathT:
    """Return a checkpoint path after requiring an Edit model marker."""
    if (
        not isinstance(checkpoint_path, str)
        or not checkpoint_path
        or checkpoint_path != checkpoint_path.strip()
    ):
        raise EditOnlyInputError(_CHECKPOINT_ERROR)

    normalized_path = checkpoint_path.rstrip("/\\")
    checkpoint_name = re.split(r"[/\\]", normalized_path)[-1]
    if not _EDIT_CHECKPOINT_NAME.fullmatch(checkpoint_name):
        raise EditOnlyInputError(_CHECKPOINT_ERROR)

    return checkpoint_path
