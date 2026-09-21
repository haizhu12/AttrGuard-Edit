"""Flow-aware regional editing primitives."""

from .config import MAX_RESPONSE_KERNEL_SIZE, FARConfig, ResponseTransportConfig
from .flow import (
    analytic_scale_weights,
    build_multiscale_response_flow,
    build_response_flow,
    compose_pri_flow,
    low_pass_flow,
    normalized_denoising_progress,
)
from .io import (
    load_causal_region_map,
    load_pri_region_map,
    load_pri_region_map_list,
    stack_pri_region_maps,
)
from .losses import (
    PRILoss,
    PRILossConfig,
    PRILossEpochAccumulator,
    PRILossReport,
    PRILossSufficientStatistics,
    aggregate_pri_loss_reports,
    compute_pri_class_frequencies,
    compute_pri_class_weights,
    normalized_inverse_sqrt_class_weights,
)
from .predictor import SemanticPRIPredictor, masked_qwen_mean
from .predictor_types import PRIRegionPrediction
from .pseudo_labels import (
    AbsoluteChangeCalibration,
    ChangeEvidence,
    PRIPseudoLabelConfig,
    PRIPseudoLabelResult,
    PseudoLabelRejectionReason,
    ResponseSupervision,
    SpatialDecomposabilityAssessment,
    build_vgg16_perceptual_extractor,
    generate_pri_pseudo_labels,
)
from .regions import CausalRegionProbabilities, PRIRegionProbabilities
from .runtime import (
    FARRuntimeContext,
    FARStepResult,
    apply_far_step,
    project_pri_probabilities,
)

__all__ = [
    "MAX_RESPONSE_KERNEL_SIZE",
    "AbsoluteChangeCalibration",
    "CausalRegionProbabilities",
    "ChangeEvidence",
    "FARConfig",
    "FARRuntimeContext",
    "FARStepResult",
    "PRILoss",
    "PRILossConfig",
    "PRILossEpochAccumulator",
    "PRILossReport",
    "PRILossSufficientStatistics",
    "PRIPseudoLabelConfig",
    "PRIPseudoLabelResult",
    "PRIRegionPrediction",
    "PRIRegionProbabilities",
    "PseudoLabelRejectionReason",
    "ResponseSupervision",
    "ResponseTransportConfig",
    "SemanticPRIPredictor",
    "SpatialDecomposabilityAssessment",
    "aggregate_pri_loss_reports",
    "analytic_scale_weights",
    "apply_far_step",
    "build_multiscale_response_flow",
    "build_response_flow",
    "build_vgg16_perceptual_extractor",
    "compose_pri_flow",
    "compute_pri_class_frequencies",
    "compute_pri_class_weights",
    "generate_pri_pseudo_labels",
    "load_causal_region_map",
    "load_pri_region_map",
    "load_pri_region_map_list",
    "low_pass_flow",
    "masked_qwen_mean",
    "normalized_denoising_progress",
    "normalized_inverse_sqrt_class_weights",
    "project_pri_probabilities",
    "stack_pri_region_maps",
]
