"""Shared mirrored-test inference-mode vocabulary."""

# Kept in sync with statistical_comparison_core.MirroredTestMixin.
# Drift-guard tests assert equality without importing SCC private attributes in
# production code.
_ALLOWED_INFERENCE_MODES = frozenset(
    {"comparison", "ranking", "ranking_no_ties"}
)
