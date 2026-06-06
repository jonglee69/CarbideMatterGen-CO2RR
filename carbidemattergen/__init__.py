"""CarbideMatterGen — physics-aware fine-tuning of MatterGen for the inverse
design of non-platinum-group HER electrocatalysts (medium/high-entropy
transition-metal carbides with Pt-like hydrogen adsorption)."""

__version__ = "0.1.0"

from . import hydrogen, extra_descriptors, features, labeling  # noqa: F401
from . import co2rr  # noqa: F401  (CO2RR-to-CO composition proxies; sibling of hydrogen)
from . import solid_solution  # noqa: F401  (Stage A': SQS solid-solution construction)
