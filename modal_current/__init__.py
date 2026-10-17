"""Modal Current: POD and DMD of a documented synthetic spatiotemporal field.

The package implements:

* a generator for a SYNTHETIC three-mode spatiotemporal field with documented
  amplitudes, growth rates, frequencies, phases and spatial shapes
  (:mod:`modal_current.synthetic`);
* Proper Orthogonal Decomposition via the SVD of the centered snapshot
  matrix (:mod:`modal_current.pod`);
* exact Dynamic Mode Decomposition (Tu et al., "Dynamic mode decomposition:
  Theory and applications") with optional time-delay (Hankel) embedding
  (:mod:`modal_current.dmd`).

Every dataset produced by this package is synthetic.  It contains no
physical measurement, and results obtained from it must be labelled as
synthetic demonstration data wherever they appear.
"""

from .dmd import DMDResult, compute_dmd, hankel_embed, match_eigenvalues
from .pod import PODResult, compute_pod, numerical_rank
from .synthetic import (
    DEFAULT_MODE_PARAMETERS,
    ModeSpec,
    SyntheticField,
    default_grid,
    default_times,
    generate_snapshots,
    make_default_field,
    sine_shape,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_MODE_PARAMETERS",
    "DMDResult",
    "ModeSpec",
    "PODResult",
    "SyntheticField",
    "__version__",
    "compute_dmd",
    "compute_pod",
    "default_grid",
    "default_times",
    "generate_snapshots",
    "hankel_embed",
    "make_default_field",
    "match_eigenvalues",
    "numerical_rank",
    "sine_shape",
]
