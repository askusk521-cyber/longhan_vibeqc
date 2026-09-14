"""Immutable native semilocal KS model and execution controls.

The compiler's FunctionalSpec supplies composition and ingredient requirements.
Native SCF has its own audited tail/spin domain, distinct from the compiler's
interior-only reference contract. Unsupported compositions fail before prepare.
"""

from dataclasses import asdict, dataclass, field, replace

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import GridSpec, checked_int
from vibeqc_compiler.xc.spec import FunctionalSpec, functional

SCF_DOMAIN = "semilocal-scaled-v1/pbe-spin-c2-1e-18"


@dataclass(frozen=True)
class KsOptions:
    """A snapshotted LDA/PBE composition, quadrature, and bounded XC tile.

    An absent functional resolves from the method name. RKS requires an
    unpolarized FunctionalSpec; UKS requires polarized. The only supported
    compositions have unit LDA_X/LDA_C_PW or GGA_X_PBE/GGA_C_PBE coefficients,
    with no exact exchange. A different model requires a new prepared owner.
    """

    functional: FunctionalSpec | None = None
    grid: GridSpec = field(default_factory=GridSpec)
    tile_points: int = 256
    scf_domain: str = SCF_DOMAIN

    def __post_init__(self):
        if self.functional is not None and not isinstance(
            self.functional, FunctionalSpec
        ):
            raise TypeError("KS functional must be a FunctionalSpec")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("KS grid must be a GridSpec")
        checked_int(self.tile_points, "KS XC tile points")
        if self.scf_domain != SCF_DOMAIN:
            raise NotImplementedError("unsupported native KS tail/spin domain policy")

    @property
    def ao_order(self):
        """SCF needs the potential; only GGA composition needs first AO jets."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        return int("sigma" in self.functional.ingredients)

    def to_payload(self):
        """Keep composition provenance and the effective SCF domain explicit."""
        if self.functional is None:
            raise ValueError("resolve KS options against a method first")
        return {
            "functional": self.functional.to_payload(),
            "scf_domain": self.scf_domain,
            "grid": asdict(self.grid),
            "tile_points": self.tile_points,
            "required_ao_order": self.ao_order,
            "required_ingredients": self.functional.ingredients,
            "scalar_derivative_order": 1,
            "observable": "scf-energy",
        }

    @property
    def identity(self):
        return canonical_hash(self.to_payload())


def resolve_ks_options(method, options=None):
    """Validate model/spin before resource estimates or native allocation."""
    if method not in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"):
        raise ValueError("KS options require a native LDA/PBE RKS/UKS method")
    options = KsOptions() if options is None else options
    if not isinstance(options, KsOptions):
        raise TypeError("ks_options must be KsOptions")
    expected = functional(
        "PBE" if method.startswith("pbe") else "LDA_XC_PW",
        spin="polarized" if method.endswith("uks") else "unpolarized",
    )
    resolved = expected if options.functional is None else options.functional
    # Identifiers are descriptive; only audited component/parameter identity
    # determines supported mathematics. Zero or modified terms are not ignored.
    if (
        resolved.spin != expected.spin
        or resolved.version != expected.version
        or sorted(resolved.components) != sorted(expected.components)
        or resolved.exact_exchange
        or resolved.range_omega
        or resolved.long_range_exchange
    ):
        raise NotImplementedError(
            "KS FunctionalSpec does not match the method's supported composition/spin"
        )
    return replace(options, functional=resolved)


def native_ks_options(options):
    """Pack a short-lived C descriptor; ctypes retains its radius-array owner."""
    import ctypes

    from . import _native

    grid = options.grid
    radii = None
    if grid.element_radii:
        radii = (ctypes.c_double * 119)(*[1.0] * 119)
        for z, radius in grid.element_radii:
            radii[z] = radius
    return _native.KsOptionsDescriptor(
        ctypes.sizeof(_native.KsOptionsDescriptor),
        _native.ABI_VERSION,
        1,
        grid.version,
        grid.radial_points,
        grid.angular_polar,
        grid.angular_azimuth,
        grid.partition_iterations,
        grid.coincident_tolerance,
        options.tile_points,
        radii,
        119 if radii is not None else 0,
    )
