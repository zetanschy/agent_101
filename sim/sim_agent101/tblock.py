"""The printed T's geometry — the one definition, shared by the sim and the printout.

Free of isaaclab on purpose, so the printable-target generator can import exactly the
numbers the simulation uses. They diverged once, and the mismatch only surfaced when
the printed goal was held against the real part.

Measured off t_20_factor_0.5.stl:
    x levels  -50, -12.5, 12.5, 50    -> 100 mm bar, 25 mm stem
    y levels  -87.5, -12.5, 12.5      -> 25 mm bar depth, 75 mm stem
    z         0 .. 20                 -> 20 mm thick
    volume    87.5 cm^3
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TBlockGeometry:
    """All in metres. A 10 x 10 cm T, 20 mm thick.

    The mesh origin sits at the middle of the crossbar, NOT at the centroid — the
    stem hangs off toward -y. Anything reasoning about "where the T is" wants the
    centroid, so it is derived here once rather than re-guessed per call site.
    """

    bar_width: float = 0.100      # x extent of the crossbar
    bar_depth: float = 0.025      # y extent of the crossbar
    stem_width: float = 0.025     # x extent of the stem
    stem_length: float = 0.075    # y extent of the stem, hanging toward -y
    thickness: float = 0.020      # z extent
    volume_m3: float = 87.5e-6    # by mesh integration

    @property
    def centroid_offset(self) -> tuple[float, float, float]:
        """Centroid in mesh coordinates: the point a push should be measured against."""
        bar_a = self.bar_width * self.bar_depth
        stem_a = self.stem_width * self.stem_length
        cy = (bar_a * 0.0 + stem_a * -(self.bar_depth / 2 + self.stem_length / 2)) / (bar_a + stem_a)
        return (0.0, cy, self.thickness / 2)

    @property
    def footprint_mm(self) -> tuple[float, float]:
        return (self.bar_width * 1000, (self.bar_depth + self.stem_length) * 1000)


T_BLOCK_GEOMETRY = TBlockGeometry()
