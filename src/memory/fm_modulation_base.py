"""FM modulation of proto-identity carrier by input signal.

Frequency-domain FM synthesis where input signal magnitude modulates
carrier phase, producing sidebands that encode the interaction between
carrier (learned knowledge) and input (query).
"""

import numpy as np


class FMModulationBase:
    """FM modulation/demodulation for proto-identity (H, W, 4) arrays.

    Channel layout: X=mag*cos(phase), Y=mag*sin(phase), Z=mag, W=(phase+pi)/(2*pi)

    The sidebands ARE the Sophia field -- emergent frequencies from the
    interference of input with accumulated knowledge structure.
    """

    def __init__(self, modulation_depth: float = 0.5, coupling: float = 0.3):
        self.modulation_depth = modulation_depth
        self.coupling = coupling

    def _extract_mag_phase(self, signal: np.ndarray):
        """Extract magnitude and phase from proto-identity channels."""
        mag = np.maximum(signal[..., 2], 0.0)
        phase = signal[..., 3] * (2.0 * np.pi) - np.pi
        return mag, phase

    def _reconstruct(self, mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Reconstruct all 4 XYZW channels from magnitude and phase."""
        result = np.empty(mag.shape + (4,), dtype=np.float32)
        result[..., 0] = mag * np.cos(phase)
        result[..., 1] = mag * np.sin(phase)
        result[..., 2] = mag
        result[..., 3] = (phase + np.pi) / (2.0 * np.pi)
        return result

    def _normalize_magnitude(self, mag: np.ndarray) -> np.ndarray:
        """Normalize magnitude to [0, 1] for controlled modulation index."""
        max_val = mag.max()
        if max_val < 1e-10:
            return np.zeros_like(mag)
        return mag / max_val

    def modulate(
        self,
        carrier: np.ndarray,
        input_signal: np.ndarray,
        modulation_depth: float = None,
    ) -> np.ndarray:
        """FM modulate carrier with input signal.

        Args:
            carrier: (H, W, 4) proto-unity carrier.
            input_signal: (H, W, 4) input proto-identity.
            modulation_depth: Override default modulation depth.

        Returns:
            modulated: (H, W, 4) modulated proto-identity with sidebands.
        """
        depth = modulation_depth if modulation_depth is not None else self.modulation_depth

        carrier_mag, carrier_phase = self._extract_mag_phase(carrier)
        input_mag, _ = self._extract_mag_phase(input_signal)

        input_norm = self._normalize_magnitude(input_mag)

        modulated_phase = carrier_phase + depth * input_norm
        modulated_mag = carrier_mag * (1.0 + self.coupling * input_norm)

        return self._reconstruct(modulated_mag, modulated_phase)

    def demodulate(
        self,
        modulated: np.ndarray,
        carrier: np.ndarray,
        modulation_depth: float = None,
    ) -> np.ndarray:
        """Extract input signal from modulated proto-identity using carrier.

        Mathematical inverse of modulate().

        Args:
            modulated: (H, W, 4) modulated proto-identity.
            carrier: (H, W, 4) proto-unity carrier used in modulation.
            modulation_depth: Must match the depth used in modulation.

        Returns:
            signal: (H, W, 4) extracted input signal (magnitude-only recovery).
        """
        depth = modulation_depth if modulation_depth is not None else self.modulation_depth

        mod_mag, mod_phase = self._extract_mag_phase(modulated)
        carrier_mag, carrier_phase = self._extract_mag_phase(carrier)

        # Recover normalized input magnitude from amplitude channel
        safe_carrier = np.where(carrier_mag > 1e-10, carrier_mag, 1.0)
        ratio = mod_mag / safe_carrier
        input_norm = np.where(carrier_mag > 1e-10, (ratio - 1.0) / self.coupling, 0.0)
        input_norm = np.clip(input_norm, 0.0, 1.0)

        # Recover phase from phase difference (secondary cross-check)
        phase_diff = mod_phase - carrier_phase
        if abs(depth) > 1e-10:
            input_norm_phase = np.clip(phase_diff / depth, 0.0, 1.0)
            # Average both estimates for robustness
            input_norm = 0.5 * (input_norm + input_norm_phase)

        # Reconstruct as a proto-identity with recovered magnitude and zero phase
        recovered_phase = np.zeros_like(input_norm)
        return self._reconstruct(input_norm, recovered_phase)

    def compute_sideband_energy(
        self, carrier: np.ndarray, modulated: np.ndarray
    ) -> float:
        """Measure new spectral energy created by modulation.

        Returns ratio of sideband energy to total energy.
        Higher values indicate more novel composition.
        """
        carrier_spectrum = np.fft.fft2(carrier[..., 0] + 1j * carrier[..., 1])
        modulated_spectrum = np.fft.fft2(modulated[..., 0] + 1j * modulated[..., 1])

        carrier_power = np.abs(carrier_spectrum) ** 2
        modulated_power = np.abs(modulated_spectrum) ** 2

        total_energy = float(modulated_power.sum())
        if total_energy < 1e-10:
            return 0.0

        # Sideband energy = energy in modulated that exceeds carrier
        sideband_power = np.maximum(modulated_power - carrier_power, 0.0)
        sideband_energy = float(sideband_power.sum())

        return sideband_energy / total_energy
