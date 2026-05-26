"""Multi-Octave Hierarchical Proto-Identity Encoder.

Encodes inputs at multiple frequency scales (octaves) to create hierarchical
proto-identity representations with dynamic clustering.

Architecture Philosophy:
    - Each text unit (character, word, phrase) → unique proto-identity (1024×1024×4)
    - Similar units cluster via similarity wells (threshold ~0.90)
    - Hash-based frequency patterns ensure deterministic uniqueness
    - NO carrier modulation (preserves distinctiveness)
    - Storage: O(vocabulary_size) not O(corpus_size)

Octave Hierarchy:
    - Octave +4: Character level (finest granularity, ~26-100 unique protos)
    - Octave  0: Word level (~10k-50k unique protos)
    - Octave -2: Short phrase level (2-3 words)
    - Octave -4: Long phrase level (4-6 words)

Usage:
    encoder = MultiOctaveEncoder(carrier, width=512, height=512)
    units = encoder.encode_text_hierarchical(text, octaves=[4, 0])
    # Returns list of OctaveUnit objects with proto_identity and frequency
"""

import numpy as np
import hashlib
from typing import List, Tuple
from dataclasses import dataclass
from src.pipeline.fft_text_encoder import FFTTextEncoder


@dataclass
class OctaveUnit:
    """A unit at a specific octave level.

    Storage: proto_identity (H×W×L frequency image) + metadata only.
    Proto-identity contains the FFT-encoded text - no separate text storage needed!
    """
    octave: int  # Octave level (+4, 0, -2, etc.)
    proto_identity: np.ndarray  # 512×512×4 proto (contains FFT-encoded text)
    frequency: np.ndarray  # 512×512×2 [magnitude, phase]


class MultiOctaveEncoder:
    """Encodes inputs at multiple octaves with dynamic clustering.

    Note: carrier parameter kept for API compatibility but not used.
    Modulation step is skipped to preserve proto-identity uniqueness.
    """

    def __init__(self, carrier: np.ndarray, width: int = 512, height: int = 512, layers: int = 4):
        """Initialize multi-octave encoder.

        Args:
            carrier: Carrier (kept for compatibility, not used)
            width: Proto-identity width (default: 512)
            height: Proto-identity height (default: 512)
            layers: Number of layers (default: 4)
        """
        self.width = width
        self.height = height
        self.layers = layers
        # Initialize FFT encoder for text encoding
        self.fft_encoder = FFTTextEncoder(width=width, height=height)

    def encode_text_hierarchical(
        self,
        text: str,
        octaves: List[int] = [4, 0]
    ) -> List[OctaveUnit]:
        """Encode text at multiple octave levels.

        Args:
            text: Input text
            octaves: List of octave levels to encode at

        Returns:
            List of OctaveUnit objects
        """
        all_units = []

        for octave in octaves:
            # Decompose text at this octave level
            units_at_octave = self._decompose_at_octave(text, octave)

            # Encode each unit
            for unit_text in units_at_octave:
                proto, freq = self._encode_unit_to_proto(unit_text, octave)

                unit = OctaveUnit(
                    octave=octave,
                    proto_identity=proto,
                    frequency=freq
                )
                all_units.append(unit)

        return all_units

    def _decompose_at_octave(self, text: str, octave: int) -> List[str]:
        """Decompose text into units at specified octave.

        Octave hierarchy:
        - +4: Character level
        - 0: Word level
        - -2: Short phrase level (2-3 words)
        - -4: Long phrase level (4-6 words)

        Args:
            text: Input text
            octave: Octave level

        Returns:
            List of units at this octave
        """
        if octave == 4:
            # Character level - each character is a unit
            return list(text)

        elif octave == 0:
            # Word level - split on whitespace and punctuation
            import re
            # Split on whitespace and keep punctuation separate
            tokens = re.findall(r'\w+|[^\w\s]', text)
            return tokens

        elif octave == -2:
            # Short phrase level (2-3 words)
            words = text.split()
            phrases = []
            for i in range(len(words) - 1):
                phrases.append(' '.join(words[i:i+2]))
            for i in range(len(words) - 2):
                phrases.append(' '.join(words[i:i+3]))
            return phrases

        elif octave == -4:
            # Long phrase level (4-6 words)
            words = text.split()
            phrases = []
            for window_size in [4, 5, 6]:
                for i in range(len(words) - window_size + 1):
                    phrases.append(' '.join(words[i:i+window_size]))
            return phrases

        else:
            raise ValueError(f"Unsupported octave: {octave}")

    def _encode_unit_to_proto(
        self,
        unit: str,
        octave: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a single unit to proto-identity using FFT.

        Uses FFT-based encoding to ensure reversible text encoding.

        Args:
            unit: Text unit (character, word, phrase)
            octave: Octave level (not used in FFT encoding, kept for API)

        Returns:
            (proto_identity, frequency) tuple
            - proto_identity: (H, W, 4) containing FFT-encoded text
            - frequency: (H, W, 2) [magnitude, phase]
        """
        # Use FFT encoder to get proto-identity directly
        proto_identity = self.fft_encoder.encode_text(unit)

        # Extract frequency spectrum from proto for compatibility
        # Proto format: [X=mag*cos(phase), Y=mag*sin(phase), Z=magnitude, W=normalized_phase]
        magnitude = proto_identity[:, :, 2]
        phase = (proto_identity[:, :, 3] * 2 * np.pi) - np.pi  # Denormalize phase
        freq_spectrum = np.stack([magnitude, phase], axis=-1).astype(np.float32)

        return proto_identity, freq_spectrum

    # NOTE: _unit_to_frequency_pattern method deprecated - replaced by FFT encoding


def compute_proto_similarity(
    proto1: np.ndarray,
    proto2: np.ndarray
) -> float:
    """Compute similarity between two proto-identities.

    Uses cosine similarity in flattened proto space.

    Args:
        proto1: First proto (H, W, 4)
        proto2: Second proto (H, W, 4)

    Returns:
        Similarity score [0, 1]
    """
    # Flatten and normalize
    v1 = proto1.flatten()
    v2 = proto2.flatten()

    # Cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1) + 1e-8
    norm2 = np.linalg.norm(v2) + 1e-8

    similarity = dot_product / (norm1 * norm2)

    # Map to [0, 1]
    return (similarity + 1.0) / 2.0
