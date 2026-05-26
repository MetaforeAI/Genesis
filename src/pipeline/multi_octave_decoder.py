"""Multi-Octave Hierarchical Proto-Identity Decoder.

Reconstructs outputs from multiple octave levels by combining information
from character, word, and phrase level proto-identities.

Architecture:
    - Queries memory at multiple octave levels
    - Weights results by similarity × resonance strength
    - Reconstructs from finest → coarsest granularity
    - NO inverse modulation needed (proto = frequency pattern directly)

Reconstruction Strategy:
    1. Query each octave level (character, word, phrase)
    2. Weight by similarity to query × resonance_strength
    3. Extract units from metadata
    4. Assemble hierarchically (characters → words preference)

Usage:
    decoder = MultiOctaveDecoder(carrier)
    response = decoder.decode_from_memory(query_proto, voxel_cloud)
    # Returns reconstructed text from memory
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.memory.voxel_cloud import VoxelCloud, ProtoIdentityEntry
from src.memory.voxel_cloud_clustering import query_by_octave
from src.pipeline.fft_text_decoder import FFTTextDecoder


@dataclass
class OctaveDecodingConfig:
    """Configuration for multi-octave decoding."""
    similarity_threshold: float = 0.3
    synthesis_temperature: float = 1.0
    energy_threshold: float = 0.2
    modulation_depth: float = 0.5
    # Octave blending weights
    octave_weights: Dict[int, float] = None

    def __post_init__(self):
        if self.octave_weights is None:
            # Default: equal weighting for character and word octaves
            self.octave_weights = {
                4: 0.5,  # Character level
                0: 0.5   # Word level
            }


class MultiOctaveDecoder:
    """Decodes proto-identities hierarchically from multiple octaves.

    Note: carrier parameter kept for API compatibility but not used.
    No demodulation needed since protos ARE frequency patterns directly.
    """

    def __init__(
        self,
        carrier: np.ndarray,
        config: Optional[OctaveDecodingConfig] = None
    ):
        """Initialize multi-octave decoder.

        Args:
            carrier: Carrier (kept for compatibility, not used)
            config: Optional decoding configuration
        """
        self.config = config if config is not None else OctaveDecodingConfig()
        # Initialize FFT decoder for text reconstruction
        self.fft_decoder = FFTTextDecoder(width=512, height=512)

    def decode_from_memory(
        self,
        query_proto: np.ndarray,
        voxel_cloud: VoxelCloud,
        max_results_per_octave: int = 10
    ) -> str:
        """Decode text by querying memory at multiple octaves.

        Args:
            query_proto: Query proto-identity (H, W, 4)
            voxel_cloud: VoxelCloud containing stored proto-identities
            max_results_per_octave: Max results per octave level

        Returns:
            Reconstructed text
        """
        # Query each octave level
        octave_results = {}

        for octave in self.config.octave_weights.keys():
            results = query_by_octave(
                voxel_cloud,
                query_proto,
                octave,
                max_results=max_results_per_octave
            )
            octave_results[octave] = results

        # Reconstruct text hierarchically
        text = self._hierarchical_reconstruction(octave_results)

        return text

    def decode_to_summary(
        self,
        query_proto: np.ndarray,
        visible_protos: List[ProtoIdentityEntry]
    ) -> str:
        """Generate response via weighted synthesis from proto-identities.

        Similar to decode_from_memory but operates on a pre-filtered list
        of visible protos rather than querying the voxel cloud.

        Args:
            query_proto: Query proto-identity (H, W, 4)
            visible_protos: List of visible ProtoIdentityEntry objects

        Returns:
            Generated text response
        """
        if not visible_protos:
            return "[no context]"

        # Group protos by octave
        octave_groups = {}
        for entry in visible_protos:
            octave = entry.octave
            if octave not in octave_groups:
                octave_groups[octave] = []
            octave_groups[octave].append(entry)

        # For each octave, compute similarities and create results list
        octave_results = {}
        for octave, entries in octave_groups.items():
            results = []
            for entry in entries:
                # Compute similarity
                from src.memory.voxel_cloud_clustering import compute_proto_similarity
                similarity = compute_proto_similarity(query_proto, entry.proto_identity)
                results.append((entry, similarity))

            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            octave_results[octave] = results

        # Reconstruct text hierarchically
        text = self._hierarchical_reconstruction(octave_results)

        return text

    def _hierarchical_reconstruction(
        self,
        octave_results: Dict[int, List[tuple]]
    ) -> str:
        """Reconstruct text from multiple octave levels.

        Strategy:
        1. Start with highest octave (finest granularity, e.g., character level)
        2. Use lower octaves to provide context and structure
        3. Blend using octave weights

        Args:
            octave_results: Dict mapping octave → [(entry, similarity), ...]

        Returns:
            Reconstructed text
        """
        # If we have character level (octave +4), use it as primary
        if 4 in octave_results and len(octave_results[4]) > 0:
            return self._reconstruct_from_characters(octave_results)

        # If we have word level (octave 0), use it
        if 0 in octave_results and len(octave_results[0]) > 0:
            return self._reconstruct_from_words(octave_results)

        # No results at any octave
        return "[silence]"

    def _reconstruct_from_characters(
        self,
        octave_results: Dict[int, List[tuple]]
    ) -> str:
        """Reconstruct text from character-level protos using FFT decoder.

        Args:
            octave_results: Results at each octave level

        Returns:
            Reconstructed text
        """
        char_results = octave_results.get(4, [])
        word_results = octave_results.get(0, [])

        # Get characters from character-level protos using FFT decoder
        chars = []
        for entry, similarity in char_results[:100]:  # Limit to 100 chars
            # Weight by similarity * resonance strength
            weight = similarity * entry.resonance_strength
            if weight > 0.1:  # Threshold to filter weak matches
                # Decode text from proto-identity using FFT
                try:
                    decoded_text = self.fft_decoder.decode_text(entry.proto_identity)
                    chars.append(decoded_text)
                except Exception:
                    # Skip entries that can't be decoded
                    continue

        # Use word-level context to organize characters
        if len(word_results) > 0 and len(chars) > 0:
            # Get words for structure using FFT decoder
            words = []
            for entry, similarity in word_results[:20]:  # Top 20 words
                weight = similarity * entry.resonance_strength
                if weight > 0.1:
                    try:
                        decoded_word = self.fft_decoder.decode_text(entry.proto_identity)
                        words.append(decoded_word)
                    except Exception:
                        continue

            # If we have words, use them; otherwise return character sequence
            if len(words) > 0:
                return ' '.join(words)

        # Fall back to character sequence
        if len(chars) > 0:
            return ''.join(chars)

        return "[silence]"

    def _reconstruct_from_words(
        self,
        octave_results: Dict[int, List[tuple]]
    ) -> str:
        """Reconstruct text from word-level protos using FFT decoder.

        Args:
            octave_results: Results at each octave level

        Returns:
            Reconstructed text
        """
        word_results = octave_results.get(0, [])

        words = []
        for entry, similarity in word_results[:50]:  # Top 50 words
            # Weight by similarity * resonance strength
            weight = similarity * entry.resonance_strength
            if weight > 0.1:
                try:
                    decoded_word = self.fft_decoder.decode_text(entry.proto_identity)
                    words.append(decoded_word)
                except Exception:
                    # Skip entries that can't be decoded
                    continue

        if len(words) > 0:
            return ' '.join(words)

        return "[silence]"
