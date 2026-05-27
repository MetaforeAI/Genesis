#!/usr/bin/env python3
"""Test multi-octave encoding with dynamic clustering.

This validates the core hypothesis:
1. Character 'e' converges to same proto across documents
2. Similar characters cluster into similarity wells
3. Word-level protos also cluster appropriately
4. Resonance strength tracks pattern frequency
"""

import numpy as np
import pytest
import sys
from pathlib import Path
from collections import defaultdict

pytest.skip(
    "References OctaveUnit.unit attribute that doesn't exist by design "
    "(zero-text-storage invariant). Rewrite scheduled for Phase E.",
    allow_module_level=True,
)

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.multi_octave_encoder import MultiOctaveEncoder
from src.memory.voxel_cloud import VoxelCloud
from src.memory.voxel_cloud_clustering import (
    add_or_strengthen_proto,
    get_octave_statistics,
    compute_proto_similarity
)

# Create carrier
carrier = np.random.randn(512, 512, 4).astype(np.float32)
carrier = carrier / (np.abs(carrier).max() + 1e-8)

# Initialize
encoder = MultiOctaveEncoder(carrier=carrier, width=512, height=512)
voxel_cloud = VoxelCloud(width=512, height=512)

print("=" * 80)
print("MULTI-OCTAVE CLUSTERING TEST")
print("=" * 80)

# Test documents
documents = [
    "Hello world",
    "Hello there",
    "The way that can be spoken of is not the constant way",
    "Power comes from within, not from without",
    "Wisdom is knowing what you do not know"
]

print(f"\nEncoding {len(documents)} documents at 2 octaves (character + word)")
print("-" * 80)

# Track character proto mapping
char_proto_map = {}  # character → first proto it encountered
char_convergence = defaultdict(list)  # character → list of similarities to first proto

for doc_idx, doc in enumerate(documents, 1):
    print(f"\n[{doc_idx}/{len(documents)}] Processing: \"{doc[:50]}...\"")

    # Encode at character and word octaves
    octave_units = encoder.encode_text_hierarchical(doc, octaves=[4, 0])

    # Add to voxel cloud with clustering
    for unit in octave_units:
        entry, is_new = add_or_strengthen_proto(
            voxel_cloud,
            unit.proto_identity,
            unit.frequency,
            unit.octave,
            unit.unit,
            modality='text',
            similarity_threshold=0.90
        )

        # Track character convergence
        if unit.octave == 4:  # Character level
            char = unit.unit

            if char not in char_proto_map:
                # First time seeing this character
                char_proto_map[char] = entry.proto_identity.copy()
                char_convergence[char].append(1.0)  # Perfect similarity to self
            else:
                # Compute similarity to first proto for this character
                similarity = compute_proto_similarity(
                    unit.proto_identity,
                    char_proto_map[char]
                )
                char_convergence[char].append(similarity)

    print(f"  Total protos in memory: {len(voxel_cloud.entries)}")

# Analysis
print("\n" + "=" * 80)
print("CLUSTERING ANALYSIS")
print("=" * 80)

stats = get_octave_statistics(voxel_cloud)

print(f"\nOctave Statistics:")
print(f"  Octave +4 (characters): {stats['octave_counts'][4]} unique protos")
print(f"  Octave 0 (words): {stats['octave_counts'][0]} unique protos")
print(f"\nAverage Resonance:")
print(f"  Octave +4: {stats['octave_resonance'][4]:.2f}")
print(f"  Octave 0: {stats['octave_resonance'][0]:.2f}")

# Character convergence analysis
print("\n" + "-" * 80)
print("CHARACTER CONVERGENCE ANALYSIS")
print("-" * 80)

# Common characters to analyze
common_chars = ['e', 'o', 't', 'h', 'a', 'n', 'i', 's']
print("\nFor common characters, measuring similarity to first occurrence:\n")

convergence_passed = True

for char in common_chars:
    if char in char_convergence and len(char_convergence[char]) > 1:
        similarities = char_convergence[char][1:]  # Skip first (always 1.0)
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        count = len(char_convergence[char])

        # Find the proto for this character
        proto_resonance = 0
        for entry in voxel_cloud.entries:
            if entry.octave == 4 and entry.metadata.get('unit') == char:
                proto_resonance = entry.resonance_strength
                break

        status = "✅" if avg_sim >= 0.85 else "⚠️"
        print(f"  '{char}': count={count}, resonance={proto_resonance}, "
              f"avg_sim={avg_sim:.3f}, range=[{min_sim:.3f}, {max_sim:.3f}] {status}")

        if avg_sim < 0.85:
            convergence_passed = False

# Word-level analysis
print("\n" + "-" * 80)
print("WORD-LEVEL CLUSTERING")
print("-" * 80)

word_stats = {}
for entry in voxel_cloud.entries:
    if entry.octave == 0:  # Word level
        word = entry.metadata.get('unit', '')
        word_stats[word] = entry.resonance_strength

# Sort by resonance
sorted_words = sorted(word_stats.items(), key=lambda x: -x[1])[:20]

print("\nTop 20 words by resonance strength:")
for word, resonance in sorted_words:
    print(f"  '{word}': {resonance}")

# Success criteria
print("\n" + "=" * 80)
print("SUCCESS CRITERIA")
print("=" * 80)

print(f"\n1. Character convergence (avg_sim >= 0.85): {'✅ PASS' if convergence_passed else '❌ FAIL'}")
print(f"2. Character protos < 100: {'✅ PASS' if stats['octave_counts'][4] < 100 else '❌ FAIL'}")
print(f"3. Word protos > 0: {'✅ PASS' if stats['octave_counts'][0] > 0 else '❌ FAIL'}")
print(f"4. Common chars have high resonance: ", end='')

high_resonance_chars = sum(1 for char in common_chars
                           if any(e.octave == 4 and e.metadata.get('unit') == char and e.resonance_strength >= 3
                                 for e in voxel_cloud.entries))
print(f"{'✅ PASS' if high_resonance_chars >= 4 else '⚠️ PARTIAL'} ({high_resonance_chars}/{len(common_chars)})")

# Storage efficiency
total_chars = sum(len(doc) for doc in documents)
total_words = sum(len(doc.split()) for doc in documents)
char_compression = (1.0 - stats['octave_counts'][4] / total_chars) * 100
word_compression = (1.0 - stats['octave_counts'][0] / total_words) * 100

print(f"\n5. Storage efficiency:")
print(f"   Character level: {stats['octave_counts'][4]}/{total_chars} protos "
      f"({char_compression:.1f}% compression)")
print(f"   Word level: {stats['octave_counts'][0]}/{total_words} protos "
      f"({word_compression:.1f}% compression)")

overall_pass = (
    convergence_passed and
    stats['octave_counts'][4] < 100 and
    stats['octave_counts'][0] > 0 and
    high_resonance_chars >= 4
)

print(f"\n{'='*80}")
print(f"OVERALL: {'✅ TEST PASSED' if overall_pass else '⚠️ NEEDS TUNING'}")
print(f"{'='*80}")
