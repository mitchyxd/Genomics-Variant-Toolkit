"""Pairwise alignment from first principles, plus a seed-and-extend read mapper.

Three algorithms, each implemented rather than imported:

* **Needleman-Wunsch** (J. Mol. Biol. 48:443, 1970) - global alignment, used
  here to compare full gene sequences.
* **Smith-Waterman** (J. Mol. Biol. 147:195, 1981) - local alignment, used to
  find the best-matching sub-region, e.g. a domain inside a larger protein.
* **Seed-and-extend mapping** - the strategy behind BWA and Bowtie.  Exact
  k-mer seeds are looked up in an index of the reference to propose candidate
  positions, and only those candidates are scored.  Aligning every read
  against every reference position with full dynamic programming would be
  O(reads x genome x read_length) and is completely impractical; seeding turns
  it into a hash lookup plus a handful of local checks.

Both DP routines use affine-free (linear) gap costs for clarity, with the
traceback returning the aligned strings so results can be eyeballed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .sequence import reverse_complement

MATCH, MISMATCH, GAP = 2, -3, -5


@dataclass
class Alignment:
    score: float
    aligned_a: str
    aligned_b: str
    start_a: int = 0
    start_b: int = 0

    @property
    def identity(self) -> float:
        pairs = [(x, y) for x, y in zip(self.aligned_a, self.aligned_b)]
        if not pairs:
            return 0.0
        return sum(x == y for x, y in pairs) / len(pairs)

    def pretty(self, width: int = 60) -> str:
        mid = "".join("|" if x == y else (" " if "-" in (x, y) else ".")
                      for x, y in zip(self.aligned_a, self.aligned_b))
        blocks = []
        for i in range(0, len(self.aligned_a), width):
            blocks.append(f"{self.aligned_a[i:i+width]}\n{mid[i:i+width]}\n"
                          f"{self.aligned_b[i:i+width]}")
        return "\n\n".join(blocks)


def _score(a: str, b: str, match: int, mismatch: int) -> int:
    return match if a == b else mismatch


def needleman_wunsch(a: str, b: str, match: int = MATCH,
                     mismatch: int = MISMATCH, gap: int = GAP) -> Alignment:
    """Global alignment.  O(len(a) * len(b)) time and memory."""
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    H[:, 0] = np.arange(n + 1) * gap
    H[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            H[i, j] = max(H[i - 1, j - 1] + _score(ai, b[j - 1], match, mismatch),
                          H[i - 1, j] + gap,
                          H[i, j - 1] + gap)

    # Traceback from the bottom-right corner
    i, j, out_a, out_b = n, m, [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and H[i, j] == H[i - 1, j - 1] + _score(
                a[i - 1], b[j - 1], match, mismatch):
            out_a.append(a[i - 1]); out_b.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and H[i, j] == H[i - 1, j] + gap:
            out_a.append(a[i - 1]); out_b.append("-"); i -= 1
        else:
            out_a.append("-"); out_b.append(b[j - 1]); j -= 1
    return Alignment(float(H[n, m]), "".join(reversed(out_a)),
                     "".join(reversed(out_b)))


def smith_waterman(a: str, b: str, match: int = MATCH,
                   mismatch: int = MISMATCH, gap: int = GAP) -> Alignment:
    """Local alignment: cells are floored at zero, traceback starts at the max."""
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    best, best_pos = 0, (0, 0)

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            val = max(0,
                      H[i - 1, j - 1] + _score(ai, b[j - 1], match, mismatch),
                      H[i - 1, j] + gap,
                      H[i, j - 1] + gap)
            H[i, j] = val
            if val > best:
                best, best_pos = val, (i, j)

    i, j = best_pos
    out_a, out_b = [], []
    while i > 0 and j > 0 and H[i, j] > 0:
        if H[i, j] == H[i - 1, j - 1] + _score(a[i - 1], b[j - 1], match, mismatch):
            out_a.append(a[i - 1]); out_b.append(b[j - 1]); i -= 1; j -= 1
        elif H[i, j] == H[i - 1, j] + gap:
            out_a.append(a[i - 1]); out_b.append("-"); i -= 1
        else:
            out_a.append("-"); out_b.append(b[j - 1]); j -= 1
    return Alignment(float(best), "".join(reversed(out_a)),
                     "".join(reversed(out_b)), start_a=i, start_b=j)


# ---------------------------------------------------------------------------
# Seed-and-extend read mapping
# ---------------------------------------------------------------------------
@dataclass
class Mapping:
    read_id: str
    ref_pos: int            # 0-based leftmost reference coordinate
    strand: str
    mismatches: int
    mapq: int
    read_seq: str           # oriented to the forward reference strand
    qual: np.ndarray | None = None


def build_kmer_index(reference: str, k: int = 14) -> dict[str, list[int]]:
    """Map every k-mer of the reference to the positions where it occurs."""
    index: dict[str, list[int]] = defaultdict(list)
    ref = reference.upper()
    for i in range(len(ref) - k + 1):
        index[ref[i:i + k]].append(i)
    return index


def _count_mismatches(ref: str, read: str, pos: int, cap: int) -> int:
    """Hamming distance with early exit once the cap is exceeded."""
    if pos < 0 or pos + len(read) > len(ref):
        return cap + 1
    mm = 0
    for x, y in zip(ref[pos:pos + len(read)], read):
        if x != y:
            mm += 1
            if mm > cap:
                return mm
    return mm


def map_read(reference: str, index: dict[str, list[int]], read: str,
             k: int = 14, max_mismatch_rate: float = 0.12,
             n_seeds: int = 4) -> tuple[int, str, int] | None:
    """Map one read.  Returns (position, strand, mismatches) or None.

    Several seeds spread along the read are tried, because a single seed
    overlapping a sequencing error or a variant would fail to match anything.
    """
    cap = max(1, int(max_mismatch_rate * len(read)))
    best: tuple[int, str, int] | None = None

    for strand, s in (("+", read.upper()), ("-", reverse_complement(read.upper()))):
        if len(s) < k:
            continue
        offsets = np.linspace(0, len(s) - k, n_seeds).astype(int)
        seen: set[int] = set()
        for off in offsets:
            for hit in index.get(s[off:off + k], ()):
                pos = hit - int(off)
                if pos in seen:
                    continue
                seen.add(pos)
                mm = _count_mismatches(reference, s, pos, cap)
                if mm <= cap and (best is None or mm < best[2]):
                    best = (pos, strand, mm)
                    if mm == 0:
                        return best
    return best


def map_reads(reference: str, reads, k: int = 14,
              max_mismatch_rate: float = 0.12) -> list[Mapping]:
    """Map a collection of SeqRecord reads to the reference.

    MAPQ is a Phred-scaled confidence.  Here it is derived from how well the
    read matches: a perfect match is capped at 60 (the BWA convention) and the
    score falls as mismatches accumulate.  Real mappers additionally compare
    the best hit against the second-best, which is what actually distinguishes
    a uniquely-mapping read from a repeat.
    """
    ref = reference.upper()
    index = build_kmer_index(ref, k)
    out: list[Mapping] = []
    for r in reads:
        hit = map_read(ref, index, r.seq, k, max_mismatch_rate)
        if hit is None:
            continue
        pos, strand, mm = hit
        seq = r.seq.upper() if strand == "+" else reverse_complement(r.seq.upper())
        qual = r.qual
        if qual is not None and strand == "-":
            qual = qual[::-1]
        mapq = int(np.clip(60 - 12 * mm, 0, 60))
        out.append(Mapping(r.id, pos, strand, mm, mapq, seq, qual))
    return out
