"""Simulate a reference locus, a variant donor genome, and sequencing reads.

The simulator gives ground truth for every stage, so variant-calling precision
and recall can be measured exactly rather than estimated against another
caller's output.

Read simulation models the parts of Illumina sequencing that actually matter
for variant calling:

* **Position-dependent quality decay.**  Base quality falls along the read as
  phasing errors accumulate, so errors cluster at read ends.
* **Errors drawn from the quality string.**  A base with Q20 is miscalled 1%
  of the time.  Callers that trust quality scores are only correct if the
  scores mean what they claim.
* **Diploid sampling.**  Each read is drawn from one of two haplotypes, so a
  heterozygous site shows roughly 50% alternate allele fraction with genuine
  binomial scatter - which is what makes low-coverage genotyping hard.
* **Strand and coverage variation**, so strand-bias filtering has something
  to act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sequence import (CODON_TABLE, START_CODON, STOP_CODONS, SeqRecord,
                       reverse_complement, translate)

BASES = "ACGT"


@dataclass
class Variant:
    """A ground-truth variant.  POS is 0-based internally, 1-based in VCF."""
    pos: int
    ref: str
    alt: str
    genotype: str          # '0/1' heterozygous or '1/1' homozygous alternate
    kind: str              # 'snv' | 'insertion' | 'deletion'

    def __repr__(self) -> str:
        return f"Variant({self.pos + 1}{self.ref}>{self.alt} {self.genotype})"


@dataclass
class Locus:
    reference: str
    cds_start: int
    cds_end: int
    haplotypes: tuple[str, str]
    variants: list[Variant]
    meta: dict = field(default_factory=dict)

    @property
    def cds(self) -> str:
        return self.reference[self.cds_start:self.cds_end]

    @property
    def protein(self) -> str:
        return translate(self.cds, 0, to_stop=True)


def _random_cds(n_codons: int, rng: np.random.Generator, gc: float = 0.5) -> str:
    """A coding sequence with a start codon, no internal stops, and one stop."""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])   # A C G T
    codons = [START_CODON]
    while len(codons) < n_codons - 1:
        c = "".join(rng.choice(list(BASES), size=3, p=p))
        if c in STOP_CODONS or c not in CODON_TABLE:
            continue
        codons.append(c)
    codons.append(str(rng.choice(sorted(STOP_CODONS))))
    return "".join(codons)


def make_reference(length: int = 6000, n_codons: int = 400, gc: float = 0.45,
                   seed: int = 0) -> tuple[str, int, int]:
    """Reference sequence containing a single coding region."""
    rng = np.random.default_rng(seed)
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    cds = _random_cds(n_codons, rng, gc)
    flank = (length - len(cds)) // 2
    left = "".join(rng.choice(list(BASES), size=flank, p=p))
    right = "".join(rng.choice(list(BASES), size=length - len(cds) - flank, p=p))
    return left + cds + right, flank, flank + len(cds)


def introduce_variants(reference: str, cds_start: int, cds_end: int,
                       n_snv: int = 25, n_indel: int = 4,
                       het_fraction: float = 0.6,
                       seed: int = 1) -> tuple[tuple[str, str], list[Variant]]:
    """Build a diploid donor genome carrying known variants.

    Variants are placed at least 150 bp apart so each can be assessed
    independently: two variants inside one read length would let a single
    mapping failure destroy both, confounding the evaluation.
    """
    rng = np.random.default_rng(seed)
    n = len(reference)
    chosen: list[int] = []

    def pick_position(margin: int = 250) -> int | None:
        for _ in range(500):
            pos = int(rng.integers(margin, n - margin))
            if all(abs(pos - q) > 150 for q in chosen):
                chosen.append(pos)
                return pos
        return None

    variants: list[Variant] = []

    for _ in range(n_snv):
        pos = pick_position()
        if pos is None:
            break
        ref_base = reference[pos]
        alt = str(rng.choice([b for b in BASES if b != ref_base]))
        gt = "0/1" if rng.random() < het_fraction else "1/1"
        variants.append(Variant(pos, ref_base, alt, gt, "snv"))

    for _ in range(n_indel):
        pos = pick_position()
        if pos is None:
            break
        gt = "0/1" if rng.random() < het_fraction else "1/1"
        if rng.random() < 0.5:
            ins = "".join(rng.choice(list(BASES), size=int(rng.integers(1, 4))))
            variants.append(Variant(pos, reference[pos], reference[pos] + ins,
                                    gt, "insertion"))
        else:
            ln = int(rng.integers(1, 4))
            variants.append(Variant(pos, reference[pos:pos + ln + 1],
                                    reference[pos], gt, "deletion"))

    variants.sort(key=lambda v: v.pos)

    def build(haplotype_index: int) -> str:
        out, cursor = [], 0
        for v in variants:
            carries = v.genotype == "1/1" or haplotype_index == 1
            out.append(reference[cursor:v.pos])
            if carries:
                out.append(v.alt)
            else:
                out.append(v.ref)
            cursor = v.pos + len(v.ref)
        out.append(reference[cursor:])
        return "".join(out)

    return (build(0), build(1)), variants


def simulate_reads(haplotypes: tuple[str, str], n_reads: int = 4000,
                   read_length: int = 100, mean_quality: float = 34.0,
                   quality_decay: float = 0.09, seed: int = 2
                   ) -> list[SeqRecord]:
    """Simulate single-end reads with quality-driven errors."""
    rng = np.random.default_rng(seed)
    reads: list[SeqRecord] = []

    # Quality falls along the read; add per-read scatter for run variability
    base_profile = mean_quality - quality_decay * np.arange(read_length)

    for i in range(n_reads):
        hap = haplotypes[int(rng.integers(2))]
        if len(hap) <= read_length:
            continue
        start = int(rng.integers(0, len(hap) - read_length))
        seq = list(hap[start:start + read_length])

        qual = np.clip(base_profile + rng.normal(0, 2.5), 2, 41)
        err_p = 10.0 ** (-qual / 10.0)
        hits = rng.random(read_length) < err_p
        for j in np.flatnonzero(hits):
            seq[j] = str(rng.choice([b for b in BASES if b != seq[j]]))
        seq = "".join(seq)

        strand = "+"
        if rng.random() < 0.5:
            seq = reverse_complement(seq)
            qual = qual[::-1]
            strand = "-"

        reads.append(SeqRecord(f"read{i:06d}", seq,
                               f"hap_strand={strand}",
                               qual.round().astype(np.int16)))
    return reads


def simulate_locus(length: int = 6000, n_codons: int = 400, n_snv: int = 25,
                   n_indel: int = 4, coverage: float = 60.0,
                   read_length: int = 100, seed: int = 0) -> tuple[Locus, list[SeqRecord]]:
    """Convenience wrapper: reference + donor haplotypes + reads."""
    reference, cds_start, cds_end = make_reference(length, n_codons, seed=seed)
    haplotypes, variants = introduce_variants(
        reference, cds_start, cds_end, n_snv, n_indel, seed=seed + 1)
    n_reads = int(coverage * length / read_length)
    reads = simulate_reads(haplotypes, n_reads, read_length, seed=seed + 2)
    locus = Locus(reference, cds_start, cds_end, haplotypes, variants,
                  meta={"length": length, "coverage": coverage,
                        "read_length": read_length, "n_reads": len(reads),
                        "n_snv": sum(v.kind == "snv" for v in variants),
                        "n_indel": sum(v.kind != "snv" for v in variants),
                        "seed": seed})
    return locus, reads
