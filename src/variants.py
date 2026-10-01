"""Variant calling from mapped reads, VCF output, and coding-effect annotation.

Calling uses a Bayesian genotype model rather than a fixed allele-fraction
cut-off.  For each site the likelihood of the observed base calls is computed
under three genotypes - homozygous reference, heterozygous, homozygous
alternate - weighting every read by its own error probability derived from the
base quality.  QUAL is then the Phred-scaled posterior probability that the
site is *not* homozygous reference.

Why this beats "call it if 20% of reads carry the alt": a 20% threshold treats
a site with 5x coverage and one alt read the same as a site with 100x coverage
and 20 alt reads, and it ignores whether those alt bases were high or low
quality.  The likelihood model handles both automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .sequence import CODON_TABLE, phred_to_error

BASES = "ACGT"
AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
       "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
       "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
       "Y": "Tyr", "V": "Val", "*": "Ter", "X": "Xaa"}


@dataclass
class PileupColumn:
    pos: int
    ref_base: str
    bases: list[str] = field(default_factory=list)
    quals: list[int] = field(default_factory=list)
    strands: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.bases)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for b in self.bases:
            c[b] += 1
        return dict(c)


@dataclass
class Call:
    pos: int
    ref: str
    alt: str
    qual: float
    depth: int
    alt_depth: int
    genotype: str
    allele_fraction: float
    strand_bias: float
    filters: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.filters


def build_pileup(reference: str, mappings, min_mapq: int = 20,
                 min_baseq: int = 13) -> dict[int, PileupColumn]:
    """Stack mapped reads into per-position base observations.

    Reads below `min_mapq` and bases below `min_baseq` are discarded before
    they can influence a call - the standard first line of defence, since a
    misplaced read contributes systematically wrong bases at every position it
    covers, not just random noise.
    """
    columns: dict[int, PileupColumn] = {}
    for m in mappings:
        if m.mapq < min_mapq:
            continue
        for offset, base in enumerate(m.read_seq):
            pos = m.ref_pos + offset
            if pos < 0 or pos >= len(reference) or base not in BASES:
                continue
            q = int(m.qual[offset]) if m.qual is not None else 40
            if q < min_baseq:
                continue
            col = columns.get(pos)
            if col is None:
                col = columns[pos] = PileupColumn(pos, reference[pos])
            col.bases.append(base)
            col.quals.append(q)
            col.strands.append(m.strand)
    return columns


def _genotype_likelihoods(bases: list[str], quals: list[int], ref: str,
                          alt: str) -> tuple[float, float, float]:
    """log10 likelihoods of the observations under 0/0, 0/1 and 1/1.

    For a read with error probability e:
        P(base | homozygous for that allele)     = 1 - e
        P(base | homozygous for the other)       = e / 3
        P(base | heterozygous)                   = average of the two
    """
    errs = phred_to_error(np.array(quals))
    ll = np.zeros(3)
    for base, e in zip(bases, errs):
        if base == ref:
            p = np.array([1 - e, 0.5 * (1 - e) + 0.5 * (e / 3), e / 3])
        elif base == alt:
            p = np.array([e / 3, 0.5 * (e / 3) + 0.5 * (1 - e), 1 - e])
        else:
            p = np.array([e / 3, e / 3, e / 3])
        ll += np.log10(np.clip(p, 1e-12, None))
    return float(ll[0]), float(ll[1]), float(ll[2])


def _strand_bias(col: PileupColumn, alt: str) -> float:
    """Fraction of alt reads on the more common strand (0.5 = unbiased).

    A real heterozygous site is sampled evenly from both strands; an artefact
    from one strand's chemistry is not.
    """
    fwd = sum(1 for b, s in zip(col.bases, col.strands) if b == alt and s == "+")
    rev = sum(1 for b, s in zip(col.bases, col.strands) if b == alt and s == "-")
    tot = fwd + rev
    return max(fwd, rev) / tot if tot else 0.5


def call_variants(reference: str, columns: dict[int, PileupColumn],
                  min_qual: float = 30.0, min_depth: int = 8,
                  min_alt_depth: int = 3, max_strand_bias: float = 0.95,
                  prior_het: float = 1e-3) -> list[Call]:
    """Call SNVs at every covered position."""
    calls: list[Call] = []
    log_priors = np.log10([1 - prior_het - prior_het / 2, prior_het,
                           prior_het / 2])

    for pos in sorted(columns):
        col = columns[pos]
        if col.depth < min_depth or col.ref_base not in BASES:
            continue
        counts = col.counts()
        alts = [(b, c) for b, c in counts.items()
                if b != col.ref_base and b in BASES]
        if not alts:
            continue
        alt, alt_count = max(alts, key=lambda x: x[1])

        ll = np.array(_genotype_likelihoods(col.bases, col.quals,
                                            col.ref_base, alt))
        post = ll + log_priors
        post -= post.max()
        probs = 10 ** post
        probs /= probs.sum()

        p_ref = float(probs[0])
        qual = -10.0 * np.log10(max(p_ref, 1e-30))
        gt_idx = int(np.argmax(probs))
        genotype = ["0/0", "0/1", "1/1"][gt_idx]
        if genotype == "0/0":
            continue

        af = alt_count / col.depth
        sb = _strand_bias(col, alt)

        filters = []
        if qual < min_qual:
            filters.append("LowQual")
        if col.depth < min_depth:
            filters.append("LowDepth")
        if alt_count < min_alt_depth:
            filters.append("LowAltDepth")
        if sb > max_strand_bias:
            filters.append("StrandBias")

        calls.append(Call(pos, col.ref_base, alt, round(qual, 2), col.depth,
                          alt_count, genotype, round(af, 4), round(sb, 3),
                          filters))
    return calls


# ---------------------------------------------------------------------------
# Alignment-based variant discovery (handles indels)
# ---------------------------------------------------------------------------
def variants_from_alignment(aligned_ref: str, aligned_alt: str,
                            ref_offset: int = 0) -> list[dict]:
    """Read variants straight out of a global alignment.

    The seed-and-extend mapper is ungapped, so it cannot discover insertions
    or deletions from reads.  Comparing two assembled sequences with
    Needleman-Wunsch does find them, and left-aligns them the way VCF requires.
    """
    out: list[dict] = []
    ref_pos = ref_offset
    i = 0
    while i < len(aligned_ref):
        r, a = aligned_ref[i], aligned_alt[i]
        if r == a:
            ref_pos += 1
            i += 1
        elif r != "-" and a != "-":
            out.append({"pos": ref_pos, "ref": r, "alt": a, "kind": "snv"})
            ref_pos += 1
            i += 1
        elif r == "-":                              # insertion in the donor
            j = i
            ins = []
            while j < len(aligned_ref) and aligned_ref[j] == "-":
                ins.append(aligned_alt[j])
                j += 1
            anchor = ref_pos - 1
            out.append({"pos": anchor, "ref": aligned_ref[:i].replace("-", "")[-1:],
                        "alt": aligned_ref[:i].replace("-", "")[-1:] + "".join(ins),
                        "kind": "insertion"})
            i = j
        else:                                       # deletion in the donor
            j = i
            dele = []
            while j < len(aligned_alt) and aligned_alt[j] == "-":
                dele.append(aligned_ref[j])
                j += 1
            prev = aligned_ref[:i].replace("-", "")[-1:]
            out.append({"pos": ref_pos - 1, "ref": prev + "".join(dele),
                        "alt": prev, "kind": "deletion"})
            ref_pos += len(dele)
            i = j
    return out


# ---------------------------------------------------------------------------
# Coding-effect annotation
# ---------------------------------------------------------------------------
def annotate_coding_effect(reference: str, pos: int, ref: str, alt: str,
                           cds_start: int, cds_end: int) -> dict:
    """Classify a variant's consequence on the protein.

    Returns HGVS-style c. and p. notation alongside the consequence term.
    """
    out = {"consequence": "intergenic", "hgvs_c": "", "hgvs_p": "",
           "codon_ref": "", "codon_alt": "", "aa_ref": "", "aa_alt": "",
           "codon_number": None}
    if not (cds_start <= pos < cds_end):
        region = "upstream" if pos < cds_start else "downstream"
        out["consequence"] = f"{region}_variant"
        return out

    cds_pos = pos - cds_start
    out["hgvs_c"] = f"c.{cds_pos + 1}{ref}>{alt}"

    if len(ref) != len(alt):
        shift = abs(len(alt) - len(ref))
        out["consequence"] = ("frameshift_variant" if shift % 3
                              else "inframe_indel")
        out["hgvs_c"] = (f"c.{cds_pos + 1}_"
                         f"{cds_pos + max(len(ref), len(alt))}"
                         f"{'ins' if len(alt) > len(ref) else 'del'}")
        return out

    codon_index = cds_pos // 3
    offset = cds_pos % 3
    codon_start = cds_start + codon_index * 3
    codon_ref = reference[codon_start:codon_start + 3]
    if len(codon_ref) < 3:
        return out
    codon_alt = codon_ref[:offset] + alt + codon_ref[offset + 1:]

    aa_ref = CODON_TABLE.get(codon_ref, "X")
    aa_alt = CODON_TABLE.get(codon_alt, "X")

    if aa_ref == aa_alt:
        consequence = "synonymous_variant"
    elif aa_alt == "*":
        consequence = "stop_gained"
    elif aa_ref == "*":
        consequence = "stop_lost"
    elif codon_index == 0:
        consequence = "start_lost"
    else:
        consequence = "missense_variant"

    out.update({
        "consequence": consequence,
        "codon_ref": codon_ref, "codon_alt": codon_alt,
        "aa_ref": aa_ref, "aa_alt": aa_alt,
        "codon_number": codon_index + 1,
        "hgvs_p": (f"p.{AA3.get(aa_ref, 'Xaa')}{codon_index + 1}"
                   f"{AA3.get(aa_alt, 'Xaa')}"),
    })
    return out


# ---------------------------------------------------------------------------
# VCF output
# ---------------------------------------------------------------------------
VCF_HEADER = """##fileformat=VCFv4.2
##source=genomics-variant-toolkit
##reference={reference_name}
##contig=<ID={contig},length={length}>
##FILTER=<ID=LowQual,Description="QUAL below threshold">
##FILTER=<ID=LowDepth,Description="Read depth below threshold">
##FILTER=<ID=LowAltDepth,Description="Too few alternate-supporting reads">
##FILTER=<ID=StrandBias,Description="Alternate allele seen on one strand only">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth">
##INFO=<ID=AF,Number=1,Type=Float,Description="Alternate allele fraction">
##INFO=<ID=SB,Number=1,Type=Float,Description="Strand bias of alternate reads">
##INFO=<ID=CSQ,Number=1,Type=String,Description="Consequence|HGVSc|HGVSp">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}
"""


def write_vcf(path: str | Path, calls: list[Call], contig: str = "chr1",
              length: int = 0, sample: str = "SAMPLE",
              annotations: dict[int, dict] | None = None,
              reference_name: str = "simulated_reference") -> None:
    """Write a spec-compliant VCF 4.2 file (POS is 1-based)."""
    annotations = annotations or {}
    with open(path, "w") as fh:
        fh.write(VCF_HEADER.format(contig=contig, length=length, sample=sample,
                                   reference_name=reference_name))
        for c in sorted(calls, key=lambda x: x.pos):
            ann = annotations.get(c.pos, {})
            csq = "|".join([ann.get("consequence", "."),
                            ann.get("hgvs_c", "."), ann.get("hgvs_p", ".")])
            info = (f"DP={c.depth};AF={c.allele_fraction};"
                    f"SB={c.strand_bias};CSQ={csq}")
            flt = ";".join(c.filters) if c.filters else "PASS"
            ad = f"{c.depth - c.alt_depth},{c.alt_depth}"
            fh.write(f"{contig}\t{c.pos + 1}\t.\t{c.ref}\t{c.alt}\t"
                     f"{c.qual}\t{flt}\t{info}\tGT:DP:AD\t"
                     f"{c.genotype}:{c.depth}:{ad}\n")


def evaluate_calls(truth: list, calls: list[Call], pass_only: bool = True
                   ) -> dict:
    """Precision / recall / F1 against the known truth set, plus genotype accuracy."""
    truth_snv = {v.pos: v for v in truth if v.kind == "snv"}
    called = [c for c in calls if c.passed or not pass_only]
    called_map = {c.pos: c for c in called}

    tp = sorted(set(truth_snv) & set(called_map))
    fp = sorted(set(called_map) - set(truth_snv))
    fn = sorted(set(truth_snv) - set(called_map))

    correct_allele = sum(1 for p in tp if called_map[p].alt == truth_snv[p].alt)
    correct_gt = sum(1 for p in tp
                     if called_map[p].genotype == truth_snv[p].genotype)

    # Precision is undefined when nothing was called - reporting 0.0 would
    # wrongly suggest the caller made mistakes rather than staying silent.
    precision = len(tp) / len(called) if called else float("nan")
    recall = len(tp) / len(truth_snv) if truth_snv else float("nan")
    denom = (precision if called else 0.0) + (recall if truth_snv else 0.0)
    f1 = (2 * precision * recall / denom) if (called and denom > 0) else 0.0
    return {
        "n_truth_snv": len(truth_snv), "n_called": len(called),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "allele_accuracy": correct_allele / len(tp) if tp else float("nan"),
        "genotype_accuracy": correct_gt / len(tp) if tp else float("nan"),
        "fp_positions": fp[:20], "fn_positions": fn[:20],
    }
