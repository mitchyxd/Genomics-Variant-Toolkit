"""Core sequence handling: FASTA/FASTQ parsing, translation, ORF discovery.

Deliberately dependency-free (standard library + numpy only).  Biopython does
all of this, but implementing it is the point: the genetic code, reading
frames, reverse complementation and Phred encoding are the primitives every
downstream analysis silently depends on, and getting one of them subtly wrong
is a classic source of irreproducible results.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Standard genetic code (NCBI translation table 1)
CODON_TABLE: dict[str, str] = {}
_BASES = "TCAG"
_AAS = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for _i, _b1 in enumerate(_BASES):
    for _j, _b2 in enumerate(_BASES):
        for _k, _b3 in enumerate(_BASES):
            CODON_TABLE[_b1 + _b2 + _b3] = _AAS[_i * 16 + _j * 4 + _k]

START_CODON = "ATG"
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Kyte & Doolittle hydropathy index (J. Mol. Biol. 157:105-132, 1982)
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Average isotopic residue masses, Da (monomer mass minus water)
AA_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "Q": 128.1307, "E": 129.1155, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
WATER_MASS = 18.0153


@dataclass
class SeqRecord:
    id: str
    seq: str
    description: str = ""
    qual: np.ndarray | None = None      # Phred scores, if from FASTQ

    def __len__(self) -> int:
        return len(self.seq)


# ---------------------------------------------------------------------------
# Basic sequence operations
# ---------------------------------------------------------------------------
def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def gc_content(seq: str) -> float:
    """Fraction of G+C, ignoring ambiguous bases."""
    s = seq.upper()
    valid = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / valid if valid else 0.0


def gc_skew(seq: str) -> float:
    """(G-C)/(G+C).  Changes sign at the replication origin in bacteria."""
    s = seq.upper()
    g, c = s.count("G"), s.count("C")
    return (g - c) / (g + c) if (g + c) else 0.0


def sliding_gc(seq: str, window: int = 100, step: int = 20
               ) -> tuple[np.ndarray, np.ndarray]:
    pos, vals = [], []
    for i in range(0, max(1, len(seq) - window + 1), step):
        pos.append(i + window // 2)
        vals.append(gc_content(seq[i:i + window]))
    return np.array(pos), np.array(vals)


def translate(seq: str, frame: int = 0, to_stop: bool = False) -> str:
    """Translate nucleotides to one-letter amino acids.

    Incomplete trailing codons are dropped; unknown codons (containing N)
    become 'X' rather than raising, which is what real pipelines need.
    """
    s = seq.upper()[frame:]
    out = []
    for i in range(0, len(s) - len(s) % 3, 3):
        aa = CODON_TABLE.get(s[i:i + 3], "X")
        if aa == "*" and to_stop:
            break
        out.append(aa)
    return "".join(out)


def codon_usage(seq: str, frame: int = 0) -> dict[str, int]:
    s = seq.upper()[frame:]
    counts: dict[str, int] = {}
    for i in range(0, len(s) - len(s) % 3, 3):
        codon = s[i:i + 3]
        if codon in CODON_TABLE:
            counts[codon] = counts.get(codon, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# ORF discovery
# ---------------------------------------------------------------------------
@dataclass
class ORF:
    start: int          # 0-based, on the given strand's coordinates
    end: int            # exclusive, includes the stop codon
    strand: str         # '+' or '-'
    frame: int          # 0, 1 or 2
    protein: str

    @property
    def length_nt(self) -> int:
        return self.end - self.start

    @property
    def length_aa(self) -> int:
        return len(self.protein)


def find_orfs(seq: str, min_aa: int = 50, both_strands: bool = True
              ) -> list[ORF]:
    """All open reading frames: ATG to in-frame stop, in all reading frames.

    Coordinates are always reported on the *forward* strand so that ORFs from
    both strands can be compared and plotted together - converting reverse
    strand coordinates back is an easy off-by-one to get wrong.
    """
    seq = seq.upper()
    n = len(seq)
    orfs: list[ORF] = []
    strands = [("+", seq)] + ([("-", reverse_complement(seq))] if both_strands else [])

    for strand, s in strands:
        for frame in range(3):
            i = frame
            while i < len(s) - 2:
                if s[i:i + 3] != START_CODON:
                    i += 3
                    continue
                j = i
                while j < len(s) - 2:
                    if s[j:j + 3] in STOP_CODONS:
                        break
                    j += 3
                if j >= len(s) - 2:      # ran off the end without a stop
                    i += 3
                    continue
                protein = translate(s[i:j], 0)
                if len(protein) >= min_aa:
                    if strand == "+":
                        start, end = i, j + 3
                    else:
                        start, end = n - (j + 3), n - i
                    orfs.append(ORF(start, end, strand, frame, protein))
                i = j + 3
    return sorted(orfs, key=lambda o: (o.start, o.strand))


# ---------------------------------------------------------------------------
# Protein-level properties
# ---------------------------------------------------------------------------
def molecular_weight(protein: str) -> float:
    """Average molecular weight in daltons."""
    p = protein.replace("*", "")
    if not p:
        return 0.0
    return sum(AA_MASS.get(a, 0.0) for a in p) + WATER_MASS


def hydropathy_profile(protein: str, window: int = 9) -> np.ndarray:
    """Kyte-Doolittle sliding-window hydropathy.

    Sustained values above ~1.6 over a 19-residue window are the classic
    signature of a transmembrane helix.
    """
    vals = np.array([HYDROPATHY.get(a, 0.0) for a in protein])
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid")


def isoelectric_point(protein: str) -> float:
    """pI by bisection on the net-charge curve (EMBOSS pK values)."""
    pk_pos = {"K": 10.8, "R": 12.5, "H": 6.5}
    pk_neg = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}
    n_term, c_term = 8.6, 3.6

    def charge(ph: float) -> float:
        q = 1.0 / (1.0 + 10 ** (ph - n_term))
        q -= 1.0 / (1.0 + 10 ** (c_term - ph))
        for aa, pk in pk_pos.items():
            q += protein.count(aa) / (1.0 + 10 ** (ph - pk))
        for aa, pk in pk_neg.items():
            q -= protein.count(aa) / (1.0 + 10 ** (pk - ph))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------
def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r")


def read_fasta(path: str | Path) -> list[SeqRecord]:
    records, name, desc, chunks = [], None, "", []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    records.append(SeqRecord(name, "".join(chunks), desc))
                header = line[1:].split(None, 1)
                name = header[0] if header else ""
                desc = header[1] if len(header) > 1 else ""
                chunks = []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        records.append(SeqRecord(name, "".join(chunks), desc))
    return records


def write_fasta(records: list[SeqRecord], path: str | Path, width: int = 60) -> None:
    with open(path, "w") as fh:
        for r in records:
            header = f">{r.id}" + (f" {r.description}" if r.description else "")
            fh.write(header + "\n")
            for i in range(0, len(r.seq), width):
                fh.write(r.seq[i:i + width] + "\n")


def read_fastq(path: str | Path) -> list[SeqRecord]:
    """Parse FASTQ with Phred+33 quality encoding (Sanger / Illumina 1.8+)."""
    records = []
    with _open(path) as fh:
        while True:
            header = fh.readline().rstrip("\n")
            if not header:
                break
            seq = fh.readline().rstrip("\n")
            fh.readline()                      # '+' separator
            qual = fh.readline().rstrip("\n")
            name = header[1:].split(None, 1)
            records.append(SeqRecord(
                name[0], seq, name[1] if len(name) > 1 else "",
                np.array([ord(c) - 33 for c in qual], dtype=np.int16)))
    return records


def write_fastq(records: list[SeqRecord], path: str | Path) -> None:
    with open(path, "w") as fh:
        for r in records:
            q = r.qual if r.qual is not None else np.full(len(r.seq), 40)
            fh.write(f"@{r.id}\n{r.seq}\n+\n"
                     + "".join(chr(int(v) + 33) for v in q) + "\n")


def phred_to_error(q: np.ndarray | int) -> np.ndarray | float:
    """Phred score -> probability the base call is wrong:  P = 10^(-Q/10)."""
    return 10.0 ** (-np.asarray(q, dtype=float) / 10.0)


def error_to_phred(p: float) -> float:
    return -10.0 * np.log10(max(p, 1e-12))
