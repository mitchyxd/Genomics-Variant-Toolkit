"""Molecular phylogenetics: sequence evolution, distance estimation, tree building.

The pipeline is deliberately end-to-end and self-validating: sequences are
evolved along a *known* tree, the tree is then reconstructed from the
sequences alone, and the reconstruction is scored against the truth with the
Robinson-Foulds metric.  That closes the loop - most tutorials build a tree
and simply assert it looks reasonable.

Distances come from two routes:

* **Alignment-free k-mer distance.**  Compares k-mer composition profiles.
  Fast (no alignment at all), which is why it underpins tools like Mash and
  sourmash for large-scale genome comparison.
* **Jukes-Cantor corrected p-distance.**  The observed fraction of differing
  sites saturates as sequences diverge, because a site can mutate twice and
  land back where it started.  The JC69 correction,
  d = -3/4 ln(1 - 4p/3), converts observed difference into estimated
  substitutions per site and is what makes distant relationships come out at
  the right depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASES = "ACGT"


# ---------------------------------------------------------------------------
# Tree representation
# ---------------------------------------------------------------------------
@dataclass
class Node:
    name: str = ""
    children: list = field(default_factory=list)
    branch_length: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> list[str]:
        if self.is_leaf:
            return [self.name]
        out = []
        for c in self.children:
            out.extend(c.leaves())
        return sorted(out)

    def newick(self) -> str:
        if self.is_leaf:
            return f"{self.name}:{self.branch_length:.4f}"
        inner = ",".join(c.newick() for c in self.children)
        return f"({inner}):{self.branch_length:.4f}"

    def splits(self, all_leaves: set[str] | None = None) -> set[frozenset]:
        """Non-trivial bipartitions, the basis of the Robinson-Foulds metric.

        Splits are *canonicalised for unrooted comparison*: a bipartition
        {A,B | C,D} is the same biological statement however the tree happens
        to be rooted, so each clade is represented by whichever side excludes
        a fixed anchor taxon.  Skipping this step makes any two trees with
        different root placements look different even when their topologies
        are identical - which silently inflates RF distance.
        """
        all_leaves = all_leaves or set(self.leaves())
        anchor = min(all_leaves)
        out: set[frozenset] = set()

        def walk(node: Node) -> set[str]:
            if node.is_leaf:
                return {node.name}
            clade: set[str] = set()
            for c in node.children:
                clade |= walk(c)
            if 1 < len(clade) < len(all_leaves):
                side = clade if anchor not in clade else (all_leaves - clade)
                if 1 < len(side) < len(all_leaves):
                    out.add(frozenset(side))
            return clade

        walk(self)
        return out


def robinson_foulds(t1: Node, t2: Node) -> dict:
    """Symmetric difference of bipartitions between two unrooted topologies."""
    leaves = set(t1.leaves())
    s1, s2 = t1.splits(leaves), t2.splits(leaves)
    shared = s1 & s2
    total = len(s1) + len(s2)
    return {
        "rf_distance": len(s1 ^ s2),
        "max_rf": total,
        "normalised_rf": (len(s1 ^ s2) / total) if total else 0.0,
        "shared_splits": len(shared),
        "n_splits_true": len(s1),
        "n_splits_inferred": len(s2),
    }


# ---------------------------------------------------------------------------
# Sequence evolution
# ---------------------------------------------------------------------------
def mutate(seq: str, distance: float, rng: np.random.Generator,
           transition_bias: float = 2.0) -> str:
    """Evolve a sequence for a given branch length under a K80-like model.

    Transitions (A<->G, C<->T) occur more often than transversions - roughly
    2:1 in real genomes because purine-purine and pyrimidine-pyrimidine
    swaps are chemically easier.  Ignoring this biases distance estimates.
    """
    arr = np.array(list(seq.upper()))
    n_sites = len(arr)
    n_mut = rng.poisson(distance * n_sites)
    if n_mut == 0:
        return "".join(arr)
    sites = rng.integers(0, n_sites, size=n_mut)
    transition = {"A": "G", "G": "A", "C": "T", "T": "C"}
    for s in sites:
        base = arr[s]
        if base not in BASES:
            continue
        if rng.random() < transition_bias / (transition_bias + 2):
            arr[s] = transition[base]
        else:
            arr[s] = rng.choice([b for b in BASES
                                 if b != base and b != transition[base]])
    return "".join(arr)


def evolve_along_tree(root_seq: str, topology: dict, rng: np.random.Generator
                      ) -> tuple[dict[str, str], Node]:
    """Evolve sequences down a nested topology specification.

    `topology` is a nested dict:  {"name": (branch_length, [children...])}
    with leaves given as {"name": (branch_length, [])}.
    """
    sequences: dict[str, str] = {}

    def walk(spec: dict, parent_seq: str) -> Node:
        (name, (blen, children)), = spec.items()
        seq = mutate(parent_seq, blen, rng)
        node = Node(name=name, branch_length=blen)
        if not children:
            sequences[name] = seq
        for child in children:
            node.children.append(walk(child, seq))
        return node

    tree = walk(topology, root_seq)
    return sequences, tree


def rate_heterogeneous_topology() -> dict:
    """The same 8-taxon shape, but with one clade evolving ~5x faster.

    This is the scenario that breaks UPGMA.  UPGMA assumes a molecular clock,
    so a fast-evolving lineage looks distant from everything and gets pulled
    to the base of the tree ("long-branch attraction").  Neighbour joining
    subtracts each taxon's average divergence before pairing, so it recovers
    the correct topology anyway.
    """
    def leaf(name, b):
        return {name: (b, [])}
    return {"root": (0.0, [
        {"cladeA": (0.05, [
            {"A1_2": (0.03, [leaf("A1", 0.02), leaf("A2", 0.02)])},
            {"A3_4": (0.03, [leaf("A3", 0.025), leaf("A4", 0.03)])},
        ])},
        {"cladeB": (0.06, [
            {"B1_2": (0.04, [leaf("B1", 0.02), leaf("B2", 0.25)])},
            {"B3_4": (0.035, [leaf("B3", 0.30), leaf("B4", 0.02)])},
        ])},
    ])}


def default_topology() -> dict:
    """A fixed 8-taxon tree with two well-separated clades."""
    def leaf(name, b):
        return {name: (b, [])}
    return {"root": (0.0, [
        {"cladeA": (0.05, [
            {"A1_2": (0.03, [leaf("A1", 0.02), leaf("A2", 0.02)])},
            {"A3_4": (0.03, [leaf("A3", 0.025), leaf("A4", 0.03)])},
        ])},
        {"cladeB": (0.06, [
            {"B1_2": (0.04, [leaf("B1", 0.02), leaf("B2", 0.03)])},
            {"B3_4": (0.035, [leaf("B3", 0.03), leaf("B4", 0.02)])},
        ])},
    ])}


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def kmer_profile(seq: str, k: int = 5) -> dict[str, int]:
    prof: dict[str, int] = {}
    s = seq.upper()
    for i in range(len(s) - k + 1):
        kmer = s[i:i + k]
        if all(c in BASES for c in kmer):
            prof[kmer] = prof.get(kmer, 0) + 1
    return prof


def kmer_distance(a: str, b: str, k: int = 5) -> float:
    """Cosine distance between k-mer frequency profiles (alignment-free)."""
    pa, pb = kmer_profile(a, k), kmer_profile(b, k)
    keys = set(pa) | set(pb)
    va = np.array([pa.get(x, 0) for x in keys], dtype=float)
    vb = np.array([pb.get(x, 0) for x in keys], dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(1.0 - np.dot(va, vb) / denom) if denom else 1.0


def p_distance(a: str, b: str) -> float:
    """Fraction of differing sites over the compared length."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    diff = sum(1 for i in range(n) if a[i] != b[i])
    return diff / n


def jukes_cantor(p: float) -> float:
    """JC69 correction: observed difference -> substitutions per site."""
    if p >= 0.75:
        return float("inf")
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)


def distance_matrix(sequences: dict[str, str], method: str = "jc69",
                    k: int = 5) -> tuple[np.ndarray, list[str]]:
    names = sorted(sequences)
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sequences[names[i]], sequences[names[j]]
            if method == "kmer":
                d = kmer_distance(a, b, k)
            elif method == "p":
                d = p_distance(a, b)
            elif method == "jc69":
                d = jukes_cantor(p_distance(a, b))
            else:
                raise ValueError(f"unknown method: {method}")
            D[i, j] = D[j, i] = d
    return D, names


# ---------------------------------------------------------------------------
# Tree building
# ---------------------------------------------------------------------------
def upgma(D: np.ndarray, names: list[str]) -> Node:
    """UPGMA clustering.  Assumes a molecular clock (equal rates on all lineages)."""
    D = D.astype(float).copy()
    clusters = {i: Node(name=names[i]) for i in range(len(names))}
    sizes = {i: 1 for i in clusters}
    heights = {i: 0.0 for i in clusters}
    active = list(clusters)
    next_id = len(names)

    while len(active) > 1:
        best, pair = np.inf, None
        for x in range(len(active)):
            for y in range(x + 1, len(active)):
                i, j = active[x], active[y]
                if D[i, j] < best:
                    best, pair = D[i, j], (i, j)
        i, j = pair
        height = best / 2.0
        node = Node(name=f"node{next_id}")
        for child, cid in ((clusters[i], i), (clusters[j], j)):
            child.branch_length = max(height - heights[cid], 0.0)
            node.children.append(child)

        # Extend the matrix with the merged cluster
        new = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
        new[:D.shape[0], :D.shape[1]] = D
        for m in active:
            if m in (i, j):
                continue
            d = (D[i, m] * sizes[i] + D[j, m] * sizes[j]) / (sizes[i] + sizes[j])
            new[next_id, m] = new[m, next_id] = d
        D = new

        clusters[next_id] = node
        sizes[next_id] = sizes[i] + sizes[j]
        heights[next_id] = height
        active = [a for a in active if a not in (i, j)] + [next_id]
        next_id += 1

    return clusters[active[0]]


def neighbour_joining(D: np.ndarray, names: list[str]) -> Node:
    """Saitou-Nei neighbour joining, which does *not* assume a molecular clock.

    UPGMA misplaces taxa whenever lineages evolve at different rates (long-
    branch attraction); NJ corrects for that by subtracting each taxon's
    average divergence from everything else before choosing the closest pair.
    """
    D = D.astype(float).copy()
    nodes = {i: Node(name=names[i]) for i in range(len(names))}
    active = list(nodes)
    next_id = len(names)

    while len(active) > 2:
        n = len(active)
        r = {i: sum(D[i, j] for j in active if j != i) / (n - 2) for i in active}
        best, pair = np.inf, None
        for x in range(n):
            for y in range(x + 1, n):
                i, j = active[x], active[y]
                q = D[i, j] - r[i] - r[j]
                if q < best:
                    best, pair = q, (i, j)
        i, j = pair
        li = 0.5 * D[i, j] + 0.5 * (r[i] - r[j])
        lj = D[i, j] - li
        node = Node(name=f"node{next_id}")
        nodes[i].branch_length = max(li, 0.0)
        nodes[j].branch_length = max(lj, 0.0)
        node.children = [nodes[i], nodes[j]]

        new = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
        new[:D.shape[0], :D.shape[1]] = D
        for m in active:
            if m in (i, j):
                continue
            new[next_id, m] = new[m, next_id] = 0.5 * (D[i, m] + D[j, m] - D[i, j])
        D = new
        nodes[next_id] = node
        active = [a for a in active if a not in (i, j)] + [next_id]
        next_id += 1

    i, j = active
    nodes[i].branch_length = max(D[i, j], 0.0)
    root = Node(name="root", children=[nodes[i], nodes[j]])
    return root
