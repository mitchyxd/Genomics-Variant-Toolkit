"""End-to-end genomics pipeline.

    simulate locus & reads -> ORF/protein analysis -> read mapping ->
    variant calling -> coding annotation -> VCF -> coverage titration ->
    indel discovery by global alignment -> phylogenetic reconstruction

Run with:  python -m src.pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import align, phylo, sequence, simulate, variants

RESULTS = Path(__file__).resolve().parent.parent / "results"
PLT_STYLE = {"figure.dpi": 125, "savefig.dpi": 125, "font.size": 9,
             "axes.grid": True, "grid.alpha": 0.25,
             "axes.spines.top": False, "axes.spines.right": False}

COVERAGE_LADDER = (3, 5, 8, 12, 16, 20, 30, 45, 60)


# ---------------------------------------------------------------------------
def analyse_reference(locus) -> dict:
    ref = locus.reference
    orfs = sequence.find_orfs(ref, min_aa=50)
    protein = locus.protein
    longest = max(orfs, key=lambda o: o.length_aa) if orfs else None
    return {
        "length_bp": len(ref),
        "gc_content": round(sequence.gc_content(ref), 4),
        "gc_skew": round(sequence.gc_skew(ref), 4),
        "n_orfs_min50aa": len(orfs),
        "true_cds_start": locus.cds_start,
        "true_cds_end": locus.cds_end,
        "true_protein_length_aa": len(protein),
        "longest_orf_aa": longest.length_aa if longest else 0,
        "longest_orf_strand": longest.strand if longest else "",
        "longest_orf_recovers_true_cds": bool(
            longest and abs(longest.start - locus.cds_start) <= 3
            and longest.strand == "+"),
        "protein_mw_da": round(sequence.molecular_weight(protein), 1),
        "protein_pi": sequence.isoelectric_point(protein),
    }, orfs


def call_at_coverage(seed: int, coverage: float, length: int, n_snv: int,
                     n_indel: int) -> dict:
    locus, reads = simulate.simulate_locus(
        length=length, n_snv=n_snv, n_indel=n_indel,
        coverage=coverage, seed=seed)
    mappings = align.map_reads(locus.reference, reads, k=14)
    columns = variants.build_pileup(locus.reference, mappings)
    calls = variants.call_variants(locus.reference, columns)
    ev = variants.evaluate_calls(locus.variants, calls)
    depths = [c.depth for c in columns.values()]
    ev.update({
        "coverage_requested": coverage,
        "mean_depth": float(np.mean(depths)) if depths else 0.0,
        "mapping_rate": len(mappings) / len(reads) if reads else 0.0,
    })
    return ev


def discover_indels_by_alignment(locus, window: int = 300) -> pd.DataFrame:
    """Recover indels with Needleman-Wunsch on windows around each truth indel.

    The seed-and-extend mapper is ungapped, so indels are invisible to it.
    Aligning assembled sequence against the reference does find them.  Full
    6 kb x 6 kb dynamic programming would be 36 million cells in pure Python,
    so each indel is examined in a local window - which is also what real
    callers do (local reassembly, as in GATK HaplotypeCaller).
    """
    rows = []
    hap = locus.haplotypes[1]          # haplotype carrying every variant
    offset = 0
    for v in locus.variants:
        if v.kind == "snv":
            offset += 0
            continue
        lo = max(0, v.pos - window)
        hi = min(len(locus.reference), v.pos + window)
        hap_lo = max(0, lo + offset)
        hap_hi = min(len(hap), hi + offset + (len(v.alt) - len(v.ref)))
        aln = align.needleman_wunsch(locus.reference[lo:hi], hap[hap_lo:hap_hi])
        found = variants.variants_from_alignment(aln.aligned_a, aln.aligned_b,
                                                 ref_offset=lo)
        indels = [f for f in found if f["kind"] != "snv"]
        best = min(indels, key=lambda f: abs(f["pos"] - v.pos)) if indels else None
        rows.append({
            "truth_pos": v.pos + 1, "truth_kind": v.kind,
            "truth_ref": v.ref, "truth_alt": v.alt,
            "detected": best is not None,
            "detected_pos": (best["pos"] + 1) if best else None,
            "detected_kind": best["kind"] if best else None,
            "position_error_bp": abs(best["pos"] - v.pos) if best else None,
            "alignment_identity": round(aln.identity, 4),
        })
        offset += len(v.alt) - len(v.ref)
    return pd.DataFrame(rows)


def run_phylogenetics(seed: int = 5) -> tuple[dict, dict]:
    """Reconstruct trees under clock-like and rate-heterogeneous evolution."""
    root_seq, _, _ = simulate.make_reference(4000, 300, seed=seed)
    scenarios = {"clock_like": phylo.default_topology(),
                 "rate_heterogeneous": phylo.rate_heterogeneous_topology()}
    results, artefacts = {}, {}
    for label, topo in scenarios.items():
        rng = np.random.default_rng(seed)
        seqs, true_tree = phylo.evolve_along_tree(root_seq, topo, rng)
        entry = {}
        for method in ("p", "jc69", "kmer"):
            D, names = phylo.distance_matrix(seqs, method=method)
            for builder, bname in ((phylo.upgma, "upgma"),
                                   (phylo.neighbour_joining, "nj")):
                tree = builder(D, names)
                rf = phylo.robinson_foulds(true_tree, tree)
                entry[f"{method}_{bname}"] = {
                    "rf_distance": rf["rf_distance"],
                    "normalised_rf": round(rf["normalised_rf"], 3),
                    "shared_splits": rf["shared_splits"],
                    "n_splits_true": rf["n_splits_true"],
                    "correct": rf["rf_distance"] == 0,
                }
                if method == "jc69":
                    artefacts[f"{label}_{bname}"] = tree
        D, names = phylo.distance_matrix(seqs, method="jc69")
        artefacts[f"{label}_matrix"] = (D, names)
        artefacts[f"{label}_true"] = true_tree
        results[label] = entry
    return results, artefacts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_reference(locus, orfs, path):
    ref = locus.reference
    pos, gc = sequence.sliding_gc(ref, window=200, step=25)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2),
                           gridspec_kw={"height_ratios": [2, 1.3]})
    ax[0].plot(pos, gc * 100, lw=1, color="#264653")
    ax[0].axhline(sequence.gc_content(ref) * 100, color="grey", ls="--", lw=0.8,
                  label=f"mean {sequence.gc_content(ref)*100:.1f}%")
    ax[0].axvspan(locus.cds_start, locus.cds_end, color="#2a9d8f", alpha=0.15,
                  label="true CDS")
    ax[0].set(ylabel="GC (%)", title="Sliding-window GC content (200 bp window)")
    ax[0].legend(fontsize=8)

    lanes = {("+", 0): 5, ("+", 1): 4, ("+", 2): 3,
             ("-", 0): 2, ("-", 1): 1, ("-", 2): 0}
    for o in orfs:
        y = lanes[(o.strand, o.frame)]
        ax[1].add_patch(plt.Rectangle((o.start, y - 0.32), o.length_nt, 0.64,
                                      color="#e76f51" if o.strand == "+" else "#457b9d",
                                      alpha=0.8))
    ax[1].axvspan(locus.cds_start, locus.cds_end, color="#2a9d8f", alpha=0.12)
    ax[1].set(xlim=(0, len(ref)), ylim=(-0.8, 5.8), xlabel="position (bp)",
              title=f"Open reading frames ≥50 aa  (n={len(orfs)}; "
                    f"orange = forward, blue = reverse)")
    ax[1].set_yticks(list(lanes.values()),
                     [f"{s}{f}" for s, f in lanes])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_reads_and_depth(reads, mappings, locus, path):
    quals = np.array([r.qual for r in reads[:2000]])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(quals.mean(axis=0), color="#264653", lw=1.6)
    ax[0].fill_between(np.arange(quals.shape[1]),
                       np.percentile(quals, 10, axis=0),
                       np.percentile(quals, 90, axis=0),
                       alpha=0.2, color="#264653")
    ax[0].set(xlabel="position in read (bp)", ylabel="Phred quality",
              title="Base quality decays along the read")

    err = sequence.phred_to_error(quals.mean(axis=0))
    ax[1].semilogy(err * 100, color="#e76f51", lw=1.6)
    ax[1].set(xlabel="position in read (bp)", ylabel="expected error (%)",
              title="Implied per-base error rate")

    depths = np.zeros(len(locus.reference))
    for m in mappings:
        depths[m.ref_pos:m.ref_pos + len(m.read_seq)] += 1
    ax[2].plot(depths, lw=0.6, color="#2a9d8f")
    ax[2].axhline(depths.mean(), color="k", ls="--", lw=0.8,
                  label=f"mean {depths.mean():.0f}x")
    for v in locus.variants:
        ax[2].axvline(v.pos, color="crimson", alpha=0.35, lw=0.7)
    ax[2].set(xlabel="reference position (bp)", ylabel="depth",
              title="Coverage depth (red = true variants)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_coverage_titration(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df.mean_depth, df.recall, "o-", label="recall (sensitivity)",
               color="#e76f51")
    ax[0].plot(df.mean_depth, df.precision, "s-", label="precision", color="#264653")
    ax[0].plot(df.mean_depth, df.f1, "^--", label="F1", color="#2a9d8f")
    ax[0].axvline(30, color="grey", ls=":", lw=1)
    ax[0].annotate("30x: germline WGS standard", (30, 0.3), rotation=90,
                   fontsize=7.5, color="grey", ha="right")
    ax[0].set(xlabel="mean depth (x)", ylabel="score", ylim=(-0.03, 1.05),
              title="SNV calling vs. sequencing depth")
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].plot(df.mean_depth, df.genotype_accuracy, "o-", color="#457b9d")
    ax[1].set(xlabel="mean depth (x)", ylabel="genotype concordance",
              ylim=(-0.03, 1.05),
              title="Correctly distinguishing het from hom-alt")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_calls(calls, truth_positions, annotations, path):
    df = pd.DataFrame([{"pos": c.pos, "qual": c.qual, "af": c.allele_fraction,
                        "depth": c.depth, "genotype": c.genotype,
                        "truth": c.pos in truth_positions} for c in calls])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for gt, colour in (("0/1", "#457b9d"), ("1/1", "#e63946")):
        sub = df[df["genotype"] == gt]
        ax[0].scatter(sub.depth, sub.af, s=26, alpha=0.75, color=colour,
                      label=f"called {gt}")
    ax[0].axhline(0.5, color="grey", ls="--", lw=0.8)
    ax[0].axhline(1.0, color="grey", ls="--", lw=0.8)
    ax[0].set(xlabel="depth", ylabel="alternate allele fraction", ylim=(0, 1.08),
              title="Allele fraction separates het from hom-alt")
    ax[0].legend(fontsize=8)

    ax[1].hist(df.qual, bins=25, color="#2a9d8f", edgecolor="white")
    ax[1].set(xlabel="QUAL (Phred posterior)", ylabel="variants",
              title="Call confidence")

    counts = pd.Series([a["consequence"] for a in annotations.values()]).value_counts()
    ax[2].barh(counts.index[::-1], counts.values[::-1], color="#e76f51")
    ax[2].set(xlabel="variants", title="Predicted coding consequence")
    ax[2].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_tree(ax, root, title):
    """Simple rectangular dendrogram layout for a Node tree."""
    leaves = root.leaves()
    ypos = {name: i for i, name in enumerate(leaves)}

    def layout(node, x0):
        x = x0 + node.branch_length
        if node.is_leaf:
            y = ypos[node.name]
            ax.text(x + 0.004, y, " " + node.name, va="center", fontsize=8)
            return y, x
        ys = []
        for c in node.children:
            yc, xc = layout(c, x)
            ax.plot([x, xc], [yc, yc], color="#264653", lw=1.2)
            ys.append(yc)
        ax.plot([x, x], [min(ys), max(ys)], color="#264653", lw=1.2)
        return float(np.mean(ys)), x

    layout(root, 0.0)
    ax.set(title=title, yticks=[], xlabel="substitutions per site")
    ax.set_ylim(-0.7, len(leaves) - 0.3)
    ax.grid(axis="x", alpha=0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def fig_phylogenetics(artefacts, results, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4))
    for row, scenario in enumerate(("clock_like", "rate_heterogeneous")):
        D, names = artefacts[f"{scenario}_matrix"]
        im = axes[row, 0].imshow(D, cmap="viridis")
        axes[row, 0].set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        axes[row, 0].set_yticks(range(len(names)), names, fontsize=7)
        axes[row, 0].set_title(f"{scenario.replace('_', ' ')}\nJC69 distance matrix")
        axes[row, 0].grid(False)
        fig.colorbar(im, ax=axes[row, 0], fraction=0.046)

        for col, builder in enumerate(("upgma", "nj"), start=1):
            key = f"{scenario}_{builder}"
            r = results[scenario][f"jc69_{builder}"]
            status = "topology recovered" if r["correct"] else \
                     f"WRONG (RF={r['rf_distance']})"
            _draw_tree(axes[row, col], artefacts[key],
                       f"{builder.upper()} — {status}")
    fig.suptitle("Tree reconstruction: UPGMA assumes a molecular clock, "
                 "neighbour joining does not", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(length: int = 6000, n_snv: int = 25, n_indel: int = 4,
         coverage: float = 60.0, seed: int = 0) -> dict:
    plt.rcParams.update(PLT_STYLE)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"[1/7] Simulating a {length} bp locus with {n_snv} SNVs "
          f"and {n_indel} indels ...")
    locus, reads = simulate.simulate_locus(
        length=length, n_snv=n_snv, n_indel=n_indel,
        coverage=coverage, seed=seed)
    sequence.write_fasta([sequence.SeqRecord("chr1", locus.reference,
                                             "simulated reference")],
                         RESULTS / "reference.fasta")
    sequence.write_fastq(reads[:500], RESULTS / "reads_sample.fastq")
    print(f"      {len(reads)} reads; {locus.meta['n_snv']} SNVs, "
          f"{locus.meta['n_indel']} indels placed")

    print("[2/7] Annotating the reference: ORFs, composition, protein ...")
    ref_stats, orfs = analyse_reference(locus)
    print(f"      GC={ref_stats['gc_content']*100:.1f}%  "
          f"{ref_stats['n_orfs_min50aa']} ORFs ≥50 aa;  longest = "
          f"{ref_stats['longest_orf_aa']} aa  "
          f"(recovers true CDS: {ref_stats['longest_orf_recovers_true_cds']})")
    pd.DataFrame([{"start": o.start + 1, "end": o.end, "strand": o.strand,
                   "frame": o.frame, "length_aa": o.length_aa,
                   "protein_prefix": o.protein[:30]} for o in orfs]
                 ).to_csv(RESULTS / "orfs.csv", index=False)

    print("[3/7] Mapping reads (seed-and-extend) ...")
    mappings = align.map_reads(locus.reference, reads, k=14)
    print(f"      mapped {len(mappings)}/{len(reads)} "
          f"({len(mappings)/len(reads)*100:.1f}%); "
          f"mean MAPQ {np.mean([m.mapq for m in mappings]):.1f}")

    print("[4/7] Calling variants and annotating consequences ...")
    columns = variants.build_pileup(locus.reference, mappings)
    calls = variants.call_variants(locus.reference, columns)
    annotations = {
        c.pos: variants.annotate_coding_effect(
            locus.reference, c.pos, c.ref, c.alt, locus.cds_start, locus.cds_end)
        for c in calls}
    variants.write_vcf(RESULTS / "variants.vcf", calls, contig="chr1",
                       length=len(locus.reference), sample="DONOR",
                       annotations=annotations)
    ev = variants.evaluate_calls(locus.variants, calls)
    print(f"      precision={ev['precision']:.3f}  recall={ev['recall']:.3f}  "
          f"F1={ev['f1']:.3f}  genotype concordance={ev['genotype_accuracy']:.3f}")

    calls_df = pd.DataFrame([{
        "pos": c.pos + 1, "ref": c.ref, "alt": c.alt, "qual": c.qual,
        "depth": c.depth, "alt_depth": c.alt_depth, "af": c.allele_fraction,
        "genotype": c.genotype, "strand_bias": c.strand_bias,
        "filter": ";".join(c.filters) or "PASS",
        **{k: v for k, v in annotations[c.pos].items()
           if k in ("consequence", "hgvs_c", "hgvs_p", "codon_number")}}
        for c in calls])
    calls_df.to_csv(RESULTS / "variant_calls.csv", index=False)
    consequence_counts = calls_df.consequence.value_counts().to_dict()
    print("      consequences: " + ", ".join(f"{k}={v}" for k, v
                                             in consequence_counts.items()))

    print("[5/7] Coverage titration ...")
    tit = pd.DataFrame([call_at_coverage(seed, c, length, n_snv, n_indel)
                        for c in COVERAGE_LADDER])
    tit.drop(columns=["fp_positions", "fn_positions"]).to_csv(
        RESULTS / "coverage_titration.csv", index=False)
    for _, r in tit.iterrows():
        print(f"      {r.mean_depth:5.1f}x  P={r.precision:.3f} "
              f"R={r.recall:.3f} F1={r.f1:.3f}")
    sufficient = tit[tit.recall >= 0.95]
    depth_95 = float(sufficient.mean_depth.min()) if len(sufficient) else float("nan")
    print(f"      minimum depth for ≥95% recall: {depth_95:.1f}x")

    print("[6/7] Discovering indels by global alignment ...")
    indel_df = discover_indels_by_alignment(locus)
    indel_df.to_csv(RESULTS / "indel_discovery.csv", index=False)
    detected = int(indel_df.detected.sum()) if len(indel_df) else 0
    print(f"      recovered {detected}/{len(indel_df)} indels "
          f"(read-based caller finds 0: it is ungapped)")

    print("[7/7] Phylogenetic reconstruction ...")
    phylo_results, artefacts = run_phylogenetics(seed=5)
    for scenario, entry in phylo_results.items():
        ok = [k for k, v in entry.items() if v["correct"]]
        print(f"      {scenario}: correct topology from {len(ok)}/{len(entry)} "
              f"method combinations -> {', '.join(ok) if ok else 'none'}")

    print("      writing figures ...")
    fig_reference(locus, orfs, RESULTS / "fig1_reference_annotation.png")
    fig_reads_and_depth(reads, mappings, locus, RESULTS / "fig2_reads_and_depth.png")
    fig_coverage_titration(tit, RESULTS / "fig3_coverage_titration.png")
    fig_calls(calls, {v.pos for v in locus.variants}, annotations,
              RESULTS / "fig4_variant_calls.png")
    fig_phylogenetics(artefacts, phylo_results, RESULTS / "fig5_phylogenetics.png")

    out = {
        "locus": locus.meta,
        "reference_annotation": ref_stats,
        "mapping": {"n_reads": len(reads), "n_mapped": len(mappings),
                    "mapping_rate": round(len(mappings) / len(reads), 4),
                    "mean_mapq": round(float(np.mean([m.mapq for m in mappings])), 2)},
        "variant_calling": {k: v for k, v in ev.items()
                            if not k.endswith("positions")},
        "consequences": consequence_counts,
        "coverage_titration": {"min_depth_for_95pct_recall": depth_95,
                               "ladder": tit[["mean_depth", "precision",
                                              "recall", "f1",
                                              "genotype_accuracy"]].round(4)
                               .to_dict("records")},
        "indel_discovery": {"n_truth": int(len(indel_df)),
                            "n_recovered": detected},
        "phylogenetics": phylo_results,
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nDone in {out['runtime_s']}s. Results in {RESULTS}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genomics variant toolkit")
    ap.add_argument("--length", type=int, default=6000)
    ap.add_argument("--n-snv", type=int, default=25)
    ap.add_argument("--n-indel", type=int, default=4)
    ap.add_argument("--coverage", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.length, a.n_snv, a.n_indel, a.coverage, a.seed)
