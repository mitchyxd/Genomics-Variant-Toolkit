"""Unit tests for the genomics variant toolkit.  Run with:  pytest -q"""

import numpy as np
import pytest

from src import align, phylo, sequence, simulate, variants


# --------------------------------------------------------------------------
# Sequence primitives
# --------------------------------------------------------------------------
def test_genetic_code_is_complete_and_correct():
    assert len(sequence.CODON_TABLE) == 64
    assert sequence.CODON_TABLE["ATG"] == "M"          # start / methionine
    assert sequence.CODON_TABLE["TGG"] == "W"          # only Trp codon
    assert all(sequence.CODON_TABLE[c] == "*" for c in ("TAA", "TAG", "TGA"))
    # Leucine and serine are each encoded by six codons
    assert sum(v == "L" for v in sequence.CODON_TABLE.values()) == 6
    assert sum(v == "S" for v in sequence.CODON_TABLE.values()) == 6


def test_reverse_complement_is_its_own_inverse():
    s = "ATGCGTNACGT"
    assert sequence.reverse_complement(sequence.reverse_complement(s)) == s
    assert sequence.reverse_complement("ATGC") == "GCAT"


def test_gc_content_and_skew():
    assert sequence.gc_content("GGCC") == pytest.approx(1.0)
    assert sequence.gc_content("ATAT") == pytest.approx(0.0)
    assert sequence.gc_content("ATGC") == pytest.approx(0.5)
    assert sequence.gc_skew("GGGGCCCC") == pytest.approx(0.0)
    assert sequence.gc_skew("GGGG") == pytest.approx(1.0)


def test_translation_handles_frames_and_ambiguity():
    assert sequence.translate("ATGGCCTAA") == "MA*"
    assert sequence.translate("ATGGCCTAA", to_stop=True) == "MA"
    assert sequence.translate("AATGGCCTAA", frame=1) == "MA*"
    assert sequence.translate("ATGNNNTAA") == "MX*"
    assert sequence.translate("ATGGC") == "M"          # partial codon dropped


def test_orf_finder_recovers_a_planted_gene():
    gene = "ATG" + "GCC" * 60 + "TAA"
    seq = "TTTTTTTTTT" + gene + "AAAAAAAAAA"
    orfs = sequence.find_orfs(seq, min_aa=50, both_strands=False)
    assert len(orfs) == 1
    assert orfs[0].start == 10
    assert orfs[0].end == 10 + len(gene)
    assert orfs[0].protein == "M" + "A" * 60


def test_orf_finder_reports_reverse_strand_in_forward_coordinates():
    gene = "ATG" + "GCC" * 60 + "TAA"
    seq = "TTTTTTTTTT" + sequence.reverse_complement(gene) + "AAAAAAAAAA"
    orfs = [o for o in sequence.find_orfs(seq, min_aa=50) if o.strand == "-"]
    assert len(orfs) == 1
    assert orfs[0].start == 10 and orfs[0].end == 10 + len(gene)


def test_phred_and_error_probabilities_round_trip():
    assert sequence.phred_to_error(10) == pytest.approx(0.1)
    assert sequence.phred_to_error(20) == pytest.approx(0.01)
    assert sequence.phred_to_error(30) == pytest.approx(0.001)
    assert sequence.error_to_phred(0.001) == pytest.approx(30.0)


def test_protein_properties():
    # Glycine has the smallest residue mass; a free glycine weighs ~75 Da
    assert sequence.molecular_weight("G") == pytest.approx(75.07, abs=0.05)
    acidic = sequence.isoelectric_point("DDDDEEEE")
    basic = sequence.isoelectric_point("KKKKRRRR")
    assert acidic < 5.0 < 9.0 < basic


def test_hydropathy_flags_a_hydrophobic_stretch():
    profile = sequence.hydropathy_profile("IIIIIIIIIVVVVVVVVV", window=9)
    assert profile.max() > 3.5
    assert sequence.hydropathy_profile("DDDDDDDDDEEEEEEEEE", window=9).max() < 0


def test_fasta_and_fastq_round_trip(tmp_path):
    recs = [sequence.SeqRecord("s1", "ATGCATGCATGC", "first"),
            sequence.SeqRecord("s2", "GGGGCCCC", "second")]
    p = tmp_path / "x.fasta"
    sequence.write_fasta(recs, p)
    back = sequence.read_fasta(p)
    assert [r.id for r in back] == ["s1", "s2"]
    assert [r.seq for r in back] == [r.seq for r in recs]

    fq = [sequence.SeqRecord("r1", "ACGT", "", np.array([40, 30, 20, 10]))]
    q = tmp_path / "x.fastq"
    sequence.write_fastq(fq, q)
    back = sequence.read_fastq(q)
    assert back[0].seq == "ACGT"
    assert list(back[0].qual) == [40, 30, 20, 10]


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------
def test_global_alignment_of_identical_sequences_is_gapless():
    a = "ACGTACGTAC"
    aln = align.needleman_wunsch(a, a)
    assert aln.identity == pytest.approx(1.0)
    assert "-" not in aln.aligned_a and "-" not in aln.aligned_b
    assert aln.score == len(a) * align.MATCH


def test_global_alignment_places_a_deletion():
    ref = "ACGTACGTACGT"
    donor = "ACGTAGTACGT"          # 'C' at index 5 deleted
    aln = align.needleman_wunsch(ref, donor)
    assert aln.aligned_b.count("-") == 1
    assert aln.aligned_a.replace("-", "") == ref
    assert aln.aligned_b.replace("-", "") == donor


def test_local_alignment_finds_an_embedded_motif():
    motif = "GATTACAGATTACA"
    aln = align.smith_waterman("TTTTT" + motif + "TTTTT",
                               "AAAAAAA" + motif + "AAAAAAA")
    assert motif in aln.aligned_a.replace("-", "")
    assert aln.identity == pytest.approx(1.0)


def test_kmer_index_and_exact_read_mapping():
    rng = np.random.default_rng(0)
    ref = "".join(rng.choice(list("ACGT"), size=2000))
    index = align.build_kmer_index(ref, k=14)
    read = ref[500:600]
    pos, strand, mm = align.map_read(ref, index, read, k=14)
    assert (pos, strand, mm) == (500, "+", 0)


def test_reverse_strand_reads_map_to_the_forward_coordinate():
    rng = np.random.default_rng(1)
    ref = "".join(rng.choice(list("ACGT"), size=2000))
    index = align.build_kmer_index(ref, k=14)
    read = sequence.reverse_complement(ref[800:900])
    pos, strand, mm = align.map_read(ref, index, read, k=14)
    assert pos == 800 and strand == "-" and mm == 0


def test_mapper_tolerates_mismatches_but_rejects_random_sequence():
    rng = np.random.default_rng(2)
    ref = "".join(rng.choice(list("ACGT"), size=3000))
    index = align.build_kmer_index(ref, k=14)
    read = list(ref[1000:1100])
    read[50] = "A" if read[50] != "A" else "C"
    hit = align.map_read(ref, index, "".join(read), k=14)
    assert hit is not None and hit[0] == 1000 and hit[2] == 1
    junk = "".join(rng.choice(list("ACGT"), size=100))
    assert align.map_read(ref, index, junk, k=14) is None


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def test_reference_contains_a_clean_coding_sequence():
    ref, start, end = simulate.make_reference(3000, 200, seed=0)
    cds = ref[start:end]
    assert len(ref) == 3000
    assert cds.startswith("ATG")
    assert cds[-3:] in sequence.STOP_CODONS
    protein = sequence.translate(cds, 0)
    assert protein.count("*") == 1 and protein.endswith("*")


def test_homozygous_variants_appear_on_both_haplotypes():
    ref, s, e = simulate.make_reference(3000, 200, seed=0)
    (h0, h1), vs = simulate.introduce_variants(ref, s, e, n_snv=20, n_indel=0,
                                               seed=1)
    hom = [v for v in vs if v.genotype == "1/1"]
    het = [v for v in vs if v.genotype == "0/1"]
    assert hom and het
    for v in hom:
        assert h0[v.pos] == v.alt and h1[v.pos] == v.alt
    for v in het:
        assert h0[v.pos] == v.ref and h1[v.pos] == v.alt


def test_read_quality_decays_along_the_read():
    reads = simulate.simulate_reads(("ACGT" * 500, "ACGT" * 500), n_reads=300,
                                    read_length=100, seed=3)
    q = np.array([r.qual for r in reads if r.description.endswith("+")])
    assert q[:, :20].mean() > q[:, -20:].mean()


# --------------------------------------------------------------------------
# Variant calling
# --------------------------------------------------------------------------
def test_genotype_likelihood_prefers_het_at_balanced_allele_fraction():
    bases = ["A"] * 20 + ["G"] * 20
    quals = [35] * 40
    hom_ref, het, hom_alt = variants._genotype_likelihoods(bases, quals, "A", "G")
    assert het > hom_ref and het > hom_alt


def test_genotype_likelihood_prefers_hom_alt_when_all_reads_are_alt():
    bases = ["G"] * 40
    hom_ref, het, hom_alt = variants._genotype_likelihoods(bases, [35] * 40,
                                                           "A", "G")
    assert hom_alt > het > hom_ref


def test_annotation_classifies_consequences_correctly():
    #                  0  1  2   3  4  5
    ref = "AAA" + "ATG" + "AAA" + "TGG" + "TAA"
    cds_start, cds_end = 3, len(ref)

    # AAA -> AAG is still Lys: synonymous
    syn = variants.annotate_coding_effect(ref, 8, "A", "G", cds_start, cds_end)
    assert syn["consequence"] == "synonymous_variant"

    # AAA -> GAA changes Lys to Glu: missense
    mis = variants.annotate_coding_effect(ref, 6, "A", "G", cds_start, cds_end)
    assert mis["consequence"] == "missense_variant"
    assert mis["hgvs_p"] == "p.Lys2Glu"

    # TGG -> TGA introduces a stop codon: nonsense
    non = variants.annotate_coding_effect(ref, 11, "G", "A", cds_start, cds_end)
    assert non["consequence"] == "stop_gained"
    assert non["aa_alt"] == "*"

    # Outside the CDS
    out = variants.annotate_coding_effect(ref, 1, "A", "T", cds_start, cds_end)
    assert out["consequence"] == "upstream_variant"


def test_frameshift_versus_inframe_indel():
    ref = "AAA" + "ATG" + "AAACCCGGG" + "TAA"
    fs = variants.annotate_coding_effect(ref, 7, "A", "AT", 3, len(ref))
    assert fs["consequence"] == "frameshift_variant"
    inframe = variants.annotate_coding_effect(ref, 7, "A", "ATTT", 3, len(ref))
    assert inframe["consequence"] == "inframe_indel"


def test_end_to_end_calling_recovers_all_variants_at_high_depth():
    locus, reads = simulate.simulate_locus(length=4000, n_snv=15, n_indel=0,
                                           coverage=60, seed=4)
    maps = align.map_reads(locus.reference, reads, k=14)
    cols = variants.build_pileup(locus.reference, maps)
    calls = variants.call_variants(locus.reference, cols)
    ev = variants.evaluate_calls(locus.variants, calls)
    assert ev["recall"] >= 0.9
    assert ev["precision"] >= 0.9
    assert ev["genotype_accuracy"] >= 0.85


def test_low_depth_reduces_recall_but_not_precision():
    """The caller must stay conservative when evidence is thin."""
    kw = dict(length=4000, n_snv=15, n_indel=0, seed=4)
    lo = simulate.simulate_locus(coverage=5, **kw)
    hi = simulate.simulate_locus(coverage=50, **kw)
    out = []
    for locus, reads in (lo, hi):
        maps = align.map_reads(locus.reference, reads, k=14)
        cols = variants.build_pileup(locus.reference, maps)
        calls = variants.call_variants(locus.reference, cols)
        out.append(variants.evaluate_calls(locus.variants, calls))
    assert out[0]["recall"] < out[1]["recall"]
    # At 5x the caller may legitimately make no calls at all, in which case
    # precision is undefined rather than zero.  What must never happen is a
    # confident wrong call.
    assert out[0]["n_called"] == 0 or out[0]["precision"] >= 0.9
    assert out[1]["precision"] >= 0.9


def test_variants_from_alignment_recovers_snv_and_indels():
    ref = "ACGTACGTACGTACGT"
    aln_ref = "ACGTACGTACGTACGT"
    aln_alt = "ACGTAAGTACGTACGT"          # single substitution
    found = variants.variants_from_alignment(aln_ref, aln_alt)
    assert len(found) == 1 and found[0]["kind"] == "snv" and found[0]["pos"] == 5

    found = variants.variants_from_alignment("ACGT-ACGT", "ACGTAACGT")
    assert found[0]["kind"] == "insertion"
    found = variants.variants_from_alignment("ACGTAACGT", "ACGT-ACGT")
    assert found[0]["kind"] == "deletion"


def test_vcf_output_is_well_formed(tmp_path):
    calls = [variants.Call(99, "A", "G", 55.0, 30, 15, "0/1", 0.5, 0.5, [])]
    ann = {99: {"consequence": "missense_variant", "hgvs_c": "c.10A>G",
                "hgvs_p": "p.Lys4Arg"}}
    p = tmp_path / "out.vcf"
    variants.write_vcf(p, calls, contig="chr1", length=1000, annotations=ann)
    lines = p.read_text().splitlines()
    assert lines[0] == "##fileformat=VCFv4.2"
    header = [l for l in lines if l.startswith("#CHROM")][0].split("\t")
    assert header[:5] == ["#CHROM", "POS", "ID", "REF", "ALT"]
    body = [l for l in lines if not l.startswith("#")][0].split("\t")
    assert body[1] == "100"            # 0-based 99 -> 1-based 100
    assert body[3] == "A" and body[4] == "G"
    assert body[6] == "PASS"
    assert "missense_variant" in body[7]


# --------------------------------------------------------------------------
# Phylogenetics
# --------------------------------------------------------------------------
def test_jukes_cantor_exceeds_raw_p_distance_and_saturates():
    assert phylo.jukes_cantor(0.0) == pytest.approx(0.0)
    for p in (0.05, 0.15, 0.30):
        assert phylo.jukes_cantor(p) > p
    assert phylo.jukes_cantor(0.80) == float("inf")


def test_distances_of_identical_sequences_are_zero():
    s = "ACGTACGTAAGGCCTT" * 20
    assert phylo.p_distance(s, s) == 0.0
    assert phylo.kmer_distance(s, s) == pytest.approx(0.0, abs=1e-9)


def test_mutate_produces_more_change_at_longer_distance():
    rng = np.random.default_rng(0)
    s = "".join(rng.choice(list("ACGT"), size=3000))
    near = phylo.p_distance(s, phylo.mutate(s, 0.01, rng))
    far = phylo.p_distance(s, phylo.mutate(s, 0.20, rng))
    assert near < far


def test_robinson_foulds_of_a_tree_against_itself_is_zero():
    rng = np.random.default_rng(0)
    root = "".join(rng.choice(list("ACGT"), size=2000))
    _, tree = phylo.evolve_along_tree(root, phylo.default_topology(), rng)
    rf = phylo.robinson_foulds(tree, tree)
    assert rf["rf_distance"] == 0 and rf["normalised_rf"] == 0.0


def test_both_builders_recover_a_clock_like_tree():
    rng = np.random.default_rng(0)
    root = "".join(rng.choice(list("ACGT"), size=4000))
    seqs, true_tree = phylo.evolve_along_tree(root, phylo.default_topology(), rng)
    D, names = phylo.distance_matrix(seqs, method="jc69")
    for builder in (phylo.upgma, phylo.neighbour_joining):
        assert phylo.robinson_foulds(true_tree, builder(D, names))["rf_distance"] == 0


def test_neighbour_joining_beats_upgma_under_rate_heterogeneity():
    """The documented failure mode of UPGMA must actually be reproduced."""
    rng = np.random.default_rng(0)
    root = "".join(rng.choice(list("ACGT"), size=4000))
    seqs, true_tree = phylo.evolve_along_tree(
        root, phylo.rate_heterogeneous_topology(), rng)
    D, names = phylo.distance_matrix(seqs, method="jc69")
    rf_upgma = phylo.robinson_foulds(true_tree, phylo.upgma(D, names))
    rf_nj = phylo.robinson_foulds(true_tree, phylo.neighbour_joining(D, names))
    assert rf_nj["rf_distance"] < rf_upgma["rf_distance"]
    assert rf_nj["rf_distance"] == 0


def test_tree_newick_output_is_parenthesis_balanced():
    rng = np.random.default_rng(0)
    root = "".join(rng.choice(list("ACGT"), size=1000))
    _, tree = phylo.evolve_along_tree(root, phylo.default_topology(), rng)
    nwk = tree.newick()
    assert nwk.count("(") == nwk.count(")")
    for taxon in ("A1", "A2", "B3", "B4"):
        assert taxon in nwk
