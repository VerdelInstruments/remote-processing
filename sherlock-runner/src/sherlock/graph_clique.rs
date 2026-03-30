//! Graph-clique isotope detection, matching the production (Python/NetworkX)
//! algorithm.
//!
//! Production builds an undirected graph where peaks are nodes and edges
//! connect peaks whose spacing matches any target isotope separation (1/z Th
//! for z = 1..max_charge).  Maximal cliques (via `nx.find_cliques`) identify
//! isotope envelopes.  The charge for each clique is determined from the
//! median pairwise spacing.
//!
//! This module replaces the union-find approach that was previously used in
//! Rust.  Union-find merges connected *components* (A-B-C all merged even if
//! A and C don't match), whereas clique detection requires all pairs within
//! the group to be connected — matching the production semantics exactly.

use std::collections::HashSet;

use log::warn;

/// Maximum clique size to process.  Real isotope envelopes rarely exceed
/// ~10 peaks.  Matching the production guard in `detect.py`.
const MAX_CLIQUE_SIZE: usize = 20;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect isotope envelopes using the graph-clique algorithm.
///
/// Given a set of TOF centroid values, builds an adjacency graph and finds
/// maximal cliques using Bron-Kerbosch with pivoting.  Returns a charge
/// assignment for every peak (0 = not part of any isotope envelope) and the
/// number of isotope groups found.
///
/// This is the shared core used by `isotope.rs`, `swim_group.rs`, and
/// `precursors.rs`.
pub fn detect_isotopes_graph_clique(
    centroids: &[f64],
    tolerance: f64,
    max_charge: usize,
) -> (Vec<i32>, usize) {
    let n = centroids.len();
    if n < 2 {
        return (vec![0; n], 0);
    }

    let targets: Vec<f64> = (1..=max_charge).map(|z| 1.0 / z as f64).collect();
    let max_d = targets[0] + 1.1 * tolerance;

    // Sort peaks by centroid for the sweep
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        centroids[a]
            .partial_cmp(&centroids[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build adjacency list
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for i in 0..order.len() {
        let gi = order[i];
        let tof_i = centroids[gi];
        for j in (i + 1)..order.len() {
            let gj = order[j];
            let d = centroids[gj] - tof_i;
            if d > max_d {
                break; // sorted — no further pairs within range
            }
            let matches = targets.iter().any(|&t| (t - d).abs() < tolerance);
            if matches {
                adj[gi].insert(gj);
                adj[gj].insert(gi);
            }
        }
    }

    // Find maximal cliques via Bron-Kerbosch with pivoting
    let mut cliques: Vec<Vec<usize>> = Vec::new();
    let p: HashSet<usize> = (0..n).filter(|&i| !adj[i].is_empty()).collect();
    let x: HashSet<usize> = HashSet::new();
    let r: Vec<usize> = Vec::new();

    bron_kerbosch(&r, &p, &x, &adj, &mut cliques);

    // Assign charges from cliques
    let mut charges = vec![0i32; n];
    let mut n_groups = 0usize;

    // Sort cliques largest-first so larger envelopes take priority
    cliques.sort_by(|a, b| b.len().cmp(&a.len()));

    for clique in &cliques {
        if clique.len() < 2 {
            continue;
        }
        if clique.len() > MAX_CLIQUE_SIZE {
            warn!(
                "Skipping clique of size {} (max {})",
                clique.len(),
                MAX_CLIQUE_SIZE
            );
            continue;
        }

        // Compute median spacing between all pairs in the clique
        let mut pair_spacings: Vec<f64> = Vec::new();
        for i in 0..clique.len() {
            for j in (i + 1)..clique.len() {
                let d = (centroids[clique[i]] - centroids[clique[j]]).abs();
                pair_spacings.push(d);
            }
        }

        if pair_spacings.is_empty() {
            continue;
        }

        pair_spacings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med_spacing = pair_spacings[pair_spacings.len() / 2];
        let charge = if med_spacing > 0.0 {
            (1.0 / med_spacing).round().max(1.0) as i32
        } else {
            0
        };

        if charge == 0 {
            continue;
        }

        n_groups += 1;
        for &idx in clique {
            charges[idx] = charge;
        }
    }

    (charges, n_groups)
}

// ---------------------------------------------------------------------------
// Bron-Kerbosch with pivot
// ---------------------------------------------------------------------------

/// Bron-Kerbosch algorithm with pivot selection for finding all maximal cliques.
///
/// - `r`: current clique being built
/// - `p`: candidate vertices that can extend the clique
/// - `x`: vertices already processed (used to ensure maximality)
/// - `adj`: adjacency list
/// - `cliques`: accumulates all maximal cliques found
fn bron_kerbosch(
    r: &[usize],
    p: &HashSet<usize>,
    x: &HashSet<usize>,
    adj: &[HashSet<usize>],
    cliques: &mut Vec<Vec<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if r.len() >= 2 {
            cliques.push(r.to_vec());
        }
        return;
    }

    if p.is_empty() {
        return; // x is non-empty, so r is not maximal
    }

    // Choose pivot: vertex in P ∪ X with most neighbours in P
    let pivot = p
        .iter()
        .chain(x.iter())
        .max_by_key(|&&v| adj[v].intersection(p).count())
        .copied()
        .unwrap(); // safe: p is non-empty

    // Vertices to explore: P \ N(pivot)
    let pivot_neighbours = &adj[pivot];
    let candidates: Vec<usize> = p.difference(pivot_neighbours).copied().collect();

    let mut p_mut = p.clone();
    let mut x_mut = x.clone();

    for v in candidates {
        let mut new_r = r.to_vec();
        new_r.push(v);

        let new_p: HashSet<usize> = p_mut.intersection(&adj[v]).copied().collect();
        let new_x: HashSet<usize> = x_mut.intersection(&adj[v]).copied().collect();

        bron_kerbosch(&new_r, &new_p, &new_x, adj, cliques);

        p_mut.remove(&v);
        x_mut.insert(v);
    }
}

// ---------------------------------------------------------------------------
// Convenience wrapper for indexed subsets
// ---------------------------------------------------------------------------

/// Run graph-clique isotope detection on a subset of peaks identified by
/// indices into a shared centroids array.
///
/// Returns `(Vec<(index, charge)>, n_groups)`.
pub fn detect_isotopes_in_group(
    group_indices: &[usize],
    centroids_tof: &[f64],
    tolerance: f64,
    max_charge: usize,
) -> (Vec<(usize, i32)>, usize) {
    let n = group_indices.len();
    if n < 2 {
        return (
            group_indices.iter().map(|&idx| (idx, 0i32)).collect(),
            0,
        );
    }

    // Extract the subset of centroids
    let local_centroids: Vec<f64> = group_indices
        .iter()
        .map(|&idx| centroids_tof[idx])
        .collect();

    let (local_charges, n_groups) =
        detect_isotopes_graph_clique(&local_centroids, tolerance, max_charge);

    // Map back to global indices
    let charges: Vec<(usize, i32)> = group_indices
        .iter()
        .zip(local_charges.iter())
        .map(|(&idx, &c)| (idx, c))
        .collect();

    (charges, n_groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_doubly_charged() {
        // Three peaks at 0.5 Th spacing -> charge 2
        let centroids = vec![100.0, 100.5, 101.0];
        let (charges, n_groups) = detect_isotopes_graph_clique(&centroids, 0.01, 5);
        assert_eq!(n_groups, 1);
        assert!(charges.iter().all(|&c| c == 2));
    }

    #[test]
    fn test_singly_charged() {
        // Three peaks at 1.0 Th spacing -> charge 1
        // Note: spacing between peaks 0 and 2 is 2.0, which does NOT match
        // any 1/z target.  So the graph has edges 0-1 and 1-2 but not 0-2.
        // The maximal cliques are {0,1} and {1,2} — both assigned charge 1.
        // This is 2 clique-groups (unlike union-find which would merge all 3).
        let centroids = vec![200.0, 201.0, 202.0];
        let (charges, n_groups) = detect_isotopes_graph_clique(&centroids, 0.01, 5);
        assert_eq!(n_groups, 2);
        assert!(charges.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_no_isotope() {
        // Peaks too far apart
        let centroids = vec![100.0, 105.0, 110.0];
        let (charges, n_groups) = detect_isotopes_graph_clique(&centroids, 0.01, 5);
        assert_eq!(n_groups, 0);
        assert!(charges.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_chain_not_merged() {
        // A at 100.0, B at 101.0, C at 102.5
        // A-B: spacing 1.0 (charge 1 match)
        // B-C: spacing 1.5 (no match for any 1/z with z=1..5)
        // A-C: spacing 2.5 (no match)
        // Union-find would merge A-B then B-C into one component.
        // Graph-clique should keep A-B as one clique and C separate.
        let centroids = vec![100.0, 101.0, 102.5];
        let (charges, n_groups) = detect_isotopes_graph_clique(&centroids, 0.01, 5);
        assert_eq!(n_groups, 1);
        assert_eq!(charges[0], 1); // A
        assert_eq!(charges[1], 1); // B
        assert_eq!(charges[2], 0); // C not in any clique
    }

    #[test]
    fn test_group_wrapper() {
        let centroids = vec![50.0, 100.0, 100.5, 101.0, 200.0];
        let group_indices = vec![1, 2, 3]; // the doubly-charged triplet
        let (charges, n_groups) = detect_isotopes_in_group(&group_indices, &centroids, 0.01, 5);
        assert_eq!(n_groups, 1);
        for &(idx, c) in &charges {
            assert_eq!(c, 2, "peak at index {} should be charge 2", idx);
        }
    }
}
