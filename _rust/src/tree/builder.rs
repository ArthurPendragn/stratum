use super::exact::gini;
use super::model::{TreeModel, TREE_LEAF};

const EPSILON: f64 = f64::EPSILON;

// Per-node statistics gathered from the current training slice.
#[derive(Clone)]
pub(crate) struct NodeStats {
    pub(crate) class_weights: Vec<f64>,
    pub(crate) weight: f64,
    pub(crate) impurity: f64,
}

// A candidate split returned by the split search implementation.
#[derive(Clone, Debug)]
pub(crate) struct Split {
    pub(crate) feature: usize,
    pub(crate) threshold: f64,
    pub(crate) missing_go_to_left: bool,
    pub(crate) improvement: f64,
    pub(crate) left_impurity: f64,
    pub(crate) right_impurity: f64,
}

// Split-search output plus any constants that should be carried forward.
pub(crate) struct SplitSearch {
    pub(crate) split: Option<Split>,
    pub(crate) constants: Vec<usize>,
}

// Pluggable strategy used by `build_tree` to choose the best node split.
pub(crate) trait SplitFinder {
    fn best_split(
        &mut self,
        x: &[f32],
        n_features: usize,
        rows: &[usize],
        y: &[usize],
        weights: &[f64],
        n_classes: usize,
        parent: &NodeStats,
        known_constants: &[usize],
    ) -> Result<SplitSearch, String>;
}

// Hyperparameters and basic shape constraints for tree construction.
pub(crate) struct BuildParams {
    pub(crate) n_features: usize,
    pub(crate) n_classes: usize,
    pub(crate) max_depth: usize,
    pub(crate) min_samples_split: usize,
    pub(crate) min_samples_leaf: usize,
    pub(crate) min_impurity_decrease: f64,
}

// Temporary mutable storage used while assembling the final `TreeModel`.
struct MutableTree {
    left: Vec<i64>,
    right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    missing_left: Vec<bool>,
    impurity: Vec<f64>,
    n_samples: Vec<i64>,
    weighted_samples: Vec<f64>,
    values: Vec<f64>,
    importances: Vec<f64>,
    max_depth: usize,
    n_leaves: usize,
}

impl MutableTree {
    // Create an empty tree with per-feature importance storage.
    fn new(n_features: usize) -> Self {
        Self {
            left: vec![],
            right: vec![],
            feature: vec![],
            threshold: vec![],
            missing_left: vec![],
            impurity: vec![],
            n_samples: vec![],
            weighted_samples: vec![],
            values: vec![],
            importances: vec![0.0; n_features],
            max_depth: 0,
            n_leaves: 0,
        }
    }

    // Append a node and initialize it as a leaf placeholder.
    fn add_node(&mut self, stats: &NodeStats, n_samples: usize, n_classes: usize) -> usize {
        let node = self.left.len();
        self.left.push(TREE_LEAF);
        self.right.push(TREE_LEAF);
        self.feature.push(-2);
        self.threshold.push(-2.0);
        self.missing_left.push(false);
        self.impurity.push(stats.impurity);
        self.n_samples.push(n_samples as i64);
        self.weighted_samples.push(stats.weight);
        self.values
            .extend(stats.class_weights.iter().map(|v| v / stats.weight));
        debug_assert_eq!(self.values.len(), (node + 1) * n_classes);
        node
    }
}

// Build a decision tree by expanding active row slices from a work stack.
pub(crate) fn build_tree<F: SplitFinder>(
    x: &[f32],
    y: &[usize],
    weights: &[f64],
    params: BuildParams,
    finder: &mut F,
) -> Result<TreeModel, String> {
    // Validate input shapes and discard zero-weight rows up front.
    let n_rows = y.len();
    if x.len() != n_rows * params.n_features || weights.len() != n_rows {
        return Err("training arrays have inconsistent shapes".into());
    }
    let mut rows: Vec<usize> = (0..n_rows).filter(|&row| weights[row] != 0.0).collect();
    if rows.is_empty() {
        return Err("at least one positive-weight row is required".into());
    }
    let total_weight: f64 = rows.iter().map(|&row| weights[row]).sum();

    // The stack stores the active row intervals that still need processing.
    let mut tree = MutableTree::new(params.n_features);
    let active_rows = rows.len();
    let mut work = vec![BuildFrame {
        start: 0,
        end: active_rows,
        depth: 0,
        constants: Vec::new(),
        parent: None,
    }];

    // Expand nodes depth-first until every frame becomes a leaf or split.
    while let Some(frame) = work.pop() {
        // Compute impurity statistics for the current slice and create a node.
        let stats = node_stats(&rows[frame.start..frame.end], y, weights, params.n_classes);
        let node = tree.add_node(&stats, frame.end - frame.start, params.n_classes);

        // Link this node back to its parent, if any.
        if let Some((parent, is_left)) = frame.parent {
            if is_left {
                tree.left[parent] = node as i64;
            } else {
                tree.right[parent] = node as i64;
            }
        }

        // Track the deepest frame visited so far.
        tree.max_depth = tree.max_depth.max(frame.depth);

        // Stop immediately if the node is too shallow, too small, or pure.
        let should_stop = frame.depth >= params.max_depth
            || frame.end - frame.start < params.min_samples_split
            || frame.end - frame.start < 2 * params.min_samples_leaf
            || stats.impurity <= 0.0;
        if should_stop {
            tree.n_leaves += 1;
            continue;
        }

        // Ask the configured splitter for the best candidate on this slice.
        let search = finder.best_split(
            x,
            params.n_features,
            &rows[frame.start..frame.end],
            y,
            weights,
            params.n_classes,
            &stats,
            &frame.constants,
        )?;
        let Some(split) = search.split else {
            tree.n_leaves += 1;
            continue;
        };

        // Compare the local gain against the global impurity-decrease threshold.
        let weighted_improvement = stats.weight / total_weight * split.improvement;
        if weighted_improvement + EPSILON < params.min_impurity_decrease {
            tree.n_leaves += 1;
            continue;
        }

        // Partition the active rows in-place (memory-efficient) so both child slices stay contiguous.
        let mut boundary = frame.start;
        let mut right = frame.end;
        while boundary < right {
            let row = rows[boundary];
            if x[row * params.n_features + split.feature] as f64 <= split.threshold {
                boundary += 1;
            } else {
                right -= 1;
                rows.swap(boundary, right);
            }
        }
        if boundary == frame.start || boundary == frame.end {
            return Err("split produced an empty child".into());
        }

        // Store split metadata and accumulate feature importance.
        tree.feature[node] = split.feature as i64;
        tree.threshold[node] = split.threshold;
        tree.missing_left[node] = split.missing_go_to_left;
        tree.importances[split.feature] += stats.weight * split.improvement;

        // Push right first so the left subtree is built next. This preserves
        // sklearn's depth-first RNG consumption and preorder node numbering.
        work.push(BuildFrame {
            start: boundary,
            end: frame.end,
            depth: frame.depth + 1,
            constants: search.constants.clone(),
            parent: Some((node, false)),
        });
        work.push(BuildFrame {
            start: frame.start,
            end: boundary,
            depth: frame.depth + 1,
            constants: search.constants,
            parent: Some((node, true)),
        });
        let _ = (split.left_impurity, split.right_impurity);
    }

    // Normalize feature importances to sum to one when there was any gain.
    if total_weight > 0.0 {
        let sum: f64 = tree.importances.iter().sum();
        if sum > 0.0 {
            for value in &mut tree.importances {
                *value /= sum;
            }
        }
    }

    // Materialize the immutable model and validate its internal consistency.
    let model = TreeModel {
        children_left: tree.left,
        children_right: tree.right,
        feature: tree.feature,
        threshold: tree.threshold,
        missing_go_to_left: tree.missing_left,
        impurity: tree.impurity,
        n_node_samples: tree.n_samples,
        weighted_n_node_samples: tree.weighted_samples,
        values: tree.values,
        n_classes: params.n_classes,
        n_features: params.n_features,
        max_depth: tree.max_depth,
        n_leaves: tree.n_leaves,
        feature_importances: tree.importances,
    };
    model.validate()?;
    Ok(model)
}

// A stack frame describing one contiguous slice of active rows.
struct BuildFrame {
    start: usize,
    end: usize,
    depth: usize,
    constants: Vec<usize>,
    parent: Option<(usize, bool)>,
}

// Compute class counts, total weight, and impurity for a node slice.
fn node_stats(rows: &[usize], y: &[usize], weights: &[f64], n_classes: usize) -> NodeStats {
    let mut class_weights = vec![0.0; n_classes];
    let mut weight = 0.0;
    for &row in rows {
        class_weights[y[row]] += weights[row];
        weight += weights[row];
    }
    NodeStats {
        impurity: gini(&class_weights, weight),
        class_weights,
        weight,
    }
}

// A small unit test that verifies preorder layout and weighted accounting.
#[cfg(test)]
mod tests {
    use super::*;

    struct RootSplit;
    impl SplitFinder for RootSplit {
        fn best_split(
            &mut self,
            _x: &[f32],
            _nf: usize,
            rows: &[usize],
            _y: &[usize],
            _w: &[f64],
            _nc: usize,
            _parent: &NodeStats,
            constants: &[usize],
        ) -> Result<SplitSearch, String> {
            Ok(SplitSearch {
                split: (rows.len() > 2).then_some(Split {
                    feature: 0,
                    threshold: 1.5,
                    missing_go_to_left: false,
                    improvement: 0.5,
                    left_impurity: 0.0,
                    right_impurity: 0.0,
                }),
                constants: constants.to_vec(),
            })
        }
    }

    #[test]
    fn builds_preorder_topology_and_weighted_statistics() {
        let mut finder = RootSplit;
        let model = build_tree(
            &[0.0, 1.0, 2.0, 3.0],
            &[0, 0, 1, 1],
            &[2.0, 1.0, 1.0, 3.0],
            BuildParams {
                n_features: 1,
                n_classes: 2,
                max_depth: 1,
                min_samples_split: 2,
                min_samples_leaf: 1,
                min_impurity_decrease: 0.0,
            },
            &mut finder,
        )
        .unwrap();
        assert_eq!(model.children_left, vec![1, -1, -1]);
        assert_eq!(model.children_right, vec![2, -1, -1]);
        assert_eq!(model.n_node_samples, vec![4, 2, 2]);
        assert_eq!(model.weighted_n_node_samples, vec![7.0, 3.0, 4.0]);
    }
}
