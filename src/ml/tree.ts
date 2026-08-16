/**
 * Random Forest Classifier (binary, positive class = "Approved").
 *
 * A real implementation: CART-style decision trees grown with Gini impurity,
 * bootstrap bagging, and random feature subsampling (max_features = sqrt), the
 * same recipe as scikit-learn's RandomForestClassifier. Trees are stored as
 * flat JSON-serializable node arrays so the trained model can be persisted and
 * reloaded for inference with no external runtime dependencies.
 */

import { mulberry32 } from "./rng";

export interface RandomForestParams {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures: "sqrt" | number;
  randomState: number;
}

export const DEFAULT_RF_PARAMS: RandomForestParams = {
  nEstimators: 80,
  maxDepth: 12,
  minSamplesSplit: 6,
  minSamplesLeaf: 6,
  maxFeatures: "sqrt",
  randomState: 42,
};

interface InternalNode {
  f: number;
  t: number;
  left: number;
  right: number;
  /** Sample count reaching this node (used for feature importance). */
  n: number;
  /** Gini impurity at this node. */
  gini: number;
}

interface LeafNode {
  counts: [number, number];
  p: number;
  n: number;
}

export type TreeNode = InternalNode | LeafNode;

function gini(n0: number, n1: number): number {
  const total = n0 + n1;
  if (total === 0) return 0;
  const p0 = n0 / total;
  const p1 = n1 / total;
  return 1 - p0 * p0 - p1 * p1;
}

interface BuildContext {
  X: number[][];
  y: number[];
  params: RandomForestParams;
  rng: () => number;
  nodes: TreeNode[];
  nodeCounts: number[];
}

function growNode(ctx: BuildContext, indices: number[], depth: number, featurePool: number[]): number {
  const { X, y, params, rng, nodes, nodeCounts } = ctx;

  let n0 = 0;
  let n1 = 0;
  for (const i of indices) {
    if (y[i] === 1) n1++;
    else n0++;
  }
  const n = indices.length;
  const nodeIdx = nodes.length;
  nodes.push({ counts: [n0, n1], p: n1 / Math.max(n, 1), n });
  nodeCounts.push(n);

  const impurity = gini(n0, n1);
  const canSplit =
    depth < params.maxDepth &&
    n >= params.minSamplesSplit &&
    n0 > 0 &&
    n1 > 0;

  if (!canSplit) return nodeIdx;

  // Random feature subset for this node.
  const nFeatures = params.maxFeatures === "sqrt"
    ? Math.max(1, Math.round(Math.sqrt(X[0].length)))
    : Math.min(params.maxFeatures, X[0].length);
  const pool = [...featurePool];
  for (let k = pool.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [pool[k], pool[j]] = [pool[j], pool[k]];
  }
  const candidates = pool.slice(0, nFeatures);

  let best: { f: number; t: number; gain: number } | null = null;
  let bestImpurity = impurity;

  for (const f of candidates) {
    // (value, label) pairs for this feature.
    const pairs: [number, number][] = new Array(indices.length);
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      pairs[k] = [X[i][f], y[i]];
    }
    pairs.sort((a, b) => a[0] - b[0]);

    let l0 = 0;
    let l1 = 0;
    for (let k = 0; k < pairs.length - 1; k++) {
      const [v, label] = pairs[k];
      if (label === 1) l1++;
      else l0++;
      if (v === pairs[k + 1][0]) continue; // don't split inside a run of equal values
      const nL = l0 + l1;
      const nR = n - nL;
      if (nL < params.minSamplesLeaf || nR < params.minSamplesLeaf) continue;
      const imp = (nL / n) * gini(l0, l1) + (nR / n) * gini(n0 - l0, n1 - l1);
      if (imp < bestImpurity - 1e-12) {
        bestImpurity = imp;
        best = { f, t: (v + pairs[k + 1][0]) / 2, gain: impurity - imp };
      }
    }
  }

  if (!best) return nodeIdx;

  const leftIdx: number[] = [];
  const rightIdx: number[] = [];
  for (const i of indices) {
    if (X[i][best.f] <= best.t) leftIdx.push(i);
    else rightIdx.push(i);
  }
  if (leftIdx.length === 0 || rightIdx.length === 0) return nodeIdx;

  // Replace the leaf placeholder with an internal node.
  const internal: InternalNode = {
    f: best.f,
    t: best.t,
    left: -1,
    right: -1,
    n,
    gini: impurity,
  };
  nodes[nodeIdx] = internal;

  const left = growNode(ctx, leftIdx, depth + 1, pool);
  const right = growNode(ctx, rightIdx, depth + 1, pool);
  internal.left = left;
  internal.right = right;
  return nodeIdx;
}

export class RandomForestClassifier {
  readonly params: RandomForestParams;
  readonly trees: TreeNode[][];
  readonly nFeatures: number;

  constructor(trees: TreeNode[][], params: RandomForestParams, nFeatures: number) {
    this.trees = trees;
    this.params = params;
    this.nFeatures = nFeatures;
  }

  static train(X: number[][], y: number[], params: Partial<RandomForestParams> = {}): RandomForestClassifier {
    const p: RandomForestParams = { ...DEFAULT_RF_PARAMS, ...params };
    const rng = mulberry32(p.randomState);
    const n = X.length;
    const nFeatures = X[0].length;
    const featurePool = Array.from({ length: nFeatures }, (_, i) => i);
    const trees: TreeNode[][] = [];

    for (let t = 0; t < p.nEstimators; t++) {
      // Bootstrap sample (with replacement).
      const indices = new Array<number>(n);
      for (let i = 0; i < n; i++) indices[i] = Math.floor(rng() * n);
      const ctx: BuildContext = {
        X,
        y,
        params: p,
        rng,
        nodes: [],
        nodeCounts: [],
      };
      growNode(ctx, indices, 0, featurePool);
      trees.push(ctx.nodes);
    }

    return new RandomForestClassifier(trees, p, nFeatures);
  }

  predictProba(x: number[]): number {
    let sum = 0;
    for (const tree of this.trees) {
      let node = tree[0];
      let guard = 0;
      while ("f" in node && guard < 200) {
        const next = x[node.f] <= node.t ? node.left : node.right;
        node = tree[next];
        guard++;
      }
      // After the descent the node is guaranteed to be a leaf.
      const leaf = node as LeafNode;
      sum += leaf.p;
    }
    return sum / this.trees.length;
  }

  predict(x: number[]): number {
    return this.predictProba(x) >= 0.5 ? 1 : 0;
  }

  /**
   * Mean impurity decrease per feature, normalized to sum to 1
   * (equivalent to scikit-learn's feature_importances_).
   */
  featureImportance(): number[] {
    const nFeatures = this.nFeatures;
    const importance = new Array<number>(nFeatures).fill(0);
    let total = 0;

    for (const tree of this.trees) {
      const treeTotal = tree[0]?.n ?? 0;
      for (const node of tree) {
        if (!("f" in node)) continue;
        const left = tree[node.left];
        const right = tree[node.right];
        const nL = left?.n ?? 0;
        const nR = right?.n ?? 0;
        const weighted =
          (node.n / treeTotal) *
          (node.gini - (nL / node.n) * giniOf(left) - (nR / node.n) * giniOf(right));
        importance[node.f] += Math.max(0, weighted);
        total += Math.max(0, weighted);
      }
    }

    if (total > 0) {
      for (let i = 0; i < nFeatures; i++) importance[i] /= total;
    }
    return importance;
  }
}

function giniOf(node: TreeNode | undefined): number {
  if (!node) return 0;
  if ("gini" in node) return node.gini;
  const [n0, n1] = node.counts;
  return gini(n0, n1);
}
