/**
 * Logistic Regression (binary, positive class = "Approved"), trained with
 * batch gradient descent and L2 regularization. Features are standardized by
 * the shared preprocessing pipeline before this model runs.
 */

export interface LogisticParams {
  learningRate: number;
  epochs: number;
  lambda: number; // L2 regularization strength
  randomState: number;
}

export const DEFAULT_LR_PARAMS: LogisticParams = {
  learningRate: 0.25,
  epochs: 800,
  lambda: 0.01,
  randomState: 42,
};

const sigmoid = (z: number) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));

export class LogisticRegression {
  readonly weights: number[];
  readonly bias: number;
  readonly params: LogisticParams;
  readonly iterations: number;
  readonly finalLoss: number;

  private constructor(weights: number[], bias: number, params: LogisticParams, iterations: number, finalLoss: number) {
    this.weights = weights;
    this.bias = bias;
    this.params = params;
    this.iterations = iterations;
    this.finalLoss = finalLoss;
  }

  /** Rebuild a trained model from serialized weights (for inference). */
  static fromSerialized(weights: number[], bias: number): LogisticRegression {
    return new LogisticRegression(weights, bias, DEFAULT_LR_PARAMS, 0, 0);
  }

  static train(X: number[][], y: number[], params: Partial<LogisticParams> = {}): LogisticRegression {
    const p: LogisticParams = { ...DEFAULT_LR_PARAMS, ...params };
    const n = X.length;
    const d = X[0].length;
    let weights = new Array<number>(d).fill(0);
    let bias = 0;
    let iterations = 0;
    let lastLoss = Infinity;

    for (let epoch = 0; epoch < p.epochs; epoch++) {
      // Forward pass.
      const probas = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        let z = bias;
        const row = X[i];
        for (let j = 0; j < d; j++) z += weights[j] * row[j];
        probas[i] = sigmoid(z);
      }

      // Gradients.
      const gradW = new Array<number>(d).fill(0);
      let gradB = 0;
      for (let i = 0; i < n; i++) {
        const diff = probas[i] - y[i];
        gradB += diff;
        const row = X[i];
        for (let j = 0; j < d; j++) gradW[j] += diff * row[j];
      }
      for (let j = 0; j < d; j++) gradW[j] = gradW[j] / n + (p.lambda / n) * weights[j];
      gradB /= n;

      // Update.
      for (let j = 0; j < d; j++) weights[j] -= p.learningRate * gradW[j];
      bias -= p.learningRate * gradB;

      // Log loss (for diagnostics / early stopping).
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const pI = Math.min(Math.max(probas[i], 1e-12), 1 - 1e-12);
        loss += -(y[i] * Math.log(pI) + (1 - y[i]) * Math.log(1 - pI));
      }
      loss /= n;
      iterations = epoch + 1;

      if (Math.abs(loss - lastLoss) < 1e-7) {
        lastLoss = loss;
        break;
      }
      lastLoss = loss;
    }

    return new LogisticRegression(weights, bias, p, iterations, lastLoss);
  }

  predictProba(x: number[]): number {
    let z = this.bias;
    for (let j = 0; j < this.weights.length; j++) z += this.weights[j] * x[j];
    return sigmoid(z);
  }

  predict(x: number[]): number {
    return this.predictProba(x) >= 0.5 ? 1 : 0;
  }
}
