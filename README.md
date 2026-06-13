# Entropy Gradient

**An information geometry library that computes gradients of Shannon entropy, KL divergence, and Jensen-Shannon divergence** — providing the analytical tools for maximum entropy optimization, information-theoretic model fitting, and simplex projection.

## Why It Matters

Entropy is the fundamental measure of uncertainty in information theory. Its gradient tells you *how to change a distribution to increase or decrease uncertainty* — which is exactly what you need for:

- **Maximum entropy modeling** — Find the distribution with maximum entropy subject to constraints (the principle of maximum entropy, Jaynes 1957). This is how language models, ecological niche models, and Bayesian priors are constructed.
- **Information-theoretic optimization** — Minimizing KL divergence is the objective function in variational inference, used extensively in modern machine learning (VAEs, expectation propagation).
- **Natural gradient descent** — Using KL divergence as a distance metric in parameter space, which converges faster than Euclidean gradient descent for probabilistic models.
- **Information bottleneck** — Finding optimal compressed representations that preserve task-relevant information.

**Key functions and their applications:**
- `shannon_entropy` — Measures uncertainty. Uniform distribution has maximum entropy.
- `entropy_gradient` — Direction of steepest entropy increase. Used in max-entropy optimization.
- `kl_divergence` — Measures how much one distribution diverges from another. Used in variational inference and model comparison.
- `jensen_shannon_divergence` — Symmetric, bounded version of KL divergence. Used for distribution clustering and similarity.
- `project_to_simplex` — Projects an arbitrary point onto the probability simplex (ensures non-negative, sums to 1). Used in constrained optimization.

## How It Works

**Shannon entropy:** `H(p) = −Σ pᵢ ln(pᵢ)`. For a uniform distribution over n outcomes, H = ln(n). For a deterministic distribution (one outcome with probability 1), H = 0. The entropy function is concave — averaging distributions increases entropy.

**Entropy gradient:** `∂H/∂pᵢ = −ln(pᵢ) − 1`. This is the derivative of the entropy formula. At pᵢ near 0, the gradient diverges to +∞ (adding probability mass to an impossible event dramatically increases entropy). At pᵢ near 1, the gradient approaches −1 (making a near-certain event less certain increases entropy).

**KL divergence:** `D_KL(p‖q) = Σ pᵢ ln(pᵢ/qᵢ)`. Always non-negative (Gibb's inequality), zero iff p = q. Not symmetric — D_KL(p‖q) ≠ D_KL(q‖p). The asymmetry matters: it measures the cost of using q to approximate p.

**Jensen-Shannon divergence:** `JSD(p,q) = ½D_KL(p‖m) + ½D_KL(q‖m)` where m = (p+q)/2. Symmetric, bounded in [0, ln(2)], and always finite. The square root of JSD is a proper metric (satisfies triangle inequality).

**Simplex projection:** Uses the algorithm from Wang & Carreira-Perpiñán (2013): sort coordinates descending, find the threshold τ such that Σ max(vᵢ − τ, 0) = 1, then project. This ensures the result is a valid probability distribution (non-negative, sums to 1).

## Quick Start

```rust
use entropy_gradient::{shannon_entropy, kl_divergence, entropy_gradient, project_to_simplex};

// Entropy of different distributions
let uniform = vec![0.25, 0.25, 0.25, 0.25];
let peaked = vec![0.97, 0.01, 0.01, 0.01];
println!("Uniform entropy: {:.4}", shannon_entropy(&uniform));  // ≈ 1.386 (ln 4)
println!("Peaked entropy: {:.4}", shannon_entropy(&peaked));     // much lower

// KL divergence: how different are two distributions?
let p = vec![0.3, 0.5, 0.2];
let q = vec![0.4, 0.4, 0.2];
println!("D_KL(p||q) = {:.6}", kl_divergence(&p, &q));

// Gradient for max-entropy optimization
let grad = entropy_gradient(&p);
println!("Entropy gradient: {:?}", grad);

// Project an arbitrary vector to the probability simplex
let v = vec![0.5, 0.5, 0.5];
let p = project_to_simplex(&v);
println!("Simplex projection: {:?} (sum = {:.4})", p, p.iter().sum::<f64>());
```

## API

### Entropy Functions
- `shannon_entropy(p: &[f64]) -> f64` — Shannon entropy H(p) = −Σ pᵢ ln pᵢ. O(n)
- `entropy_gradient(p: &[f64]) -> Vec<f64>` — ∂H/∂pᵢ = −ln(pᵢ) − 1. O(n)
- `cross_entropy(p: &[f64], q: &[f64]) -> f64` — H(p, q) = −Σ pᵢ ln qᵢ. O(n)
- `kl_divergence(p, q) -> f64` — D_KL(p‖q) = cross_entropy − entropy. O(n)
- `kl_gradient_q(p, q) -> Vec<f64>` — Gradient of KL w.r.t. q. O(n)
- `jensen_shannon_divergence(p, q) -> f64` — Symmetric, bounded [0, ln2]. O(n)

### Optimization Utilities
- `max_entropy_uniform(n: usize) -> Vec<f64>` — Uniform distribution over n outcomes
- `project_to_simplex(v: &[f64]) -> Vec<f64>` — Euclidean projection onto probability simplex. O(n log n)

## Architecture Notes

This library provides information-theoretic analysis tools for SuperInstance's machine learning and optimization toolkit. It supports variational inference, maximum entropy modeling, and natural gradient optimization across the ML infrastructure.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## License

MIT
