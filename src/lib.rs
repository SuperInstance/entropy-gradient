//! Entropy Gradient
//!
//! Compute gradients of Shannon entropy with respect to probability parameters,
//! useful for maximum entropy optimization and information geometry.

/// Compute Shannon entropy of a probability distribution.
pub fn shannon_entropy(p: &[f64]) -> f64 {
    let total: f64 = p.iter().sum();
    p.iter()
        .filter(|&&pi| pi > 0.0)
        .map(|&pi| {
            let pn = pi / total;
            -pn * pn.ln()
        })
        .sum()
}

/// Gradient of Shannon entropy with respect to each p_i.
///
/// ∂H/∂p_i = -ln(p_i) - 1  (for normalized distributions summing to 1)
pub fn entropy_gradient(p: &[f64]) -> Vec<f64> {
    p.iter()
        .map(|&pi| {
            if pi > 0.0 {
                -(pi.ln() + 1.0)
            } else {
                f64::INFINITY // gradient is +∞ at boundary
            }
        })
        .collect()
}

/// Cross-entropy H(p, q) = -Σ p_i ln(q_i).
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");
    p.iter()
        .zip(q.iter())
        .filter(|&(_, &qi)| qi > 0.0)
        .map(|(&pi, &qi)| -pi * qi.ln())
        .sum()
}

/// KL divergence D_KL(p || q) = Σ p_i ln(p_i / q_i).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    cross_entropy(p, q) - shannon_entropy(p)
}

/// Gradient of KL divergence w.r.t. q.
pub fn kl_gradient_q(p: &[f64], q: &[f64]) -> Vec<f64> {
    assert_eq!(p.len(), q.len());
    q.iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| {
            if qi > 0.0 {
                -pi / qi
            } else if pi > 0.0 {
                f64::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect()
}

/// Jensen-Shannon divergence (symmetric, bounded [0, ln(2)]).
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| 0.5 * (pi + qi)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Maximum entropy distribution over n outcomes with given constraints.
/// Returns uniform distribution (no constraints beyond normalization).
pub fn max_entropy_uniform(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

/// Project probabilities to the simplex using the algorithm from
/// Wang & Carreira-Perpinán (2013).
pub fn project_to_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cumsum = 0.0;
    let mut rho = 0;
    let mut tau = 0.0;
    for (j, &s) in sorted.iter().enumerate() {
        cumsum += s;
        let t = (cumsum - 1.0) / ((j + 1) as f64);
        if s - t > 0.0 {
            rho = j;
            tau = t;
        }
    }

    v.iter().map(|&vi| (vi - tau).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_entropy() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&p);
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_kl_same_distribution() {
        let p = vec![0.3, 0.5, 0.2];
        assert!(kl_divergence(&p, &p).abs() < 1e-10);
    }

    #[test]
    fn test_jsd_bounded() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let jsd = jensen_shannon_divergence(&p, &q);
        assert!((jsd - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_simplex_projection() {
        let v = vec![0.5, 0.5, 0.5];
        let p = project_to_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

/// FNV-1a 64 — the digest every substrate in the SuperInstance fleet agrees on.
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// True if this crate's FNV-1a still agrees with the rest of the fleet.
pub fn canary_holds() -> bool {
    fnv1a64("café Δ 日本語".as_bytes()) == 0x024a555471370b18d
}
