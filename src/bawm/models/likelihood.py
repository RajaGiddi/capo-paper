"""
N-feature Bayes factor model with bivariate t-copula correction.

Each feature contributes an independent log Bayes factor. A copula
correction term adjusts for pairwise dependence between containment
and cosine similarity (the two most correlated features).

Features and their likelihood-ratio forms:
  1. Containment J           — Beta vs Beta
  2. Chain coverage          — Beta vs Beta
  3. Chain ratio f_s/f_t     — Beta vs Beta (reversed: high ratio = repeat)
  4. End-anchoring           — Beta vs Beta
  5. Unique containment      — Beta vs Beta
  6. Encoder cosine          — Gaussian vs Gaussian
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import beta as beta_dist, norm, t as t_dist
from scipy.special import gammaln


@dataclass
class FeatureDistribution:
    """Parameters for one feature's genuine and null distributions."""
    name: str
    # Distribution type: 'beta' or 'gaussian'
    dist_type: str = 'beta'
    # Beta parameters (used if dist_type == 'beta')
    a_genuine: float = 2.0
    b_genuine: float = 2.0
    a_null: float = 2.0
    b_null: float = 2.0
    # Gaussian parameters (used if dist_type == 'gaussian')
    mu_genuine: float = 0.0
    sigma_genuine: float = 1.0
    mu_null: float = 0.0
    sigma_null: float = 1.0
    # Reliability weight (1.0 = full contribution)
    weight: float = 1.0


@dataclass
class MultiFeatureBFModel:
    """
    N-feature Bayes factor model with optional copula correction.
    """
    features: list[FeatureDistribution] = field(default_factory=list)
    logit_pi: float = 0.0  # prior log-odds

    # Copula correction for containment-cosine dependence
    copula_nu: float = 5.0       # t-copula degrees of freedom
    copula_rho_genuine: float = 0.0  # correlation under H1
    copula_rho_null: float = 0.0     # correlation under H0
    copula_enabled: bool = False

    # Feature name -> index mapping
    _name_to_idx: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._name_to_idx = {f.name: i for i, f in enumerate(self.features)}


def _fit_beta_mom(values):
    """Fit Beta(a, b) via method of moments."""
    values = np.array(values, dtype=float)
    values = np.clip(values, 1e-6, 1 - 1e-6)
    mu = np.mean(values)
    var = np.var(values) + 1e-8
    mu = np.clip(mu, 0.01, 0.99)
    common = max(mu * (1 - mu) / var - 1, 2.0)
    return float(mu * common), float((1 - mu) * common)


def _fit_gaussian(values):
    """Fit Gaussian from sample mean/std."""
    values = np.array(values, dtype=float)
    mu = float(np.mean(values)) if len(values) > 0 else 0.0
    sigma = max(float(np.std(values)), 0.01) if len(values) > 1 else 0.3
    return mu, sigma


def _estimate_copula_rho(x1, x2):
    """Estimate Spearman rank correlation for copula."""
    if len(x1) < 10:
        return 0.0
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x1, x2)
    return float(np.clip(rho, -0.95, 0.95))


def fit_multi_feature_model(
    genuine_features: dict[str, list],
    null_features: dict[str, list],
    pi_estimate: float,
    feature_configs: list[dict] = None,
) -> MultiFeatureBFModel:
    """
    Fit the multi-feature BF model from calibration data.

    genuine_features: {feature_name: [values for genuine pairs]}
    null_features:    {feature_name: [values for null pairs]}
    feature_configs:  list of {name, dist_type, weight} dicts
    """
    if feature_configs is None:
        feature_configs = [
            {'name': 'containment',        'dist_type': 'beta',     'weight': 1.0},
            {'name': 'chain_coverage',     'dist_type': 'beta',     'weight': 1.0},
            {'name': 'cosine_sim',         'dist_type': 'gaussian', 'weight': 0.5},
        ]

    fitted_features = []

    for cfg in feature_configs:
        name = cfg['name']
        dist_type = cfg.get('dist_type', 'beta')
        weight = cfg.get('weight', 1.0)

        g_vals = genuine_features.get(name, [])
        n_vals = null_features.get(name, [])

        fd = FeatureDistribution(name=name, dist_type=dist_type, weight=weight)

        if dist_type == 'beta':
            if len(g_vals) > 10:
                fd.a_genuine, fd.b_genuine = _fit_beta_mom(g_vals)
            else:
                fd.a_genuine, fd.b_genuine = 2.0, 2.0

            if len(n_vals) > 10:
                fd.a_null, fd.b_null = _fit_beta_mom(n_vals)
            else:
                fd.a_null, fd.b_null = 1.0, 10.0
        else:  # gaussian
            fd.mu_genuine, fd.sigma_genuine = _fit_gaussian(g_vals)
            fd.mu_null, fd.sigma_null = _fit_gaussian(n_vals)

        fitted_features.append(fd)

    logit_pi = float(np.log(max(pi_estimate, 0.01) / max(1 - pi_estimate, 0.01)))

    model = MultiFeatureBFModel(features=fitted_features, logit_pi=logit_pi)

    # Fit copula if containment and cosine are both present
    if 'containment' in genuine_features and 'cosine_sim' in genuine_features:
        g_cont = genuine_features['containment']
        g_cos = genuine_features['cosine_sim']
        n_cont = null_features.get('containment', [])
        n_cos = null_features.get('cosine_sim', [])

        if len(g_cont) > 20 and len(g_cos) > 20:
            model.copula_rho_genuine = _estimate_copula_rho(g_cont, g_cos)
            model.copula_rho_null = _estimate_copula_rho(n_cont, n_cos) if len(n_cont) > 20 else 0.0
            model.copula_enabled = True

    return model


def compute_feature_log_bf(value: float, feat: FeatureDistribution) -> float:
    """Compute log Bayes factor for a single feature."""
    if feat.dist_type == 'beta':
        v = np.clip(value, 1e-6, 1 - 1e-6)
        log_g = beta_dist.logpdf(v, feat.a_genuine, feat.b_genuine)
        log_n = beta_dist.logpdf(v, feat.a_null, feat.b_null)
    else:  # gaussian
        log_g = norm.logpdf(value, feat.mu_genuine, feat.sigma_genuine)
        log_n = norm.logpdf(value, feat.mu_null, feat.sigma_null)

    return float(log_g - log_n)


def _copula_correction(u1_g, u2_g, u1_n, u2_n, model: MultiFeatureBFModel) -> float:
    """
    Bivariate t-copula correction term.

    Computes log[c_genuine(u1, u2) / c_null(u1, u2)] where c is the
    copula density. This corrects the independent-BF sum for dependence
    between containment and cosine.

    u1, u2 are the CDF-transformed feature values under each hypothesis.
    """
    if not model.copula_enabled:
        return 0.0

    nu = model.copula_nu

    def _t_copula_log_density(u1, u2, rho):
        """Log density of bivariate t-copula."""
        if abs(rho) < 0.01:
            return 0.0  # near-independent, no correction

        # Quantile transform through t-distribution
        t1 = t_dist.ppf(np.clip(u1, 1e-6, 1-1e-6), nu)
        t2 = t_dist.ppf(np.clip(u2, 1e-6, 1-1e-6), nu)

        # Bivariate t-copula density (log form)
        det = 1 - rho**2
        Q = (t1**2 - 2*rho*t1*t2 + t2**2) / det
        log_c = (
            np.log(max(det, 1e-12)) * (-0.5)
            + gammaln((nu + 2) / 2) - gammaln(nu / 2)
            - np.log(nu * np.pi)
            + (-0.5) * np.log(det)
            + (-(nu + 2) / 2) * np.log(1 + Q / nu)
            - (-(nu + 1) / 2) * (np.log(1 + t1**2 / nu) + np.log(1 + t2**2 / nu))
        )
        return float(log_c)

    log_c_g = _t_copula_log_density(u1_g, u2_g, model.copula_rho_genuine)
    log_c_n = _t_copula_log_density(u1_n, u2_n, model.copula_rho_null)

    return log_c_g - log_c_n


def compute_log_bayes_factor(feature_values: dict, model: MultiFeatureBFModel) -> float:
    """
    Compute combined log Bayes factor for a candidate pair.

    feature_values: {feature_name: float}

    Returns: logit(p) = sum(w_k * log_bf_k) + copula_correction + logit(pi)
    """
    log_bf_total = 0.0

    for feat in model.features:
        if feat.name not in feature_values:
            continue
        value = feature_values[feat.name]
        log_bf = compute_feature_log_bf(value, feat)
        log_bf_total += feat.weight * log_bf

    # Copula correction between containment and cosine
    if model.copula_enabled:
        cont_val = feature_values.get('containment', 0.5)
        cos_val = feature_values.get('cosine_sim', 0.0)

        cont_feat = model.features[model._name_to_idx.get('containment', 0)]
        cos_feat = model.features[model._name_to_idx.get('cosine_sim', -1)]

        # CDF-transform under each hypothesis
        cont_v = np.clip(cont_val, 1e-6, 1-1e-6)
        u1_g = beta_dist.cdf(cont_v, cont_feat.a_genuine, cont_feat.b_genuine)
        u1_n = beta_dist.cdf(cont_v, cont_feat.a_null, cont_feat.b_null)
        u2_g = norm.cdf(cos_val, cos_feat.mu_genuine, cos_feat.sigma_genuine)
        u2_n = norm.cdf(cos_val, cos_feat.mu_null, cos_feat.sigma_null)

        correction = _copula_correction(u1_g, u2_g, u1_n, u2_n, model)
        log_bf_total += correction

    return float(log_bf_total + model.logit_pi)


def print_model_summary(model: MultiFeatureBFModel):
    """Print fitted model parameters for diagnostics."""
    print("\n=== Multi-Feature BF Model ===")
    print(f"Prior: pi={1/(1+np.exp(-model.logit_pi)):.3f} (logit={model.logit_pi:.3f})")
    print(f"Copula: enabled={model.copula_enabled}, "
          f"rho_g={model.copula_rho_genuine:.3f}, rho_n={model.copula_rho_null:.3f}")
    print()
    for feat in model.features:
        if feat.dist_type == 'beta':
            g_mean = feat.a_genuine / (feat.a_genuine + feat.b_genuine)
            n_mean = feat.a_null / (feat.a_null + feat.b_null)
            print(f"  {feat.name:25s} [w={feat.weight:.1f}] "
                  f"Beta({feat.a_genuine:.1f},{feat.b_genuine:.1f}) mean={g_mean:.3f} | "
                  f"Beta({feat.a_null:.1f},{feat.b_null:.1f}) mean={n_mean:.3f} | "
                  f"sep={g_mean-n_mean:.3f}")
        else:
            print(f"  {feat.name:25s} [w={feat.weight:.1f}] "
                  f"N({feat.mu_genuine:.3f},{feat.sigma_genuine:.3f}) | "
                  f"N({feat.mu_null:.3f},{feat.sigma_null:.3f}) | "
                  f"sep={feat.mu_genuine-feat.mu_null:.3f}")
    print()
