# Advanced Volatility Modeling: From Dynamic GARCH Innovation to Component Model Validation

## Executive Summary

This repository documents a comprehensive research project investigating advanced volatility modeling techniques for financial time series. The research began with an innovative hypothesis about dynamic baseline volatility in GARCH models, evolved through systematic empirical testing and diagnostic analysis, and culminated in the validation of Component GARCH models for practical risk management applications.

**Key Contributions:**
- Development and rigorous testing of novel ARMA-driven dynamic GARCH approach
- Comprehensive diagnostic framework revealing fundamental limitations of ARMA-on-residuals methods
- Systematic empirical validation of Component GARCH models through industry-standard risk management backtests
- Documentation of complete research methodology from hypothesis formation to practical validation

---

## Table of Contents

1. [Research Motivation and Objectives](#research-motivation-and-objectives)
2. [Theoretical Background](#theoretical-background)
3. [Phase 1: Dynamic GARCH Innovation](#phase-1-dynamic-garch-innovation)
4. [Phase 2: Empirical Testing and Diagnostic Analysis](#phase-2-empirical-testing-and-diagnostic-analysis)
5. [Phase 3: Component GARCH Framework](#phase-3-component-garch-framework)
6. [Phase 4: Risk Management Validation](#phase-4-risk-management-validation)
7. [Mathematical Appendix](#mathematical-appendix)
8. [Implementation Guide](#implementation-guide)
9. [Results and Conclusions](#results-and-conclusions)

---

## Research Motivation and Objectives

### The Central Question

**Can we improve volatility forecasting by making the GARCH baseline parameter dynamic rather than constant?**

### Theoretical Motivation

Standard GARCH models assume a constant long-run volatility level. However, financial markets exhibit regime changes where the underlying "normal" level of risk evolves over time. This research investigates whether explicitly modeling this time-varying baseline can improve volatility forecasts and risk management applications.

### Primary Objectives

1. **Develop** a theoretically motivated approach to dynamic baseline volatility
2. **Test** the empirical performance against standard benchmarks
3. **Diagnose** any failures to understand fundamental limitations
4. **Validate** final models through practical risk management applications

---

## Theoretical Background

### Standard GARCH(1,1) Model

The baseline GARCH(1,1) model for conditional variance is:

```
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```

**Where:**
- `σ²_t` = conditional variance at time t
- `ω` = constant intercept (baseline volatility)
- `α` = ARCH parameter (reaction to shocks)
- `β` = GARCH parameter (volatility persistence)
- `ε²_{t-1}` = squared innovation from previous period

**Long-run unconditional variance:**
```
σ²_∞ = ω / (1 - α - β)
```

**Stationarity condition:** `α + β < 1`

### Limitations of Standard GARCH

1. **Constant Long-run Variance:** The parameter ω implies a fixed baseline volatility level
2. **Regime Insensitivity:** Cannot adapt to structural changes in market conditions
3. **Static Risk Assessment:** Long-run risk level remains constant regardless of economic conditions

### Two-Pass ARMA-GARCH Methodology

Traditional approach for separating mean and variance dynamics:

**Pass 1 - Mean Equation (ARMA):**
```
R_t = μ + φ₁R_{t-1} + θ₁a_{t-1} + a_t
```

**Pass 2 - Variance Equation (GARCH on residuals):**
```
σ²_t = ω + α a²_{t-1} + β σ²_{t-1}
```

Where `a_t` are the residuals from the mean equation.

---

## Phase 1: Dynamic GARCH Innovation

### 1.1 Core Hypothesis

**Hypothesis:** The GARCH intercept ω should be time-varying and driven by a separate ARMA process modeling the dynamics of squared residuals.

### 1.2 The Three-Stage ARMA-Driven Dynamic GARCH Model

#### Stage 1: Mean Equation
```
R_t = μ + φ₁R_{t-1} + θ₁a_{t-1} + a_t
```

**Purpose:** Capture serial correlation in returns and extract residuals `a_t`

#### Stage 2: Variance Innovation Equation
```
log(a²_t + c) = ω_A + α_A log(a²_{t-1} + c) + β_A v_{t-1} + v_t
```

**Key Features:**
- **Log transformation:** Ensures positivity and stabilizes variance
- **Small constant c:** Prevents log(0) numerical issues
- **ARMA(1,1) structure:** Captures complex patterns in variance evolution

**Theoretical Justification:**
- Models the time-varying component of baseline volatility
- Captures long-memory effects in volatility clustering
- Provides one-step-ahead forecasts for dynamic intercept

#### Stage 3: Dynamic GARCH Equation
```
ω_t = E_{t-1}[exp(ω_A + α_A log(a²_{t-1} + c) + β_A v_{t-1}) - c]
```

```
σ²_t = ω_t + α_G a²_{t-1} + β_G σ²_{t-1}
```

**Theoretical Appeal:**
- **Adaptive baseline:** ω_t changes with market conditions
- **Regime sensitivity:** Model automatically adjusts to different risk environments
- **Unified framework:** Combines ARMA pattern recognition with GARCH efficiency

### 1.3 Mathematical Properties

#### Long-run Variance Consistency

For model consistency, both approaches should yield the same long-run variance:

**ARMA long-run variance:**
```
σ²_∞(ARMA) = ω_A / (1 - α_A)
```

**GARCH long-run variance:**
```
σ²_∞(GARCH) = E[ω_t] / (1 - α_G - β_G)
```

**Linking condition:**
```
ω_A / (1 - α_A) = E[ω_t] / (1 - α_G - β_G)
```

#### Bias Correction for Log Transformation

**Jensen's Inequality Problem:**
```
E[exp(X)] ≠ exp(E[X])
```

**Bias correction formula:**
```
E[exp(X)] = exp(E[X] + Var(X)/2)
```

Applied to our case:
```
ω_t = exp(ω_A + α_A log(a²_{t-1} + c) + β_A v_{t-1} + σ²_v/2) - c
```

---

## Phase 2: Empirical Testing and Diagnostic Analysis

### 2.1 Backtesting Methodology

#### Performance Metrics

**1. Root Mean Square Error (RMSE):**
```
RMSE = sqrt(1/T × Σ(σ²_{t,realized} - σ²_{t,forecast})²)
```

**2. Diebold-Mariano Test for Forecast Accuracy:**
```
H₀: E[d_t] = 0
d_t = L(e_{1,t}) - L(e_{2,t})
```
Where `L(e_{i,t})` is the loss function for model i.

**3. Mean Absolute Error (MAE):**
```
MAE = 1/T × Σ|σ²_{t,realized} - σ²_{t,forecast}|
```

#### Out-of-Sample Testing Framework

- **Training Period:** First 80% of sample
- **Testing Period:** Remaining 20% of sample
- **Rolling Window:** 1000-observation estimation window
- **Forecast Horizons:** 1-day and 10-day ahead
- **Volatility Proxy:** Daily squared returns (a²_t) are used as the proxy for realized variance, a standard approach in the GARCH literature.

### 2.2 Empirical Findings: The Failure

#### Performance Results

**Diebold-Mariano Test Results:**
- **1-day forecasts:** p-value > 0.70
- **10-day forecasts:** p-value > 0.65
- **Interpretation:** No statistically significant difference from standard GARCH

**RMSE Comparison:**
- **Dynamic GARCH:** Marginally worse than standard GARCH
- **Conclusion:** The added complexity provided no forecasting benefit. While standard for capturing mean and variance dynamics separately, this sequential approach can lead to bias accumulation and is not designed to decompose the variance into its underlying frequency components.

  Systematic demonstration of the theoretical and empirical failure of a novel ARMA-driven dynamic GARCH approach
  
### 2.3 Diagnostic Analysis: Uncovering the Fundamental Flaw

#### Critical Diagnostic Discoveries

**1. Massive Scale Misalignment**
```
Mean(ω_t) ≈ 30 × ω_benchmark
```
**Translation:** The dynamic intercept was operating on a completely different scale

**2. The "Spikiness" Problem**

**Expected behavior:** ω_t should be smooth, slow-moving baseline
**Actual behavior:** ω_t was highly volatile, resembling daily volatility patterns

**Diagnostic plot reveals:**
- High correlation between ω_t and daily volatility
- Frequent dramatic spikes in baseline level
- No clear trend or regime-based evolution

**3. Double-Counting of Short-term Dynamics**

**The Core Problem:** ARMA on a²_t captures the same short-term volatility clustering that GARCH is designed to model.

**Mathematical Explanation:**
```
a²_t = σ²_t × z²_t
```
Where `z_t ~ iid(0,1)`. When we model a²_t with ARMA, we're modeling:
```
σ²_t × z²_t = ω_A + α_A (σ²_{t-1} × z²_{t-1}) + β_A v_{t-1} + v_t
```

This creates confounding between the true volatility dynamics (σ²_t) and the random shocks (z²_t).

#### Theoretical Diagnosis: Why ARMA-on-Residuals Fails

**Fundamental Issue:** Inability to decompose variance into frequency components

**The residuals a²_t contain:**
- High-frequency noise (z²_t effects)
- Medium-frequency volatility clustering
- Low-frequency regime changes

**ARMA models all three together, cannot isolate the baseline component we need.**

---

## Phase 3: Component GARCH Framework

### 3.1 Theoretical Pivot

Based on diagnostic findings, research pivoted to Component GARCH (CGARCH) model of Engle and Lee (1999), which explicitly addresses the dynamic baseline volatility problem through proper mathematical decomposition.

### 3.2 Component GARCH Model Specification

#### Variance Decomposition

**Total conditional variance:**
```
σ²_t = q_t + (σ²_t - q_t)
```

Where:
- `q_t` = permanent component (time-varying long-run variance)
- `(σ²_t - q_t)` = transitory component (short-run deviations)

#### Long-run Component Equation
```
q_t = ω + ρ(q_{t-1} - ω) + φ(a²_{t-1} - σ²_{t-1})
```

**Parameter Interpretations:**
- `ω` = long-run average variance level
- `ρ` = persistence of long-run component (typically ρ ≈ 0.99)
- `φ` = adjustment speed based on variance surprises
- `(a²_{t-1} - σ²_{t-1})` = variance forecast error (surprise component)

#### Short-run Component Equation
```
σ²_t = q_t + α(a²_{t-1} - q_{t-1}) + β(σ²_{t-1} - q_{t-1})
```

**Parameter Interpretations:**
- `α` = reaction to transitory shocks
- `β` = persistence of transitory component
- Mean reversion target is `q_t` (time-varying), not constant ω

### 3.3 Why CGARCH Solves the Original Problem

#### 1. Proper Frequency Decomposition
- **High ρ (≈0.99):** Forces q_t to be smooth, slow-moving
- **Low φ:** Gradual adjustment to regime changes
- **Solves spikiness problem** from original approach

#### 2. Correct Scale Relationships
- **Joint MLE estimation:** Ensures all parameters are on consistent scales
- **No calibration failures** like the ARMA approach

#### 3. Theoretical Coherence
- **Unified framework:** All dynamics estimated simultaneously
- **Economically interpretable:** Clear separation of short-run vs. long-run effects

### 3.4 Relationship to Original Dynamic ω Concept

The CGARCH model achieves the original research objective through mathematically superior means:

**Original concept:** σ²_t = ω_t + α ε²_{t-1} + β σ²_{t-1}
**CGARCH realization:** σ²_t = q_t + α(a²_{t-1} - q_{t-1}) + β(σ²_{t-1} - q_{t-1})

**Key insight:** q_t serves as the dynamic baseline, but with proper mathematical constraints ensuring smooth evolution.

---

## Phase 4: Risk Management Validation

### 4.1 Value-at-Risk (VaR) Backtesting Framework

#### VaR Model Specification

**1% VaR at time t:**
```
VaR_{t,0.01} = μ_t + σ_t × F^{-1}_{0.01}
```

Where:
- `μ_t` = conditional mean forecast
- `σ_t` = conditional volatility forecast  
- `F^{-1}_{0.01}` = 1st percentile of assumed error distribution

#### Distribution Assumptions Tested

**1. Normal Distribution:**
```
z_t ~ N(0,1)
F^{-1}_{0.01} = Φ^{-1}(0.01) ≈ -2.33
```

**2. Student's t-Distribution:**
```
z_t ~ t_ν (standardized)
F^{-1}_{0.01} = t^{-1}_{ν,0.01}
```
Where ν is the degrees of freedom parameter estimated via MLE.

#### Backtesting Statistical Tests

**1. Kupiec Test (Unconditional Coverage):**
```
H₀: p = p* (VaR breaches occur at expected rate)
LR_{UC} = -2 log(L_R / L_U) ~ χ²(1)
```

Where:
- `p*` = expected breach rate (0.01 for 1% VaR)
- `p` = observed breach rate
- `L_R` = likelihood under H₀
- `L_U` = unrestricted likelihood

**2. Christoffersen Test (Independence):**
```
H₀: VaR breaches are independent
LR_{IND} = -2 log(L_IND / L_U) ~ χ²(1)
```

Tests whether VaR breaches cluster together (bad) or occur independently (good).

**3. Joint Test (Conditional Coverage):**
```
LR_{CC} = LR_{UC} + LR_{IND} ~ χ²(2)
```

### 4.2 Empirical Results

#### Initial Results with Normal Distribution

**Standard GARCH-Normal:**
- **Actual breaches:** ~2.1% (expected: 1.0%)
- **Kupiec test:** p-value < 0.01 (FAIL)
- **Diagnosis:** Severe underestimation of tail risk

**CGARCH-Normal:**
- **Actual breaches:** ~2.0% (expected: 1.0%)  
- **Kupiec test:** p-value < 0.01 (FAIL)
- **Diagnosis:** Distribution assumption is the primary issue

#### Final Results with Student's t-Distribution

**Standard GARCH-t:**
- **Kupiec test:** p-value = 0.0296 (FAIL at 5% level)
- **Christoffersen test:** p-value = 0.1245 (PASS)
- **Conclusion:** While it passes the independence test, it fails the crucial unconditional coverage test, indicating it is not a reliable risk model as it produces too many breaches.

**Component GARCH-t:**
- **Kupiec test:** p-value = 0.1553 (PASS)
- **Christoffersen test:** p-value = 0.1833 (PASS)
- **Joint test:** p-value = 0.2741 (PASS)
- **Conclusion:** Passes all industry-standard risk management tests

---

## Mathematical Appendix

### A.1 Complete Model Specifications

#### Failed Dynamic GARCH Model

**Stage 1 - Mean Equation:**
```
R_t = μ + φ₁R_{t-1} + θ₁a_{t-1} + a_t
```

**Stage 2 - Log-Variance ARMA:**
```
log(a²_t + c) = ω_A + α_A log(a²_{t-1} + c) + β_A v_{t-1} + v_t
```

**Stage 3 - Dynamic GARCH:**
```
ω_t = exp(ω_A + α_A log(a²_{t-1} + c) + β_A v_{t-1} + σ²_v/2) - c
σ²_t = ω_t + α_G a²_{t-1} + β_G σ²_{t-1}
```

#### Successful Component GARCH Model

**Long-run Component:**
```
q_t = ω + ρ(q_{t-1} - ω) + φ(a²_{t-1} - σ²_{t-1})
```

**Short-run Component:**
```
σ²_t = q_t + α(a²_{t-1} - q_{t-1}) + β(σ²_{t-1} - q_{t-1})
```

**Distribution Specification:**
```
a_t | F_{t-1} ~ t_ν(0, σ²_t)
```

### A.2 Estimation Methodology

#### Maximum Likelihood Estimation

**Log-likelihood function for CGARCH-t:**
```
L(θ) = Σ_{t=1}^T [log Γ((ν+1)/2) - log Γ(ν/2) - (1/2)log(π(ν-2)) 
       - (1/2)log(σ²_t) - ((ν+1)/2)log(1 + a²_t/(σ²_t(ν-2)))]
```

Where `θ = {ω, ρ, φ, α, β, ν}` is the parameter vector.

#### Parameter Constraints

**Stationarity conditions:**
- `0 < ρ < 1` (long-run component persistence)
- `α + β < 1` (short-run component stationarity)
- `α, β ≥ 0` (non-negativity)
- `ν > 2` (finite variance for t-distribution)

### A.3 Static Linking Formula Derivation

#### Variance Targeting Approach

**Step 1:** Extract long-run variance from ARMA model
```
σ²_∞(ARMA) = ω_A / (1 - α_A)
```

**Step 2:** Set equal to GARCH long-run variance
```
ω_A / (1 - α_A) = ω_G / (1 - α_G - β_G)
```

**Step 3:** Solve for linked constant
```
ω_linked = ω_A × [(1 - α_G - β_G) / (1 - α_A)]
```

**Step 4:** Implement hybrid model
```
σ²_t = ω_linked + α_G ε²_{t-1} + β_G σ²_{t-1}
```

#### Alternative Dynamic Linking

**Time-varying intercept:**
```
ω_t = ω̂_A + δ(a²_{t-1} - σ²_{t-1})
```

**Interpretation:** Baseline adjusts based on forecast errors from previous period.

---

## Implementation Guide

### Software Requirements

- **R Version:** 4.0+ 
- **Key Packages:**
  - `rugarch` for GARCH model estimation
  - `forecast` for ARMA model fitting
  - `rugarch` for backtesting procedures
  - `PerformanceAnalytics` for risk metrics

### Implementation Steps

#### Step 1: Data Preparation
```r
# Load and clean return data
returns <- diff(log(prices))
returns <- returns[!is.na(returns)]

# Basic descriptive statistics
summary(returns)
jarque.bera.test(returns)  # Test for normality
```

#### Step 2: Standard GARCH Benchmark
```r
# GARCH(1,1) specification
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(1,1)),
  distribution.model = "std"
)

# Estimation
garch_fit <- ugarchfit(garch_spec, returns)
```

#### Step 3: Component GARCH Implementation
```r
# CGARCH specification
cgarch_spec <- ugarchspec(
  variance.model = list(model = "csGARCH"),
  mean.model = list(armaOrder = c(1,1)),
  distribution.model = "std"
)

# Estimation
cgarch_fit <- ugarchfit(cgarch_spec, returns)
```

#### Step 4: VaR Backtesting
```r
# Generate VaR forecasts
var_forecasts <- quantile(fitted_model, probs = 0.01)

# Backtest results
kupiec_test <- VaRTest(returns, var_forecasts, alpha = 0.01)
christoffersen_test <- DurTest(returns, var_forecasts, alpha = 0.01)
```

---

## Results and Conclusions

### Key Empirical Findings

#### 1. Dynamic GARCH Approach Failure
- **Performance:** No improvement over standard GARCH
- **Root Cause:** ARMA-on-residuals cannot properly decompose variance frequencies
- **Scale Issues:** Dynamic intercept operated on wrong magnitude
- **Theoretical Flaw:** Double-counting of short-term volatility dynamics

#### 2. Component GARCH Success
- **Volatility Modeling:** Superior handling of regime changes through q_t component
- **Risk Management:** Only model to pass comprehensive VaR backtests
- **Practical Validation:** Meets industry standards for risk model validation

#### 3. Distribution Assumption Criticality
- **Normal Distribution:** Both models fail VaR tests
- **Student's t:** Only CGARCH-t passes all tests
- **Insight:** Model structure AND distribution assumption both required for success

### Theoretical Contributions

#### 1. Systematic Documentation of ARMA-on-Residuals Failure
**Contribution:** Detailed analysis of why an intuitive approach fails
**Value:** Prevents future researchers from pursuing this dead end
**Insight:** Proper frequency decomposition requires joint estimation, not sequential approaches

#### 2. Comprehensive Validation Framework
**Methodology:** Complete pipeline from hypothesis to practical validation
**Testing:** Multiple metrics including industry-standard risk management backtests
**Rigor:** Both statistical significance and practical relevance assessed

#### 3. Two-Pillar Framework for Volatility Models
**Pillar 1:** Correct volatility dynamics (structure)
**Pillar 2:** Correct distributional assumptions
**Evidence:** Neither alone is sufficient for practical applications

### Practical Implications

#### For Risk Managers
- **Standard GARCH insufficient** for regulatory VaR requirements
- **Component models necessary** for regime-adaptive risk measurement
- **Fat-tailed distributions essential** for tail risk assessment

#### For Researchers  
- **Joint estimation preferred** over multi-stage approaches for volatility modeling
- **Diagnostic analysis crucial** for identifying model misspecification
- **Practical validation necessary** beyond pure statistical metrics

### Future Research Directions

#### 1. Alternative Dynamic Baseline Approaches
- **GARCH-MIDAS:** Incorporate macroeconomic variables in long-run component
- **Regime-Switching:** Explicit modeling of discrete regime changes
- **Machine Learning:** Neural network approaches to time-varying parameters

#### 2. Multi-Asset Extensions
- **DCC-CGARCH:** Dynamic correlations with component variance structure
- **Portfolio Applications:** Optimal portfolio construction with component models

#### 3. High-Frequency Extensions
- **Realized Volatility:** Incorporate intraday information in component models
- **Jump Detection:** Separate jump and diffusive components

---

## Repository Structure


## Acknowledgments and References

### Theoretical Foundations
- **Engle & Lee (1999):** "A Long-Run and Short-Run Component Model of Stock Return Volatility"
- **Bollerslev (1986):** "Generalized Autoregressive Conditional Heteroskedasticity"
- **Kupiec (1995):** "Techniques for Verifying the Accuracy of Risk Measurement Models"

### Methodological Contributions
- **Diebold & Mariano (1995):** Forecast accuracy testing framework
- **Christoffersen (1998):** VaR backtesting procedures
- **Hansen & Lunde (2005):** Volatility model comparison techniques

---

## License and Usage

This research is provided for educational and research purposes. All code is available under MIT license. Please cite this repository if you use any components in your own research.

**Citation:**
```
Sunil S. (2025). Advanced Volatility Modeling: From Dynamic GARCH Innovation 
to Component Model Validation. GitHub Repository. 
https://github.com/Shreyas70773/volatility-research-project
```

---

## Contact and Collaboration

For questions about the methodology, implementation details, or potential collaborations, please open an issue in this repository or contact [your contact information].

**Keywords:** Volatility Modeling, GARCH, Component GARCH, Risk Management, VaR Backtesting, Financial Econometrics, Time Series Analysis
