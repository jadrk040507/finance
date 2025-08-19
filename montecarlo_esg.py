# montecarlo_esg_full.py
import numpy as np
import streamlit as st
from scipy.stats import norm, t, beta, gamma, pareto, poisson, expon, lognorm
import matplotlib.pyplot as plt

st.set_page_config(page_title="ESG Monte Carlo Risk", layout="wide")
st.title("Monte Carlo ESG Risk Simulator")

# ---------------------- Inputs -----------------------
dist_type = st.selectbox("Base Distribution (X)", [
    "Normal", "t-Student", "LogNormal", "Poisson", "Exponential", "Gamma", "Beta"
])
n_sims = st.number_input("Number of simulations", value=10000, step=1000)

# Base distribution parameters
if dist_type == "Normal":
    mu = st.number_input("μ (mean)", value=0.0)
    sigma = st.number_input("σ (std)", value=1.0)
    X = norm.rvs(loc=mu, scale=sigma, size=n_sims)

elif dist_type == "t-Student":
    df = st.number_input("Degrees of freedom", value=4)
    X = t.rvs(df=df, size=n_sims)

elif dist_type == "LogNormal":
    mean = st.number_input("Log-mean", value=0.0)
    std  = st.number_input("Log-std", value=1.0)
    X = lognorm.rvs(s=std, scale=np.exp(mean), size=n_sims)

elif dist_type == "Poisson":
    lam = st.number_input("λ (rate)", value=2.0)
    X = poisson.rvs(mu=lam, size=n_sims)

elif dist_type == "Exponential":
    rate = st.number_input("λ (1/mean)", value=1.0)
    X = expon.rvs(scale=1/rate, size=n_sims)

elif dist_type == "Gamma":
    shape = st.number_input("k (shape)", value=2.0)
    scale = st.number_input("θ (scale)", value=1.0)
    X = gamma.rvs(a=shape, scale=scale, size=n_sims)

elif dist_type == "Beta":
    a = st.number_input("α (shape1)", value=2.0)
    b = st.number_input("β (shape2)", value=5.0)
    X = beta.rvs(a=a, b=b, size=n_sims)

# -------------- Shock settings -----------------------
st.subheader("Shock Settings")
shock_form = st.selectbox("Shock Type", [
    "Additive", "Multiplicative", "Parameter Shock"])

if shock_form != "Parameter Shock":
    shock_dist = st.selectbox("Shock Distribution", ["Normal","Pareto"])
    if shock_dist == "Normal":
        mu_s = st.number_input("Shock μ", value=0.0)
        sigma_s = st.number_input("Shock σ", value=1.0)
        S = norm.rvs(loc=mu_s, scale=sigma_s, size=n_sims)
    else:
        alpha_s = st.number_input("Pareto α", value=3.0)
        S = pareto.rvs(b=alpha_s, size=n_sims)

    if shock_form == "Additive":
        Xp = X + S
    else:  # multiplicative
        Xp = X * (1 + S)

else:
    # parametric shock
    param_pct = st.number_input(
        "Shock Size on parameter (%)", value=20.0)
    direction = st.radio("Direction", ["Increase","Decrease"])
    factor = 1 + param_pct/100 if direction == "Increase" else 1 - param_pct/100

    if dist_type == "Poisson":
        lam_new = factor*lam
        Xp = poisson.rvs(mu=lam_new, size=n_sims)
    elif dist_type == "Normal":
        mu_new = factor*mu
        Xp = norm.rvs(loc=mu_new, scale=sigma, size=n_sims)
    else:
        st.warning("Parametric shock not implemented for this dist → using additive default.")
        Xp = X         # fallback

# -------------- Risk metrics -------------------------
alpha = st.slider("Confidence level", 0.90, 0.99, 0.95)
VaR = np.quantile(Xp, alpha)
ES  = Xp[Xp >= VaR].mean()
EL  = Xp.mean()

col1, col2, col3 = st.columns(3)
col1.metric("VaR", round(VaR,3))
col2.metric("Expected Shortfall", round(ES,3))
col3.metric("Expected Loss", round(EL,3))

# --------------- Plot --------------------------------
fig,ax = plt.subplots()
ax.hist(Xp, bins=50, density=True, alpha=0.7)
ax.axvline(VaR,color='r',linestyle='--', label='VaR')
ax.set_title("Simulated Distribution")
ax.legend()
st.pyplot(fig)
