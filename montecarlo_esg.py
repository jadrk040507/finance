import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, poisson, randint, beta, norm, lognorm, expon, gamma, triang, uniform, skew, kurtosis

# Funciones
def frequency_function(frequency_type, params, size):
    if frequency_type == 'bernoulli':
        return bernoulli.rvs(p=params['p'], size=size)
    elif frequency_type == 'binomial':
        return binom.rvs(n=params['n'], p=params['p'], size=size)
    elif frequency_type == 'poisson':
        return poisson.rvs(mu=params['mu'], size=size)
    elif frequency_type == 'uniform_int':
        return randint.rvs(low=params['low'], high=params['high'], size=size)
    else:
        raise ValueError("Unsupported frequency function")

def pert_rvs(a, b, c, lambd, size):
    mu = (a + lambd * b + c) / (lambd + 2)
    sigma = (c - a) / (lambd + 2) * np.sqrt(1 / (lambd + 3))
    alpha = ((mu - a) * (2 * c - a - mu)) / ((c - mu) * (mu - a))
    beta_ = alpha * (c - mu) / (mu - a)
    return beta.rvs(alpha, beta_, loc=a, scale=c - a, size=size)

def pert_percentile_rvs(a, b, c, p, size):
    alpha = 1 + p * (b - a) / (c - a)
    beta_ = 1 + p * (c - b) / (c - a)
    return beta.rvs(alpha, beta_, loc=a, scale=c - a, size=size)

def triangular_perc_rvs(a, b, c, p, size):
    c_rel = (b - a) / (c - a)
    return triang.rvs(c_rel, loc=a, scale=c - a, size=size)

def severity_function(severity_type, params, size):
    if severity_type == 'normal':
        return norm.rvs(loc=params['loc'], scale=params['scale'], size=size)
    elif severity_type == 'lognormal':
        sigma = np.log(1 + (params['sigma'] / params['mu'])**2)**0.5
        mu = np.log(params['mu']) - 0.5 * sigma**2
        return lognorm.rvs(s=sigma, scale=np.exp(mu), size=size)
    elif severity_type == 'exponential':
        return expon.rvs(scale=params['scale'], size=size)
    elif severity_type == 'gamma':
        return gamma.rvs(a=params['a'], scale=params['scale'], size=size)
    elif severity_type == 'triangular':
        return triang.rvs(c=params['c'], loc=params['loc'], scale=params['scale'], size=size)
    elif severity_type == 'uniform':
        return uniform.rvs(loc=params['loc'], scale=params['scale'], size=size)
    elif severity_type == 'pert':
        return pert_rvs(params['a'], params['b'], params['c'], params['lambda'], size=size)
    elif severity_type == 'pert_percentile':
        return pert_percentile_rvs(params['a'], params['b'], params['c'], params['p'], size=size)
    elif severity_type == 'triangular_perc':
        return triangular_perc_rvs(params['a'], params['b'], params['c'], params['p'], size=size)
    else:
        raise ValueError("Unsupported severity function")

def montecarlo_simulator(frequency_type, frequency_params, severity_type, severity_params, percentile, iterations):
    losses = []
    records = []
    for _ in range(iterations):
        frequency = frequency_function(frequency_type, frequency_params, 1)[0]
        severity = severity_function(severity_type, severity_params, frequency)
        total_loss = np.sum(severity)
        losses.append(total_loss)
        record = {'frequency': frequency}
        record.update({f"sev_{k}": v for k, v in severity_params.items()})
        records.append(record)
    df = pd.DataFrame(records)
    df['loss'] = losses
    losses_array = np.array(losses)
    stats = {
        'mean_loss': np.mean(losses_array),
        'std_loss': np.std(losses_array),
        'var_loss': np.percentile(losses_array, percentile * 100),
        'unexpected_loss': np.percentile(losses_array, percentile * 100) - np.mean(losses_array),
        'skewness': skew(losses_array),
        'kurtosis': kurtosis(losses_array)
    }
    return df, stats

# Streamlit app
st.title("Simulador Montecarlo para Valuación de Riesgos")

# Inputs Frecuencia
st.sidebar.header("Frecuencia")
freq_type = st.sidebar.selectbox("Tipo de función de frecuencia", ["bernoulli", "binomial", "poisson", "uniform_int"])
freq_params = {}
if freq_type == 'bernoulli':
    freq_params['p'] = st.sidebar.number_input("p (probabilidad de éxito, entre 0 y 1)", min_value=0.0, max_value=1.0, value=0.5)
elif freq_type == 'binomial':
    freq_params['n'] = st.sidebar.number_input("n (número de ensayos)", min_value=1, value=10)
    freq_params['p'] = st.sidebar.number_input("p (probabilidad de éxito, entre 0 y 1)", min_value=0.0, max_value=1.0, value=0.5)
elif freq_type == 'poisson':
    freq_params['mu'] = st.sidebar.number_input("mu (tasa media de eventos)", min_value=0.0, value=5.0)
elif freq_type == 'uniform_int':
    freq_params['low'] = st.sidebar.number_input("low (mínimo, incluido)", value=1)
    freq_params['high'] = st.sidebar.number_input("high (máximo, excluido)", value=10)

# Inputs Severidad
st.sidebar.header("Severidad")
sev_type = st.sidebar.selectbox("Tipo de función de severidad", ["normal", "lognormal", "exponential", "gamma", "triangular", "uniform", "pert", "pert_percentile", "triangular_perc"])
sev_params = {}
if sev_type == 'normal':
    sev_params['loc'] = st.sidebar.number_input("loc (media)", value=0.0)
    sev_params['scale'] = st.sidebar.number_input("scale (desviación estándar)", min_value=0.0, value=1.0)
elif sev_type == 'lognormal':
    sev_params['mu'] = st.sidebar.number_input("mu (media geométrica)", min_value=0.1, value=1000.0)
    sev_params['sigma'] = st.sidebar.number_input("sigma (desviación estándar geométrica)", min_value=0.0, value=0.25)
elif sev_type == 'exponential':
    sev_params['scale'] = st.sidebar.number_input("scale (media)", min_value=0.1, value=1.0)
elif sev_type == 'gamma':
    sev_params['a'] = st.sidebar.number_input("a (parámetro de forma)", min_value=0.1, value=2.0)
    sev_params['scale'] = st.sidebar.number_input("scale (parámetro de escala)", min_value=0.1, value=1.0)
elif sev_type == 'triangular':
    sev_params['c'] = st.sidebar.number_input("c (modo relativo entre 0 y 1)", min_value=0.0, max_value=1.0, value=0.5)
    sev_params['loc'] = st.sidebar.number_input("loc (valor mínimo)", value=0.0)
    sev_params['scale'] = st.sidebar.number_input("scale (rango: máximo - mínimo)", min_value=0.1, value=1.0)
elif sev_type == 'uniform':
    sev_params['loc'] = st.sidebar.number_input("loc (mínimo)", value=0.0)
    sev_params['scale'] = st.sidebar.number_input("scale (rango: máximo - mínimo)", min_value=0.1, value=1.0)
elif sev_type in ['pert', 'pert_percentile', 'triangular_perc']:
    sev_params['a'] = st.sidebar.number_input("a (mínimo)", value=0.0)
    sev_params['b'] = st.sidebar.number_input("b (más probable)", value=1.0)
    sev_params['c'] = st.sidebar.number_input("c (máximo)", value=2.0)
    if sev_type == 'pert':
        sev_params['lambda'] = st.sidebar.number_input("lambda (parámetro de forma, usualmente 4)", min_value=0.0, value=4.0)
    else:
        sev_params['p'] = st.sidebar.number_input("p (percentil, entre 0 y 1)", min_value=0.0, max_value=1.0, value=0.5)

percentile = st.sidebar.slider("Percentil VaR (ej. 0.95 para 95%)", min_value=0.0, max_value=1.0, value=0.95)
iterations = st.sidebar.number_input("Número de iteraciones", min_value=100, value=1000)

# Simulación y Gráficas
if st.button("Ejecutar Simulación"):
    df, stats = montecarlo_simulator(freq_type, freq_params, sev_type, sev_params, percentile, iterations)

    st.subheader("Resultados estadísticos del modelo")
    st.write(f"Media (pérdida esperada): {stats['mean_loss']:,.2f}")
    st.write(f"Desviación estándar: {stats['std_loss']:,.2f}")
    st.write(f"VaR (percentil seleccionado): {stats['var_loss']:,.2f}")
    st.write(f"Pérdida no esperada: {stats['unexpected_loss']:,.2f}")
    st.write(f"Asimetría: {stats['skewness']:.4f}")
    st.write(f"Curtosis: {stats['kurtosis']:.4f}")

    # Histograma
    fig, ax = plt.subplots()
    ax.hist(df['loss'], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax.axvline(stats['mean_loss'], color='red', linestyle='--', linewidth=2, label=f"Media: {stats['mean_loss']:,.2f}")
    ax.axvline(stats['var_loss'], color='green', linestyle='--', linewidth=2, label=f"VaR: {stats['var_loss']:,.2f}")
    ax.axvline(stats['mean_loss'] + stats['unexpected_loss'], color='purple', linestyle='--', linewidth=2, label=f"Pérdida no esperada")
    ax.set_xlabel("Pérdida")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    st.pyplot(fig)

    # Gráfica S
    sorted_losses = np.sort(df['loss'])
    percentiles_line = np.linspace(0, 100, len(sorted_losses))
    fig_s, ax_s = plt.subplots()
    ax_s.plot(sorted_losses, percentiles_line, color='dodgerblue', linewidth=2)
    ax_s.axvline(stats['var_loss'], color='green', linestyle='--', label=f"VaR: {stats['var_loss']:,.2f}")
    ax_s.axvline(stats['mean_loss'], color='red', linestyle='--', label=f"Media: {stats['mean_loss']:,.2f}")
    ax_s.set_xlabel("Pérdida")
    ax_s.set_ylabel("Percentil")
    ax_s.legend()
    st.pyplot(fig_s)

    # Tornado correlación
    corr = df.corr()['loss'].drop('loss').sort_values()
    fig_corr, ax_corr = plt.subplots()
    corr.plot.barh(ax=ax_corr, color='coral')
    ax_corr.set_xlabel("Correlación")
    st.pyplot(fig_corr)

    # Tornado sensibilidad
    ranges = {}
    for col in df.columns.drop('loss'):
        p5 = df.groupby(col)['loss'].quantile(0.05).mean()
        p95 = df.groupby(col)['loss'].quantile(0.95).mean()
        ranges[col] = p95 - p5
    sens_df = pd.Series(ranges).sort_values()
    fig_sens, ax_sens = plt.subplots()
    sens_df.plot.barh(ax=ax_sens, color='teal')
    ax_sens.set_xlabel("Rango (p95 - p5)")
    st.pyplot(fig_sens)

    # Scatter plot
    xcol = st.selectbox("Selecciona input para X", df.columns.drop('loss'))
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df[xcol], df['loss'], alpha=0.5, color='mediumvioletred')
    ax_scatter.set_xlabel(xcol)
    ax_scatter.set_ylabel("Pérdida")
    st.pyplot(fig_scatter)
