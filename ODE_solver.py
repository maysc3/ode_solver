import streamlit as st
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os

def run_desmos_server():
    os.chdir("desmos_view")  # Set to where your index.html lives
    with TCPServer(("", 5500), SimpleHTTPRequestHandler) as httpd:
        httpd.serve_forever()

    try:
        threading.Thread(target=run_desmos_server, daemon=True).start()
    except OSError:
        pass  # Likely already running
    
    # Open browser tab
    webbrowser.open_new_tab("http://localhost:5500/index.html")


import numpy as np
from scipy.special import gamma as scipy_gamma 
import plotly.graph_objects as go
from scipy.integrate import quad
import re



# --- Utility functions ---
def preprocess_expr(expr):
    expr = re.sub(r'(?<=[0-9\)])(?=[a-zA-Z(])', '*', expr)  # 2x -> 2*x
    expr = expr.replace('^', '**')
    expr = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', expr)  # .5 -> 0.5
    return expr

def delta_approx(t, center=0.1, width=1e-3):
    return (np.abs(t - center) < width) / width

# --- Input controls ---
st.set_page_config(layout="wide")
st.title("Second-Order Linear ODE Solver")

graph_col, control_col, func_col = st.columns([5, 2, 2])
with control_col:
    st.markdown("### Initial Conditions and Coefficients")
    def dual_input(label, default, min_val, max_val, step=0.1, key=""):
        col1, col2 = st.columns([3, 1])
        with col1:
            slider_val = st.slider(label, min_value=min_val, max_value=max_val, value=default, step=step, key=f"slider_{key}")
        with col2:
            typed_val = st.number_input(f"Exact {label}", value=slider_val, step=step, key=f"num_{key}")
        return typed_val if typed_val != slider_val else slider_val

    y0 = dual_input("y₀", 1.0, -10.0, 10.0, key="y0")
    y0p = dual_input("y₀'", 1.0, -10.0, 10.0, key="y0p")
    b = dual_input("b (for y')", -1.0, -5.0, 5.0, key="b")
    c = dual_input("c (for y)", -2.0, -5.0, 5.0, key="c")

with func_col:
    st.markdown("### Forcing Functions")
    x_expr = st.text_input("Input Forcing Function x(t)", "sin(t)")
    compare_expr = st.text_input("Comparison Function (Optional)", "cos(t)")
    show_compare = st.toggle("Show Comparison Function", value=True)

    # --- Time domain settings ---
    t_min = st.number_input("Start time", value=0.0)
    t_max = st.number_input("End time", value=20.0)
    t_steps = st.number_input("Number of time steps", value=400, step=50)

t_vals = np.linspace(t_min, t_max, int(t_steps))

# --- Evaluate x(t) and compare(t) ---
try:
    env = {
        "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "sqrt": np.sqrt, "t": t_vals, "pi": np.pi, "e": np.e,
        "gamma": scipy_gamma,
        "heaviside": lambda t: np.heaviside(t, 1.0),
        "delta": lambda t: delta_approx(t)
    }
    x_func = eval(preprocess_expr(x_expr), env)
except Exception as e:
    st.error(f"Error in x(t): {e}")
    st.stop()

try:
    compare_func = eval(preprocess_expr(compare_expr), env) if show_compare else None
except Exception as e:
    st.warning(f"Compare function error: {e}")
    compare_func = None

# --- Solve the ODE ---
a = 1  # Coefficient of y''
roots = np.roots([a, b, c])

# Impulse response h(t)
def h(t):
    if np.isreal(roots[0]) and np.isreal(roots[1]) and not np.isclose(roots[0], roots[1]):
        r1, r2 = roots
        return (np.exp(r1 * t) - np.exp(r2 * t)) / (r1 - r2)
    elif np.isclose(roots[0], roots[1]):
        r = roots[0].real
        return t * np.exp(r * t) * (t >= 0)
    else:
        lam = roots[0].real
        mu = roots[0].imag
        return (np.exp(lam * t) * np.sin(mu * t)) / mu * (t >= 0)

# Homogeneous solution
t = t_vals
if np.isreal(roots[0]) and np.isreal(roots[1]) and not np.isclose(roots[0], roots[1]):
    r1, r2 = roots
    A = y0-(a * y0p + (a * r2 + b) * y0) / (r2 - r1)
    B = (a * y0p + (a * r2 + b) * y0) / (r2 - r1)
    y_hom = A * np.exp(r1 * t_vals) + B * np.exp(r2 * t_vals)
elif np.isclose(roots[0], roots[1]):
    r = roots[0].real
    A = y0
    B = y0p + r * y0 + b*y0
    y_hom = (A + B * t_vals) * np.exp(r * t_vals)
else:
    lam = roots[0].real
    mu = roots[0].imag
    A = y0
    B = (y0p + b * y0)
    y_hom = (A * np.exp(lam * t_vals)) * (np.cos(mu * t_vals) + (lam * np.sin(mu * t_vals))/mu) + (B * np.exp(lam * t_vals) * np.sin(mu * t_vals))/mu

# Particular solution by convolution
y_particular = np.zeros_like(t_vals)
for i, t_i in enumerate(t_vals):
    integrand = lambda tau: h(t_i - tau) * eval(preprocess_expr(x_expr), {
        "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "t": tau, "sqrt": np.sqrt, "pi": np.pi, "e": np.e,
        "gamma": scipy_gamma,
        "heaviside": lambda t: np.heaviside(t, 1.0),
        "delta": lambda t: delta_approx(t)
    })
    try:
        y_particular[i], _ = quad(integrand, 0, t_i)
    except:
        y_particular[i] = np.nan

# Total solution
y_total = y_hom + y_particular

# --- Plotly Graph ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_vals, y=y_total, mode='lines', name='y(t)', line=dict(width=3)))
fig.add_trace(go.Scatter(x=t_vals, y=x_func, mode='lines', name='x(t)', line=dict(dash='dash')))
if show_compare and compare_func is not None:
    fig.add_trace(go.Scatter(x=t_vals, y=compare_func, mode='lines', name='Compare', line=dict(dash='dot')))

fig.update_layout(title="ODE Solution", xaxis_title="t", yaxis_title="y(t)",
                  template="plotly_white", height=600)
with graph_col:
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.caption("Supports sin, cos, exp, sqrt, pi, e, gamma(x), heaviside(t), delta(t), etc.")

import json
import webbrowser

if st.button("View in Desmos"):
    export_data = {
        "t": t_vals.tolist(),
        "y": y_total.tolist(),
        "x": x_func.tolist(),
    }
    with open("desmos_view/data/solution_data.json", "w") as f:
        json.dump(export_data, f)
    st.success("Exported to solution_data.json")



    # Open the HTML viewer (served by a local server like python -m http.server)
    viewer_url = "http://localhost:5500/desmos_view/index.html"
    try:
        webbrowser.open_new_tab(viewer_url)
        st.info("Opened Desmos viewer in new tab.")
    except Exception as e:
        st.warning(f"Failed to open browser tab. Please open manually: {viewer_url}")
