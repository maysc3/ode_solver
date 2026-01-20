import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy.abc import t 
import re
import math
from sympy import (
    symbols, sin, cos, tan, exp, sqrt, Heaviside, gamma, integrate, simplify,
    sympify, latex, Poly, roots, lambdify, solve, Rational, expand, trigsimp, fraction,
    lcm, denom, numer, Add, cancel, expand_mul,
    together, nsimplify, assuming, Q, laplace_transform, inverse_laplace_transform, powsimp, apart, collect,
    DiracDelta
)

# --- Utility functions ---
def preprocess_expr(expr):
    expr = re.sub(r'(?<=[0-9\)])(?=[a-zA-Z(])', '*', expr)
    expr = re.sub(r'e\^\((.*?)\)', r'exp(\1)', expr)
    expr = re.sub(r'e\^(\w+)', r'exp(\1)', expr)
    expr = expr.replace('^', '**')
    expr = re.sub(r'(?<!\d)\.(\d+)', r'0.\1', expr)
    expr = re.sub(r'delta\s*\(', 'DiracDelta(', expr)
    expr = expr.replace('H(', 'Heaviside(')
    return expr

def compute_symbolic_convolution(h_sym, x_sym, tau, t_sym):
    with assuming(Q.positive(t_sym)):
        return integrate(h_sym * x_sym, (tau, 0, t_sym))

def compute_roots_char_poly(b, c):
    t_sym = symbols('t')
    poly = Poly(t_sym**2 + b * t_sym + c, t_sym)
    return list(roots(poly).keys())

def compute_laplace_solution(b, c, x_sym, t_sym):
    s = symbols('s')
    tau = symbols('tau')
    try:
        X_s = laplace_transform(x_sym, t_sym, s, noconds=True)
        Y_s = X_s / (s**2 + b * s + c)
        y_t = inverse_laplace_transform(Y_s, s, t_sym)
        # Check if inversion failed (result contains InverseLaplaceTransform)
        if 'InverseLaplaceTransform' in str(y_t):
            raise ValueError("Laplace inversion failed, falling back to convolution")
        return simplify(y_t)
    except:
        # Fallback: use convolution with impulse response
        # For y'' + by' + cy = x(t), convolve x with impulse response h
        r_vals = compute_roots_char_poly(b, c)
        if len(r_vals) == 2 and r_vals[0] != r_vals[1]:
            r1, r2 = r_vals
            h_sym = (exp(r1*(t_sym - tau)) - exp(r2*(t_sym - tau))) / (r1 - r2) * Heaviside(t_sym - tau)
        elif len(r_vals) == 1:
            r = r_vals[0]
            h_sym = (t_sym - tau) * exp(r*(t_sym - tau)) * Heaviside(t_sym - tau)
        else:
            lam = r_vals[0].as_real_imag()[0]
            mu = r_vals[0].as_real_imag()[1]
            h_sym = exp(lam*(t_sym - tau)) * sin(mu*(t_sym - tau)) / mu * Heaviside(t_sym - tau)
        
        y_particular = compute_symbolic_convolution(h_sym, x_sym, tau, t_sym)
        return simplify(y_particular)

def symbolic_solution(y_hom_sym, y_particular_sym, t_sym, y0, y0p):
    A, B = symbols('A B')
    y_hom_prime = y_hom_sym.diff(t_sym)
    eq1 = y_hom_sym.subs(t_sym, 0) - y0
    eq2 = y_hom_prime.subs(t_sym, 0) - y0p
    sol_dict = simplify(solve((eq1, eq2), (A, B)))
    A_val, B_val = sol_dict[A], sol_dict[B]
    solution = y_hom_sym + y_particular_sym
    subbed = solution.subs({A: A_val, B: B_val})
    solved = together(expand(subbed)).replace(Heaviside(t_sym), 1)
    return solved

def compute_residual(y_total_sym, b_sym, c_sym, x_sym, t_sym):
    """Compute the residual of the differential equation: y'' + by' + cy - x(t)."""
    y_prime = y_total_sym.diff(t_sym)
    y_double_prime = y_prime.diff(t_sym)
    residual = y_double_prime + b_sym * y_prime + c_sym * y_total_sym - x_sym
    return simplify(residual)

def evaluate_residual(residual_expr, t_vals):
    """Evaluate residual at numeric time points."""
    try:
        t_sym = list(residual_expr.free_symbols)[0]
        residual_func = lambdify(t_sym, residual_expr, modules=[{"Heaviside": lambda t: np.heaviside(t, 1), "DiracDelta": lambda t: 0}, "numpy"])
        return np.abs(residual_func(t_vals))
    except:
        return np.zeros_like(t_vals)


# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("Second-Order Linear ODE Solver")

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### Initial Conditions and Coefficients")
    b = st.number_input("b (for y')", value=-1.0)
    c = st.number_input("c (for y)", value=-2.0)
    y0 = st.number_input("y₀", value=1.0)
    y0p = st.number_input("y₀'", value=1.0)
   
    st.markdown("### Forcing Function")
    x_expr = st.text_input("Input Forcing Function x(t)", "sin(t)")
    
    with st.expander("📌 Quick Insert Examples", expanded=False):
        st.write("**Common functions:**")
        col1_btn, col2_btn, col3_btn, col4_btn = st.columns(4)
        with col1_btn:
            st.code("delta(t)")
        with col2_btn:
            st.code("H(t)")
        with col3_btn:
            st.code("exp(-t)")
        with col4_btn:
            st.code("sin(t)")
        
        col5_btn, col6_btn, col7_btn, col8_btn = st.columns(4)
        with col5_btn:
            st.code("cos(t)")
        with col6_btn:
            st.code("tan(t)")
        with col7_btn:
            st.code("1")
        with col8_btn:
            st.code("t")
        
        st.write("**Combine them:** `t + sin(t)`, `delta(t-2) + H(t)`, `exp(-t)*cos(t)`, etc.")

    st.markdown("### Time and Scaling Settings")
    t_min = st.number_input("Start time", value=0.0)
    t_max = st.number_input("End time", value=10.0)
    y_range = st.number_input("Y-Axis Range (±)", value=10.0)

    st.markdown("### Analysis Options")
    check_residual = st.checkbox("Check Solution Residual", value=False)

    t_range = t_max - t_min
    t_steps = int(min(max(t_range * 20, 100), 1000))
    t_vals = np.linspace(t_min, t_max, t_steps)

# --- Symbolic Setup ---
t_sym, tau = symbols('t tau', real=True)
safe_locals = {'t': t_sym, 'tau': tau, 'sin': sin, 'cos': cos, 'tan': tan, 'exp': exp, 'Heaviside': Heaviside, 'gamma': gamma, 'DiracDelta': DiracDelta}
x_sym = nsimplify(sympify(preprocess_expr(x_expr), locals=safe_locals), rational=True)
b_sym = Rational(b)
c_sym = Rational(c)
y0_sym = Rational(y0)
y0p_sym = Rational(y0p)

# --- Evaluation Environment ---
try:
    env = {
        "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
        "sqrt": np.sqrt, "t": t_vals, "pi": np.pi, "e": np.e,
        "gamma": lambda t: np.vectorize(math.gamma)(t),
        "heaviside": lambda t: np.heaviside(t, 1.0),
        "Heaviside": lambda t: np.heaviside(t, 1.0),
        "DiracDelta": lambda t: np.zeros_like(t),  # Approximation for numerical eval
    }
    x_func = eval(preprocess_expr(x_expr), env)
except Exception as e:
    st.error(f"Error in x(t): {e}")
    st.stop()


# --- Roots ---
a = 1
r_values = compute_roots_char_poly(b_sym, c_sym)
roots_numeric = np.roots([a, float(b), float(c)])

A, B = symbols('A B')
if len(r_values) == 2 and r_values[0] != r_values[1]:
    r1, r2 = r_values
    y_hom_sym = A * exp(r1 * t_sym) + B * exp(r2 * t_sym)
elif len(r_values) == 1:
    r = r_values[0]
    y_hom_sym = (A + B * t_sym) * exp(r * t_sym)
else:
    lam = r_values[0].as_real_imag()[0]
    mu = r_values[0].as_real_imag()[1]
    y_hom_sym = (A * exp(lam * t_sym)) * (cos(mu * t_sym) + (lam * sin(mu * t_sym))/mu) + (B * exp(lam * t_sym) * sin(mu * t_sym))/mu

# --- Impulse Response ---
if len(roots_numeric) == 2 and not np.isclose(roots_numeric[0], roots_numeric[1]):
    r1, r2 = roots_numeric
    h_sym = (exp(r1*(t_sym - tau)) - exp(r2*(t_sym - tau))) / (r1 - r2) * Heaviside(t_sym - tau)
elif np.isclose(roots_numeric[0], roots_numeric[1]):
    r = roots_numeric[0].real
    h_sym = (t_sym - tau) * exp(r*(t_sym - tau)) * Heaviside(t_sym - tau)
else:
    lam = roots_numeric[0].real
    mu = roots_numeric[0].imag
    h_sym = exp(lam*(t_sym - tau)) * sin(mu*(t_sym - tau)) / mu * Heaviside(t_sym - tau)

# --- Particular Solution ---
y_particular_sym = compute_laplace_solution(b_sym, c_sym, x_sym, t_sym)
y_total_sym = symbolic_solution( y_hom_sym, y_particular_sym, t_sym, y0_sym, y0p_sym)

# --- Numeric Evaluation ---
try:
    y_total_vals = np.array([float(y_total_sym.subs(t_sym, val).evalf()) for val in t_vals])
except:
    y_total_vals = np.zeros_like(t_vals)

# Prepare forcing expression for display and lambdify
forcing_str = str(x_sym)  # Ensure it's a string
forcing_expr = sympify(forcing_str.replace("^", "**"), locals=safe_locals)  # Convert to sympy expr

# Get symbolic solution and its derivative
t = list(y_total_sym.free_symbols)[0] 
y_final_expr = y_total_sym
y_prime_final = y_final_expr.diff(t_sym)

# Lambdify
f_lambdified = lambdify(t, forcing_expr, modules=[{"Heaviside": lambda t: np.heaviside(t, 1), "DiracDelta": lambda t: np.zeros_like(t), "tan": np.tan},"numpy"])
y_lambdified = lambdify(t, y_total_sym, modules=[{"I": 1j, "Heaviside": lambda t: np.heaviside(t, 1), "DiracDelta": lambda t: np.zeros_like(t), "tan": np.tan},"numpy"])
y_prime_lambdified = lambdify(t, y_prime_final, modules=[{"I": 1j, "Heaviside": lambda t: np.heaviside(t, 1), "DiracDelta": lambda t: np.zeros_like(t), "tan": np.tan},'numpy'])

# Evaluate over time
y_vals = y_lambdified(t_vals)
y_prime_vals = y_prime_lambdified(t_vals)

# --- Plot Layout ---
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_vals, y=y_total_vals, mode='lines', name='y(t)', line=dict(width=3)))
    fig.update_layout(title="ODE Solution", xaxis_title="t", yaxis_title="y(t)",
                      template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=y_vals,
        y=y_prime_vals,
        mode='lines',
        line=dict(color='royalblue'),
        name="Phase Trajectory"
    ))

    fig2.update_layout(
        title="Phase Plane",
        xaxis=dict(title="y(t)"),
        yaxis=dict(title="y'(t)"),
        template="plotly_white",
        width=600,
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig2)
# --- Math Display ---
st.markdown("### Differential Equation")
forcing_latex = latex(forcing_expr)
eq_latex = fr"y'' + {b}y' + {c}y = {forcing_latex}\quad y(0) = {y0}, y'(0) ={y0p}"
st.latex(eq_latex)

st.markdown("### Symbolic Solution")
st.latex(r"y(t) = " + latex(y_total_sym))


# --- Residual Checking ---
if check_residual:
    st.markdown("### Solution Residual Analysis")
    
    residual_sym = compute_residual(y_total_sym, b_sym, c_sym, x_sym, t_sym)
    
    # Check if forcing function contains problematic terms
    if 'DiracDelta' in str(residual_sym) or 'delta' in str(residual_sym):
        st.latex(r"\text{Residual} = y''(t) + by'(t) + cy(t) - x(t) = " + latex(residual_sym))
        st.info("ℹ️ Residual analysis not available for delta function forcing. The symbolic residual is shown above.")
    else:
        try:
            st.latex(r"\text{Residual} = y''(t) + by'(t) + cy(t) - x(t) = " + latex(residual_sym))
            
            residual_vals = evaluate_residual(residual_sym, t_vals)
            max_residual = np.max(residual_vals)
            mean_residual = np.mean(residual_vals)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Max Residual", f"{max_residual:.2e}")
            with col_res2:
                st.metric("Mean Residual", f"{mean_residual:.2e}")
            
            fig_residual = go.Figure()
            fig_residual.add_trace(go.Scatter(x=t_vals, y=residual_vals, mode='lines', name='|Residual|', 
                                             line=dict(color='red', width=2)))
            fig_residual.update_layout(title="Solution Residual", xaxis_title="t", yaxis_title="|Residual|",
                                      yaxis_type="log", template="plotly_white", height=400)
            st.plotly_chart(fig_residual, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute residual: {str(e)[:100]}")


st.caption("Solves second-order linear ODEs using symbolic convolution and Laplace transform automatically.")
