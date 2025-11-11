import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

st.title("Interactive ODE Lab")

# --- User inputs ---
vars_str = st.text_input("Variables (comma separated)", "x,y")
eqs_str = st.text_area("RHS (one per line)", "-0.5*x - y\nx - 0.5*y")
ics_str = st.text_input("Initial conditions (comma separated)", "1,0")
solver = st.selectbox("Solver", ["Analytical", "Numerical"])
t0 = st.number_input("Start time t0", value=0.0)
t1 = st.number_input("End time t1", value=20.0)
npts = st.slider("Number of points", min_value=50, max_value=2000, value=200)

# Parse inputs
vars_list = [v.strip() for v in vars_str.split(',')]
ics_list = [float(v.strip()) for v in ics_str.split(',')]
eqs_list = [line.strip() for line in eqs_str.splitlines() if line.strip() != '']

# --- Functions ---
SAFE_GLOBALS = {'np': np, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e,
                'abs': abs, 'pow': pow}

def make_rhs_function(vars_list, eqs_list):
    var_map = {vars_list[i]: f"x{i}" for i in range(len(vars_list))}
    compiled = []
    for line in eqs_list:
        expr = line
        for name, repl in var_map.items():
            expr = expr.replace(name, repl)
        code = compile(expr, '<string>', 'eval')
        compiled.append(code)
    def rhs(t,Y):
        local = {f"x{i}": Y[i] for i in range(len(Y))}
        local['t'] = t
        return np.array([float(eval(c, SAFE_GLOBALS, local)) for c in compiled])
    return rhs

def rk4_integrate(rhs, t_eval, y0):
    Y = np.zeros((len(t_eval), len(y0)))
    Y[0,:] = y0
    for i in range(len(t_eval)-1):
        t = t_eval[i]
        dt = t_eval[i+1] - t_eval[i]
        y = Y[i,:]
        k1 = rhs(t, y)
        k2 = rhs(t + dt/2, y + dt*k1/2)
        k3 = rhs(t + dt/2, y + dt*k2/2)
        k4 = rhs(t + dt, y + dt*k3)
        Y[i+1,:] = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return t_eval, Y.T

# --- Simulation ---
t_eval = np.linspace(t0, t1, npts)

if solver == 'Analytical':
    t = sp.symbols('t')
    funcs = [sp.Function(v)(t) for v in vars_list]
    local_dict = {v: funcs[i] for i,v in enumerate(vars_list)}
    eq_syms = []
    try:
        for i,line in enumerate(eqs_list):
            rhs_expr = sp.sympify(line, locals=local_dict)
            eq_syms.append(sp.Eq(funcs[i].diff(t), rhs_expr))
        sols = [sp.dsolve(eq, funcs[i]) for i, eq in enumerate(eq_syms)]
        st.markdown("**Analytical solution:**")
        for s in sols:
            st.text(s)
    except Exception as e:
        st.warning(f"Symbolic solution failed, fallback to Numerical. ({e})")
        solver = 'Numerical'

if solver == 'Numerical':
    rhs = make_rhs_function(vars_list, eqs_list)
    try:
        T, Y = rk4_integrate(rhs, t_eval, ics_list)
        nvars = len(vars_list)
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        # Time series
        for i in range(nvars):
            ax[0].plot(T, Y[i,:], label=vars_list[i])
        ax[0].set_xlabel('t'); ax[0].set_ylabel('Value'); ax[0].set_title('Time series')
        ax[0].legend(); ax[0].grid(True)
        # Phase portrait
        if nvars == 1:
            ax[1].plot(T,Y[0,:])
            ax[1].set_xlabel('t'); ax[1].set_title(f'{vars_list[0]}(t)')
        elif nvars == 2:
            x, y = Y[0,:], Y[1,:]
            ax[1].plot(x,y,'-k',lw=1.2,label='trajectory')
            ax[1].scatter([x[0]],[y[0]],color='green',label='start')
            ax[1].scatter([x[-1]],[y[-1]],color='red',label='end')
            ax[1].set_xlabel(vars_list[0]); ax[1].set_ylabel(vars_list[1])
            ax[1].set_title('Phase portrait'); ax[1].legend(); ax[1].grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Numerical simulation failed: {e}")

