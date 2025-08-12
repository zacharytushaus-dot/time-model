
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from engine import Inputs, Tier, run_monte_carlo, IntervCost

st.set_page_config(page_title="Time Utility Model", layout="wide")

# Persist results across reruns so charts don't error before you run.
if "results" not in st.session_state:
    st.session_state.results = None

st.title("Time Utility Model")

# ------------------ Sidebar: profile & presets ------------------
st.sidebar.header("1) Your Profile & Presets")

age = st.sidebar.number_input("Your Age", min_value=18, max_value=95, value=21, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])

preset = st.sidebar.selectbox(
    "Preset",
    ["None", "Athlete", "Budget Health", "Max Longevity"],
    index=1
)

# Default risk multipliers (HR×adherence at 100%)
BASE_RISK_MULT = {
    "Consistent Sleep": 0.88,
    "Frequent Exercise": 0.68,
    "Mediterranean Diet": 0.77,
    "Meditation": 0.93,
    "Red-Light Therapy": 0.98,
    "Smoking Currently": 2.5, # harmful, editable in UI later if we want
}

# Preset → default toggle set
PRESET_TOGGLES = {
    "None":               {"Consistent Sleep": False, "Frequent Exercise": False, "Mediterranean Diet": False, "Meditation": False, "Red-Light Therapy": False, "Smoking Currently": False},
    "Athlete":            {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True,  "Meditation": True,  "Red-Light Therapy": False, "Smoking Currently": False},
    "Budget Health":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": False, "Meditation": True,  "Red-Light Therapy": False, "Smoking Currently": False},
    "Max Longevity":      {"Consistent Sleep": True,  "Frequent Exercise": True,  "Mediterranean Diet": True,  "Meditation": True,  "Red-Light Therapy": True, "Smoking Currently": False},
}

st.sidebar.header("2) Your Health Habits")
toggles = {}
for name, default_on in PRESET_TOGGLES[preset].items():
    toggles[name] = st.sidebar.checkbox(name, value=default_on)

adherence = st.sidebar.slider("Adherence to Your Habits (%)", 0.0, 1.0, 1.0, 0.05)
st.sidebar.caption("How likely you are to commit to your habits, from 0 to 100%")

# Final multipliers to feed engine (Excel rule)

# Canonical keys so UI labels, costs, and HR multipliers stay in sync
CANON = {
    "Consistent Sleep":    "sleep",
    "Frequent Exercise":       "exercise",
    "Mediterranean Diet": "mediterraneandiet",
    "Meditation":               "meditation",
    "Red-Light Therapy":        "redlight",
    "Smoking Currently":        "smoker",
}

intervention_on = {}
lifestyle_HRs = {}
for name, base in BASE_RISK_MULT.items():
    key = CANON[name]
    on = bool(toggles[name])
    intervention_on[key] = on
    # benefit-weighting (80% adherence = 80% of the benefit)
    if on and base != 1.0:
        m = 1.0 - adherence * (1.0 - base)
    else:
        m = 1.0
    lifestyle_HRs[key] = float(m)

# ------------------ Sidebar: habit costs (annual only) ------------------
with st.sidebar.expander("Health Habit Costs", expanded=False):
    st.caption("One field per habit: dollars per year (negative allowed if it saves you money). We apply healthcare inflation automatically each year.")

    # canonical keys used everywhere (from CANON above)
    ORDER = ["sleep", "exercise", "redlight", "mediterraneandiet", "meditation", "smoker"]
    LABEL_FOR = {v: k for k, v in CANON.items()}  # reverse map: key -> UI label

    # Defaults: previous one-time + recurring, converted to annual (keeps totals comparable)
    ANNUAL_DEFAULT = {
        "sleep":              50,   # ≈ $50/yr
        "exercise":           240.0,      # gym, etc.
        "redlight":           150,  # ≈ $157/yr
        "mediterraneandiet":  5400.0,     # extra groceries vs baseline
        "meditation":         70.0,       # app subscription
        "smoker":             800.0,      # cigarettes; make negative if quitting = savings
    }

    annual_inputs = {}
    for key in ORDER:
        label = LABEL_FOR[key]
        enabled = toggles[label]
        annual_inputs[key] = st.number_input(
            f"{label} — $ per year",
            value=float(ANNUAL_DEFAULT[key]),
            step=50.0,
            min_value=-1_000_000.0,  # allow negative for savings
            key=f"{key}_annual",
            disabled=not enabled,
        )

    # Feed the engine: horizon=1, one_time=0, recurring=annual
    intervention_costs = {
        k: IntervCost(horizon=1, one_time=0.0, recurring=float(annual_inputs[k]))
        for k in ORDER
    }

# ------------------ Sidebar: finance ------------------
st.sidebar.header("3) Your Finances")
start_capital = st.sidebar.number_input("Starting Capital ($)", min_value=0, value=10_000, step=1_000)
di0 = st.sidebar.number_input("Yearly Spending Budget ($)", min_value=0, value=10_000, step=1_000)
ret = st.sidebar.slider("Portfolio Return (% per year)", 0.0, 15.0, 5.0, 0.1) / 100.0
income_growth = st.sidebar.slider("Spending Growth (% per year)", 0.0, 10.0, 3.0, 0.1) / 100.0
hc_infl = st.sidebar.slider("Healthcare Inflation (% per year)", 0.0, 10.0, 3.0, 0.1) / 100.0

# ------------------ Sidebar: breakthroughs ------------------
st.sidebar.header("4) Future Treatments, Ranked from Tiers 1-3")
def tier_inputs(label, p0, g_pp, cap, cost, years_gained):
    st.sidebar.subheader(label)
    kid = label.lower().replace(" ", "_")
    c = st.sidebar.number_input("Estimated Cost Today ($)", min_value=0, value=cost, step=1_000, key=f"{kid}_cost")
    y = st.sidebar.number_input("Years Added (after purchase)", min_value=0.0, value=years_gained, step=0.1, key=f"{kid}_years")
    p = st.sidebar.slider("Annual Odds of Occuring (%)", 0.0, 50.0, p0*100.0, 0.01, key=f"{kid}_p0") / 100.0
    g = st.sidebar.slider("Increase in Odds per Missed Year (%)", 0.0, 5.0, g_pp*100.0, 0.01, key=f"{kid}_gpp") / 100.0
    cap_ = st.sidebar.slider("Max Annual Chance of Occuring (%)", 0.0, 100.0, cap*100.0, 0.1, key=f"{kid}_cap") / 100.0
    return Tier(
        cost_today=c,
        years_gain=y,
        base_prob=p,
        growth_per_year=g,
        cap_prob=cap_,
    )

tier1 = tier_inputs("Tier 1", p0=0.03,  g_pp=0.0015, cap=0.10, cost=12_000,     years_gained=0.7)
tier2 = tier_inputs("Tier 2", p0=0.006, g_pp=0.0015, cap=0.10, cost=550_000,   years_gained=2.5)
tier3 = tier_inputs("Tier 3", p0=0.0012,g_pp=0.0015, cap=0.10, cost=2_400_000, years_gained=7.0)

# ------------------ Sidebar: longevity params ------------------
st.sidebar.header("5) Longevity Parameters")
lambda_plateau = st.sidebar.number_input("Late-age Risk Plateau λ", min_value=0.0, max_value=5.0, value=0.6, step=0.05)
drift_days = st.sidebar.number_input("Frontier Age Drift (days per year)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
le_improve = st.sidebar.slider("Life Expectancy Growth (% per year)", 0.0, 2.0, 0.2, 0.01) / 100.0
max_age_today = st.sidebar.number_input("Frontier Age Today", min_value=100.0, max_value=130.0, value=119.0, step=0.5)

# ------------------ Sidebar: simulation ------------------
st.sidebar.header("6) Simulation")
draws = st.sidebar.slider("Simulation Runs", 1000, 50000, 5000, step=1000)
seed = st.sidebar.number_input("Random Seed (reproducible)", min_value=0, max_value=1_000_000, value=49, step=1)

if st.sidebar.button("Run Simulation", type="primary"):
    inputs = Inputs(
        start_age=int(age),
        sex=sex,
        draws=int(draws),
        investment_return=float(ret),
        start_capital=float(start_capital),
        # Excel-parity finance
        discretionary_income=float(di0),
        income_growth=float(income_growth),
        annual_contrib=0.0,                  # ignored when discretionary_income is provided
        contrib_growth=0.0,
        lambdaP=float(lambda_plateau),
        frontier_drift_days=float(drift_days),
        le_trend=float(le_improve),
        max_age_today=int(max_age_today),
        hc_inflation=float(hc_infl),
        lifestyle_HRs=lifestyle_HRs,
        adherence=float(adherence),
        tiers=[tier1, tier2, tier3],
        tier_repeatable=True,                 # repeatable breakthroughs
        intervention_costs=intervention_costs,
        intervention_on=intervention_on,
        grid_max_age=max(int(age) + 126, 170),
        seed=int(seed),
    )
    st.session_state.results = run_monte_carlo(inputs)

# ------------------ Render results (if any) ------------------
out = st.session_state.results

if out is None:
    st.info("Describe yourself on the left, then click **Run Simulation**")
    st.stop()  # ← do not execute the rest of the page until results exist
else:
    col1, col2, col3 = st.columns(3)
    median_life = float(np.median(out["projected_life"]))
    p5, p95 = np.percentile(out["projected_life"], [5,95])
    median_net = float(np.median(out["net_worth"]))

    with col1:
        st.metric("Projected Life (median)", f"{median_life:.0f} years",
                  help="The age when your biological age first equals or exceeds society's average life-expectancy for that calendar year")
        # As a placeholder, personal gain ≈ median life − current age
        st.metric("Personal gain (median)", f"{median_life - age:.2f} years")

    with col2:
        st.metric("Net Worth (median, $MM)", f"{median_net:,.2f}")
        st.metric("Simulations Ran", f"{draws:,}")

    with col3:
        st.metric("90% range (5th–95th percentile)", f"{p5:.0f}–{p95:.0f} years",
                  help="Middle 90% of simulated lifespans; 5% of draws fall below the left value and 5% above the right")
    
# Histograms
c1, c2 = st.columns(2)
with c1:
    hist_mode = st.radio("Life histogram", ["Monte-Carlo (hazard)", "Threshold crossing"], index=0,
                         help="Monte-Carlo: lifespans sampled from yearly death probabilities. Threshold: lifespans when biological age exceeds society's average life-expectancy")
    if hist_mode.startswith("Monte"):
        life_data = out["projected_life_mc"]
        life_title = "Monte‑Carlo Projected Life (1‑year bins)"
        # keep 1‑year bins for the hazard MC
        nbins = int(np.ptp(life_data)) + 1 if np.ptp(life_data) >= 1 else 10
        fig_life = px.histogram(life_data, nbins=nbins, title=life_title)
    else:
        # Prefer fractional crossing if present; fall back to integer
        life_data = out.get("projected_life_frac", out["projected_life"])
        life_title = "Projected Lifespans Across Simulations"
        fig_life = px.histogram(life_data, title=life_title)
        # finer bars look better with fractional values
        fig_life.update_traces(xbins=dict(size=1))

    # start x‑axis at the user’s age for both modes
    min_x = int(age)
    fig_life.update_xaxes(range=[min_x, None])
    st.plotly_chart(fig_life, use_container_width=True)

with c2:
    # --- Wealth vs life: scatter or density ---
    life_for_net = out.get("projected_life_frac", out["projected_life"])  # use fractional crossing if present
    nw = out["net_worth"]

    # Count tech purchases per draw (any positive years_added means a purchase that year)
    purchases = (out["tech_years_by_age"] > 0).sum(axis=1)
    # Bucket for clean legend
    bucket = np.where(purchases == 0, "0",
              np.where(purchases == 1, "1",
              np.where(purchases <= 3, "2-3", "4+")))
    df_nw = pd.DataFrame({"Life": life_for_net, "NetWorth": nw, "Purchases": bucket})

    view = st.radio("Wealth view", ["Scatter", "Heatmap"], index=0, horizontal=True)
    if view == "Scatter":
        fig_nw = px.scatter(
            df_nw, x="Life", y="NetWorth", color="Purchases",
            title="Net Worth vs Projected Life (per draw)",
            labels={"Life": "Projected life (years)", "NetWorth": "Net worth at death ($MM)"},
            opacity=0.55
        )
    else:
        fig_nw = px.density_heatmap(
            df_nw, x="Life", y="NetWorth",
            nbinsx=40, nbinsy=40, color_continuous_scale="Blues",
            title="Net Worth vs Projected Life — density",
            labels={"Life": "Projected life (years)", "NetWorth": "Net worth at death ($MM)"}
        )

    # Start x-axis at the user's current age for clarity
    fig_nw.update_xaxes(range=[int(age), None])
    st.plotly_chart(fig_nw, use_container_width=True)

# Years‑added chart: Excel‑match mode by default with options
    mode = st.selectbox(
        "Years-added chart",
        ["Your Health Habits", "Interventions + tech (alive-weighted)", "Tech only (alive-weighted)"],
        index=0,
        help="Excel's dashboard chart shows interventions only. 'Alive-weighted' tech counts only when a draw is alive that year"
    )

    yrs_int = out["yrs_added_interventions"]                           # (T,)

    # Alive‑weighted expected tech by age
    # Use the 1-D threshold series returned by the engine
    le_series = out.get("threshold_series")  # shape (T,)
    # Back-compat if you ever run an older engine that used a different name
    if le_series is None:
        le_series = out.get("le_threshold_series")

    alive_mask = (out["bio_age"] < le_series[None, :]).astype(float)  # (D, T)
    exp_tech_by_age = (out["tech_years_by_age"] * alive_mask).mean(axis=0)

    if mode.startswith("Your Health Habits"):
        y = yrs_int
    elif mode.startswith("Interventions + tech"):
        y = yrs_int + exp_tech_by_age
    else:  # Tech only
        y = exp_tech_by_age

    fig_yrs = px.area(x=out["chrono_age"], y=y,
                      labels={"x": "Age", "y": "Expected Years Added"},
                      title=mode)
    st.plotly_chart(fig_yrs, use_container_width=True)

    # Optional diagnostic
    with st.expander("Diagnostics: biological age vs LE threshold (mean across draws)"):
        mean_bio = out["bio_age"].mean(axis=0)        # (T,)
        le_series = out.get("threshold_series")       # (T,)
        if le_series is None:
            le_series = out.get("le_threshold_series")

        df = pd.DataFrame({
            "Age": out["chrono_age"],
            "Biological age (mean)": mean_bio,
            "LE threshold": le_series,                # already (T,) – no .mean(axis=0)
        })
        fig_diag = px.line(
            df, x="Age", y=["Biological age (mean)", "LE threshold"],
            title="Biological Age vs Average Life Expectancy"
        )
        st.plotly_chart(fig_diag, use_container_width=True)
    
    # Finance diagnostics
    with st.expander("Personal Finance Details, First 20 Years"):
        df_fin = pd.DataFrame({
            "Age": out["chrono_age"],
            "Discretionary income": out["discretionary_income_by_year"],
            "Total health spend": out["health_spend_by_year"],
            "Contribution (DI − spend)": out["contrib_by_year"],
        })
        st.dataframe(df_fin.head(20))
