import wandb
import pandas as pd

# -------------------------------------------------------------
# 1. SETUP
# -------------------------------------------------------------
entity = "bhuvaneshreddy-bits-pilani"
project = "saidl-transformer"

print("[*] Connecting to W&B API...")
api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

data = []

# -------------------------------------------------------------
# 2. FETCH EVAL DATA FROM W&B
# -------------------------------------------------------------
print("[*] Snagging extrapolation metrics...")
for run in runs:
    # Only target the evaluation runs that the mega-cell created
    if run.name.startswith("EVAL-"):
        # Strip the "EVAL-" prefix for the clean LaTeX name
        base_name = run.name.replace("EVAL-", "").replace("_", "\\_")

        # Pull the exact keys shown in your W&B screenshot
        ppl_512 = run.summary.get("extrapolation/ppl_L512", "N/A")
        ppl_1024 = run.summary.get("extrapolation/ppl_L1024", "N/A")
        ppl_2048 = run.summary.get("extrapolation/ppl_L2048", "N/A")
        
        tp_512 = run.summary.get("extrapolation/throughput_L512", "N/A")
        tp_1024 = run.summary.get("extrapolation/throughput_L1024", "N/A")
        tp_2048 = run.summary.get("extrapolation/throughput_L2048", "N/A")

        # Clean formatting for the LaTeX output
        # Clean formatting for the LaTeX output
        def fmt_ppl(val):
            if val == "N/A": return val
            
            # Catch any variation of infinity or old crash strings
            if val == float('inf') or str(val).lower() in ['inf', 'infinity', 'crash (out of bounds)', 'crash']: 
                return "Crash"
            
            try:
                return round(float(val), 2)
            except (ValueError, TypeError):
                return str(val) # Fallback just in case W&B throws out something totally random

        def fmt_tp(val):
            if val == "N/A" or val == 0.0 or str(val).lower() == 'nan': return "-"
            try:
                return f"{int(float(val)):,}"
            except (ValueError, TypeError):
                return str(val)
            
        data.append({
            "Model": base_name,
            "PPL (512)": fmt_ppl(ppl_512),
            "PPL (1024)": fmt_ppl(ppl_1024),
            "PPL (2048)": fmt_ppl(ppl_2048),
            "TP 512 (tok/s)": fmt_tp(tp_512),
            "TP 1024 (tok/s)": fmt_tp(tp_1024),
            "TP 2048 (tok/s)": fmt_tp(tp_2048)
        })

if not data:
    print("⚠️ No EVAL runs found. Double check your W&B project sync.")

# -------------------------------------------------------------
# 3. FORMAT AS LATEX TABLE
# -------------------------------------------------------------
df = pd.DataFrame(data)

# Sort alphabetically so it's consistent
df = df.sort_values("Model")

print("\n" + "="*70)
print("🚀 COPY-PASTE THIS INTO OVERLEAF")
print("="*70 + "\n")

# Generate clean LaTeX code using booktabs
latex_code = df.to_latex(
    index=False,
    column_format="l" + "c" * 6,
    escape=False,
    position="H",
    caption="Extrapolation test results (Perplexity and Throughput) across increasing context lengths.",
    label="tab:extrapolation_results"
)

# Add standard booktabs formatting for academic styling
latex_code = latex_code.replace("\\toprule", "\\toprule\n\\midrule").replace("\\bottomrule", "\\midrule\n\\bottomrule")
print(latex_code)