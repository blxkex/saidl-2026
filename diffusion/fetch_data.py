import wandb
import pandas as pd

def main():
    api = wandb.Api()
    print("Fetching from W&B...")
    runs = api.runs("bhuvaneshreddy-bits-pilani/saidl-diffusion")
    
    # Helper to grab the final VALID chunk of a fragmented run
    def get_latest_run(name):
        # Filter out failed runs first
        matches = [r for r in runs if r.name == name and r.state != "failed"]
        if not matches:
            # Fallback if all of them are somehow marked failed/crashed
            matches = [r for r in runs if r.name == name]
            if not matches: return None
            
        # Sort by created_at which is universally available on all runs
        return sorted(matches, key=lambda r: r.created_at, reverse=True)[0]

    baseline_run = get_latest_run("baseline-dit-b8")
    predictor_run = get_latest_run("difficulty-predictor")
    
    if not baseline_run or not predictor_run:
        print("Couldn't find the runs. Check your project path.")
        return

    # Grab the summaries
    b_sum = baseline_run.summary
    p_sum = predictor_run.summary

    # Map the rows. 
    # IMPORTANT: If your RACD and Global metrics print as "-", it means the 
    # dictionary keys below (e.g. "eval/fid_global") don't match exactly what 
    # you named them in your evaluation script. Just tweak the strings here to match!
    table_data = [
        {
            "Configuration": "Baseline DiT",
            "FID": b_sum.get("eval/fid", "N/A"),
            "CMMD": b_sum.get("eval/cmmd", "N/A"),
            "Gen Time (s)": b_sum.get("eval/gen_time_per_image", "N/A")
        },
        {
            "Configuration": "Global Cyclic",
            "FID": p_sum.get("eval/fid_global", "N/A"),
            "CMMD": p_sum.get("eval/cmmd_global", "N/A"),
            "Gen Time (s)": p_sum.get("eval/gen_time_global", "N/A")
        },
        {
            "Configuration": "RACD ($\\tau=0.3$)",
            "FID": p_sum.get("eval/fid_racd_0.3", "N/A"),
            "CMMD": p_sum.get("eval/cmmd_racd_0.3", "N/A"),
            "Gen Time (s)": p_sum.get("eval/gen_time_racd_0.3", "N/A")
        },
        {
            "Configuration": "RACD ($\\tau=0.5$)",
            "FID": p_sum.get("eval/fid_racd_0.5", "N/A"),
            "CMMD": p_sum.get("eval/cmmd_racd_0.5", "N/A"),
            "Gen Time (s)": p_sum.get("eval/gen_time_racd_0.5", "N/A")
        },
        {
            "Configuration": "RACD ($\\tau=0.7$)",
            "FID": p_sum.get("eval/fid_racd_0.7", "N/A"),
            "CMMD": p_sum.get("eval/cmmd_racd_0.7", "N/A"),
            "Gen Time (s)": p_sum.get("eval/gen_time_racd_0.7", "N/A")
        }
    ]

    df = pd.DataFrame(table_data)

    # Force numeric types to floats for clean 2-decimal rounding
    for col in ["FID", "CMMD", "Gen Time (s)"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    latex_table = df.to_latex(
        index=False, 
        float_format="%.2f", 
        na_rep="-",
        column_format="lccc", 
        escape=False # Keeps the LaTeX math symbols in the names intact
    )
    
    print("\n--- YOUR LATEX TABLE ---\n")
    print(latex_table)
    print("\n------------------------\n")

if __name__ == "__main__":
    main()