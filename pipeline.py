import pandas as pd
import pathlib
from trading_tools import (
    calculate_signal_probabilities, download_multiple_tickers,
    create_summary_table
)

# Folder where your market CSVs are stored
MARKET_DATA_DIR = pathlib.Path("./markets")
OUTPUT_DIR = pathlib.Path("./processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_tickers_from_csv(file_path):
    df = pd.read_csv(file_path, skiprows=1, header=None)
    return df[0].str.strip().tolist()

def run_pipeline():
    print("🚀 Starting Dynamic Market Pipeline...")
    
    # Find all CSV files in the directory
    csv_files = list(MARKET_DATA_DIR.glob("*.csv"))
    
    for csv_file in csv_files:
        market_name = csv_file.stem  # e.g., 'UK_STABLE'
        print(f"--- Processing {market_name} ---")
        
        try:
            # 1. Load and Download
            tickers = load_tickers_from_csv(csv_file)
            data = download_multiple_tickers(tickers)
            
            # 2. Calculate Signals
            results = calculate_signal_probabilities(data)
            results_df = pd.concat(results.values(), keys=results.keys(), names=['Ticker'])
            
            # 3. Create Summary
            summary_df = create_summary_table(results)
            
            # 4. Save using a consistent naming convention
            results_df.to_pickle(OUTPUT_DIR / f"{market_name}_results.pkl")
            summary_df.to_pickle(OUTPUT_DIR / f"{market_name}_summary.pkl")
            
            print(f"Saved {market_name}")
            
        except Exception as e:
            print(f"Error processing {market_name}: {str(e)}")

if __name__ == "__main__":
    run_pipeline()
