import streamlit as st
import pandas as pd
import pathlib
import os
from trading_tools import plot_interactive_signals

st.set_page_config(page_title="Trading Signals", layout="wide")

# Directory where pipeline.py saves processed data
DATA_DIR = pathlib.Path("./processed_data")

@st.cache_data
def load_market_data(market_name):
    """Loads pickles and converts results into a dictionary of DataFrames safely."""
    res_path = DATA_DIR / f"{market_name}_results.pkl"
    sum_path = DATA_DIR / f"{market_name}_summary.pkl"
    
    if res_path.exists() and sum_path.exists():
        results_df = pd.read_pickle(res_path)
        summary_df = pd.read_pickle(sum_path)
        
        results_dict = {}
        
        # Group by Ticker to split the MultiIndex back into individual frames
        for ticker, df in results_df.groupby(level='Ticker'):
            # Drop the Ticker level from index first
            temp_df = df.reset_index(level='Ticker', drop=True)
            
            # SAFE RESET: If 'Date' is already a column, don't try to move index to 'Date'
            # This prevents the DuplicateError: 'Date' 2 times
            if 'Date' in temp_df.columns:
                clean_df = temp_df.copy()
            else:
                clean_df = temp_df.reset_index()

            # Final check to rename the index column if it became 'index' or 'level_1'
            rename_dict = {
                col: 'Date' for col in clean_df.columns 
                if col.lower() in ['index', 'level_1', 'datetime'] and 'Date' not in clean_df.columns
            }
            clean_df = clean_df.rename(columns=rename_dict)
            
            # Ensure column names are unique by removing any exact duplicates that might remain
            clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]
            
            results_dict[ticker] = clean_df
            
        return results_dict, summary_df
    
    return None, None

def main():
    st.title("Interactive Trading Signals Dashboard")

    if not DATA_DIR.exists():
        st.error(f"Directory {DATA_DIR} not found. Please run pipeline.py first.")
        return

    available_summaries = list(DATA_DIR.glob("*_summary.pkl"))
    market_names = sorted([f.name.replace("_summary.pkl", "") for f in available_summaries])

    if not market_names:
        st.error("No processed data found. Please run pipeline.py.")
        return

    tabs = st.tabs(market_names)

    for tab, name in zip(tabs, market_names):
        with tab:
            results_dict, summary_df = load_market_data(name)
            
            if results_dict is None:
                st.warning(f"Could not load data for {name}")
                continue

            # --- LEADERBOARDS ---
            st.subheader(f"{name} Signal Leaderboards")
            cols = st.columns(3)
            
            # Shared configuration to make columns fit the container width
            col_cfg = {
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Latest_Close": st.column_config.NumberColumn("Latest Close", width="small", format="%.4f"),
                "Latest_Buy_Prob": st.column_config.NumberColumn("Buy %", format="%.2f"),
                "Latest_Sell_Prob": st.column_config.NumberColumn("Sell %", format="%.2f"),
                "Latest_Diff_Prob": st.column_config.NumberColumn("Strength", format="%.2f")
            }

            with cols[0]:
                st.write("**Top 10 Buy Probabilities**")
                st.dataframe(
                    summary_df.sort_values('Latest_Buy_Prob', ascending=False).head(10), 
                    hide_index=True,
                    use_container_width=True,
                    column_config=col_cfg
                )
            
            with cols[1]:
                st.write("**Top 10 Sell Probabilities**")
                st.dataframe(
                    summary_df.sort_values('Latest_Sell_Prob', ascending=False).head(10), 
                    hide_index=True,
                    use_container_width=True,
                    column_config=col_cfg
                )
            
            with cols[2]:
                st.write("**Top 10 Strengths Difference**")
                st.dataframe(
                    summary_df.sort_values('Latest_Diff_Prob', ascending=False).head(10), 
                    hide_index=True,
                    use_container_width=True,
                    column_config=col_cfg
                )

            st.divider()

            # --- VISUALISATION ---
            st.subheader("Dynamic Signal Visualisation")
            g1, g2 = st.columns(2)
            with g1:
                st.write("### Buy Signal Analysis")
                fig_buy = plot_interactive_signals(results_dict, probability_column='Buy_Probability')
                st.plotly_chart(fig_buy, use_container_width=True)
            
            with g2:
                st.write("### Sell Signal Analysis")
                fig_sell = plot_interactive_signals(results_dict, probability_column='Sell_Probability')
                st.plotly_chart(fig_sell, use_container_width=True)

if __name__ == "__main__":
    main()
