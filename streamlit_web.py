# import os
# import json
# import pandas as pd
# import numpy as np
# import streamlit as st
# from streamlit_echarts import st_echarts

# # Load historical data
# @st.cache_data
# def load_historical_data():
#     def load_and_concat_data(base_path, years):
#         all_dfs = []
#         year_mapping = {'019': '2019', '020': '2021', '022': '2022', '023': '2023'}
#         for short_year in years:
#             file_path = os.path.join(base_path, f'ssfs-sale-transactions-{short_year}_en.json')
#             if not os.path.isfile(file_path):
#                 st.warning(f"File for {short_year} not found. Skipping.")
#                 continue
#             try:
#                 with open(file_path, 'r') as f:
#                     data = json.load(f)
#                 temp_df = pd.DataFrame(data.get("records", []))
#                 if temp_df.empty:
#                     st.warning(f"No records in {short_year} file.")
#                     continue
#                 full_year = year_mapping.get(short_year, short_year)
#                 temp_df['Data_Year'] = full_year
#                 if 'Court' in temp_df.columns and 'Court/Estate' not in temp_df.columns:
#                     temp_df = temp_df.rename(columns={'Court': 'Court/Estate'})
#                 temp_df['Floor'] = pd.to_numeric(temp_df['Floor'], errors='coerce').fillna(0).astype(int)
#                 all_dfs.append(temp_df)
#             except Exception as e:
#                 st.error(f"Error loading {short_year}: {str(e)}")
#         if not all_dfs:
#             st.error("No historical data loaded.")
#             return pd.DataFrame()
#         df = pd.concat(all_dfs, axis=0, ignore_index=True)
#         df['Date'] = pd.to_datetime(df['Date of Agreement for Sale and Purchase (ASP)'], 
#                                    format='%d/%m/%Y', errors='coerce')
#         df['Flat_ID'] = df['Court/Estate'].fillna('') + '_' + df['Floor'].astype(str).fillna('')
#         return df.sort_values('Date').reset_index(drop=True)

#     curr_dir = os.path.join(os.path.dirname(__file__), 'datasets')
#     data_years = ['019', '020', '022', '023']
#     return load_and_concat_data(curr_dir, data_years)

# # Load prediction data
# @st.cache_data
# def load_prediction_data():
#     csv_path = 'enhanced_prediction_results_2025_2026.csv'
#     if not os.path.isfile(csv_path):
#         st.error(f"Prediction file {csv_path} not found.")
#         return pd.DataFrame()
#     try:
#         df = pd.read_csv(csv_path)
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#         required_cols = ['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 'Predicted_2025_Price']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             st.error(f"Missing columns in prediction data: {missing_cols}")
#         else:
#             df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce').fillna(0).astype(int)
#             df['Flat_ID'] = df['Court/Estate'].fillna('') + '_' + df['Floor'].astype(str).fillna('')
#             if 'Lower_CI' not in df.columns:
#                 df['Lower_CI'] = df['Predicted_2025_Price'] * 0.9
#                 df['Upper_CI'] = df['Predicted_2025_Price'] * 1.1
#         return df
#     except Exception as e:
#         st.error(f"Error loading prediction data: {str(e)}")
#         return pd.DataFrame()

# # Main app
# def main():
#     st.markdown("""
#         <style>
#         .title { font-size: 2.5em; color: #FF5722; text-align: center; }
#         .subheader { color: #1976D2; }
#         .stButton>button { background-color: #4CAF50; color: white; }
#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown('<div class="title">2025-2026 HOS Flat Price Prediction & Analysis (R² 0.958)</div>', unsafe_allow_html=True)

#     with st.spinner("Loading data..."):
#         historical_df = load_historical_data()
#         prediction_df = load_prediction_data()

#     if historical_df.empty and prediction_df.empty:
#         st.error("Both datasets failed to load.")
#         return

#     # Sidebar debug
#     st.sidebar.subheader("Debug Info")
#     st.sidebar.write("Total Historical Rows:", len(historical_df))
#     st.sidebar.write("Historical Courts:", len(historical_df['Court/Estate'].unique()) if not historical_df.empty else 0)
#     st.sidebar.write("Total Prediction Rows:", len(prediction_df))
#     st.sidebar.write("Prediction Courts:", len(prediction_df['Court/Estate'].unique()) if not prediction_df.empty else 0)

#     all_courts = sorted(set(historical_df['Court/Estate'].unique() if not historical_df.empty else []).union(
#         prediction_df['Court/Estate'].unique() if not prediction_df.empty else []))
#     if not all_courts:
#         st.error("No courts available.")
#         return

#     selected_courts = st.multiselect("Select Court/Estate(s)", all_courts)

#     # Filters
#     st.sidebar.subheader("Filters")
#     min_price = int(min(historical_df['Transaction Price'].min() if not historical_df.empty else 0,
#                        prediction_df['Predicted_2025_Price'].min() if not prediction_df.empty else 0))
#     max_price = int(max(historical_df['Transaction Price'].max() if not historical_df.empty else 0,
#                        prediction_df['Predicted_2025_Price'].max() if not prediction_df.empty else 0))
#     price_range = st.sidebar.slider("Price Range (HKD)", min_price, max_price, (min_price, max_price))

#     min_floor = int(min(historical_df['Floor'].min() if not historical_df['Floor'].isna().all() else 0,
#                        prediction_df['Floor'].min() if not prediction_df['Floor'].isna().all() else 0))
#     max_floor = int(max(historical_df['Floor'].max() if not historical_df['Floor'].isna().all() else 0,
#                        prediction_df['Floor'].max() if not prediction_df['Floor'].isna().all() else 0))
#     floor_range = st.sidebar.slider("Floor Range", min_floor, max_floor, (min_floor, max_floor))

#     # Filter data for listings
#     court_historical = historical_df[historical_df['Court/Estate'].isin(selected_courts)].copy() if not historical_df.empty else pd.DataFrame()
#     court_predictions = prediction_df[prediction_df['Court/Estate'].isin(selected_courts)].copy() if not prediction_df.empty else pd.DataFrame()

#     if not court_historical.empty:
#         court_historical = court_historical[
#             (court_historical['Transaction Price'] >= price_range[0]) & 
#             (court_historical['Transaction Price'] <= price_range[1]) &
#             (court_historical['Floor'] >= floor_range[0]) & 
#             (court_historical['Floor'] <= floor_range[1])
#         ]
#     if not court_predictions.empty:
#         court_predictions = court_predictions[
#             (court_predictions['Predicted_2025_Price'] >= price_range[0]) & 
#             (court_predictions['Predicted_2025_Price'] <= price_range[1]) &
#             (court_predictions['Floor'] >= floor_range[0]) & 
#             (court_predictions['Floor'] <= floor_range[1])
#         ]

#     tab1, tab2 = st.tabs(["Price Listings", "Visualizations"])

#     with tab1:
#         st.markdown('<h3 class="subheader">Price Listings</h3>', unsafe_allow_html=True)
#         if selected_courts:
#             for court in selected_courts:
#                 st.write(f"**{court} Predictions**")
#                 court_pred = court_predictions[court_predictions['Court/Estate'] == court].copy()
#                 if not court_pred.empty:
#                     court_pred = court_pred[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
#                                             'Date', 'Predicted_2025_Price', 'Lower_CI', 'Upper_CI']]
#                     court_pred = court_pred.sort_values('Predicted_2025_Price', ascending=False)
#                     court_pred[['Predicted_2025_Price', 'Lower_CI', 'Upper_CI']] = court_pred[
#                         ['Predicted_2025_Price', 'Lower_CI', 'Upper_CI']].round(0).astype(int)
#                     court_pred = court_pred.rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
#                     st.dataframe(court_pred)
#                 else:
#                     st.warning(f"No prediction data for {court} after filtering.")
#         else:
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write("**Top 9 Hottest Courts (by Prediction Count)**")
#                 hottest = prediction_df['Court/Estate'].value_counts().head(9)
#                 hottest_data = prediction_df[prediction_df['Court/Estate'].isin(hottest.index)].sort_values(
#                     'Predicted_2025_Price', ascending=False).drop_duplicates('Court/Estate').head(9)
#                 hottest_data = hottest_data[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
#                                             'Predicted_2025_Price']].rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
#                 st.dataframe(hottest_data)
#             with col2:
#                 st.write("**Top 9 Most Expensive Courts (by Avg Predicted Price)**")
#                 expensive = prediction_df.groupby('Court/Estate')['Predicted_2025_Price'].mean().sort_values(ascending=False).head(9)
#                 expensive_data = prediction_df[prediction_df['Court/Estate'].isin(expensive.index)].sort_values(
#                     'Predicted_2025_Price', ascending=False).drop_duplicates('Court/Estate').head(9)
#                 expensive_data = expensive_data[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
#                                                 'Predicted_2025_Price']].rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
#                 st.dataframe(expensive_data)

#     with tab2:
#         st.markdown('<h3 class="subheader">Visualizations (All Courts)</h3>', unsafe_allow_html=True)
        
#         # Debug totals
#         st.write("Debug: Historical Rows for Viz:", len(historical_df))
#         st.write("Debug: Prediction Rows for Viz:", len(prediction_df))

#         # Top 10 Regional Avg Prices (Historical)
#         st.subheader("Top 10 Average Prices (Historical)")
#         if not historical_df.empty:
#             top_areas = historical_df.groupby('Court/Estate')['Transaction Price'].mean().sort_values(ascending=False).head(10)
#             options1 = {
#                 "xAxis": {"type": "category", "data": list(top_areas.index)},
#                 "yAxis": {"type": "value"},
#                 "series": [{"data": [int(y) for y in top_areas.fillna(0)], "type": "bar"}]
#             }
#             st_echarts(options=options1, height="400px")
           
        

#         # Annual Volume
#         st.subheader("Annual Transaction Volume")
#         if not historical_df.empty:
#             yearly_counts = historical_df['Data_Year'].value_counts().sort_index()
#             options2 = {
#                 "xAxis": {"type": "category", "data": list(yearly_counts.index.astype(str))},
#                 "yAxis": {"type": "value"},
#                 "series": [{"data": [int(y) for y in yearly_counts.fillna(0)], "type": "bar"}]
#             }
#             st_echarts(options=options2, height="400px")


#         # Monthly Trend
#         st.subheader("Monthly Price Trend (All Courts)")
#         if not (historical_df.empty and prediction_df.empty):
#             historical = historical_df[['Date', 'Transaction Price']].copy() if not historical_df.empty else pd.DataFrame()
#             historical['Type'] = 'Historical'
#             future = prediction_df[['Date', 'Predicted_2025_Price']].copy() if not prediction_df.empty else pd.DataFrame()
#             if not future.empty:
#                 future.columns = ['Date', 'Transaction Price']
#                 future['Type'] = 'Predicted'
#             combined = pd.concat([historical, future])
#             if not combined.empty:
#                 combined['YearMonth'] = pd.to_datetime(combined['Date']).dt.to_period('M').astype(str)
#                 monthly_trend = combined.groupby(['YearMonth', 'Type'])['Transaction Price'].mean().unstack().fillna(0).reset_index()
#                 if not monthly_trend.empty:
#                     trend_x = list(monthly_trend['YearMonth'])
#                     hist_y = [int(y) for y in monthly_trend.get('Historical', pd.Series([0] * len(trend_x))).fillna(0)]
#                     pred_y = [int(y) for y in monthly_trend.get('Predicted', pd.Series([0] * len(trend_x))).fillna(0)]
#                     options3 = {
#                         "xAxis": {"type": "category", "data": trend_x},
#                         "yAxis": {"type": "value"},
#                         "series": [
#                             {"name": "Historical", "type": "line", "data": hist_y},
#                             {"name": "Predicted", "type": "line", "data": pred_y}
#                         ]
#                     }
#                     st_echarts(options=options3, height="400px")

# if __name__ == "__main__":
#     main()





import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

# Load historical data
@st.cache_data
def load_historical_data():
    def load_and_concat_data(base_path, years):
        all_dfs = []
        year_mapping = {'019': '2019', '020': '2021', '022': '2022', '023': '2023'}
        for short_year in years:
            file_path = os.path.join(base_path, f'ssfs-sale-transactions-{short_year}_en.json')
            if not os.path.isfile(file_path):
                st.warning(f"File for {short_year} not found. Skipping.")
                continue
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                temp_df = pd.DataFrame(data.get("records", []))
                if temp_df.empty:
                    st.warning(f"No records in {short_year} file.")
                    continue
                full_year = year_mapping.get(short_year, short_year)
                temp_df['Data_Year'] = full_year
                if 'Court' in temp_df.columns and 'Court/Estate' not in temp_df.columns:
                    temp_df = temp_df.rename(columns={'Court': 'Court/Estate'})
                temp_df['Floor'] = pd.to_numeric(temp_df['Floor'], errors='coerce').fillna(0).astype(int)
                all_dfs.append(temp_df)
            except Exception as e:
                st.error(f"Error loading {short_year}: {str(e)}")
        if not all_dfs:
            st.error("No historical data loaded.")
            return pd.DataFrame()
        df = pd.concat(all_dfs, axis=0, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date of Agreement for Sale and Purchase (ASP)'], 
                                   format='%d/%m/%Y', errors='coerce')
        df['Flat_ID'] = df['Court/Estate'].fillna('') + '_' + df['Floor'].astype(str).fillna('')
        return df.sort_values('Date').reset_index(drop=True)

    curr_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    data_years = ['019', '020', '022', '023']
    return load_and_concat_data(curr_dir, data_years)

# Load prediction data
@st.cache_data
def load_prediction_data():
    csv_path = 'enhanced_prediction_results_2025_2026.csv'
    if not os.path.isfile(csv_path):
        st.error(f"Prediction file {csv_path} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        required_cols = ['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 'Predicted_2025_Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in prediction data: {missing_cols}")
        else:
            df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce').fillna(0).astype(int)
            df['Flat_ID'] = df['Court/Estate'].fillna('') + '_' + df['Floor'].astype(str).fillna('')
            if 'Lower_CI' not in df.columns:
                df['Lower_CI'] = df['Predicted_2025_Price'] * 0.9
                df['Upper_CI'] = df['Predicted_2025_Price'] * 1.1
        return df
    except Exception as e:
        st.error(f"Error loading prediction data: {str(e)}")
        return pd.DataFrame()

# Main app
def main():
    st.markdown("""
        <style>
        .title { font-size: 2.5em; color: #FF5722; text-align: center; }
        .subheader { color: #1976D2; font-size: 1.5em; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stSidebar .sidebar-content { background-color: #F5F5F5; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">2025-2026 HOS Flat Price Prediction & Analysis (R² 0.958)</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        historical_df = load_historical_data()
        prediction_df = load_prediction_data()

    if historical_df.empty and prediction_df.empty:
        st.error("Both datasets failed to load.")
        return

    # Sidebar debug
    st.sidebar.subheader("Debug Info")
    st.sidebar.write("Total Historical Rows:", len(historical_df))
    st.sidebar.write("Historical Courts:", len(historical_df['Court/Estate'].unique()) if not historical_df.empty else 0)
    st.sidebar.write("Total Prediction Rows:", len(prediction_df))
    st.sidebar.write("Prediction Courts:", len(prediction_df['Court/Estate'].unique()) if not prediction_df.empty else 0)

    all_courts = sorted(set(historical_df['Court/Estate'].unique() if not historical_df.empty else []).union(
        prediction_df['Court/Estate'].unique() if not prediction_df.empty else []))
    if not all_courts:
        st.error("No courts available.")
        return

    # Filters
    st.sidebar.subheader("**Filters**")
    selected_courts = st.sidebar.multiselect("**Select Court/Estate(s)**", all_courts)

    min_price = int(min(historical_df['Transaction Price'].min() if not historical_df.empty else 0,
                       prediction_df['Predicted_2025_Price'].min() if not prediction_df.empty else 0))
    max_price = int(max(historical_df['Transaction Price'].max() if not historical_df.empty else 0,
                       prediction_df['Predicted_2025_Price'].max() if not prediction_df.empty else 0))
    price_range = st.sidebar.slider("**Price Range (HKD)**", min_price, max_price, (min_price, max_price))

    min_floor = int(min(historical_df['Floor'].min() if not historical_df['Floor'].isna().all() else 0,
                       prediction_df['Floor'].min() if not prediction_df['Floor'].isna().all() else 0))
    max_floor = int(max(historical_df['Floor'].max() if not historical_df['Floor'].isna().all() else 0,
                       prediction_df['Floor'].max() if not prediction_df['Floor'].isna().all() else 0))
    floor_range = st.sidebar.slider("**Floor Range**", min_floor, max_floor, (min_floor, max_floor))

    if st.sidebar.button("Reset Filters"):
        selected_courts = []
        price_range = (min_price, max_price)
        floor_range = (min_floor, max_floor)

    # Filter data for listings
    court_historical = historical_df[historical_df['Court/Estate'].isin(selected_courts)].copy() if not historical_df.empty else pd.DataFrame()
    court_predictions = prediction_df[prediction_df['Court/Estate'].isin(selected_courts)].copy() if not prediction_df.empty else pd.DataFrame()

    if not court_historical.empty:
        court_historical = court_historical[
            (court_historical['Transaction Price'] >= price_range[0]) & 
            (court_historical['Transaction Price'] <= price_range[1]) &
            (court_historical['Floor'] >= floor_range[0]) & 
            (court_historical['Floor'] <= floor_range[1])
        ]
    if not court_predictions.empty:
        court_predictions = court_predictions[
            (court_predictions['Predicted_2025_Price'] >= price_range[0]) & 
            (court_predictions['Predicted_2025_Price'] <= price_range[1]) &
            (court_predictions['Floor'] >= floor_range[0]) & 
            (court_predictions['Floor'] <= floor_range[1])
        ]

    # Main content
    st.subheader("Price Listings")
    if selected_courts:
        # Stats Dashboard
        st.write("**Selected Courts Stats**")
        max_price_all = int(court_predictions['Predicted_2025_Price'].max()) if not court_predictions.empty else 0
        avg_price_all = int(court_predictions['Predicted_2025_Price'].mean()) if not court_predictions.empty else 0
        total_flats = len(court_predictions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Price", f"{max_price_all:,} HKD")
        col2.metric("Average Price", f"{avg_price_all:,} HKD")
        col3.metric("Total Flats", total_flats)

        # Comparison Summary Table
        st.write("**Selected Courts Comparison**")
        summary_data = []
        for court in selected_courts:
            court_pred = court_predictions[court_predictions['Court/Estate'] == court].copy()
            if not court_pred.empty:
                summary_data.append({
                    'Court/Estate': court,
                    'Min Predicted Price': int(court_pred['Predicted_2025_Price'].min()) if not court_pred['Predicted_2025_Price'].empty else 0,
                    'Max Predicted Price': int(court_pred['Predicted_2025_Price'].max()) if not court_pred['Predicted_2025_Price'].empty else 0,
                    'Avg Predicted Price': int(court_pred['Predicted_2025_Price'].mean()) if not court_pred['Predicted_2025_Price'].empty else 0,
                    'Total Flats': len(court_pred)
                })
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.highlight_max(subset=['Max Predicted Price', 'Avg Predicted Price', 'Total Flats'], color='yellow')
                         .highlight_min(subset=['Min Predicted Price'], color='lightcoral'))

        # Comparison Chart
        if summary_data:
            st.write("**Predicted Price Comparison Across Selected Courts**")
            options_compare = {
                "title": {"text": "Avg Predicted Price Comparison", "left": "center", "textStyle": {"fontSize": 20}},
                "xAxis": {"type": "category", "data": summary_df['Court/Estate'].tolist(), "axisLabel": {"rotate": 45}},
                "yAxis": {"type": "value", "name": "Avg Price (HKD)"},
                "series": [{"data": summary_df['Avg Predicted Price'].tolist(), "type": "bar", "color": "#FFD700"}],
                "tooltip": {"trigger": "axis"},
                "grid": {"containLabel": True}
            }
            st_echarts(options=options_compare, height="600px", key="comparison_chart")

        # Detailed Listings
        st.write("**Detailed Predictions**")
        all_preds = []
        for court in selected_courts:
            court_pred = court_predictions[court_predictions['Court/Estate'] == court].copy()
            if not court_pred.empty:
                court_pred = court_pred[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
                                        'Date', 'Predicted_2025_Price', 'Lower_CI', 'Upper_CI']]
                court_pred = court_pred.sort_values('Predicted_2025_Price', ascending=False)
                court_pred[['Predicted_2025_Price', 'Lower_CI', 'Upper_CI']] = court_pred[
                    ['Predicted_2025_Price', 'Lower_CI', 'Upper_CI']].round(0).astype(int)
                court_pred = court_pred.rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
                st.dataframe(court_pred.style.highlight_max(subset=['Predicted_2025_Price'], color='yellow'))
                all_preds.append(court_pred)
        
        if all_preds:
            all_preds_df = pd.concat(all_preds)
            csv = all_preds_df.to_csv(index=False)
            st.download_button("Download All Selected Courts Predictions", csv, "all_selected_predictions.csv", "text/csv")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 9 Hottest Courts (by Prediction Count)**")
            hottest = prediction_df['Court/Estate'].value_counts().head(9)
            hottest_data = prediction_df[prediction_df['Court/Estate'].isin(hottest.index)].sort_values(
                'Predicted_2025_Price', ascending=False).drop_duplicates('Court/Estate').head(9)
            hottest_data = hottest_data[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
                                        'Predicted_2025_Price']].rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
            st.dataframe(hottest_data)
        with col2:
            st.write("**Top 9 Most Expensive Courts (by Avg Predicted Price)**")
            expensive = prediction_df.groupby('Court/Estate')['Predicted_2025_Price'].mean().sort_values(ascending=False).head(9)
            expensive_data = prediction_df[prediction_df['Court/Estate'].isin(expensive.index)].sort_values(
                'Predicted_2025_Price', ascending=False).drop_duplicates('Court/Estate').head(9)
            expensive_data = expensive_data[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 
                                            'Predicted_2025_Price']].rename(columns={'Saleable Area of Flats (sq. m.)': 'Area (sq. m.)'})
            st.dataframe(expensive_data)

    st.subheader("Visualizations (All Courts)")
    
    # Top 10 Regional Avg Prices (Historical)
    st.subheader("Top 10 Average Prices (Historical)")
    if not historical_df.empty:
        top_areas = historical_df.groupby('Court/Estate')['Transaction Price'].mean().sort_values(ascending=False).head(10)
        options1 = {
            "title": {"text": "Top 10 Average Prices (Historical)", "left": "center", "textStyle": {"fontSize": 20}},
            "xAxis": {"type": "category", "data": list(top_areas.index), "axisLabel": {"rotate": 45}},
            "yAxis": {"type": "value", "name": "Avg Price (HKD)"},
            "series": [{"data": [int(y) for y in top_areas.fillna(0)], "type": "bar", "color": "#00FF00"}],
            "tooltip": {"trigger": "axis"},
            "grid": {"containLabel": True}
        }
        st_echarts(options=options1, height="600px", key="top_10_chart")

    # Annual Volume
    st.subheader("Annual Transaction Volume")
    if not historical_df.empty:
        yearly_counts = historical_df['Data_Year'].value_counts().sort_index()
        options2 = {
            "title": {"text": "Annual Transaction Volume", "left": "center", "textStyle": {"fontSize": 20}},
            "xAxis": {"type": "category", "data": list(yearly_counts.index.astype(str))},
            "yAxis": {"type": "value", "name": "Volume"},
            "series": [{"data": [int(y) for y in yearly_counts.fillna(0)], "type": "bar", "color": "#00BFFF"}],
            "tooltip": {"trigger": "axis"},
            "grid": {"containLabel": True}
        }
        st_echarts(options=options2, height="600px", key="annual_volume_chart")

    # Monthly Trend
    st.subheader("Monthly Price Trend (All Courts)")
    if not (historical_df.empty and prediction_df.empty):
        historical = historical_df[['Date', 'Transaction Price']].copy() if not historical_df.empty else pd.DataFrame()
        historical['Type'] = 'Historical'
        future = prediction_df[['Date', 'Predicted_2025_Price']].copy() if not prediction_df.empty else pd.DataFrame()
        if not future.empty:
            future.columns = ['Date', 'Transaction Price']
            future['Type'] = 'Predicted'
        combined = pd.concat([historical, future])
        if not combined.empty:
            combined['YearMonth'] = pd.to_datetime(combined['Date']).dt.to_period('M').astype(str)
            monthly_trend = combined.groupby(['YearMonth', 'Type'])['Transaction Price'].mean().unstack().fillna(0).reset_index()
            if not monthly_trend.empty:
                trend_x = list(monthly_trend['YearMonth'])
                hist_y = [int(y) for y in monthly_trend.get('Historical', pd.Series([0] * len(trend_x))).fillna(0)]
                pred_y = [int(y) for y in monthly_trend.get('Predicted', pd.Series([0] * len(trend_x))).fillna(0)]
                options3 = {
                    "title": {"text": "Monthly Price Trend (All Courts)", "left": "center", "textStyle": {"fontSize": 20}},
                    "xAxis": {"type": "category", "data": trend_x, "axisLabel": {"rotate": 45}},
                    "yAxis": {"type": "value", "name": "Price (HKD)"},
                    "series": [
                        {"name": "Historical", "type": "line", "data": hist_y, "color": "#00BFFF"},
                        {"name": "Predicted", "type": "line", "data": pred_y, "color": "#FF4500"}
                    ],
                    "tooltip": {"trigger": "axis"},
                    "legend": {"data": ["Historical", "Predicted"], "top": "5%"},
                    "grid": {"containLabel": True},
                    "dataZoom": [{"type": "slider", "start": 0, "end": 100}]
                }
                st_echarts(options=options3, height="600px", key="monthly_trend_chart")

if __name__ == "__main__":
    main()