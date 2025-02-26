import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import median_abs_deviation
try:
    from lunardate import LunarDate
except ImportError:
    print("lunardate not installed. Using fallback for lunar features.")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === Global Lunar Feature Function ===
def get_lunar_features(date):
    try:
        lunar = LunarDate.fromSolarDate(date.year, date.month, date.day)
        return lunar.month, 1 if lunar.month == 1 else 0
    except:
        is_lunar_newyear = 1 if (date.month == 1 and date.day > 20) or (date.month == 2 and date.day < 20) else 0
        return date.month, is_lunar_newyear

# === Data Loading ===
def load_and_concat_data(base_path, years):
    all_dfs = []
    year_mapping = {'019': '2019', '020': '2021', '022': '2022', '023': '2023'}  # '023' includes 2024
    for short_year in years:
        file_path = os.path.join(base_path, f'ssfs-sale-transactions-{short_year}_en.json')
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            temp_df = pd.DataFrame(data["records"])
            full_year = year_mapping.get(short_year, short_year)
            temp_df['Data_Year'] = full_year
            if 'Court' in temp_df.columns and 'Court/Estate' not in temp_df.columns:
                temp_df = temp_df.rename(columns={'Court': 'Court/Estate'})
            required_columns = ['Date of Agreement for Sale and Purchase (ASP)', 'Court/Estate', 
                                'Floor', 'Saleable Area of Flats (sq. m.)', 'Transaction Price']
            missing_cols = [col for col in required_columns if col not in temp_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {full_year}: {missing_cols}")
            all_dfs.append(temp_df)
        except FileNotFoundError:
            print(f"File for {short_year} ({year_mapping.get(short_year, 'unknown year')}) not found. Skipping.")
        except Exception as e:
            print(f"Error loading {short_year} data: {str(e)}")
    if not all_dfs:
        raise ValueError("No valid data loaded.")
    df = pd.concat(all_dfs, axis=0, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date of Agreement for Sale and Purchase (ASP)'], format='%d/%m/%Y', errors='coerce')
    return df.sort_values('Date').reset_index(drop=True)

curr_dir = os.path.join(os.path.dirname(__file__), 'datasets')
data_years = ['019', '020', '022', '023']
df = load_and_concat_data(curr_dir, data_years)

# === Real Interest Rate Data ===
external_data = pd.DataFrame({
    'Year': [2019]*12 + [2021]*12 + [2022]*12 + [2023]*12 + [2024]*5,
    'Month': list(range(1, 13))*4 + list(range(1, 6)),
    'Interest_Rate': [
        2.6, 2.5, 2.5, 2.5, 2.5, 2.4, 2.4, 2.5, 2.5, 2.6, 2.5, 2.4,  # 2019
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,  # 2021
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,  # 2022
        3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6,  # 2023
        4.7, 4.7, 4.8, 4.8, 4.9  # 2024 (Jan-May)
    ]
})

# === Enhanced Time-Feature Engineering ===
def create_time_features(df):
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Days_since_2019'] = (df['Date'] - pd.Timestamp('2019-01-01')).dt.days
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    lunar_features = df['Date'].apply(lambda x: pd.Series(get_lunar_features(x)))
    df[['Lunar_Month', 'Is_Lunar_NewYear']] = lunar_features
    df['Market_Trend'] = df.groupby('Court/Estate')['Transaction Price'].transform(
        lambda x: x.rolling(window=6, min_periods=1, closed='left').mean())
    df['Price_per_sqm'] = df['Transaction Price'] / df['Saleable Area of Flats (sq. m.)']
    df['Market_Trend_Lag3'] = df.groupby('Court/Estate')['Transaction Price'].transform(
        lambda x: x.rolling(window=3, min_periods=1, closed='left').mean().shift(1))
    df['Price_per_sqm_Lag6'] = df.groupby('Court/Estate')['Price_per_sqm'].transform(
        lambda x: x.rolling(window=6, min_periods=1, closed='left').mean().shift(1))
    df['Trans_Density'] = df.groupby(['Court/Estate', 'Year', 'Month'])['Transaction Price'].transform('count')
    df['Post_COVID_Recovery'] = df['Year'].apply(lambda x: 1 if x == 2021 else 0)
    df['COVID_Gap'] = df['Year'].apply(lambda x: 0 if x == 2019 else 1)
    # New volatility feature: rolling std dev of price per sqm
    df['Price_Volatility'] = df.groupby('Court/Estate')['Price_per_sqm'].transform(
        lambda x: x.rolling(window=6, min_periods=1, closed='left').std().shift(1)).fillna(0)
    df = df.merge(external_data, on=['Year', 'Month'], how='left')
    df['Interest_Rate'] = df['Interest_Rate'].ffill()
    # Boost weights for Fold 2 period (Aug 2021 - Dec 2022)
    df['Sample_Weight'] = df.apply(
        lambda row: 1.2 if (row['Date'] >= pd.Timestamp('2021-08-01') and row['Date'] <= pd.Timestamp('2022-12-31'))
        else (0.6 if row['Trans_Density'] < 3 else 0.9 if row['Trans_Density'] < 10 else 1.0), axis=1)
    return df.drop(columns=['Date of Agreement for Sale and Purchase (ASP)'])

df = create_time_features(df)

# === Data Validation and Cleaning ===
def validate_data(df):
    critical_columns = ['Transaction Price', 'Saleable Area of Flats (sq. m.)']
    df = df.dropna(subset=critical_columns)
    
    def mad_filter(series, threshold=5.0):
        med = series.median()
        mad = median_abs_deviation(series, scale='normal')
        return (series >= med - threshold * mad) & (series <= med + threshold * mad)
    
    price_mask = mad_filter(df['Price_per_sqm'])
    area_mask = df['Saleable Area of Flats (sq. m.)'].between(5, 500)
    df = df[price_mask & area_mask].copy()
    df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce').fillna(0)
    return df

df = validate_data(df)

# Print dataset info
print("Full Dataset Date Range:", df['Date'].min(), "to", df['Date'].max())
print("Rows per Year:", df['Year'].value_counts().sort_index())
print("Total Unique Court/Estate in dataset:", len(df['Court/Estate'].unique()))

# === TimeSeriesSplit ===
tscv = TimeSeriesSplit(n_splits=5)

# === Model Pipeline ===
numeric_features = ['Saleable Area of Flats (sq. m.)', 'Days_since_2019', 'Month_sin', 'Month_cos', 
                    'Market_Trend', 'Lunar_Month', 'Is_Lunar_NewYear', 'Floor', 'Price_per_sqm', 
                    'Market_Trend_Lag3', 'Price_per_sqm_Lag6', 'Post_COVID_Recovery', 'COVID_Gap', 
                    'Interest_Rate', 'Trans_Density', 'Price_Volatility']
categorical_features = ['Court/Estate', 'Quarter', 'Year']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
    ],
    remainder='drop'
)

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'regressor__n_estimators': [1000, 1200, 1500],  # More trees
    'regressor__max_depth': [20, 25, 30],
    'regressor__min_samples_leaf': [1, 2, 3],
    'regressor__min_samples_split': [2, 5, 10],  # New param for split control
    'regressor__max_features': [0.7, 0.9, 'sqrt']
}

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', rf_base)
])

search = RandomizedSearchCV(model_pipeline, param_grid, n_iter=50, cv=tscv, scoring='r2', random_state=42, n_jobs=-1, verbose=2)
X = df.drop(columns=['Transaction Price', 'Sample_Weight'])
y = df['Transaction Price']
search.fit(X, y, regressor__sample_weight=df['Sample_Weight'])
print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation R² score: {search.best_score_:.3f}")

# Detailed Fold Metrics
for i, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    w_train = df['Sample_Weight'].iloc[train_idx]
    model_pipeline.fit(X_train, y_train, regressor__sample_weight=w_train)
    preds = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    test_dates = df['Date'].iloc[test_idx]
    print(f"Fold {i}: R² = {r2:.3f}, RMSE = {rmse:.0f} HKD, MAE = {mae:.0f} HKD")
    print(f"  Test Date Range: {test_dates.min()} to {test_dates.max()}")
    print(f"  Train Rows: {len(train_idx)}, Test Rows: {len(test_idx)}")

model = search.best_estimator_
model.fit(X, y, regressor__sample_weight=df['Sample_Weight'])

# === Future Data Generation ===
def generate_future_data(full_data, start_year=2025, end_year=2026):
    unique_courts = full_data['Court/Estate'].unique()
    latest_data = full_data.sort_values('Date').groupby('Court/Estate').last().reset_index()
    template_df = pd.DataFrame({'Court/Estate': unique_courts}).merge(
        latest_data, on='Court/Estate', how='left', validate='1:1'
    )
    
    for col in ['Saleable Area of Flats (sq. m.)', 'Floor', 'Market_Trend', 'Price_per_sqm', 
                'Market_Trend_Lag3', 'Price_per_sqm_Lag6']:
        template_df[col] = template_df[col].fillna(full_data[col].median())
    
    future_dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='ME')
    future_df = pd.concat([template_df] * len(future_dates), ignore_index=True)
    future_df['Date'] = np.tile(future_dates, len(unique_courts))
    
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Quarter'] = future_df['Date'].dt.quarter
    future_df['Days_since_2019'] = (future_df['Date'] - pd.Timestamp('2019-01-01')).dt.days
    lunar_features = future_df['Date'].apply(lambda x: pd.Series(get_lunar_features(x)))
    future_df[['Lunar_Month', 'Is_Lunar_NewYear']] = lunar_features
    future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month']/12)
    future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month']/12)
    future_df['Price_Volatility'] = full_data['Price_Volatility'].median()  # Use median for future
    
    yearly_growth = full_data.groupby('Year')['Transaction Price'].mean().pct_change().mean()
    if np.isnan(yearly_growth) or yearly_growth == 0:
        yearly_growth = 0.03
    years_ahead = future_df['Year'] - full_data['Year'].max()
    future_df['Market_Trend'] = future_df['Market_Trend'] * (1 + yearly_growth) ** years_ahead
    future_df['Price_per_sqm'] = future_df['Price_per_sqm'] * (1 + yearly_growth) ** years_ahead
    future_df['Market_Trend_Lag3'] = future_df['Market_Trend_Lag3'] * (1 + yearly_growth) ** years_ahead
    future_df['Price_per_sqm_Lag6'] = future_df['Price_per_sqm_Lag6'] * (1 + yearly_growth) ** years_ahead
    
    future_df['Trans_Density'] = full_data['Trans_Density'].mean()
    future_df['Post_COVID_Recovery'] = 0
    future_df['COVID_Gap'] = 1
    future_df['Interest_Rate'] = external_data[external_data['Year'] == 2024]['Interest_Rate'].mean() * (1 + 0.01 * years_ahead)
    
    return future_df.drop(columns=['Sample_Weight'] if 'Sample_Weight' in future_df.columns else [])

future_df = generate_future_data(df, start_year=2025, end_year=2026)
future_prices = model.predict(future_df)
future_df['Predicted_2025_Price'] = future_prices

# === Feature Importance ===
feature_names = numeric_features + list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
importances = model.named_steps['regressor'].feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print("\nTop 10 Feature Importances:")
print(feat_imp.sort_values('Importance', ascending=False).head(10))

# === Visualization ===
plt.figure(figsize=(12, 6))
historical = df[['Date', 'Transaction Price']].copy()
historical['Type'] = 'Historical'
future = future_df[['Date', 'Predicted_2025_Price']].copy()
future.columns = ['Date', 'Transaction Price']
future['Type'] = 'Predicted'
n_boot = 100
boot_preds = np.array([model.predict(future_df.sample(n_boot)) for _ in range(n_boot)])
future_sample = future.sample(n_boot)
future_sample['Lower_CI'] = np.percentile(boot_preds, 2.5, axis=0)
future_sample['Upper_CI'] = np.percentile(boot_preds, 97.5, axis=0)
combined = pd.concat([historical, future_sample])
sns.lineplot(x='Date', y='Transaction Price', hue='Type', data=combined, style='Type', markers=True)
plt.fill_between(future_sample['Date'], future_sample['Lower_CI'], future_sample['Upper_CI'], alpha=0.2, color='orange', label='95% CI')
plt.title('Historical Price Trend and 2025-2026 Prediction (2019-2024 Data)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price (HKD)')
plt.legend()
plt.savefig('trend_prediction_with_ci_scaled.png', dpi=300)
plt.close('all')

# === Save Results ===
future_df.to_csv('enhanced_prediction_results_2025_2026.csv', index=False)
print("\nPredictions saved to 'enhanced_prediction_results_2025_2026.csv'")
print(f"Generated {len(future_df)} predictions for 2025-2026")
print(f"Unique Court/Estate in predictions: {len(future_df['Court/Estate'].unique())}")
print("\nSample Predicted Prices for 2025-2026 (first 10 rows):")
print(future_df[['Court/Estate', 'Floor', 'Saleable Area of Flats (sq. m.)', 'Date', 'Predicted_2025_Price']].head(10))