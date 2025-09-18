
from automl_tool.automl import AutoML
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error
from automl_tool.automl import AutoML
from automl_tool.preprocessing import ts_train_test_split
from pandas.tseries.frequencies import to_offset

warnings.filterwarnings("ignore")
np.random.seed(42)

# Helpers
make_dates = lambda n, freq='ME', start='2000-01-01': pd.date_range(start, periods=n, freq=freq)

def df_from_values(values, start='2000-01-01', freq='ME'):
	return pd.DataFrame({'date': make_dates(len(values), freq, start), 'value': np.asarray(values, dtype=float)})

def ensure_regular_frequency(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value') -> pd.DataFrame:
	d = df.copy()
	d[date_col] = pd.to_datetime(d[date_col])
	d = d.sort_values(date_col).drop_duplicates(subset=[date_col])
	# Try to infer frequency; fallback to median delta if needed
	freq = pd.infer_freq(d[date_col])
	if freq is None:
		deltas = d[date_col].diff().dropna()
		if len(deltas) == 0:
			freq = 'D'
		else:
			td = deltas.median()
			try:
				freq = to_offset(td)
			except Exception:
				freq = 'D'
	full_index = pd.date_range(d[date_col].iloc[0], d[date_col].iloc[-1], freq=freq)
	d = d.set_index(date_col).reindex(full_index)
	# Interpolate missing values over time; fallback to ffill then bfill
	if d[value_col].isna().any():
		try:
			d[value_col] = d[value_col].interpolate(method='time')
		except Exception:
			d[value_col] = d[value_col].ffill().bfill()
	d = d.reset_index().rename(columns={'index': date_col})
	return d[[date_col, value_col]]

# Synthetic dataset generators (diverse shapes)
def synth_seasonal(n=1000):
	t = np.arange(n)
	y = 10*np.sin(2*np.pi*t/12) + 0.5*t + np.random.normal(0, 2, n)
	return df_from_values(y)

def synth_linear(n=1000):
	t = np.arange(n)
	y = 0.3*t + np.random.normal(0, 3, n)
	return df_from_values(y)

def synth_quadratic(n=1000):
	t = np.arange(n)
	y = 0.01*(t-n/2)**2 + np.random.normal(0, 2, n)
	return df_from_values(y)

def synth_logistic(n=1000):
	t = np.arange(n)
	midpoint = n/2
	y = 100/(1+np.exp(-(t-midpoint)/10)) + np.random.normal(0, 2, n)
	return df_from_values(y)

def synth_walk(n=1000):
	rng = np.random.default_rng(123)
	y = np.cumsum(rng.normal(0.3, 1.0, n))
	return df_from_values(y)

def synth_piecewise(n=1000):
	t = np.arange(n)
	k1 = int(n*0.3); k2 = int(n*0.65)
	base = np.piecewise(t,
						[t < k1, (t >= k1) & (t < k2), t >= k2],
						[lambda x: 0.2*x,
						lambda x: (0.2*k1) + (-0.1)*(x-k1),
						lambda x: (0.2*k1) + (-0.1)*(k2-k1) + 0.4*(x-k2)])
	y = base + np.random.normal(0, 2, n)
	return df_from_values(y)

def synth_spiky(n=1000):
	rng = np.random.default_rng(0)
	y = 20 + np.sin(2*np.pi*np.arange(n)/24) + rng.normal(0, 2, n)
	idx = rng.choice(n, size=max(10, n//30), replace=False)
	y[idx] += rng.uniform(10, 25, len(idx))
	return df_from_values(y)

def synth_multiseason(n=1000):
	t = np.arange(n)
	y = 5*np.sin(2*np.pi*t/12) + 3*np.sin(2*np.pi*t/6) + np.random.normal(0, 2, n)
	return df_from_values(y)

if __name__ == "__main__":

	# Collect datasets (n=1000 each synthetic)
	datasets = {
		'Seasonal+Trend': synth_seasonal(n=1000),
		'Linear Trend': synth_linear(n=1000),
		'Quadratic': synth_quadratic(n=1000),
		'Logistic (S-curve)': synth_logistic(n=1000),
		'Random Walk (drift)': synth_walk(n=1000),
		'Piecewise (changepoints)': synth_piecewise(n=1000),
		'Spiky Intermittent': synth_spiky(n=1000),
		'Multi-seasonal': synth_multiseason(n=1000),
	}

	# Add real datasets from statsmodels
	try:
		import statsmodels.api as sm
		# Sunspots (yearly)
		try:
			sun = sm.datasets.sunspots.load_pandas().data
			df_sun = pd.DataFrame({
				'date': pd.to_datetime(sun['YEAR'], format='%Y', errors='coerce'),
				'value': sun['SUNACTIVITY'].astype(float)
			}).dropna()
			datasets['Sunspots'] = df_sun
		except Exception:
			pass

		# Mauna Loa CO2 (weekly)
		try:
			co2 = sm.datasets.co2.load_pandas().data
			co2 = co2.copy()
			if 'co2' in co2.columns and 'date' not in co2.columns:
				co2 = co2.reset_index().rename(columns={'index': 'date', 'co2': 'value'})
			else:
				if 'date' not in co2.columns:
					co2 = co2.reset_index().rename(columns={'index': 'date'})
				if 'value' not in co2.columns:
					first_val = [c for c in co2.columns if c != 'date'][0]
					co2 = co2.rename(columns={first_val: 'value'})
			co2 = co2[['date', 'value']].dropna()
			datasets['CO2'] = co2
		except Exception:
			pass
	except Exception:
		pass

	# Add selected FRED series
	try:
		from pandas_datareader import data as pdr
		fred_codes = ['CPIAUCSL', 'UNRATE', 'INDPRO']
		for code in fred_codes:
			try:
				df_fred = pdr.DataReader(code, 'fred', start='1990-01-01')
				df_fred = df_fred.rename(columns={code: 'value'}).reset_index().rename(columns={'DATE': 'date'})
				df_fred = df_fred.dropna()
				datasets[code] = df_fred
			except Exception:
				continue
	except Exception:
		pass

	# Filter to only the datasets the user wants
	allowed = {
		'Seasonal+Trend', 'Linear Trend', 'Quadratic', 'Logistic (S-curve)',
		'Random Walk (drift)', 'Piecewise (changepoints)', 'Spiky Intermittent',
		'Multi-seasonal', 'Sunspots', 'CO2', 'CPIAUCSL', 'UNRATE', 'INDPRO'
	}
	datasets = {k: v for k, v in datasets.items() if k in allowed}

	# Normalize to equidistant dates
	for k in list(datasets.keys()):
		datasets[k] = ensure_regular_frequency(datasets[k], 'date', 'value')

	fdw, holdout_window, forecast_window = 18, 24, 1
	maes = []

	for name, df in datasets.items():
		if len(df) < fdw + holdout_window + forecast_window + 1:
			continue
		X, y = df, df['value']
		X_train, X_holdout, y_train, y_holdout = ts_train_test_split(
			X, y, 'value', 'date', fdw, holdout_window, forecast_window=forecast_window
	)
		# Skip datasets that are too short for 5-fold TimeSeriesSplit with this test size
		if len(X_train) <= 5 * holdout_window:
			continue
		automl_mod = AutoML(X_train, y_train, 'value', time_series=True)
		automl_mod.fit_pipeline(holdout_window=holdout_window)
		preds = automl_mod.fitted_pipeline.best_estimator_.predict(X_holdout)
		mae = mean_absolute_error(y_holdout, preds)
		maes.append(mae)
		print(f"{name}: {mae:.3f}")

	if maes:
		print(f"Average MAE: {float(np.mean(maes)):.3f}")