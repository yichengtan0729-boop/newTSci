"""
Model Library for Time Series Forecasting
Contains all individual model prediction functions for time series forecasting.
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
# prophet and tbats are lazily imported inside their predict_* functions
# to avoid crashing the entire model library when they are not installed.

logger = logging.getLogger(__name__)


def _get_torch_device() -> Optional[Any]:
    """Return MPS device if available and PYTORCH_USE_MPS=1, else CPU."""
    import os
    try:
        import torch
        if os.environ.get("PYTORCH_USE_MPS", "1") == "1":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return None  # fallback to cpu below


def _filter_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the kwargs that *cls.__init__* accepts (avoids unexpected-kwarg errors).

    Works with scikit-learn estimators, XGBoost, LightGBM, Prophet, etc.
    Falls back to returning an empty dict if introspection fails.
    """
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}
        # If **kwargs is present, allow everything through
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(params)
        return {k: v for k, v in params.items() if k in valid_keys}
    except (ValueError, TypeError):
        return {}


def _create_time_series_features(series: pd.Series, lookback: int = 10) -> tuple:
    """
    Create time series features for machine learning models.
    
    Args:
        series: Time series data
        lookback: Number of lag features to create
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series.iloc[i-lookback:i].values)
        y.append(series.iloc[i])
    return np.array(X), np.array(y)


def _create_enriched_features(data: Dict[str, Any], lookback: int = 10) -> tuple:
    """Build feature matrix from a data dict that may contain L2-engineered columns.

    If *data* has only ``"value"`` (or is a DataFrame), falls back to
    ``_create_time_series_features``.  Otherwise it stacks pre-computed
    columns (e.g. ``lag_1``, ``roll_mean``, ``sin_24``) alongside the
    standard lookback window from ``"value"``.

    Returns (X, y) with shapes suitable for sklearn regressors.
    """
    if isinstance(data, pd.DataFrame):
        series = data["value"].dropna() if "value" in data.columns else data.iloc[:, 0].dropna()
        return _create_time_series_features(series, lookback)

    if not isinstance(data, dict):
        series = pd.Series(np.asarray(data).flatten())
        return _create_time_series_features(series, lookback)

    vals = np.asarray(data.get("value", [])).flatten()
    extra_keys = [k for k in data if k != "value"]

    # No extras → standard path
    if not extra_keys or len(vals) == 0:
        series = pd.Series(vals)
        return _create_time_series_features(series, lookback)

    n = len(vals)
    series = pd.Series(vals)

    # Standard lag window features
    base_X, y = _create_time_series_features(series, lookback)
    if len(base_X) == 0:
        return base_X, y

    # Align extra columns to the same rows (starting from index *lookback*)
    extra_cols = []
    for k in sorted(extra_keys):
        col = np.asarray(data[k]).flatten()
        if len(col) == n:
            extra_cols.append(col[lookback:])
    if extra_cols:
        extras = np.column_stack(extra_cols)
        X = np.hstack([base_X, extras])
    else:
        X = base_X
    return X, y

def predict_arima(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """ARIMA model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'p', 'q', 'd' for ARIMA order
        horizon: Number of steps to predict into the future
        
    Returns:
        List of predicted values for the specified horizon
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        p = params.get('p', 1)
        q = params.get('q', 1)
        d = params.get('d', 1)
        
        # Fit ARIMA model
        model = ARIMA(series, order=(p, d, q))
        fitted_model = model.fit()
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"ARIMA prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_random_walk(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Random Walk model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in random walk)
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        last_value = series.iloc[-1]
        std_dev = series.diff().std()
        
        predictions = []
        current_value = last_value
        
        for _ in range(horizon):
            # Random walk: next value = current value + random noise
            noise = np.random.normal(0, std_dev)
            current_value = current_value + noise
            predictions.append(max(0, current_value))  # Ensure non-negative
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Random Walk prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_moving_average(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Moving Average model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'window_size'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        window_size = params.get('window_size', 10)
        
        # Calculate moving average
        ma = series.rolling(window=window_size).mean().iloc[-1]
        
        # Simple prediction: use the last moving average value
        predictions = [ma] * horizon
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Moving Average prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_polynomial_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Polynomial Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'degree'
        horizon: Number of steps to predict into the future
    """
    try:
        if isinstance(data, dict):
            series = pd.Series(np.asarray(data.get("value", [])).flatten()).dropna()
        elif isinstance(data, pd.DataFrame):
            series = data['value'].dropna()
        else:
            series = pd.Series(np.asarray(data).flatten()).dropna()
        degree = params.get('degree', 2)
        
        # Create polynomial features
        X = np.arange(len(series)).reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        # Fit polynomial regression
        model = LinearRegression()
        model.fit(X_poly, series)
        
        # Predict future values
        predictions = []
        for i in range(len(series), len(series) + horizon):
            X_future = poly_features.transform([[i]])
            pred = model.predict(X_future)[0]
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Polynomial Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_ridge_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Ridge Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=10)
        
        safe_params = _filter_params(Ridge, params)
        model = Ridge(**safe_params)
        model.fit(X, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Ridge Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lasso_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Lasso Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=10)
        
        safe_params = _filter_params(Lasso, params)
        model = Lasso(**safe_params)
        model.fit(X, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Lasso Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_elastic_net(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Elastic Net model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha', 'l1_ratio'
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=10)
        
        safe_params = _filter_params(ElasticNet, params)
        model = ElasticNet(**safe_params)
        model.fit(X, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Elastic Net prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_svr(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Support Vector Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for SVR
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=10)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        safe_params = _filter_params(SVR, params)
        model = SVR(**safe_params)
        model.fit(X_scaled, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        for _ in range(horizon):
            pred = model.predict(last_features_scaled)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
            last_features_scaled = scaler.transform(last_features)
        
        return predictions
        
    except Exception as e:
        logger.warning(f"SVR prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_gradient_boosting(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Gradient Boosting model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for GradientBoostingRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        # Use enriched features if data is dict with extra L2 columns
        X, y = _create_enriched_features(data, lookback=10)
        
        # Train Gradient Boosting model
        safe_params = _filter_params(GradientBoostingRegressor, params)
        model = GradientBoostingRegressor(**safe_params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Gradient Boosting prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_xgboost(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """XGBoost model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for XGBRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        import xgboost as xgb
        
        # Use enriched features if data is dict with extra L2 columns
        X, y = _create_enriched_features(data, lookback=10)
        
        # Train XGBoost model
        safe_params = _filter_params(xgb.XGBRegressor, params)
        model = xgb.XGBRegressor(**safe_params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        n_features = X.shape[1]
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update lag portion of features (first 10 cols); extras stay frozen
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred  # newest lag = prediction
        
        return predictions
        
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lightgbm(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """LightGBM model prediction for time series forecasting.

    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for LGBMRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        import lightgbm as lgb

        # Use enriched features if data is dict with extra L2 columns
        X, y = _create_enriched_features(data, lookback=10)

        # Train LightGBM model
        safe_params = _filter_params(lgb.LGBMRegressor, params)
        safe_params.setdefault("verbosity", -1)
        model = lgb.LGBMRegressor(**safe_params)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            model.fit(X, y)

        # Predict (same warning can appear when passing ndarray to predict)
        predictions = []
        last_features = X[-1].reshape(1, -1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            for _ in range(horizon):
                pred = model.predict(last_features)[0]
                predictions.append(max(0, pred))

                last_features = last_features.copy()
                last_features[0, :10] = np.roll(last_features[0, :10], -1)
                last_features[0, 9] = pred

        return predictions

    except Exception as e:
        logger.warning(f"LightGBM prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_neural_network(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Neural Network model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for MLPRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=10)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        safe_params = _filter_params(MLPRegressor, params)
        model = MLPRegressor(**safe_params)
        model.fit(X_scaled, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        for _ in range(horizon):
            pred = model.predict(last_features_scaled)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
            last_features_scaled = scaler.transform(last_features)
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Neural Network prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lstm(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """LSTM model prediction for time series forecasting using PyTorch.

    Builds a real ``torch.nn.LSTM`` network, trains it on sliding-window
    sequences from the input series, and predicts *horizon* steps ahead
    in an autoregressive fashion.

    Supported *params*:
        lookback (int, default 20): length of input sequence window.
        units (int, default 64): LSTM hidden size.
        layers (int, default 2): number of stacked LSTM layers.
        dropout (float, default 0.1): dropout between LSTM layers.
        epochs (int, default 50): training epochs.
        batch_size (int, default 32): mini-batch size.
        learning_rate (float, default 0.001): Adam learning rate.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # --- extract 1-D series ---
        if isinstance(data, dict):
            vals = np.asarray(data.get("value", [])).flatten().astype(float)
        elif isinstance(data, pd.DataFrame):
            vals = data["value"].dropna().values.astype(float)
        else:
            vals = np.asarray(data).flatten().astype(float)

        lookback = int(params.get("lookback", 20))
        if len(vals) < lookback + 2:
            return predict_default(data, params, horizon)

        # --- hyper-parameters ---
        units = int(params.get("units", 64))
        num_layers = int(params.get("layers", 2))
        dropout = float(params.get("dropout", 0.1))
        epochs = int(params.get("epochs", 50))
        batch_size = int(params.get("batch_size", 32))
        lr = float(params.get("learning_rate", 0.001))

        # --- normalise ---
        scaler = StandardScaler()
        scaled = scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        # --- sliding-window sequences ---
        X_list, y_list = [], []
        for i in range(lookback, len(scaled)):
            X_list.append(scaled[i - lookback : i])
            y_list.append(scaled[i])
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        X_t = torch.from_numpy(X_arr).unsqueeze(-1)   # (N, lookback, 1)
        y_t = torch.from_numpy(y_arr).unsqueeze(-1)    # (N, 1)

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
        )

        # --- model definition ---
        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=units,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(units, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = _get_torch_device() or torch.device("cpu")
        model = _LSTM().to(device)
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        # --- train ---
        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimiser.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimiser.step()

        # --- autoregressive prediction ---
        model.eval()
        current_seq = torch.from_numpy(
            scaled[-lookback:].astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1)  # (1, lookback, 1)

        preds_scaled: List[float] = []
        with torch.no_grad():
            for _ in range(horizon):
                pred = model(current_seq.to(device))       # (1, 1)
                val = pred.item()
                preds_scaled.append(val)
                # shift window: drop first, append prediction
                new_step = pred.unsqueeze(-1)               # (1, 1, 1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)

        # --- inverse-transform ---
        predictions = (
            scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
            .flatten()
            .tolist()
        )
        return predictions

    except Exception as e:
        logger.warning(f"LSTM prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_exponential_smoothing(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Exponential Smoothing model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for ExponentialSmoothing
        horizon: Number of steps to predict into the future
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Fit model — filter params to only valid ExponentialSmoothing kwargs
        safe_params = _filter_params(ExponentialSmoothing, params)
        model = ExponentialSmoothing(series, **safe_params)
        fitted_model = model.fit()
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"ExponentialSmoothing prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_linear_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Linear Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for LinearRegression
        horizon: Number of steps to predict into the future
    """
    try:
        X, y = _create_enriched_features(data, lookback=5)
        
        safe_params = _filter_params(LinearRegression, params)
        lr = LinearRegression(**safe_params)
        lr.fit(X, y)
        
        predictions = []
        last_features = X[-1].reshape(1, -1)
        lookback = min(5, X.shape[1])
        
        for _ in range(horizon):
            pred = lr.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :lookback] = np.roll(last_features[0, :lookback], -1)
            last_features[0, lookback - 1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"LinearRegression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_random_forest(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Random Forest model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for RandomForestRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        # Use enriched features if data is dict with extra L2 columns
        X, y = _create_enriched_features(data, lookback=10)
        
        # Train model
        safe_params = _filter_params(RandomForestRegressor, params)
        rf = RandomForestRegressor(**safe_params)
        rf.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = rf.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            last_features = last_features.copy()
            last_features[0, :10] = np.roll(last_features[0, :10], -1)
            last_features[0, 9] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"RandomForest prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_prophet(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Prophet model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for Prophet
        horizon: Number of steps to predict into the future
    """
    try:
        from prophet import Prophet  # lazy import
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Ensure we have a proper index
        if data.index.name is None:
            data.index.name = 'ds'
        
        # Prepare Prophet data format - ensure proper date handling
        df_prophet = data.reset_index()
        
        # Check if the index column contains valid dates
        if df_prophet.columns[0] == 'ds':
            # If the index is already named 'ds', use it directly
            pass
        else:
            # Create a proper date index if needed
            df_prophet.columns = ['ds', 'y']
        
        # Ensure 'ds' column contains valid dates
        if not pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
            # Try to convert to datetime
            try:
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            except Exception as e:
                logger.warning(f"Failed to convert dates for Prophet: {e}")
                # Create a simple numeric index as fallback
                df_prophet['ds'] = pd.date_range(start='2020-01-01', periods=len(df_prophet), freq='D')
        
        # Ensure 'y' column exists and contains numeric values
        if 'y' not in df_prophet.columns:
            df_prophet['y'] = data['value'].values
        
        # Remove any rows with invalid dates or values
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) == 0:
            logger.warning("No valid data for Prophet after cleaning")
            return predict_default(data, params, horizon)
        
        # Fit model with default parameters if none provided
        prophet_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive'
        }
        prophet_params.update(params)
        
        model = Prophet(**prophet_params)
        model.fit(df_prophet)
        
        # Create future dates
        last_date = df_prophet['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
        future = pd.DataFrame({'ds': future_dates})
        
        # Predict
        forecast = model.predict(future)
        predictions = forecast['yhat'].tolist()
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Prophet prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_tbats(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """TBATS model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for TBATS
        horizon: Number of steps to predict into the future
    """
    try:
        from tbats import TBATS  # lazy import        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Fit TBATS model — filter to valid TBATS kwargs
        safe_params = _filter_params(TBATS, params)
        model = TBATS(**safe_params)
        fitted_model = model.fit(series)
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"TBATS prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_theta(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Theta model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in Theta method)
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Simple Theta method implementation
        # Decompose into trend and seasonal components
        n = len(series)
        theta = 2  # Default theta parameter
        
        # Calculate trend using linear regression
        x = np.arange(n)
        trend_coef = np.polyfit(x, series, 1)
        trend = trend_coef[0] * x + trend_coef[1]
        
        # Calculate seasonal component (simple moving average)
        seasonal_period = params.get('seasonal_period', 12)
        seasonal = series.rolling(window=seasonal_period, center=True).mean()
        seasonal = seasonal.bfill().ffill()
        
        # Theta decomposition
        theta_trend = trend + (series - trend) / theta
        theta_seasonal = seasonal
        
        # Predict
        predictions = []
        for i in range(horizon):
            future_trend = trend_coef[0] * (n + i) + trend_coef[1]
            future_seasonal = seasonal.iloc[-seasonal_period + (i % seasonal_period)]
            pred = future_trend + (future_seasonal - future_trend) / theta
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Theta prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_croston(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Croston model prediction for intermittent time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        alpha = params.get('alpha', 0.4)
        
        # Croston method for intermittent demand
        # Separate demand size and inter-demand intervals
        demand_sizes = []
        intervals = []
        last_demand_idx = -1
        
        for i, value in enumerate(series):
            if value > 0:
                if last_demand_idx >= 0:
                    intervals.append(i - last_demand_idx)
                demand_sizes.append(value)
                last_demand_idx = i
        
        if len(demand_sizes) == 0:
            return [0] * horizon
        
        # Calculate Croston parameters
        avg_demand_size = np.mean(demand_sizes)
        avg_interval = np.mean(intervals) if intervals else 1
        
        # Predict
        predictions = []
        for _ in range(horizon):
            pred = avg_demand_size / avg_interval
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Croston prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_transformer(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Transformer model prediction for time series forecasting using PyTorch.

    Builds a real ``torch.nn.TransformerEncoder`` with sinusoidal positional
    encoding, trains it on sliding-window sequences, and predicts *horizon*
    steps ahead autoregressively.

    Supported *params*:
        lookback (int, default 20): length of input sequence window.
        d_model (int, default 64): internal embedding dimension.
        nhead (int, default 4): number of attention heads.
        num_layers (int, default 2): number of TransformerEncoderLayers.
        dropout (float, default 0.1): dropout rate.
        epochs (int, default 50): training epochs.
        batch_size (int, default 32): mini-batch size.
        learning_rate (float, default 0.001): Adam learning rate.
    """
    try:
        import math
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # --- extract 1-D series ---
        if isinstance(data, dict):
            vals = np.asarray(data.get("value", [])).flatten().astype(float)
        elif isinstance(data, pd.DataFrame):
            vals = data["value"].dropna().values.astype(float)
        else:
            vals = np.asarray(data).flatten().astype(float)

        lookback = int(params.get("lookback", 20))
        if len(vals) < lookback + 2:
            return predict_default(data, params, horizon)

        # --- hyper-parameters ---
        d_model = int(params.get("d_model", 64))
        nhead = int(params.get("nhead", 4))
        # d_model must be divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * max(1, d_model // nhead)
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.1))
        epochs = int(params.get("epochs", 50))
        batch_size = int(params.get("batch_size", 32))
        lr = float(params.get("learning_rate", 0.001))

        # --- normalise ---
        scaler = StandardScaler()
        scaled = scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        # --- sliding-window sequences ---
        X_list, y_list = [], []
        for i in range(lookback, len(scaled)):
            X_list.append(scaled[i - lookback : i])
            y_list.append(scaled[i])
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        X_t = torch.from_numpy(X_arr).unsqueeze(-1)   # (N, lookback, 1)
        y_t = torch.from_numpy(y_arr).unsqueeze(-1)    # (N, 1)

        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True,
        )

        # --- positional encoding ---
        class _PosEnc(nn.Module):
            def __init__(self, d_model: int, max_len: int = 5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(pos * div)
                if d_model > 1:
                    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
                self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.pe[:, : x.size(1), :]

        # --- transformer model ---
        class _TSTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(1, d_model)
                self.pos_enc = _PosEnc(d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.input_proj(x)          # (B, L, d_model)
                x = self.pos_enc(x)
                x = self.encoder(x)             # (B, L, d_model)
                return self.fc(x[:, -1, :])     # (B, 1)

        device = _get_torch_device() or torch.device("cpu")
        model = _TSTransformer().to(device)
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        # --- train ---
        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimiser.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimiser.step()

        # --- autoregressive prediction ---
        model.eval()
        current_seq = torch.from_numpy(
            scaled[-lookback:].astype(np.float32)
        ).unsqueeze(0).unsqueeze(-1)  # (1, lookback, 1)

        preds_scaled: List[float] = []
        with torch.no_grad():
            for _ in range(horizon):
                pred = model(current_seq.to(device))       # (1, 1)
                val = pred.item()
                preds_scaled.append(val)
                new_step = pred.unsqueeze(-1)               # (1, 1, 1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)

        # --- inverse-transform ---
        predictions = (
            scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
            .flatten()
            .tolist()
        )
        return predictions

    except Exception as e:
        logger.warning(f"Transformer prediction failed: {e}")
        return predict_default(data, params, horizon)



def predict_ttm(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """TTM (IBM Granite TinyTimeMixer) zero-shot prediction.

    Notes
    -----
    - Uses `granite-tsfm` / `tsfm_public` if installed.
    - If unavailable or an exception occurs, falls back to `predict_transformer`.

    Expected inputs
    ---------------
    - `data`: dict with key "value" (1-D array-like) OR a DataFrame with a 'value' column.
    - `params`: supports:
        - context_length (int): model context length (default from config/hyperparams).
        - model_path (str): HuggingFace model id (default: ibm-granite/granite-timeseries-ttm-r2).
        - revision (str|None): model revision (optional).
        - device (str): "cuda" or "cpu" (optional).
    """
    try:
        import pandas as pd
        import numpy as np
        import torch

        # 1) Extract series
        if isinstance(data, dict):
            vals = np.asarray(data.get("value", [])).flatten().astype(float)
        elif isinstance(data, pd.DataFrame):
            vals = data["value"].dropna().values.astype(float)
        else:
            vals = np.asarray(data).flatten().astype(float)

        if len(vals) < 10:
            return predict_default(data, params, horizon)

        # 2) Config
        model_path = params.get("model_path") or "ibm-granite/granite-timeseries-ttm-r2"
        revision = params.get("revision", None)
        context_length = int(params.get("context_length", 512))
        device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # 3) Build dataframe with a synthetic timestamp column (required by tsfm_public toolkit)
        # Use minute frequency to be generic; TTM is resolution-aware mainly via revision/context/pred length.
        start = pd.Timestamp("2000-01-01", tz="UTC")
        df = pd.DataFrame({
            "date": pd.date_range(start=start, periods=len(vals), freq="min"),
            "value": vals,
        })

        # 4) Load model + pipeline (zero-shot)
        from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesForecastingPipeline, TimeSeriesPreprocessor

        tsp = TimeSeriesPreprocessor(
            id_columns=[],
            timestamp_column="date",
            target_columns=["value"],
            prediction_length=int(horizon),
            context_length=int(context_length),
            scaling=True,
            scaling_type="standard",
        )

        model = TinyTimeMixerForPrediction.from_pretrained(
            model_path,
            revision=revision,
            num_input_channels=tsp.num_input_channels,
            prediction_channel_indices=tsp.prediction_channel_indices,
            exogenous_channel_indices=tsp.exogenous_channel_indices,
        )
        pipe = TimeSeriesForecastingPipeline(model=model, device=device)

        # The pipeline expects preprocessed input df
        inp = tsp.preprocess(df)
        # Forecast: returns a dataframe with prediction column naming convention "<target>_prediction"
        out_df = pipe.predict(inp)
        pred_col = "value_prediction"
        if pred_col not in out_df.columns:
            # Fall back: pick last numeric col
            numeric_cols = [c for c in out_df.columns if c != "date" and pd.api.types.is_numeric_dtype(out_df[c])]
            if not numeric_cols:
                raise RuntimeError("TTM pipeline did not return numeric predictions")
            pred_col = numeric_cols[0]

        preds = out_df[pred_col].tail(horizon).values.astype(float).tolist()
        if len(preds) < horizon:
            # pad if needed
            preds = preds + [preds[-1]] * (horizon - len(preds))
        return preds[:horizon]

    except Exception as e:
        logger.warning(f"TTM prediction failed, falling back to Transformer: {e}")
        return predict_transformer(data, params, horizon)

def predict_default(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Default prediction method for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in default method)
        horizon: Number of steps to predict into the future
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    series = data['value'].dropna()
    
    # Simple moving average prediction
    window = min(10, len(series) // 4)
    if window > 0:
        ma = series.rolling(window=window).mean().iloc[-1]
    else:
        ma = series.mean()
    
    # Add some randomness
    predictions = []
    for i in range(horizon):
        pred = ma + np.random.normal(0, series.std() * 0.1)
        predictions.append(max(0, pred))
    
    return predictions

# Model mapping dictionary
MODEL_FUNCTIONS = {
    'ARIMA': predict_arima,
    'LSTM': predict_lstm,
    'ExponentialSmoothing': predict_exponential_smoothing,
    'LinearRegression': predict_linear_regression,
    'RandomForest': predict_random_forest,
    'Prophet': predict_prophet,
    'SVR': predict_svr,
    'GradientBoosting': predict_gradient_boosting,
    'XGBoost': predict_xgboost,
    'LightGBM': predict_lightgbm,
    'NeuralNetwork': predict_neural_network,
    'TBATS': predict_tbats,
    'Theta': predict_theta,
    'Croston': predict_croston,
    'Transformer': predict_transformer,
    'TTM': predict_ttm,
    'RandomWalk': predict_random_walk,
    'MovingAverage': predict_moving_average,
    'PolynomialRegression': predict_polynomial_regression,
    'RidgeRegression': predict_ridge_regression,
    'LassoRegression': predict_lasso_regression,
    'ElasticNet': predict_elastic_net
}

def get_model_function(model_name: str):
    """Get the prediction function for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Prediction function for the model
    """
    return MODEL_FUNCTIONS.get(model_name, predict_default)
