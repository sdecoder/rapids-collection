import cudf
from cuml.tsa.arima import ARIMA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def load_dataset(name, max_batch=4):
    import os
    pdf = pd.read_csv(os.path.join("data", "time_series", "%s.csv" % name))
    return cudf.from_pandas(pdf[pdf.columns[1:max_batch+1]].astype(np.float64))

def visualize(y, pred=None, pred_start=None, lower=None, upper=None):
    n_obs, batch_size = y.shape
    col = ["#1f77b4", "#ff7f0e"]

    # Create the subplots
    c = min(batch_size, 2)
    r = (batch_size + c - 1) // c
    fig, ax = plt.subplots(r, c, squeeze=False)
    ax = ax.flatten()

    # Range for the prediction
    if pred is not None:
        pred_start = n_obs if pred_start is None else pred_start
        pred_end = pred_start + pred.shape[0]
    else:
        pred_end = n_obs

    # Plot the data
    for i in range(batch_size):
        title = y.columns[i]
        if pred is not None:
            ax[i].plot(np.r_[pred_start:pred_end],
                       pred[pred.columns[i]].to_numpy(),
                       linestyle="--", color=col[1])
        # Prediction intervals
        if lower is not None and upper is not None:
            ax[i].fill_between(np.r_[pred_start:pred_end],
                               lower[lower.columns[i]].to_numpy(),
                               upper[upper.columns[i]].to_numpy(),
                               alpha=0.2, color=col[1])
        ax[i].plot(np.r_[:n_obs], y[title].to_numpy(), color=col[0])
        ax[i].title.set_text(title)
        ax[i].set_xlim((0, pred_end))
    for i in range(batch_size, r*c):
        fig.delaxes(ax[i])
    fig.tight_layout()
    fig.patch.set_facecolor('white')
    plt.savefig("result.png")
    plt.show()

def func_migratition():

  df_mig = load_dataset("net_migrations_auckland_by_age", 4)
  visualize(df_mig)

  print(f"[trace] creating ARIMA model")
  model_mig = ARIMA(df_mig, order=(0,0,2), fit_intercept=True)
  print(f"[trace] training ARIMA model")
  model_mig.fit()

  print(f'[trzce] showing the model_mig.get_fit_params')
  print(model_mig.get_fit_params()["ma"])
  print(model_mig.ma_)
  print(model_mig.pack())

  print("log-likelihood:\n", model_mig.llf)
  print("\nAkaike Information Criterion (AIC):\n", model_mig.aic)
  print("\nCorrected Akaike Information Criterion (AICc):\n", model_mig.aicc)
  print("\nBayesian Information Criterion (BIC):\n", model_mig.bic)

  pass

def func_guest():

  print(f'[trace] working in guest_nights_by_region function:')
  df_guests = load_dataset("guest_nights_by_region", 4)
  # Create and fit an ARIMA(1,1,1)(1,1,1)12 model:
  model_guests = ARIMA(df_guests, order=(1,1,1), seasonal_order=(1,1,1,12),
                      fit_intercept=False)
  model_guests.fit()
  print(f'[trace] model fit done')

  # Forecast
  print(f'[trace] forecast the result')
  fc_guests = model_guests.forecast(40)

  # Visualize after the time step 200
  print(f'[trace] visualize result and save to disk file')
  visualize(df_guests[200:], fc_guests)

  print(f'[trace] dealing with the missing df_guests')
  df_guests_missing = df_guests[:100].copy()
  for title in df_guests_missing.columns:
      # Missing observations at the start to simulate varying lengths
      n_leading = random.randint(5, 40)
      df_guests_missing[title][:n_leading]=None
      # Random missing observations in the middle
      missing_obs = random.choices(range(n_leading, 100), k=random.randint(5, 20))
      df_guests_missing[title][missing_obs]=None

  df_guests_missing = df_guests_missing.fillna(np.nan)
    # Create and fit an ARIMA(1,1,1)(1,1,1)12 model:
  model_guests_missing = ARIMA(df_guests_missing, order=(1,1,1), seasonal_order=(1,1,1,12), fit_intercept=False)
  model_guests_missing.fit()

  print(f'[trace] predicting the missing guests')
  fc_guests_missing = model_guests_missing.predict(0, 120)
  visualize(df_guests_missing, fc_guests_missing, 0)
  # Forecast
  print(f'[trace] done')
  pass

def exogenous_variables():

  print(f'[trace] working in func:exogenous_variables')
  nb = 4

  print(f'[trace] Generate exogenous variables and coefficients')
  get_sine = lambda n, period: np.sin(np.r_[:n] * 2 * np.pi / period + np.random.uniform(0, period))
  np_exog = np.column_stack([get_sine(319, T) for T in np.random.uniform(20, 100, 2 * nb)])
  np_exog_coef = np.random.uniform(20, 200, 2 * nb)

  print(f'[trace] Create dataframes for the past and future values')
  df_exog = cudf.DataFrame(np_exog[:279])
  df_exog_fut = cudf.DataFrame(np_exog[279:])

  print(f'[trace] Add linear combination of the exogenous variables to the endogenous')
  df_guests = load_dataset("guest_nights_by_region", 4)
  df_guests_exog = df_guests.copy()
  for ib in range(nb):
      df_guests_exog[df_guests_exog.columns[ib]] += \
          np.matmul(np_exog[:279, ib*2:(ib+1)*2], np_exog_coef[ib*2:(ib+1)*2])


  print(f'[trace] Create and fit an ARIMA(1,0,1)(1,1,1)12 (c) model with exogenous variables')
  model_guests_exog = ARIMA(endog=df_guests_exog, exog=df_exog,
                            order=(1,0,1), seasonal_order=(1,1,1,12),
                            fit_intercept=True)
  model_guests_exog.fit()
  print(f'[trace] Forecast')
  fc_guests_exog = model_guests_exog.forecast(40, exog=df_exog_fut)

  print(f'[trace] Visualize after the time step 100')
  visualize(df_guests_exog[100:], fc_guests_exog)
  print(f'[trace] done')
  pass

if __name__ == '__main__':
  print(f'[trace] working in the main function')
  exogenous_variables()