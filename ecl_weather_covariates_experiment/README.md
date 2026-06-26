# ECL + Lisbon Weather Covariates Experiment

## 1. Purpose

This experiment extends the original ECL electricity load forecasting dataset
by adding real Lisbon weather and calendar covariates.

The target remains electricity consumption. The new independent variables are
weather and calendar predictors that create a richer weather-augmented
forecasting task.

This experiment is not designed primarily to test whether weather improves
forecasting accuracy. Instead, it creates a more complex ECL forecasting
setting so that traditional methods, deep learning models and
Transformer-based models can be compared under richer multivariate and
exogenous covariates.

Research focus:

```text
When the ECL forecasting task is augmented with shared regional weather and
calendar covariates, do Transformer-based models handle the richer
multivariate/exogenous setting better than traditional approaches?
```

This experiment is intentionally separate from the original baseline setup. The
existing load-only model scripts and result files should remain unchanged.
No dissertation files are modified by this experiment.

## 2. Why Lisbon Weather Is Used

The ECL dataset is based on Portuguese electricity consumption. However, the
public benchmark version does not provide exact client-level geographic
locations. Because client-level locations are unavailable, Lisbon weather is
used as a representative regional weather proxy for the Portuguese ECL dataset.

These weather variables should therefore be interpreted as shared regional
covariates, not client-specific local weather measurements. This is a
reasonable but imperfect modelling assumption: it allows a controlled
exogenous-variable experiment while acknowledging that local weather variation
across clients cannot be captured.

The weather variables are shared across all clients because the public ECL
benchmark does not include client-specific locations. They should not be
interpreted as local weather measured at each individual electricity customer.

## 3. Added Variables

Weather variables downloaded from the Open-Meteo Historical Weather API:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`

Derived weather variables:

- `heating_degree`
- `cooling_degree`

The derived variables are defined as:

```text
heating_degree = max(0, 18 - temperature_2m)
cooling_degree = max(0, temperature_2m - 22)
```

This first version does not include `cloud_cover` or `wind_speed_10m`. The
focused weather feature set is used to keep the experiment interpretable and to
avoid adding weak or noisy predictors before the core temperature, humidity and
precipitation effects are evaluated.

Calendar variables:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `dayofweek_sin`
- `dayofweek_cos`
- `month_sin`
- `month_cos`

## 4. Methodology

The dataset construction process is:

1. Load the original ECL electricity dataset from
   `../dissertation/data/ECL/electricity.csv`.
2. Preserve the existing `date` column from that benchmark file as
   `benchmark_datetime` in the generated merged output. In the current local
   processed file, this column ranges from `2016-07-01 02:00:00` to
   `2019-07-02 01:00:00`.
3. Add a separate `physical_datetime` column for the standard original ECL
   physical period, from `2012-01-01 00:00:00` to `2014-12-31 23:00:00` with
   hourly frequency.
4. Download real hourly historical weather for Lisbon using the Open-Meteo
   Historical Weather API for the `physical_datetime` period only.
5. Align Lisbon weather data with `physical_datetime`, not with
   `benchmark_datetime`.
6. Add calendar features from `physical_datetime`.
7. Merge weather and calendar covariates into a new dataset.
8. Save the merged dataset separately under this experiment's `outputs/`
   directory without changing the original file.

The distinction between the two datetime columns is intentional:

- `benchmark_datetime` preserves the date values already present in the
  processed benchmark electricity file.
- `physical_datetime` is used only to align the row sequence with the original
  2012-2014 ECL physical period and the corresponding Lisbon weather data.

## 5. Experimental Framing

The original ECL experiment is a load-only time series forecasting task.

This weather-augmented experiment changes the input setting to:

```text
past load + past shared regional weather/calendar covariates -> future load
```

The key comparison is not simply "does weather help?" The key comparison is
whether different model families perform differently when the forecasting task
contains richer multivariate and exogenous structure.

To keep the model comparison fair, active model scripts should use:

- the same weather-augmented dataset
- the same chronological 70% / 10% / 20% split
- the same selected 20 clients
- the same forecast horizons: 24h, 48h, 168h
- the same metrics: MSE, MAE, RMSE
- the same information setting: no future observed exogenous variables in the
  active fair experiment

Future observed weather should not be used unless every model is explicitly
placed in an oracle/weather-forecast-informed setting. To keep the active
scripts maximally consistent, current active models use past exogenous context
only, including historical weather and historical calendar covariates.

## 6. Limitations

Lisbon weather is only a regional proxy. It does not represent exact
client-level weather conditions because the ECL benchmark does not provide
client-level locations.

The processed local ECL file may contain a continuous hourly `date` column that
does not correspond to the standard 2012-2014 ECL benchmark calendar. This
experiment therefore preserves that column as `benchmark_datetime` and uses a
separate `physical_datetime` column for the standard benchmark physical period
when the file has the expected 26,304 hourly rows. The original dissertation
dataset file is not modified.

Future work could match each client to local weather if exact client locations
were available. Since this first version uses a focused weather feature set,
additional variables such as cloud cover or wind speed could also be tested
later as a robustness extension.

## 7. Files Produced

Source/documentation files:

- `build_ecl_lisbon_weather_features.py`
- `README.md`

Generated output files:

- `outputs/lisbon_weather_hourly.csv`
- `outputs/electricity_lisbon_weather.csv`

The merged output contains both datetime columns:

- `benchmark_datetime`: copied from the existing benchmark electricity file.
- `physical_datetime`: standard 2012-2014 hourly sequence used for weather and
  calendar alignment.

## 8. Future Modelling Plan

The existing baseline scripts should remain unchanged. Future exogenous-variable
model scripts should be created separately.

Active fair-setting model scripts:

- `models/arima/train_arima_load_only.py`
- `models/xgboost/train_xgboost_exog_past.py`
- `models/lstm/train_lstm_exog.py`
- `models/itransformer/train_itransformer_exog.py`

Model-specific treatment:

- ARIMA is retained as a load-only statistical benchmark using the same split,
  selected clients, horizons and forecast origin.
- XGBoost uses lag/rolling load features and past weather/calendar lag/rolling
  features.
- LSTM can use weather/calendar variables as additional input channels.
- iTransformer can use covariates as additional multivariate inputs, but final
  evaluation should remain focused on electricity load columns.

SARIMAX/ARIMAX variants requiring future weather are archived separately as
oracle-weather extensions, not active fair-setting models.

Active result figures can be regenerated with:

```bash
python generate_result_figures.py
```
