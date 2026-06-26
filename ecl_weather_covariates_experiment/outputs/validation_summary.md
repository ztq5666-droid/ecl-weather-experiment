# ECL + Lisbon Weather Validation Summary

## Scope

This validation summary applies only to the independent weather covariate
experiment in `ecl_weather_covariates_experiment/`.

No files inside the original `dissertation/` project were modified by the
weather data generation or merge process.

## Datetime Columns

The generated merged dataset intentionally contains two datetime columns:

- `benchmark_datetime`: copied from the existing
  `dissertation/data/ECL/electricity.csv` file.
- `physical_datetime`: assigned standard ECL physical-period timestamp used for
  Lisbon weather and calendar alignment.

Validation results:

| Field | Start | End | Rows | Duplicates | Hourly continuous |
|---|---:|---:|---:|---:|---:|
| `benchmark_datetime` | 2016-07-01 02:00:00 | 2019-07-02 01:00:00 | 26,304 | 0 | Yes |
| `physical_datetime` | 2012-01-01 00:00:00 | 2014-12-31 23:00:00 | 26,304 | 0 | Yes |

The original benchmark file's date range is preserved as `benchmark_datetime`.
The 2012-2014 `physical_datetime` sequence is used only to align with the
standard ECL physical period and Lisbon historical weather data.

## Weather Data

Weather source:

- Open-Meteo Historical Weather API
- Location: Lisbon, Portugal
- Latitude: 38.7223
- Longitude: -9.1393
- Timezone: Europe/Lisbon
- Period: 2012-01-01 00:00:00 to 2014-12-31 23:00:00
- Frequency: hourly

Weather output file:

- `outputs/lisbon_weather_hourly.csv`

Weather validation:

| Field | Value |
|---|---:|
| Rows | 26,304 |
| Columns | 6 |
| Datetime column | `physical_datetime` |
| Duplicate physical datetimes | 0 |
| Hourly continuity | Yes |
| Missing values | 0 |

## Merged Dataset

Merged output file:

- `outputs/electricity_lisbon_weather.csv`

Merged dataset validation:

| Field | Value |
|---|---:|
| Rows | 26,304 |
| Columns | 338 |
| Old `date` column present | No |
| First columns | `benchmark_datetime`, `physical_datetime` |
| Duplicate `benchmark_datetime` values | 0 |
| Duplicate `physical_datetime` values | 0 |
| Missing values in added weather/calendar covariates | 0 |
| Total missing values | 0 |

Added weather variables:

- `temperature_2m`
- `relative_humidity_2m`
- `precipitation`
- `heating_degree`
- `cooling_degree`

Added calendar variables:

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

## Interpretation Note

The Lisbon weather variables are shared regional exogenous covariates. They are
not client-specific local weather measurements because the public ECL benchmark
does not provide exact client locations.

This design is suitable for creating a richer weather-augmented ECL forecasting
task for model-family comparison. It should not be interpreted as a precise
customer-level weather exposure model or as a standalone test of whether
weather improves forecasting accuracy.
