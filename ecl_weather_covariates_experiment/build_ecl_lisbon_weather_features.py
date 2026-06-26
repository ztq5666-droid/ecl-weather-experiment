"""
Build an ECL + Lisbon weather covariate dataset.

This script keeps the original ECL file unchanged. It downloads real hourly
historical Lisbon weather from Open-Meteo, aligns the weather to a separate
physical datetime axis, adds calendar covariates, and writes separate output
files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
ECL_PATH = PROJECT_ROOT / "dissertation" / "data" / "ECL" / "electricity.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
WEATHER_OUTPUT_PATH = OUTPUT_DIR / "lisbon_weather_hourly.csv"
MERGED_OUTPUT_PATH = OUTPUT_DIR / "electricity_lisbon_weather.csv"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LISBON_LATITUDE = 38.7223
LISBON_LONGITUDE = -9.1393
LISBON_TIMEZONE = "Europe/Lisbon"

RAW_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
]
DERIVED_WEATHER_COLUMNS = [
    "heating_degree",
    "cooling_degree",
]
CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]
ADDED_WEATHER_COLUMNS = RAW_WEATHER_COLUMNS + DERIVED_WEATHER_COLUMNS
ADDED_COLUMNS = ADDED_WEATHER_COLUMNS + CALENDAR_COLUMNS
STANDARD_ECL_START = pd.Timestamp("2012-01-01 00:00:00")
STANDARD_ECL_END = pd.Timestamp("2014-12-31 23:00:00")
STANDARD_ECL_ROW_COUNT = 26304


def load_ecl_data(path: Path) -> pd.DataFrame:
    """Load ECL and add a separate standard 2012-2014 physical time axis.

    Some processed ECL benchmark files include a continuous hourly ``date``
    column that is not the original 2012-2014 physical benchmark calendar. The
    original file is never modified. The existing ``date`` column is preserved
    as ``benchmark_datetime`` in the generated output, while
    ``physical_datetime`` is used only for weather alignment and calendar
    covariates.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required ECL dataset not found: {path}\n"
            "Download the ECL benchmark dataset and place it at this path."
        )

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a 'date' column in {path}, but none was found.")

    df = df.copy()
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        missing_count = int(parsed_dates.isna().sum())
        raise ValueError(
            "The ECL date column could not be fully parsed as datetimes; "
            f"missing/invalid rows={missing_count}."
        )

    expected_dates = pd.date_range(
        STANDARD_ECL_START,
        STANDARD_ECL_END,
        freq="h",
    )
    if len(expected_dates) != STANDARD_ECL_ROW_COUNT:
        raise RuntimeError("Internal standard ECL date range configuration is invalid.")

    benchmark_datetime = parsed_dates.dt.tz_localize(None)
    has_standard_physical_row_count = len(df) == STANDARD_ECL_ROW_COUNT

    if has_standard_physical_row_count:
        if (
            benchmark_datetime.iloc[0] == STANDARD_ECL_START
            and benchmark_datetime.iloc[-1] == STANDARD_ECL_END
        ):
            print("ECL date column already matches the standard 2012-2014 benchmark range.")
        else:
            print(
                "ECL date column is preserved as benchmark_datetime, but it does not "
                "match the standard 2012-2014 physical ECL range."
            )
            print(
                "Assigning physical_datetime from the standard hourly range "
                "in memory only for weather alignment."
            )
            print(
                f"Original benchmark_datetime range in file: "
                f"{benchmark_datetime.min()} to {benchmark_datetime.max()}"
            )

        df = df.drop(columns=["date"])
        df.insert(0, "benchmark_datetime", benchmark_datetime)
        df.insert(1, "physical_datetime", expected_dates)
    else:
        raise ValueError(
            "ECL row count does not match the standard 2012-2014 benchmark period: "
            f"row count is {len(df)} rather than {STANDARD_ECL_ROW_COUNT}; "
            "cannot safely assign physical_datetime."
        )

    if df["benchmark_datetime"].duplicated().any():
        duplicate_count = int(df["benchmark_datetime"].duplicated().sum())
        raise ValueError(f"ECL contains duplicate benchmark datetimes: {duplicate_count}")
    if df["physical_datetime"].duplicated().any():
        duplicate_count = int(df["physical_datetime"].duplicated().sum())
        raise ValueError(f"ECL contains duplicate physical datetimes: {duplicate_count}")

    return df.sort_values("physical_datetime").reset_index(drop=True)


def detect_date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    """Detect start/end timestamps, row count, and number of load columns."""
    start = df["physical_datetime"].min()
    end = df["physical_datetime"].max()
    row_count = len(df)
    load_column_count = len(
        [
            col
            for col in df.columns
            if col not in {"benchmark_datetime", "physical_datetime"}
        ]
    )
    return start, end, row_count, load_column_count


def iter_date_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    """Return monthly date chunks as YYYY-MM-DD string pairs."""
    start_date = start.date().replace(day=1)
    end_date = end.date()
    month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")

    chunks: list[tuple[str, str]] = []
    for month_start in month_starts:
        chunk_start = max(month_start.date(), start.date())
        month_end = (month_start + pd.offsets.MonthEnd(0)).date()
        chunk_end = min(month_end, end.date())
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))

    if not chunks:
        chunks.append((start.date().isoformat(), end.date().isoformat()))

    return chunks


def fetch_lisbon_weather_chunk(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one Open-Meteo weather chunk for Lisbon."""
    params = {
        "latitude": LISBON_LATITUDE,
        "longitude": LISBON_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(RAW_WEATHER_COLUMNS),
        "timezone": LISBON_TIMEZONE,
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Open-Meteo request failed for {start_date} to {end_date}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            "Open-Meteo request failed "
            f"for {start_date} to {end_date}: HTTP {response.status_code} "
            f"{response.text[:500]}"
        )

    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(
            f"Open-Meteo response for {start_date} to {end_date} did not contain hourly.time."
        )

    missing_variables = [col for col in RAW_WEATHER_COLUMNS if col not in hourly]
    if missing_variables:
        raise RuntimeError(
            f"Open-Meteo response missing variables for {start_date} to {end_date}: "
            f"{missing_variables}"
        )

    weather_df = pd.DataFrame(hourly)
    expected_len = len(weather_df["time"])
    for col in RAW_WEATHER_COLUMNS:
        if len(weather_df[col]) != expected_len:
            raise RuntimeError(
                f"Open-Meteo variable length mismatch in {start_date} to {end_date}: {col}"
            )

    return weather_df


def fetch_lisbon_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch Lisbon weather over the full ECL date range in monthly chunks."""
    frames = []
    chunks = iter_date_chunks(start, end)
    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"Fetching Lisbon weather chunk {idx}/{len(chunks)}: {chunk_start} to {chunk_end}")
        frames.append(fetch_lisbon_weather_chunk(chunk_start, chunk_end))

    if not frames:
        raise RuntimeError("No weather chunks were downloaded.")

    weather_df = pd.concat(frames, ignore_index=True)
    return weather_df


def normalise_weather_time(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Open-Meteo timestamps to timezone-naive Lisbon local datetimes."""
    df = weather_df.copy()
    df["physical_datetime"] = pd.to_datetime(df["time"], errors="raise")

    if getattr(df["physical_datetime"].dt, "tz", None) is not None:
        df["physical_datetime"] = (
            df["physical_datetime"].dt.tz_convert(LISBON_TIMEZONE).dt.tz_localize(None)
        )
    else:
        # Open-Meteo returns local timestamp strings when timezone=Europe/Lisbon.
        # They are already local clock time; keep them timezone-naive for merging.
        df["physical_datetime"] = df["physical_datetime"].dt.tz_localize(None)

    df = df.drop(columns=["time"])
    df = df[["physical_datetime"] + RAW_WEATHER_COLUMNS]
    df = (
        df.drop_duplicates(subset=["physical_datetime"])
        .sort_values("physical_datetime")
        .reset_index(drop=True)
    )

    if df["physical_datetime"].duplicated().any():
        duplicate_count = int(df["physical_datetime"].duplicated().sum())
        raise ValueError(f"Weather data contains duplicate timestamps after normalisation: {duplicate_count}")

    return df


def add_derived_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Add heating and cooling degree variables."""
    df = weather_df.copy()
    df["heating_degree"] = np.maximum(0.0, 18.0 - df["temperature_2m"])
    df["cooling_degree"] = np.maximum(0.0, df["temperature_2m"] - 22.0)
    return df


def add_calendar_features(ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Add standard and cyclical calendar covariates from ECL timestamps."""
    df = ecl_df.copy()
    df["hour"] = df["physical_datetime"].dt.hour
    df["day_of_week"] = df["physical_datetime"].dt.dayofweek
    df["month"] = df["physical_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def merge_ecl_weather(ecl_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ECL load data with Lisbon weather and calendar covariates."""
    ecl_with_calendar = add_calendar_features(ecl_df)
    merged = ecl_with_calendar.merge(
        weather_df,
        on="physical_datetime",
        how="left",
        validate="one_to_one",
    )

    original_columns = list(ecl_df.columns)
    ordered_columns = original_columns + ADDED_WEATHER_COLUMNS + CALENDAR_COLUMNS
    return merged[ordered_columns]


def validate_outputs(
    original_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    added_columns: list[str],
) -> None:
    """Validate that the merged dataset preserves ECL rows and covariates."""
    if len(merged_df) != len(original_df):
        raise ValueError(
            f"Merged row count mismatch: original={len(original_df)}, merged={len(merged_df)}"
        )

    if merged_df["benchmark_datetime"].duplicated().any():
        duplicate_count = int(merged_df["benchmark_datetime"].duplicated().sum())
        raise ValueError(f"Merged dataset contains duplicate benchmark datetimes: {duplicate_count}")
    if merged_df["physical_datetime"].duplicated().any():
        duplicate_count = int(merged_df["physical_datetime"].duplicated().sum())
        raise ValueError(f"Merged dataset contains duplicate physical datetimes: {duplicate_count}")

    if not merged_df["benchmark_datetime"].equals(original_df["benchmark_datetime"]):
        raise ValueError("Merged benchmark_datetime order/content does not match the original ECL dates.")
    if not merged_df["physical_datetime"].equals(original_df["physical_datetime"]):
        raise ValueError("Merged physical_datetime order/content does not match the assigned physical dates.")

    missing_columns = [col for col in added_columns if col not in merged_df.columns]
    if missing_columns:
        raise ValueError(f"Merged dataset is missing added covariate columns: {missing_columns}")

    added_missing = merged_df[added_columns].isna().sum()
    weather_missing = added_missing[ADDED_WEATHER_COLUMNS]
    if (weather_missing > 0).any():
        raise ValueError(
            "Missing weather values remain after merge:\n"
            f"{weather_missing[weather_missing > 0].to_string()}"
        )


def _validate_weather_coverage(
    ecl_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    """Ensure weather timestamps cover every ECL timestamp."""
    weather_min = weather_df["physical_datetime"].min()
    weather_max = weather_df["physical_datetime"].max()
    if weather_min > start or weather_max < end:
        raise ValueError(
            "Weather data does not cover the full ECL physical date range: "
            f"weather={weather_min} to {weather_max}, physical_datetime={start} to {end}"
        )

    missing_dates = ecl_df.loc[
        ~ecl_df["physical_datetime"].isin(weather_df["physical_datetime"]),
        "physical_datetime",
    ]
    if not missing_dates.empty:
        sample = missing_dates.head(10).astype(str).tolist()
        raise ValueError(
            f"Weather data is missing {len(missing_dates)} ECL timestamps. Sample: {sample}"
        )


def _print_diagnostics(
    original_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    load_column_count: int,
) -> None:
    """Print clear diagnostics for the generated files."""
    print("\n=== ECL + Lisbon Weather Build Diagnostics ===")
    print(f"Original ECL shape: {original_df.shape}")
    print(
        "Benchmark datetime range from ECL file: "
        f"{original_df['benchmark_datetime'].min()} to {original_df['benchmark_datetime'].max()}"
    )
    print(f"Physical datetime range used for weather merge: {start} to {end}")
    print(f"Number of electricity load columns: {load_column_count}")
    print(
        "Weather physical_datetime range: "
        f"{weather_df['physical_datetime'].min()} to {weather_df['physical_datetime'].max()}"
    )
    print(f"Weather shape: {weather_df.shape}")
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Added weather columns: {ADDED_WEATHER_COLUMNS}")
    print(f"Added calendar columns: {CALENDAR_COLUMNS}")
    print("Missing values per added covariate:")
    print(merged_df[ADDED_COLUMNS].isna().sum().to_string())
    print("\nFirst 3 rows of merged dataset:")
    print(merged_df.head(3).to_string())
    print("\nOutput file paths:")
    print(f"- Weather-only: {WEATHER_OUTPUT_PATH}")
    print(f"- Merged dataset: {MERGED_OUTPUT_PATH}")


def _file_signature(path: Path) -> tuple[int, int]:
    """Return a simple file signature to verify the original file is unchanged."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def main() -> None:
    original_signature = _file_signature(ECL_PATH) if ECL_PATH.exists() else None

    ecl_df = load_ecl_data(ECL_PATH)
    start, end, row_count, load_column_count = detect_date_range(ecl_df)
    print(f"Loaded ECL data: rows={row_count}, load_columns={load_column_count}")
    print(
        "Benchmark datetime range from file: "
        f"{ecl_df['benchmark_datetime'].min()} to {ecl_df['benchmark_datetime'].max()}"
    )
    print(f"Physical datetime range for weather alignment: {start} to {end}")

    raw_weather = fetch_lisbon_weather(start, end)
    weather = normalise_weather_time(raw_weather)
    weather = add_derived_weather_features(weather)

    weather = weather[
        (weather["physical_datetime"] >= start)
        & (weather["physical_datetime"] <= end)
    ].reset_index(drop=True)
    _validate_weather_coverage(ecl_df, weather, start, end)

    merged = merge_ecl_weather(ecl_df, weather)
    validate_outputs(ecl_df, merged, ADDED_COLUMNS)

    if original_signature != _file_signature(ECL_PATH):
        raise RuntimeError(f"Original ECL file changed unexpectedly: {ECL_PATH}")

    WEATHER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    weather[["physical_datetime"] + ADDED_WEATHER_COLUMNS].to_csv(
        WEATHER_OUTPUT_PATH,
        index=False,
    )
    merged.to_csv(MERGED_OUTPUT_PATH, index=False)

    if original_signature != _file_signature(ECL_PATH):
        raise RuntimeError(f"Original ECL file changed unexpectedly after writing outputs: {ECL_PATH}")

    _print_diagnostics(ecl_df, weather, merged, start, end, load_column_count)


if __name__ == "__main__":
    main()
