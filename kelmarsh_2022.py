"""Kelmarsh 2022 SCADA loader adapted to the local folder layout.

This module is a drop-in replacement for the original ``kelmarsh_2022`` helper
used by ``Kelmarsh_2022_Preprocessing_Literature_Benchmark.ipynb``.  The only
change is where the raw files are read from: instead of the upstream
``kelmarsh_data/`` repository tree, it reads the Greenbyte exports stored in
``Kelmarsh_SCADA_2022_4457/``:

    Kelmarsh_SCADA_2022_4457/
        Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv   # SCADA
        ...
        Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv         # events

Both files are Greenbyte exports whose preamble is a block of ``#`` comment
lines followed by a header line.  The SCADA files carry 303 columns and mark
missing/erroneous values with ``NaN``.

All helper names, return shapes and column names match what the notebook
expects, so the notebook can run unchanged apart from path resolution.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATA_DIRNAME",
    "GRID",
    "SIGNALS",
    "TURBINE_NUMBERS",
    "data_dir",
    "scada_path",
    "events_path",
    "ensure_metadata",
    "capped_interpolate",
    "load_2022",
    "self_check",
]

DATA_DIRNAME = "Kelmarsh_SCADA_2022_4457"

# UTC 10-minute grid for the whole calendar year 2022 (52,560 slots).
GRID = pd.date_range("2022-01-01", "2023-01-01", freq="10min", tz="UTC", inclusive="left")

# Short name -> exact 2022 SCADA column header.
SIGNALS = {
    "wind": "Wind speed (m/s)",
    "direction": "Wind direction (°)",
    "power": "Power (kW)",
    "reactive": "Reactive power (kvar)",
    "ambient": "Nacelle ambient temperature (°C)",
    "bearing": "Generator bearing front temperature (°C)",
    "oil": "Gear oil temperature (°C)",
    "rotor": "Rotor speed (RPM)",
    "generator": "Generator RPM (RPM)",
    "pitch": "Blade angle (pitch position) A (°)",
    "frequency": "Grid frequency (Hz)",
    "setpoint": "Turbine Power setpoint (kW)",
}

# Turbine number in the file name -> notebook turbine id.
TURBINE_NUMBERS = {1: "KWF1", 2: "KWF2", 3: "KWF3", 4: "KWF4", 5: "KWF5", 6: "KWF6"}
NUMBER_BY_TURBINE = {v: k for k, v in TURBINE_NUMBERS.items()}

# Static asset metadata (identical to the upstream turbines.csv).  Kept inline so
# the notebook no longer depends on a separate metadata repository.
_TURBINES_ROWS = [
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 1", "Alternative Title": "KWF1",
        "Identity": "SEN 93420", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 78.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.400604, "Longitude": -0.947133, "Elevation (m)": 145.598,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 2", "Alternative Title": "KWF2",
        "Identity": "SEN 93421", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 78.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.402551, "Longitude": -0.949527, "Elevation (m)": 156.577,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 3", "Alternative Title": "KWF3",
        "Identity": "SEN 93422", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 68.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.403834, "Longitude": -0.944190, "Elevation (m)": 153.477,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 4", "Alternative Title": "KWF4",
        "Identity": "SEN 93423", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 78.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.398781, "Longitude": -0.941150, "Elevation (m)": 146.313,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 5", "Alternative Title": "KWF5",
        "Identity": "SEN 93424", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 78.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.402308, "Longitude": -0.940537, "Elevation (m)": 142.901,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
    {
        "Wind Farm": "Kelmarsh", "Title": "Kelmarsh 6", "Alternative Title": "KWF6",
        "Identity": "SEN 93425", "Manufacturer": "Senvion", "Model": "MM92",
        "Rated power (kW)": 2050, "Hub Height (m)": 68.5, "Rotor Diameter (m)": 92,
        "Latitude": 52.400687, "Longitude": -0.936093, "Elevation (m)": 135.039,
        "Country": "UK", "Commercial Operations Date": "15/04/2016",
    },
]


def data_dir(root) -> Path:
    """Return the directory holding the Greenbyte CSV exports."""
    return Path(root) / DATA_DIRNAME


def _find_export(root, kind: str, turbine: str) -> Path:
    """Locate the SCADA ('Turbine_Data') or event ('Status') CSV for a turbine."""
    number = NUMBER_BY_TURBINE[turbine]
    hits = sorted(data_dir(root).glob(f"{kind}_Kelmarsh_{number}_*"))
    if not hits:
        raise FileNotFoundError(
            f"No {kind} export for {turbine} (Kelmarsh {number}) in {data_dir(root)}"
        )
    return hits[0]


def scada_path(root, turbine: str) -> Path:
    """Path to the per-turbine SCADA CSV (used by the notebook audit cells)."""
    return _find_export(root, "Turbine_Data", turbine)


def events_path(root, turbine: str) -> Path:
    """Path to the per-turbine event/status CSV."""
    return _find_export(root, "Status", turbine)


def _read_header(path: Path):
    """Return (preamble_lines, header_fields, header_line_index) for a Greenbyte CSV."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        preamble = []
        for line_no, line in enumerate(stream):
            if not line.startswith("#") or line.startswith("# Date and time,"):
                return preamble, next(csv.reader([line])), line_no
            preamble.append(line.rstrip("\n"))
    raise ValueError(f"No CSV header found in {path}")


def _split_title_unit(header: str):
    """Split 'Power (kW)' into ('Power', 'kW'); keep header when no unit suffix."""
    if header.endswith(")") and " (" in header:
        title, unit = header.rsplit(" (", 1)
        return title, unit[:-1]
    return header, ""


def ensure_metadata(root) -> tuple[Path, Path]:
    """Create turbines.csv / signal_mapping.csv beside the raw data if absent."""
    meta = data_dir(root) / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    turbines_csv = meta / "turbines.csv"
    mapping_csv = meta / "signal_mapping.csv"

    if not turbines_csv.exists():
        pd.DataFrame(_TURBINES_ROWS).to_csv(turbines_csv, index=False)

    if not mapping_csv.exists():
        _, headers, _ = _read_header(scada_path(root, TURBINE_NUMBERS[1]))
        rows, seen = [], set()
        for header in headers[1:]:
            title, unit = _split_title_unit(header)
            if title in seen:
                continue
            seen.add(title)
            rows.append({"Greenbyte Signal ID": "", "Greenbyte Title": title, "Unit": unit})
        mapping = pd.DataFrame(rows)
        # Preserve the documented upstream unit conflict: Greenbyte labels the
        # 2022 SCADA 'Power (kW)' header, while the native mapping stores kWh.
        mapping.loc[mapping["Greenbyte Title"].eq("Power"), "Unit"] = "kWh"
        mapping.to_csv(mapping_csv, index=False)

    return turbines_csv, mapping_csv


def _load_scada_file(root, turbine: str):
    """Read one SCADA export and return (sub_frame, profile_rows, schema, audit, raw_columns)."""
    path = scada_path(root, turbine)
    _, headers, skiprows = _read_header(path)
    raw = pd.read_csv(path, skiprows=skiprows, na_values=["NaN"], low_memory=False)
    time_col = headers[0]

    timestamp = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    availability = pd.to_numeric(raw["Data Availability"], errors="coerce")
    in_year = timestamp.ge(GRID[0]) & timestamp.lt(pd.Timestamp("2023-01-01", tz="UTC"))
    valid = in_year & availability.eq(1)

    invalid_timestamps = int(timestamp.isna().sum())
    outside_2022 = int((~in_year & timestamp.notna()).sum())
    valid_duplicate_rows = int(timestamp[valid].duplicated().sum())
    valid_rows = int(valid.sum())

    signal_headers = list(SIGNALS.values())
    sub = raw.loc[valid, [time_col] + signal_headers].copy()
    sub.columns = ["timestamp"] + list(SIGNALS.keys())
    sub["timestamp"] = timestamp.loc[valid].to_numpy()
    sub["turbine"] = turbine
    for short in SIGNALS:
        sub[short] = pd.to_numeric(sub[short], errors="coerce")
    sub = sub[["turbine", "timestamp"] + list(SIGNALS.keys())].reset_index(drop=True)

    profile_rows = []
    for column in headers[1:]:
        values = pd.to_numeric(raw[column], errors="coerce")
        observed = values.loc[valid].dropna()
        profile_rows.append(
            {
                "turbine": turbine,
                "signal": column,
                "missing": int(values.loc[valid].isna().sum()),
                "valid_rows": valid_rows,
                "missing_pct": 100.0 * values.loc[valid].isna().mean(),
                "min": float(observed.min()) if len(observed) else np.nan,
                "max": float(observed.max()) if len(observed) else np.nan,
                "constant_or_empty": bool(observed.nunique() < 2),
            }
        )

    schema = {"turbine": turbine, "headers": headers}
    audit = {
        "turbine": turbine,
        "raw_rows": int(len(raw)),
        "availability_1": int(availability.eq(1).sum()),
        "invalid_timestamps": invalid_timestamps,
        "outside_2022": outside_2022,
        "valid_duplicate_rows": valid_duplicate_rows,
        "rows_2022": valid_rows,
    }
    return sub, profile_rows, schema, audit


def _load_events_file(root, turbine: str):
    path = events_path(root, turbine)
    _, _, skiprows = _read_header(path)
    frame = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    frame["Timestamp start"] = pd.to_datetime(frame["Timestamp start"], utc=True, errors="coerce")
    frame["Timestamp end"] = pd.to_datetime(frame["Timestamp end"], utc=True, errors="coerce")
    frame["turbine"] = turbine
    return frame


def load_2022(root):
    """Load all six turbines exactly like the upstream notebook helper.

    Returns
    -------
    df : DataFrame          availability-1 rows with turbine, timestamp and the 12 SIGNALS
    audit : DataFrame       per-turbine raw/availability/duplicate counts
    profile : DataFrame     per-turbine per-field missingness and range
    schemas : DataFrame     per-turbine header lists (303 fields)
    events : DataFrame      concatenated event/status logs
    """
    ensure_metadata(root)

    sub_frames, profile_rows, schemas, audits, events = [], [], [], [], []
    for turbine in TURBINE_NUMBERS.values():
        sub, prof, schema, audit = _load_scada_file(root, turbine)
        sub_frames.append(sub)
        profile_rows.extend(prof)
        schemas.append(schema)
        audits.append(audit)

        event_file = data_dir(root).glob(f"Status_Kelmarsh_{NUMBER_BY_TURBINE[turbine]}_*")
        if any(event_file):
            events.append(_load_events_file(root, turbine))

    df = pd.concat(sub_frames, ignore_index=True)
    audit = pd.DataFrame(audits)
    profile = pd.DataFrame(profile_rows)
    schemas = pd.DataFrame(schemas)
    events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    return df, audit, profile, schemas, events


def capped_interpolate(frame: pd.DataFrame, columns, max_steps: int = 2) -> pd.DataFrame:
    """Fill only complete NaN runs of length <= ``max_steps`` by time interpolation.

    Long gaps, leading/trailing gaps, and any column not listed stay untouched,
    matching the notebook's bounded whole-run policy.
    """
    out = frame.copy()
    cols = [c for c in columns if c in frame.columns]
    if not cols:
        return out
    interpolated = frame[cols].interpolate(method="time", limit_area="inside")
    for column in cols:
        isna = frame[column].isna()
        if not isna.any():
            continue
        run_id = (isna != isna.shift()).cumsum()
        run_length = isna.groupby(run_id).transform("sum")
        fillable = isna & run_length.le(max_steps)
        out.loc[fillable, column] = interpolated.loc[fillable, column]
    return out


def self_check() -> bool:
    """Lightweight sanity check run at the top of the notebook."""
    assert len(GRID) == 52560, f"Unexpected GRID length: {len(GRID)}"
    assert GRID.tz is not None, "GRID must be timezone-aware (UTC)"
    assert len(SIGNALS) == 12, "SIGNALS must contain the 12 selected channels"
    assert set(NUMBER_BY_TURBINE) == set(TURBINE_NUMBERS.values())
    return True
