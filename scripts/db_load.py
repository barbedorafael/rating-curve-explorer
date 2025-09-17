# scripts/db_load.py
import re
import sqlite3
from calendar import monthrange
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "hydrodata.sqlite"
SCHEMA_PATH = ROOT / "schema.sql"


# ----------------- tiny utils -----------------

def _first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def _to_int(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s.replace(",", ".")))
    except ValueError:
        return None


def _date_txt(v):
    try:
        d = pd.to_datetime(v, errors="coerce")
        if pd.isna(d):
            return None
        return d.strftime("%Y-%m-%d")
    except Exception:
        return None


def _melt_monthly(df: pd.DataFrame, prefix: str):
    """
    Wide monthly (prefix01..prefix31 [+ Status]) -> rows:
    (station_id, date, method:int|None, value:float, status:int|None)
    """
    value_cols = sorted([c for c in df.columns if re.fullmatch(prefix + r"\d{2}", c)])
    pairs = [(vc, f"{prefix}{int(vc[-2:]):02d}Status", int(vc[-2:])) for vc in value_cols]
    method_col = _first(df, ["TipoMedicaoCotas", "MetodoObtencaoVazoes"])
    if "Data" not in df.columns:
        return []

    df = df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    out = []
    for _, row in df.iterrows():
        station_id = row["EstacaoCodigo"]
        base = row["Data"]
        if pd.isna(base):
            continue
        y, m = base.year, base.month
        dim = monthrange(y, m)[1]
        method = None
        if method_col and row.get(method_col) not in (None, "", "NaN"):
            try:
                method = int(str(row.get(method_col)).strip())
            except Exception:
                method = None
        for vc, sc, day in pairs:
            if day > dim:
                continue
            val_raw = row.get(vc)
            if val_raw in (None, "", "NaN") or pd.isna(val_raw):
                continue
            val = _to_float(val_raw)
            if val is None:
                continue
            status_raw = row.get(sc)
            try:
                status = int(status_raw) if status_raw not in (None, "", "NaN") and not pd.isna(status_raw) else None
            except Exception:
                status = None
            out.append((station_id, f"{y:04d}-{m:02d}-{day:02d}", method, val, status))
    return out


# ----------------- loaders -----------------
def init_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.execute("PRAGMA foreign_keys = ON;")

def load_station(conn, data_dir: Path):
    """Load station metadata from _Estacao_.csv into stations table."""
    estacao_csv = data_dir / "_Estacao_.csv"
    df = pd.read_csv(estacao_csv)

    # Handle multiple stations in single CSV
    for _, station_row in df.iterrows():
        row = {
            "station_id": _to_int(station_row.get("EstacaoCodigo")),
            "name": station_row.get("Nome"),
            "station_type": station_row.get("TipoEstacao"),
            "lon": _to_float(station_row.get("Longitude")),
            "lat": _to_float(station_row.get("Latitude")),
            "basin_id": _to_int(station_row.get("BaciaCodigo")),
            "sub_basin_id": _to_int(station_row.get("SubBaciaCodigo")),
            "river_id": _to_int(station_row.get("RioCodigo")),
            "state_id": _to_int(station_row.get("EstadoCodigo")),
            "municipality_id": _to_int(station_row.get("MunicipioCodigo")),
            "responsible_id": _to_int(station_row.get("ResponsavelCodigo")),
            "responsible_unit": station_row.get("ResponsavelUnidade"),
            "responsible_jurisdiction": station_row.get("ResponsavelJurisdicao"),
            "operator_id": _to_int(station_row.get("OperadoraCodigo")),
            "operator_unit": station_row.get("OperadoraUnidade"),
            "operator_subunit": station_row.get("OperadoraSubUnidade"),
            "additional_code": station_row.get("CodigoAdicional"),
            "altitude": _to_float(station_row.get("Altitude")),
            "drainage_area": _to_float(station_row.get("AreaDrenagem")),
        }

        conn.execute("""
            INSERT OR REPLACE INTO stations (
                station_id, name, station_type, lon, lat, basin_id, sub_basin_id, river_id, state_id, municipality_id, responsible_id, responsible_unit, responsible_jurisdiction, operator_id, operator_unit,  operator_subunit, additional_code, altitude, drainage_area
            )
            VALUES (
                :station_id, :name, :station_type, :lon, :lat, :basin_id, :sub_basin_id, :river_id, :state_id, :municipality_id, :responsible_id, :responsible_unit, :responsible_jurisdiction, :operator_id, :operator_unit, :operator_subunit, :additional_code, :altitude, :drainage_area
            )
        """, row)

def load_timeseries(conn, data_dir: Path):
    for file in ["_Cotas_.csv", "_Vazoes_.csv"]:
        path = data_dir / file
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        if file == "_Cotas_.csv":
            prefix = "Cota"
        elif file == "_Vazoes_.csv":
            prefix = "Vazao"
        rows = _melt_monthly(df, prefix)
        conn.executemany(
            "INSERT OR REPLACE INTO timeseries_cota (station_id, date, method, value, status) VALUES (?, ?, ?, ?, ?)",
            rows,
        )

def load_stage_discharge(conn, data_dir: Path):
    path = data_dir / "_ResumoDescarga_.csv"

    df = pd.read_csv(path)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                int(str(r.get("EstacaoCodigo"))),
                _date_txt(r.get("Data")),
                (str(r.get("Hora")).strip()[-12:-7] if r.get("Hora") not in (None, "", "NaN") else None),
                _to_int(r.get("NivelConsistencia")),
                _to_float(r.get("Cota")),
                _to_float(r.get("Vazao")),
                _to_float(r.get("AreaMolhada")),
                _to_float(r.get("VelMedia")),
                _to_float(r.get("Largura")),
                _to_float(r.get("Profundidade")),
                _to_int(r.get("MedidorVazao")),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO stage_discharge
            (station_id, date, time, consistency, level, discharge, area, velocity, width, depth, instrument)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

def load_vertical_profiles(conn, data_dir: Path):
    # surveys
    survey_path = data_dir / "_PerfilTransversal_.csv"
    df = pd.read_csv(survey_path, dtype=str)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                int(str(r.get("RegistroID")).split(".")[0]),
                int(str(r.get("EstacaoCodigo"))),
                _date_txt(r.get("Data")),
                _to_float(r.get("NumLevantamento")),
                (r.get("TipoSecao") if r.get("TipoSecao") not in (None, "", "NaN") else None),
                _to_float(r.get("NumVerticais")),
                _to_float(r.get("DistanciaPIPF")),
                _to_float(r.get("EixoXDistMaxima")),
                _to_float(r.get("EixoXDistMinima")),
                _to_float(r.get("EixoYCotaMaxima")),
                _to_float(r.get("EixoYCotaMinima")),
                _to_float(r.get("ElmGeomPassoCota")),
                (r.get("Observacoes") if r.get("Observacoes") not in (None, "", "NaN") else None),
            )
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO vertical_profile_survey (
                survey_id, station_id, date, num_survey, section_type, num_verticals,
                dist_pipf, dist_max, dist_min, elev_max, elev_min, step_elev, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    # points
    points_path = data_dir / "_PerfilTransversalVert_.csv"
    if points_path.exists():
        dfp = pd.read_csv(points_path, dtype=str)
        rows = []
        for _, r in dfp.iterrows():
            rows.append(
                (int(str(r.get("RegistroID")).split(".")[0]), 
                 _to_float(r.get("Distancia")), 
                 _to_float(r.get("Cota"))))
        conn.executemany(
            "INSERT OR REPLACE INTO vertical_profile (survey_id, distance, elevation) VALUES (?, ?, ?)", rows
        )

def load_rating_curves(conn, data_dir: Path):
      """Load rating curve parameters from _CurvaDescarga_.csv into rating_curve 
  table."""
      path = data_dir / "_CurvaDescarga_.csv"
      if not path.exists():
          return

      df = pd.read_csv(path, dtype=str)

      rows = []
      for _, r in df.iterrows():
          rows.append((
              int(str(r.get("EstacaoCodigo"))),
              str(r.get("NumeroCurva")),
              _date_txt(r.get("PeriodoValidadeInicio")),
              _date_txt(r.get("PeriodoValidadeFim")),
              _to_int(r.get("CotaMinima")),
              _to_int(r.get("CotaMaxima")),
              _to_float(r.get("CoefH0")),
              _to_float(r.get("CoefA")),
              _to_float(r.get("CoefN")),
              _date_txt(r.get("DataIns"))
          ))

      conn.executemany("""
          INSERT OR REPLACE INTO rating_curve (
              station_id, segment_number, start_date, end_date, h_min, h_max, h0_param, a_param, n_param, date_inserted
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      """, rows)

# ----------------- main -----------------
def main():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        tables_dir = DATA_DIR / "raw_tables"
        load_station(conn, tables_dir)
        load_timeseries(conn, tables_dir)
        load_stage_discharge(conn, tables_dir)
        load_vertical_profiles(conn, tables_dir)
        load_rating_curves(conn, tables_dir)

    print(f"Loaded into {DB_PATH}")


if __name__ == "__main__":
    main()
