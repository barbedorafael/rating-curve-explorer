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

def _first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _date_txt(v):
    try:
        d = pd.to_datetime(v, errors="coerce")
        if pd.isna(d):
            return None
        return d.strftime("%Y-%m-%d")
    except Exception:
        return None


def _melt_monthly(df: pd.DataFrame, prefix: str, station_id: int):
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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.execute("PRAGMA foreign_keys = ON;")

def load_station(conn, sdir, sid):
    """Load station metadata from _Estacao_.csv into stations table."""
    estacao_csv = sdir / "_Estacao_.csv"
    df = pd.read_csv(estacao_csv)

    row = {
        "station_id": sid,  # enforce folder ID
        "name": df.at[0, "Nome"],
        "station_type": df.at[0, "TipoEstacao"],
        "lon": df.at[0, "Longitude"],
        "lat": df.at[0, "Latitude"],
        "basin_id": df.at[0, "BaciaCodigo"],
        "sub_basin_id": df.at[0, "SubBaciaCodigo"],
        "river_id": df.at[0, "RioCodigo"],
        "state_id": df.at[0, "EstadoCodigo"],
        "municipality_id": df.at[0, "MunicipioCodigo"],
        "responsible_id": df.at[0, "ResponsavelCodigo"],
        "responsible_unit": df.at[0, "ResponsavelUnidade"],
        "responsible_jurisdiction": df.at[0, "ResponsavelJurisdicao"],
        "operator_id": df.at[0, "OperadoraCodigo"],
        "operator_unit": df.at[0, "OperadoraUnidade"],
        "operator_subunit": df.at[0, "OperadoraSubUnidade"],
        "additional_code": df.at[0, "CodigoAdicional"],
        "altitude": df.at[0, "Altitude"],
        "drainage_area": df.at[0, "AreaDrenagem"],
    }

    conn.execute("""
        INSERT OR REPLACE INTO stations (
            station_id, name, station_type, lon, lat, basin_id, sub_basin_id, river_id,
            state_id, municipality_id, responsible_id, responsible_unit,
            responsible_jurisdiction, operator_id, operator_unit, operator_subunit,
            additional_code, altitude, drainage_area
        )
        VALUES (
            :station_id, :name, :station_type, :lon, :lat, :basin_id, :sub_basin_id, :river_id,
            :state_id, :municipality_id, :responsible_id, :responsible_unit,
            :responsible_jurisdiction, :operator_id, :operator_unit, :operator_subunit,
            :additional_code, :altitude, :drainage_area
        )
    """, row)

def load_timeseries(conn, station_dir: Path, station_id: int):
    for file in ["_Cotas_.csv", "_Vazoes_.csv"]:
        path = station_dir / file
        df = pd.read_csv(path, dtype=str)
        if file == "_Cotas_.csv":
            prefix = "Cota"
        elif file == "_Vazoes_.csv":
            prefix = "Vazao"
        rows = _melt_monthly(df, prefix, station_id)
        conn.executemany(
            "INSERT OR REPLACE INTO timeseries_cota (station_id, date, method, value, status) VALUES (?, ?, ?, ?, ?)",
            rows,
        )

def load_stage_discharge(conn, station_dir: Path, station_id: int):
    path = station_dir / "_ResumoDescarga_.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str)
    station_col = _first(df, ["EstacaoCodigo"])
    date_col = _first(df, ["Data"])
    time_col = _first(df, ["Hora"])
    consistency_col = _first(df, ["NivelConsistencia"])
    stage_col = _first(df, ["Cota"])
    q_col = _first(df, ["Vazao"])
    vel_col = _first(df, ["VelMedia"])
    width_col = _first(df, ["Largura"])
    depth_col = _first(df, ["Profundidade"])
    area_col = _first(df, ["AreaMolhada"])
    instr_col = _first(df, ["MedidorVazao"])

    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                int(str(r.get(station_col) or station_id).strip()),
                _date_txt(r.get(date_col)),
                (str(r.get(time_col)).strip()[-12:-7] if r.get(time_col) not in (None, "", "NaN") else None),
                _to_int(r.get(consistency_col)),
                _to_float(r.get(stage_col)),
                _to_float(r.get(q_col)),
                _to_float(r.get(area_col)),
                _to_float(r.get(vel_col)),
                _to_float(r.get(width_col)),
                _to_float(r.get(depth_col)),
                _to_int(r.get(instr_col)),
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


def load_vertical_profiles(conn, station_dir: Path, station_id: int):
    # surveys
    survey_path = station_dir / "_PerfilTransversal_.csv"
    if survey_path.exists():
        df = pd.read_csv(survey_path, dtype=str)
        survey_id_col = _first(df, ["PerfilTransversalCodigo", "RegistroID", "Codigo"])
        date_col = _first(df, ["DataLevantamento", "Data"])
        rows = []
        for _, r in df.iterrows():
            sid_raw = r.get(survey_id_col)
            if sid_raw in (None, "", "NaN") or pd.isna(sid_raw):
                continue
            try:
                survey_id = int(str(sid_raw).split(".")[0])
            except Exception:
                continue
            dt = _date_txt(r.get(date_col))
            if not dt:
                continue
            rows.append(
                (
                    survey_id,
                    station_id if r.get("EstacaoCodigo") in (None, "", "NaN") else int(str(r.get("EstacaoCodigo")).split(".")[0]),
                    dt,
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
    points_path = station_dir / "_PerfilTransversalVert_.csv"
    if points_path.exists():
        dfp = pd.read_csv(points_path, dtype=str)
        survey_fk_col = _first(dfp, ["PerfilTransversalCodigo", "RegistroID", "Codigo"])
        if survey_fk_col:
            rows = []
            for _, r in dfp.iterrows():
                sid_raw = r.get(survey_fk_col)
                if sid_raw in (None, "", "NaN") or pd.isna(sid_raw):
                    continue
                try:
                    survey_id = int(str(sid_raw).split(".")[0])
                except Exception:
                    continue
                rows.append((survey_id, _to_float(r.get("Distancia")), _to_float(r.get("Cota"))))
            conn.executemany(
                "INSERT OR REPLACE INTO vertical_profile (survey_id, distance, elevation) VALUES (?, ?, ?)", rows
            )


# ----------------- main -----------------
def main():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # walk station folders: data/<digits>/
        for sdir in sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()]):
            sid = int(sdir.name)
            load_station(conn, sdir, sid)
            load_timeseries(conn, sdir, sid)
            load_stage_discharge(conn, sdir, sid)
            load_vertical_profiles(conn, sdir, sid)

    print(f"Loaded into {DB_PATH}")


if __name__ == "__main__":
    main()
