# database.py
# SQLite layer for AI Construction Safety Platform
# Replaces JSON files for incidents and worker history
# Site_Safety.json and live_frame.jpg stay as files (Streamlit reads them live)

import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager


DB_PATH = os.path.join("data", "safety_platform.db")


def init_db():
    """
    Creates the database and all tables if they don't exist.
    Call this ONCE at pipeline startup.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    os.makedirs("data", exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            -- Every closed incident goes here
            CREATE TABLE IF NOT EXISTS incidents (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id           TEXT    NOT NULL,
                person_id           TEXT    NOT NULL,
                violation_type      TEXT    NOT NULL,
                severity            TEXT    NOT NULL,
                risk_score          INTEGER NOT NULL,
                risk_level          TEXT    NOT NULL,
                start_time          TEXT    NOT NULL,
                end_time            TEXT    NOT NULL,
                duration_seconds    REAL    NOT NULL,
                confidence          REAL    NOT NULL,
                image_path          TEXT,
                near_machinery      INTEGER DEFAULT 0,
                groq_analysis       TEXT,
                created_at          TEXT    DEFAULT (datetime('now'))
            );

            -- One row per worker per camera — upserted on every incident
            CREATE TABLE IF NOT EXISTS workers (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id               TEXT    NOT NULL,
                person_id               TEXT    NOT NULL,
                total_risk_score        INTEGER DEFAULT 0,
                risk_level              TEXT    DEFAULT 'LOW',
                unique_violation_count  INTEGER DEFAULT 0,
                total_incidents         INTEGER DEFAULT 0,
                violations_json         TEXT    DEFAULT '[]',
                last_seen               TEXT,
                UNIQUE(camera_id, person_id)
            );

            -- Alert history
            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id       TEXT    NOT NULL,
                person_id       TEXT    NOT NULL,
                alert_level     INTEGER NOT NULL,
                violation_type  TEXT    NOT NULL,
                message         TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL
            );

            -- Indexes for fast dashboard queries
            CREATE INDEX IF NOT EXISTS idx_incidents_camera
                ON incidents(camera_id);
            CREATE INDEX IF NOT EXISTS idx_incidents_person
                ON incidents(person_id);
            CREATE INDEX IF NOT EXISTS idx_incidents_created
                ON incidents(created_at);
            CREATE INDEX IF NOT EXISTS idx_workers_camera
                ON workers(camera_id);
            CREATE INDEX IF NOT EXISTS idx_alerts_camera
                ON alerts(camera_id);
        """)


@contextmanager
def get_conn():
    """
    Context manager — always closes connection cleanly.
    Uses WAL mode so Streamlit reads while pipeline writes without locking.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row          # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL") # critical for concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  WRITE FUNCTIONS  (called from pipeline)
# ─────────────────────────────────────────────

def insert_incident(camera_id: str, incident: dict) -> int:
    """
    Inserts one closed incident. Returns the new row id.
    """
    with get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO incidents (
                camera_id, person_id, violation_type, severity,
                risk_score, risk_level, start_time, end_time,
                duration_seconds, confidence, image_path,
                near_machinery, groq_analysis
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            camera_id,
            str(incident["person_id"]),
            incident["violation_type"],
            incident.get("severity", "LOW"),
            incident.get("risk_score", 0),
            incident.get("risk_level", "LOW"),
            str(incident["start_time"]),
            str(incident["end_time"]),
            float(incident["duration_seconds"]),
            float(incident.get("confidence", 0)),
            incident.get("Image_path", ""),
            1 if incident.get("near_machinery") else 0,
            incident.get("GROQ_analysis", "")
        ))
        return cursor.lastrowid


def upsert_worker(camera_id: str, person_id: str, risk_score_delta: int,
                  violation_type: str):
    """
    Insert or update worker row.
    Adds risk_score_delta to existing score.
    Merges violation_type into violations list.
    """
    with get_conn() as conn:
        # Try to get existing worker
        row = conn.execute(
            "SELECT * FROM workers WHERE camera_id=? AND person_id=?",
            (camera_id, str(person_id))
        ).fetchone()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if row is None:
            # First time seeing this worker
            violations = [violation_type]
            total_risk = risk_score_delta
            total_inc = 1
            risk_level = _risk_level(total_risk)
            conn.execute("""
                INSERT INTO workers
                    (camera_id, person_id, total_risk_score, risk_level,
                     unique_violation_count, total_incidents,
                     violations_json, last_seen)
                VALUES (?,?,?,?,?,?,?,?)
            """, (camera_id, str(person_id), total_risk, risk_level,
                  len(violations), total_inc, json.dumps(violations), now))
        else:
            violations = json.loads(row["violations_json"])
            if violation_type not in violations:
                violations.append(violation_type)
            total_risk = row["total_risk_score"] + risk_score_delta
            total_inc = row["total_incidents"] + 1
            risk_level = _risk_level(total_risk)
            conn.execute("""
                UPDATE workers SET
                    total_risk_score    = ?,
                    risk_level          = ?,
                    unique_violation_count = ?,
                    total_incidents     = ?,
                    violations_json     = ?,
                    last_seen           = ?
                WHERE camera_id=? AND person_id=?
            """, (total_risk, risk_level, len(violations), total_inc,
                  json.dumps(violations), now, camera_id, str(person_id)))


def insert_alert(camera_id: str, alert: dict):
    """Saves one alert record."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO alerts
                (camera_id, person_id, alert_level, violation_type, message, timestamp)
            VALUES (?,?,?,?,?,?)
        """, (
            camera_id,
            str(alert["person_id"]),
            alert["alert_level"],
            alert["violation_type"],
            alert["message"],
            alert["timestamp"]
        ))


# ─────────────────────────────────────────────
#  READ FUNCTIONS  (called from app.py)
# ─────────────────────────────────────────────

def get_incidents(camera_id: str, limit: int = 100) -> list:
    """Returns latest incidents for a camera as list of dicts."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM incidents
            WHERE camera_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (camera_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_workers(camera_id: str) -> list:
    """Returns all workers for a camera sorted by risk score."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM workers
            WHERE camera_id = ?
            ORDER BY total_risk_score DESC
        """, (camera_id,)).fetchall()
        return [dict(r) for r in rows]


def get_alerts(camera_id: str, limit: int = 50) -> list:
    """Returns latest alerts for a camera."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM alerts
            WHERE camera_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (camera_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_stats(camera_id: str) -> dict:
    """
    Returns summary stats for the dashboard.
    Replaces reading Site_Safety.json for historical stats.
    """
    with get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE camera_id=?", (camera_id,)
        ).fetchone()[0]

        total_risk = conn.execute(
            "SELECT SUM(risk_score) FROM incidents WHERE camera_id=?", (camera_id,)
        ).fetchone()[0] or 0

        workers_count = conn.execute(
            "SELECT COUNT(*) FROM workers WHERE camera_id=?", (camera_id,)
        ).fetchone()[0]

        high_risk_workers = conn.execute(
            "SELECT COUNT(*) FROM workers WHERE camera_id=? AND risk_level='HIGH'",
            (camera_id,)
        ).fetchone()[0]

        top_violation = conn.execute("""
            SELECT violation_type, COUNT(*) as cnt
            FROM incidents WHERE camera_id=?
            GROUP BY violation_type
            ORDER BY cnt DESC LIMIT 1
        """, (camera_id,)).fetchone()

        return {
            "total_incidents": total,
            "historical_risk_score": total_risk,
            "total_workers_tracked": workers_count,
            "high_risk_workers": high_risk_workers,
            "top_violation": top_violation["violation_type"] if top_violation else "None"
        }


# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────

def _risk_level(score: int) -> str:
    if score < 50:
        return "LOW"
    elif score < 100:
        return "MEDIUM"
    return "HIGH"