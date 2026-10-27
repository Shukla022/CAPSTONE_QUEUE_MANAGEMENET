r"""SQLite booking store for the appointment-only QMS with Dual-Role Auth.

Dependency-free (uses Python's built-in ``sqlite3``). Holds three entities:

* **users** — authenticated accounts for Patients, Doctors, and Hospital Admins.
  Supports secure SHA-256 salted password verification and profile attributes.
* **doctors** — the roster: each doctor belongs to a department and has a duty
  shift (``shift_start``/``shift_end``). ``slot_minutes`` / ``buffer_minutes``
  are fallbacks; live slots are generated from predicted service time and a
  dynamic buffer. Managed from the Hospital / Doctor Portal.
* **appointments** — the live bookings, each pinned to a patient account,
  a doctor, and a concrete clock-time slot (``slot_start``/``slot_end``),
  moving through the queue lifecycle:

      booked -> checked_in -> in_service -> done
                           \-> no_show / cancelled

The predictive layer annotates each booking (e.g. ``noshow_prob``) and the Live
Queue derives the doctor's live schedule from the current statuses and the
actual service start/finish times recorded here.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    username         TEXT    UNIQUE NOT NULL,
    password_hash    TEXT    NOT NULL,
    login_password   TEXT,
    role             TEXT    NOT NULL,  -- 'patient', 'doctor', 'admin'
    full_name        TEXT    NOT NULL,
    email            TEXT,
    phone            TEXT,
    age              INTEGER,
    gender           TEXT,
    doctor_id        INTEGER,           -- linked doctor roster id if role == 'doctor'
    department       TEXT,
    medical_history  TEXT,
    created_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name            TEXT    NOT NULL,
    patient_user_id         INTEGER,            -- linked users.id for authenticated patient
    department              TEXT    NOT NULL,
    doctor_id               INTEGER,
    appointment_date        TEXT    NOT NULL,   -- ISO date (YYYY-MM-DD)
    time_slot               TEXT,               -- coarse label (kept for reports)
    slot_start              TEXT,               -- concrete clock time "HH:MM"
    slot_end                TEXT,               -- concrete clock time "HH:MM"
    day_of_week             TEXT,
    priority_level          INTEGER DEFAULT 0,  -- 0 routine, 1 emergency
    lead_time_hours         REAL    DEFAULT 0,
    previous_no_show_count  INTEGER DEFAULT 0,
    noshow_prob             REAL,
    predicted_wait          REAL,
    predicted_arrival_delay REAL,           -- minutes early (-) / late (+) vs slot
    status                  TEXT    DEFAULT 'booked',
    check_in_time           TEXT,
    service_start           TEXT,               -- ISO timestamp service began
    service_minutes         REAL,               -- realised service duration
    notes                   TEXT,               -- consultation notes/diagnosis summary
    emergency_reason        TEXT,               -- reason for emergency priority
    assigned_by             TEXT,               -- role or user who assigned priority/modified
    created_at              TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS doctors (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    department     TEXT    NOT NULL,
    shift_start    TEXT    NOT NULL DEFAULT '09:00',
    shift_end      TEXT    NOT NULL DEFAULT '13:00',
    slot_minutes   REAL    NOT NULL DEFAULT 15,
    buffer_minutes REAL    NOT NULL DEFAULT 5,
    active         INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS doctor_availability (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id      INTEGER NOT NULL,
    available_date TEXT    NOT NULL,
    is_available   INTEGER NOT NULL DEFAULT 1,
    shift_start    TEXT,
    shift_end      TEXT,
    published_by   TEXT    NOT NULL DEFAULT 'doctor',
    created_at     TEXT    NOT NULL,
    updated_at     TEXT,
    UNIQUE(doctor_id, available_date),
    FOREIGN KEY(doctor_id) REFERENCES doctors(id)
);

CREATE TABLE IF NOT EXISTS queue_notifications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    department     TEXT    NOT NULL,
    doctor_id      INTEGER NOT NULL,
    message        TEXT    NOT NULL,
    created_at     TEXT    NOT NULL
);
"""

LIFECYCLE = ["booked", "checked_in", "in_service", "done", "no_show", "cancelled"]

# Columns added after the first release; ensured on every init for old DBs.
_APPT_MIGRATIONS = {
    "patient_user_id": "INTEGER",
    "doctor_id": "INTEGER",
    "slot_start": "TEXT",
    "slot_end": "TEXT",
    "service_start": "TEXT",
    "service_minutes": "REAL",
    "predicted_arrival_delay": "REAL",
    "notes": "TEXT",
    "emergency_reason": "TEXT",
    "assigned_by": "TEXT",
}

_USER_MIGRATIONS = {
    "login_password": "TEXT",
}


@contextmanager
def get_conn(db_path: str | Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any columns introduced after a DB was first created (idempotent)."""
    have = {r["name"] for r in conn.execute("PRAGMA table_info(appointments)")}
    for col, decl in _APPT_MIGRATIONS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE appointments ADD COLUMN {col} {decl}")

    have_users = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    for col, decl in _USER_MIGRATIONS.items():
        if col not in have_users:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")

    # Backfill login_password for existing users if NULL
    conn.execute("UPDATE users SET login_password = 'doctor123' WHERE role = 'doctor' AND (login_password IS NULL OR login_password = '')")
    conn.execute("UPDATE users SET login_password = 'admin123' WHERE username = 'admin' AND (login_password IS NULL OR login_password = '')")
    conn.execute("UPDATE users SET login_password = 'patient123' WHERE username IN ('patient_john', 'patient_alice') AND (login_password IS NULL OR login_password = '')")

    # Ensure doctor_availability table exists (doctor-published duty days)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doctor_availability (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id      INTEGER NOT NULL,
            available_date TEXT    NOT NULL,
            is_available   INTEGER NOT NULL DEFAULT 1,
            shift_start    TEXT,
            shift_end      TEXT,
            published_by   TEXT    NOT NULL DEFAULT 'doctor',
            created_at     TEXT    NOT NULL,
            updated_at     TEXT,
            UNIQUE(doctor_id, available_date),
            FOREIGN KEY(doctor_id) REFERENCES doctors(id)
        )
        """
    )
    have_avail = {r["name"] for r in conn.execute("PRAGMA table_info(doctor_availability)")}
    for col, decl in (
        ("shift_start", "TEXT"),
        ("shift_end", "TEXT"),
        ("published_by", "TEXT NOT NULL DEFAULT 'doctor'"),
    ):
        if col not in have_avail:
            conn.execute(f"ALTER TABLE doctor_availability ADD COLUMN {col} {decl}")

    now_ts = datetime.now().isoformat(timespec="seconds")

    # Keep existing booked dates bookable so historical appointments stay valid
    appt_dates = conn.execute(
        "SELECT DISTINCT doctor_id, appointment_date FROM appointments WHERE doctor_id IS NOT NULL"
    ).fetchall()
    for r in appt_dates:
        conn.execute(
            """
            INSERT OR IGNORE INTO doctor_availability
                (doctor_id, available_date, is_available, published_by, created_at, updated_at)
            VALUES (?, ?, 1, 'system', ?, ?)
            """,
            [r["doctor_id"], r["appointment_date"], now_ts, now_ts],
        )


def init_db(db_path: str | Path) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


# =====================================================================
# Authentication & User Management
# =====================================================================
def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Generate secure salted SHA-256 password hash in format salt$hexdigest."""
    if not salt:
        salt = secrets.token_hex(8)
    dk = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${dk}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hexdigest format."""
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, expected_dk = stored_hash.split("$", 1)
    actual_dk = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual_dk, expected_dk)


def create_user(db_path: str | Path, user_rec: Dict) -> Tuple[Optional[int], Optional[str]]:
    """Register a new user account. Returns (user_id, error_message)."""
    init_db(db_path)
    user_rec = dict(user_rec)
    username = str(user_rec.get("username", "")).strip().lower()
    raw_pwd = str(user_rec.get("password", "")).strip()
    role = str(user_rec.get("role", "patient")).strip().lower()

    if not username or len(username) < 3:
        return None, "Username must be at least 3 characters long."
    if not raw_pwd or len(raw_pwd) < 4:
        return None, "Password must be at least 4 characters long."
    if role not in ("patient", "doctor", "admin"):
        return None, f"Invalid role '{role}'. Must be patient, doctor, or admin."

    full_name = str(user_rec.get("full_name") or username.title()).strip()
    pwd_hash = hash_password(raw_pwd)
    login_pwd = str(user_rec.get("login_password") or raw_pwd)
    now_iso = datetime.now().isoformat(timespec="seconds")

    with get_conn(db_path) as conn:
        exist = conn.execute("SELECT id FROM users WHERE username = ?", [username]).fetchone()
        if exist:
            return None, f"Username '{username}' already exists. Please choose another."

        cur = conn.execute(
            """
            INSERT INTO users (
                username, password_hash, login_password, role, full_name, email, phone,
                age, gender, doctor_id, department, medical_history, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                username, pwd_hash, login_pwd, role, full_name,
                user_rec.get("email"), user_rec.get("phone"),
                user_rec.get("age"), user_rec.get("gender"),
                user_rec.get("doctor_id"), user_rec.get("department"),
                user_rec.get("medical_history"), now_iso,
            ],
        )
        return int(cur.lastrowid), None


def authenticate_user(db_path: str | Path, username: str, password: str) -> Optional[Dict]:
    """Validate credentials and return user profile dict if valid, else None."""
    init_db(db_path)
    username = str(username or "").strip().lower()
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", [username]).fetchone()
        if not row:
            return None
        user_dict = dict(row)
        if verify_password(password, user_dict.get("password_hash", "")):
            # Never leak password hash to application state
            user_dict.pop("password_hash", None)
            user_dict.pop("login_password", None)
            return user_dict
    return None


def get_user_by_id(db_path: str | Path, user_id: int) -> Optional[Dict]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("password_hash", None)
        d.pop("login_password", None)
        return d


def get_user_by_username(db_path: str | Path, username: str) -> Optional[Dict]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", [username.strip().lower()]).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("password_hash", None)
        d.pop("login_password", None)
        return d


def update_user_profile(db_path: str | Path, user_id: int, **fields) -> None:
    if not fields:
        return
    if "password" in fields:
        raw_p = fields.pop("password")
        fields["password_hash"] = hash_password(raw_p)
        fields["login_password"] = raw_p
    sets = ",".join(f"{k} = ?" for k in fields)
    args = list(fields.values()) + [user_id]
    with get_conn(db_path) as conn:
        conn.execute(f"UPDATE users SET {sets} WHERE id = ?", args)


def list_users(db_path: str | Path, role: Optional[str] = None) -> List[Dict]:
    init_db(db_path)
    q = "SELECT id, username, role, full_name, email, phone, age, gender, doctor_id, department, created_at FROM users"
    args = []
    if role:
        q += " WHERE role = ?"
        args.append(role.strip().lower())
    q += " ORDER BY role, full_name"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


# =====================================================================
# Doctors / roster
# =====================================================================
_DOCTOR_FIELDS = ["name", "department", "shift_start", "shift_end",
                  "slot_minutes", "buffer_minutes", "active", "created_at"]


def add_doctor(db_path: str | Path, rec: Dict) -> int:
    init_db(db_path)
    rec = dict(rec)
    rec.setdefault("active", 1)
    rec.setdefault("slot_minutes", 15)
    rec.setdefault("buffer_minutes", 5)
    rec.setdefault("shift_start", "09:00")
    rec.setdefault("shift_end", "13:00")
    rec.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    values = [rec.get(f) for f in _DOCTOR_FIELDS]
    placeholders = ",".join("?" for _ in _DOCTOR_FIELDS)
    with get_conn(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO doctors ({','.join(_DOCTOR_FIELDS)}) VALUES ({placeholders})",
            values,
        )
        # Duty days are published by the doctor when they set their slots —
        # do not auto-seed admin-side availability calendars.
        return int(cur.lastrowid)



def list_doctors(db_path: str | Path, *, department: Optional[str] = None,
                 active_only: bool = False) -> List[Dict]:
    init_db(db_path)
    q = "SELECT * FROM doctors WHERE 1=1"
    args: List = []
    if department:
        q += " AND department = ?"
        args.append(department)
    if active_only:
        q += " AND active = 1"
    q += " ORDER BY department, name"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def list_all_hospital_doctors(db_path: str | Path) -> List[Dict]:
    """Retrieve ALL hospital doctors (active and inactive) along with their linked user login credentials."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        query = """
            SELECT 
                d.id,
                d.name,
                d.department,
                d.shift_start,
                d.shift_end,
                d.slot_minutes,
                d.buffer_minutes,
                d.active,
                d.created_at,
                u.id AS user_id,
                u.username,
                u.login_password
            FROM doctors d
            LEFT JOIN users u ON u.doctor_id = d.id AND u.role = 'doctor'
            ORDER BY d.id
        """
        rows = [dict(r) for r in conn.execute(query).fetchall()]

        # Ensure every doctor has linked username and login_password
        for doc in rows:
            if not doc.get("username"):
                name_clean = doc["name"].replace("Dr.", "").replace("Dr", "").strip()
                slug = "_".join(name_clean.lower().split()) or f"doctor_{doc['id']}"
                username = f"dr_{slug}"
                # Guarantee uniqueness
                c = 1
                base_u = username
                while conn.execute("SELECT id FROM users WHERE username = ?", [username]).fetchone():
                    username = f"{base_u}_{c}"
                    c += 1

                pwd_hash = hash_password("doctor123")
                conn.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, login_password, role, full_name,
                        department, doctor_id, created_at
                    ) VALUES (?, ?, ?, 'doctor', ?, ?, ?, ?)
                    """,
                    [username, pwd_hash, "doctor123", doc["name"], doc["department"], doc["id"], datetime.now().isoformat(timespec="seconds")]
                )
                doc["username"] = username
                doc["login_password"] = "doctor123"
            elif not doc.get("login_password"):
                doc["login_password"] = "doctor123"
                conn.execute("UPDATE users SET login_password = 'doctor123' WHERE id = ?", [doc["user_id"]])
        return rows


def add_doctor_with_login(db_path: str | Path, doctor_rec: Dict, username: str, password: str) -> Tuple[int, Optional[str]]:
    """Create a new doctor roster entry and linked user login account. Returns (doctor_id, error)."""
    init_db(db_path)
    clean_username = str(username or "").strip().lower()
    clean_password = str(password or "").strip()

    if not clean_username or len(clean_username) < 3:
        return 0, "Username must be at least 3 characters long."
    if not clean_password or len(clean_password) < 4:
        return 0, "Password must be at least 4 characters long."

    with get_conn(db_path) as conn:
        exist = conn.execute("SELECT id FROM users WHERE username = ?", [clean_username]).fetchone()
        if exist:
            return 0, f"Username '{clean_username}' already exists. Choose another."

        doc_id = add_doctor(db_path, doctor_rec)

        now_iso = datetime.now().isoformat(timespec="seconds")
        pwd_hash = hash_password(clean_password)
        conn.execute(
            """
            INSERT INTO users (
                username, password_hash, login_password, role, full_name,
                department, doctor_id, created_at
            ) VALUES (?, ?, ?, 'doctor', ?, ?, ?, ?)
            """,
            [
                clean_username,
                pwd_hash,
                clean_password,
                doctor_rec.get("name"),
                doctor_rec.get("department"),
                doc_id,
                now_iso,
            ],
        )

        # Slots open only after the doctor publishes duty hours for a date.
        return doc_id, None


def set_doctor_status(db_path: str | Path, doctor_id: int, active: int | bool) -> None:
    update_doctor(db_path, doctor_id, active=1 if active else 0)


def publish_doctor_duty(
    db_path: str | Path,
    doctor_id: int,
    available_date: str,
    *,
    is_available: bool | int = True,
    shift_start: Optional[str] = None,
    shift_end: Optional[str] = None,
    published_by: str = "doctor",
    update_default_shift: bool = True,
) -> Dict:
    """Doctor publishes (or closes) on-duty hours for a calendar date.

    Publishing opens ML-packed bookable slots inside ``shift_start``–``shift_end``.
    Closing (``is_available=0``) removes the day from patient booking.
    """
    init_db(db_path)
    now_iso = datetime.now().isoformat(timespec="seconds")
    val = 1 if is_available else 0
    with get_conn(db_path) as conn:
        doc = conn.execute("SELECT * FROM doctors WHERE id = ?", [doctor_id]).fetchone()
        if not doc:
            raise ValueError("Doctor not found.")
        start = shift_start or doc["shift_start"] or "09:00"
        end = shift_end or doc["shift_end"] or "13:00"
        conn.execute(
            """
            INSERT INTO doctor_availability (
                doctor_id, available_date, is_available,
                shift_start, shift_end, published_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doctor_id, available_date) DO UPDATE SET
                is_available = excluded.is_available,
                shift_start = excluded.shift_start,
                shift_end = excluded.shift_end,
                published_by = excluded.published_by,
                updated_at = excluded.updated_at
            """,
            [doctor_id, available_date, val, start, end, published_by, now_iso, now_iso],
        )
        if update_default_shift and val and (shift_start or shift_end):
            fields = {}
            if shift_start:
                fields["shift_start"] = start
            if shift_end:
                fields["shift_end"] = end
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields)
                conn.execute(
                    f"UPDATE doctors SET {sets} WHERE id = ?",
                    [*fields.values(), doctor_id],
                )
        return {
            "doctor_id": doctor_id,
            "available_date": available_date,
            "is_available": val,
            "shift_start": start,
            "shift_end": end,
            "published_by": published_by,
        }


def set_doctor_availability(db_path: str | Path, doctor_id: int, available_date: str,
                            is_available: bool | int) -> None:
    """Compatibility wrapper — prefer :func:`publish_doctor_duty`."""
    publish_doctor_duty(
        db_path, doctor_id, available_date,
        is_available=is_available, published_by="system",
        update_default_shift=False,
    )


def get_doctor_duty(db_path: str | Path, doctor_id: int, available_date: str) -> Optional[Dict]:
    """Return the published duty record for a doctor/date, if any."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        r = conn.execute(
            """
            SELECT doctor_id, available_date, is_available, shift_start, shift_end,
                   published_by, created_at, updated_at
            FROM doctor_availability
            WHERE doctor_id = ? AND available_date = ?
            """,
            [doctor_id, available_date],
        ).fetchone()
        return dict(r) if r else None


def is_doctor_available(db_path: str | Path, doctor_id: int, available_date: str) -> bool:
    """Active doctor who has published on-duty hours for ``available_date``."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        doc = conn.execute("SELECT active FROM doctors WHERE id = ?", [doctor_id]).fetchone()
        if not doc or not doc["active"]:
            return False

        r = conn.execute(
            "SELECT is_available FROM doctor_availability WHERE doctor_id = ? AND available_date = ?",
            [doctor_id, available_date],
        ).fetchone()
        # Opt-in: slots exist only after the doctor publishes duty for that day.
        if r is None:
            return False
        return bool(r["is_available"])


def list_doctor_availability(db_path: str | Path, *,
                             doctor_id: Optional[int] = None,
                             available_date: Optional[str] = None) -> List[Dict]:
    """Retrieve date-specific availability records with doctor details."""
    init_db(db_path)
    q = """
        SELECT 
            da.id,
            da.doctor_id,
            da.available_date,
            da.is_available,
            COALESCE(da.shift_start, d.shift_start) AS shift_start,
            COALESCE(da.shift_end, d.shift_end) AS shift_end,
            COALESCE(da.published_by, 'doctor') AS published_by,
            da.created_at,
            da.updated_at,
            d.name AS doctor_name,
            d.department,
            d.active AS doctor_active
        FROM doctor_availability da
        JOIN doctors d ON d.id = da.doctor_id
        WHERE 1=1
    """
    args: List = []
    if doctor_id is not None:
        q += " AND da.doctor_id = ?"
        args.append(doctor_id)
    if available_date:
        q += " AND da.available_date = ?"
        args.append(available_date)
    q += " ORDER BY da.available_date ASC, d.department, d.name"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def list_available_doctors(db_path: str | Path, *,
                           department: Optional[str] = None,
                           available_date: Optional[str] = None) -> List[Dict]:
    """Return doctors who are BOTH active and available on the specified date."""
    init_db(db_path)
    if not available_date:
        return list_doctors(db_path, department=department, active_only=True)

    q = """
        SELECT d.*
        FROM doctors d
        JOIN doctor_availability da ON da.doctor_id = d.id
        WHERE d.active = 1
          AND da.available_date = ?
          AND da.is_available = 1
    """
    args = [available_date]
    if department:
        q += " AND d.department = ?"
        args.append(department)
    q += " ORDER BY d.department, d.name"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def get_doctor(db_path: str | Path, doctor_id: int) -> Optional[Dict]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM doctors WHERE id = ?", [doctor_id]).fetchone()
        return dict(r) if r else None


def update_doctor(db_path: str | Path, doctor_id: int, **fields) -> None:
    if not fields:
        return
    sets = ",".join(f"{k} = ?" for k in fields)
    args = list(fields.values()) + [doctor_id]
    with get_conn(db_path) as conn:
        conn.execute(f"UPDATE doctors SET {sets} WHERE id = ?", args)


def delete_doctor(db_path: str | Path, doctor_id: int) -> None:
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM doctors WHERE id = ?", [doctor_id])


def count_doctors(db_path: str | Path) -> int:
    init_db(db_path)
    with get_conn(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) c FROM doctors").fetchone()["c"])


def seed_doctors(db_path: str | Path, docs: List[Dict]) -> int:
    """Insert a starter roster only if none exist yet. Returns rows inserted."""
    if count_doctors(db_path) > 0:
        return 0
    for d in docs:
        add_doctor(db_path, d)
    return len(docs)


# =====================================================================
# Appointments
# =====================================================================
def check_booking_conflict(db_path: str | Path, doctor_id: int,
                           appointment_date: str, slot_start: str,
                           slot_end: str, exclude_id: Optional[int] = None) -> bool:
    """Return True if (slot_start, slot_end) conflicts with any existing confirmed booking."""
    init_db(db_path)
    from scheduling import to_min
    new_s = to_min(slot_start)
    new_e = to_min(slot_end)
    if new_s is None or new_e is None or new_e <= new_s:
        return False
    with get_conn(db_path) as conn:
        q = """
            SELECT id, slot_start, slot_end, status FROM appointments
            WHERE doctor_id = ? AND appointment_date = ?
              AND status IN ('booked', 'checked_in', 'in_service', 'done')
        """
        args = [doctor_id, appointment_date]
        if exclude_id is not None:
            q += " AND id != ?"
            args.append(exclude_id)
        rows = conn.execute(q, args).fetchall()
        for r in rows:
            ex_s = to_min(r["slot_start"])
            ex_e = to_min(r["slot_end"])
            if ex_s is None:
                continue
            if ex_e is None or ex_e <= ex_s:
                ex_e = ex_s + 15.0
            if max(new_s, ex_s) < min(new_e, ex_e) - 1e-6:
                return True
    return False


def add_booking(db_path: str | Path, rec: Dict) -> int:
    init_db(db_path)
    fields = ["patient_name", "patient_user_id", "department", "doctor_id",
              "appointment_date", "time_slot", "slot_start", "slot_end",
              "day_of_week", "priority_level", "lead_time_hours",
              "previous_no_show_count", "noshow_prob", "predicted_arrival_delay",
              "status", "notes", "emergency_reason", "assigned_by", "created_at"]
    rec = dict(rec)
    rec.setdefault("status", "booked")
    rec.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

    # Backend conflict check for routine appointments with defined slot_start/slot_end
    if int(rec.get("priority_level") or 0) < 1 and rec.get("doctor_id") and rec.get("appointment_date") and rec.get("slot_start") and rec.get("slot_end"):
        if check_booking_conflict(db_path, int(rec["doctor_id"]), rec["appointment_date"], rec["slot_start"], rec["slot_end"]):
            raise ValueError("Cannot book appointment: Selected slot conflicts with an existing appointment for this doctor.")

    values = [rec.get(f) for f in fields]
    placeholders = ",".join("?" for _ in fields)
    with get_conn(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO appointments ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )
        return int(cur.lastrowid)


def list_bookings(db_path: str | Path, *, department: Optional[str] = None,
                  date: Optional[str] = None, doctor_id: Optional[int] = None,
                  patient_user_id: Optional[int] = None,
                  patient_name: Optional[str] = None,
                  statuses: Optional[List[str]] = None) -> List[Dict]:
    init_db(db_path)
    q = "SELECT * FROM appointments WHERE 1=1"
    args: List = []
    if department:
        q += " AND department = ?"
        args.append(department)
    if date:
        q += " AND appointment_date = ?"
        args.append(date)
    if doctor_id is not None:
        q += " AND doctor_id = ?"
        args.append(doctor_id)
    if patient_user_id is not None:
        q += " AND patient_user_id = ?"
        args.append(patient_user_id)
    if patient_name:
        q += " AND LOWER(patient_name) = ?"
        args.append(patient_name.strip().lower())
    if statuses:
        q += f" AND status IN ({','.join('?' for _ in statuses)})"
        args.extend(statuses)
    q += " ORDER BY priority_level DESC, check_in_time IS NULL, check_in_time, id"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def get_booking(db_path: str | Path, booking_id: int) -> Optional[Dict]:
    """Retrieve an appointment booking by its unique ID."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT * FROM appointments WHERE id = ?", [int(booking_id)]).fetchone()
        return dict(r) if r else None


def _norm_name(name: str) -> str:
    return " ".join(str(name or "").split()).strip()


def list_known_patients(db_path: str | Path) -> List[Dict]:
    """Returning patients derived from past bookings (name is the key)."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT patient_name, appointment_date, department, status, id
            FROM appointments
            ORDER BY appointment_date DESC, id DESC
            """
        ).fetchall()
    by_key: Dict[str, Dict] = {}
    for r in rows:
        rec = dict(r)
        key = _norm_name(rec.get("patient_name", "")).lower()
        if not key:
            continue
        if key not in by_key:
            by_key[key] = {
                "display_name": rec["patient_name"],
                "visits": 0,
                "no_shows": 0,
                "last_date": rec["appointment_date"],
                "last_department": rec["department"],
            }
        by_key[key]["visits"] += 1
        if rec.get("status") == "no_show":
            by_key[key]["no_shows"] += 1
    return sorted(by_key.values(), key=lambda p: (p["last_date"], p["display_name"]),
                  reverse=True)


def patient_profile(db_path: str | Path, name: str) -> Optional[Dict]:
    """History for a returning patient, matched case-insensitively on name."""
    key = _norm_name(name).lower()
    if not key:
        return None
    init_db(db_path)
    with get_conn(db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM appointments ORDER BY appointment_date DESC, id DESC"
        ).fetchall()]
    matched = [r for r in rows if _norm_name(r.get("patient_name", "")).lower() == key]
    if not matched:
        return None
    return {
        "display_name": matched[0]["patient_name"],
        "visits": len(matched),
        "no_shows": sum(1 for r in matched if r.get("status") == "no_show"),
        "last_date": matched[0]["appointment_date"],
        "last_department": matched[0]["department"],
        "history": matched[:10],
    }


def update_booking(db_path: str | Path, booking_id: int, **fields) -> None:
    if not fields:
        return
    sets = ",".join(f"{k} = ?" for k in fields)
    args = list(fields.values()) + [booking_id]
    with get_conn(db_path) as conn:
        conn.execute(f"UPDATE appointments SET {sets} WHERE id = ?", args)


def check_in(db_path: str | Path, booking_id: int,
             predicted_wait: Optional[float] = None) -> None:
    booking = get_booking(db_path, booking_id)
    if not booking:
        raise ValueError(f"Booking #{booking_id} not found.")
    appt_date = booking.get("appointment_date", "")
    from datetime import date as _dt_date
    today_iso = _dt_date.today().isoformat()
    if appt_date and appt_date < today_iso:
        raise ValueError("Cannot check in: appointment date has passed.")
    fields = {"status": "checked_in",
              "check_in_time": datetime.now().isoformat(timespec="seconds")}
    if predicted_wait is not None:
        fields["predicted_wait"] = float(predicted_wait)
    update_booking(db_path, booking_id, **fields)


def start_service(db_path: str | Path, booking_id: int) -> None:
    """Move a patient into service and stamp the actual start time."""
    update_booking(db_path, booking_id, status="in_service",
                   service_start=datetime.now().isoformat(timespec="seconds"))


def complete_service(db_path: str | Path, booking_id: int,
                     service_minutes: Optional[float] = None,
                     notes: Optional[str] = None) -> None:
    """Mark done; record realised service duration and optional consultation notes."""
    fields: Dict[str, object] = {"status": "done"}
    if service_minutes is None:
        with get_conn(db_path) as conn:
            r = conn.execute("SELECT service_start FROM appointments WHERE id = ?",
                             [booking_id]).fetchone()
        started = r["service_start"] if r else None
        if started:
            try:
                delta = datetime.now() - datetime.fromisoformat(started)
                service_minutes = round(delta.total_seconds() / 60.0, 2)
            except Exception:  # noqa: BLE001
                service_minutes = None
    if service_minutes is not None:
        fields["service_minutes"] = float(service_minutes)
    if notes is not None and str(notes).strip():
        fields["notes"] = str(notes).strip()
    update_booking(db_path, booking_id, **fields)


def set_emergency_priority(db_path: str | Path, booking_id: int,
                           priority_level: int = 1,
                           reason: str = "",
                           assigned_by: str = "") -> None:
    """Set emergency priority with an audit trail (reason and assigning actor)."""
    fields = {
        "priority_level": int(priority_level),
        "emergency_reason": str(reason or "").strip(),
        "assigned_by": str(assigned_by or "").strip(),
    }
    update_booking(db_path, booking_id, **fields)


def get_patient_medical_history(db_path: str | Path, patient_user_id: int) -> List[Dict]:
    """Fetch all completed visits for a patient with doctor name, duration, and notes."""
    init_db(db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.*, d.name AS doctor_name, d.department AS doctor_department
            FROM appointments a
            LEFT JOIN doctors d ON a.doctor_id = d.id
            WHERE a.patient_user_id = ?
            ORDER BY a.appointment_date DESC, a.slot_start DESC, a.id DESC
            """,
            [patient_user_id],
        ).fetchall()
        return [dict(r) for r in rows]


def set_status(db_path: str | Path, booking_id: int, status: str) -> None:
    if status not in LIFECYCLE:
        raise ValueError(f"Unknown status '{status}'. Use one of {LIFECYCLE}.")
    with get_conn(db_path) as conn:
        r = conn.execute("SELECT status FROM appointments WHERE id = ?", [int(booking_id)]).fetchone()
        if not r:
            raise ValueError(f"Booking #{booking_id} not found.")
        current_status = r["status"]
        if current_status == "done" and status == "no_show":
            raise ValueError("Invalid transition: cannot mark a completed appointment as no-show.")
    update_booking(db_path, booking_id, status=status)


def auto_handle_expired_appointments(db_path: str | Path,
                                     today_iso: Optional[str] = None,
                                     now_min: Optional[float] = None,
                                     grace_minutes: float = 30.0) -> int:
    """Transition expired booked appointments to 'no_show'.

    - Idempotent: only updates records where status == 'booked'.
    - Past dates: if appointment_date < today_iso, mark no_show.
    - Today: if slot_end (or slot_start + fallback) + grace_minutes < now_min, mark no_show.
    - Never touches checked_in, in_service, done, or cancelled.
    - Never touches future appointments (appointment_date > today_iso).
    """
    init_db(db_path)
    from datetime import date as _dt_date
    from scheduling import to_min
    if today_iso is None:
        today_iso = _dt_date.today().isoformat()
    updated_count = 0
    with get_conn(db_path) as conn:
        # 1. Past dates
        cur = conn.execute(
            """
            UPDATE appointments
            SET status = 'no_show'
            WHERE status = 'booked' AND appointment_date < ?
            """,
            [today_iso],
        )
        updated_count += cur.rowcount

        # 2. Today's appointments that passed scheduled slot + grace
        if now_min is not None:
            today_booked = conn.execute(
                """
                SELECT id, slot_start, slot_end FROM appointments
                WHERE status = 'booked' AND appointment_date = ?
                """,
                [today_iso],
            ).fetchall()
            for r in today_booked:
                em = to_min(r["slot_end"])
                sm = to_min(r["slot_start"])
                end_time = em if em is not None else ((sm + 15.0) if sm is not None else None)
                if end_time is not None and (end_time + grace_minutes) < now_min:
                    conn.execute(
                        "UPDATE appointments SET status = 'no_show' WHERE id = ? AND status = 'booked'",
                        [r["id"]],
                    )
                    updated_count += 1
    return updated_count


def create_queue_notification(db_path: str | Path, department: str,
                              doctor_id: int, message: str) -> int:
    """Record an in-app emergency alert for waiting patients in this doctor/department queue."""
    init_db(db_path)
    now_iso = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO queue_notifications (department, doctor_id, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [department, doctor_id, message, now_iso],
        )
        return int(cur.lastrowid)


def list_queue_notifications(db_path: str | Path, *,
                             department: Optional[str] = None,
                             doctor_id: Optional[int] = None,
                             since_time: Optional[str] = None,
                             date: Optional[str] = None) -> List[Dict]:
    """Retrieve queue notifications filtered by department, doctor, and/or date."""
    init_db(db_path)
    q = "SELECT * FROM queue_notifications WHERE 1=1"
    args = []
    if department:
        q += " AND department = ?"
        args.append(department)
    if doctor_id is not None:
        q += " AND doctor_id = ?"
        args.append(doctor_id)
    if since_time:
        q += " AND created_at >= ?"
        args.append(since_time)
    if date:
        q += " AND created_at LIKE ?"
        args.append(f"{date}%")
    q += " ORDER BY id DESC"
    with get_conn(db_path) as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]


def counts_by_status(db_path: str | Path, *, date: Optional[str] = None) -> Dict[str, int]:
    init_db(db_path)
    q = "SELECT status, COUNT(*) c FROM appointments"
    args: List = []
    if date:
        q += " WHERE appointment_date = ?"
        args.append(date)
    q += " GROUP BY status"
    with get_conn(db_path) as conn:
        return {r["status"]: int(r["c"]) for r in conn.execute(q, args).fetchall()}


def clear_all(db_path: str | Path, *, include_doctors: bool = False, include_users: bool = False) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM appointments")
        if include_doctors:
            conn.execute("DELETE FROM doctors")
        if include_users:
            conn.execute("DELETE FROM users")


def seed_default_users_and_doctors(db_path: str | Path) -> None:
    """Populate authentic doctors across the 8 real departments and standard accounts."""
    init_db(db_path)
    # Admin
    if not get_user_by_username(db_path, "admin"):
        create_user(db_path, {
            "username": "admin",
            "password": "admin123",
            "role": "admin",
            "full_name": "Chief Medical Administrator",
            "department": "Administration",
        })
    # Authentic clinical doctors across the 8 real departments
    authentic_docs = [
        {"name": "Dr. Rajesh Sharma", "department": "Cardiology", "shift_start": "09:00", "shift_end": "13:00", "username": "dr_sharma"},
        {"name": "Dr. Priya Patel", "department": "General Medicine", "shift_start": "09:00", "shift_end": "13:00", "username": "dr_patel"},
        {"name": "Dr. Ananya Rao", "department": "Dermatology", "shift_start": "10:00", "shift_end": "14:00", "username": "dr_ananya"},
        {"name": "Dr. Vikram Seth", "department": "Orthopedics", "shift_start": "09:00", "shift_end": "13:00", "username": "dr_seth"},
        {"name": "Dr. Neha Deshmukh", "department": "Gynecology", "shift_start": "14:00", "shift_end": "18:00", "username": "dr_deshmukh"},
        {"name": "Dr. Arjun Nair", "department": "ENT", "shift_start": "10:00", "shift_end": "14:00", "username": "dr_nair"},
        {"name": "Dr. Kavita Reddy", "department": "Dentistry", "shift_start": "09:00", "shift_end": "13:00", "username": "dr_reddy"},
        {"name": "Dr. Suresh Menon", "department": "Ophthalmology", "shift_start": "14:00", "shift_end": "18:00", "username": "dr_menon"},
    ]
    for d in authentic_docs:
        exist = list_doctors(db_path, department=d["department"])
        if not exist:
            did = add_doctor(db_path, {
                "name": d["name"],
                "department": d["department"],
                "shift_start": d["shift_start"],
                "shift_end": d["shift_end"],
                "slot_minutes": 0,
                "buffer_minutes": 5,
                "active": 1,
            })
        else:
            did = exist[0]["id"]
        if not get_user_by_username(db_path, d["username"]):
            create_user(db_path, {
                "username": d["username"],
                "password": "doctor123",
                "role": "doctor",
                "full_name": d["name"],
                "department": d["department"],
                "doctor_id": did,
            })
    # Seed default patients (with NO department set)
    if not get_user_by_username(db_path, "patient_john"):
        create_user(db_path, {
            "username": "patient_john",
            "password": "patient123",
            "role": "patient",
            "full_name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-0101",
        })
    if not get_user_by_username(db_path, "patient_alice"):
        create_user(db_path, {
            "username": "patient_alice",
            "password": "patient123",
            "role": "patient",
            "full_name": "Alice Smith",
            "email": "alice.smith@example.com",
            "phone": "+1-555-0102",
        })

