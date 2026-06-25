from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.redis_controls import (
    invalidate_query_cache,
    redis_distributed_lock,
)


settings = get_settings()
RUNTIME_TABLES = [
    "tickets",
    "flights",
    "ticket_flights",
    "boarding_passes",
    "car_rentals",
    "hotels",
    "trip_recommendations",
]


def _sqlite_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        settings.SQLITE_DB_PATH,
        timeout=settings.SQLITE_BUSY_TIMEOUT_MS / 1000,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {settings.SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _is_postgres() -> bool:
    return settings.DATABASE_URL.startswith(("postgres://", "postgresql://"))


def using_postgres() -> bool:
    return _is_postgres()


@contextmanager
def db_connection() -> Iterator[Any]:
    if not _is_postgres():
        conn = _sqlite_connection()
        try:
            yield conn
        finally:
            conn.close()
        return

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL points to PostgreSQL, but psycopg is not installed."
        ) from exc

    with psycopg.connect(settings.DATABASE_URL) as conn:
        yield conn


def sql_placeholders(query: str) -> str:
    if _is_postgres():
        return query.replace("?", "%s")
    return query


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_placeholders(query), params)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        cursor.close()
    return [dict(zip(columns, row)) for row in rows]


def execute_write(query: str, params: tuple[Any, ...] = ()) -> int:
    lock_name = "db:write"
    with redis_distributed_lock(lock_name) as acquired:
        if not acquired:
            raise TimeoutError(f"Could not acquire distributed write lock for {lock_name}")

        rowcount = _execute_write_unlocked(query, params)
        if rowcount > 0:
            invalidate_query_cache()
        return rowcount


def _execute_write_unlocked(query: str, params: tuple[Any, ...] = ()) -> int:
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_placeholders(query), params)
        rowcount = cursor.rowcount
        conn.commit()
        cursor.close()
    return rowcount


def initialize_postgres_from_sqlite() -> None:
    if not _is_postgres():
        return

    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_no TEXT PRIMARY KEY,
                book_ref TEXT,
                passenger_id TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS flights (
                flight_id INTEGER PRIMARY KEY,
                flight_no TEXT,
                departure_airport TEXT,
                arrival_airport TEXT,
                scheduled_departure TEXT,
                scheduled_arrival TEXT,
                status TEXT,
                aircraft_code TEXT,
                actual_departure TEXT,
                actual_arrival TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_flights (
                ticket_no TEXT,
                flight_id INTEGER,
                fare_conditions TEXT,
                PRIMARY KEY (ticket_no, flight_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS boarding_passes (
                ticket_no TEXT,
                flight_id INTEGER,
                seat_no TEXT,
                PRIMARY KEY (ticket_no, flight_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS car_rentals (
                id INTEGER PRIMARY KEY,
                name TEXT,
                location TEXT,
                price_tier TEXT,
                start_date TEXT,
                end_date TEXT,
                booked INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hotels (
                id INTEGER PRIMARY KEY,
                name TEXT,
                location TEXT,
                price_tier TEXT,
                checkin_date TEXT,
                checkout_date TEXT,
                booked INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trip_recommendations (
                id INTEGER PRIMARY KEY,
                name TEXT,
                location TEXT,
                keywords TEXT,
                details TEXT,
                booked INTEGER
            )
            """
        )
        cursor.execute("SELECT COUNT(*) FROM tickets")
        has_seed_data = cursor.fetchone()[0] > 0
        conn.commit()
        cursor.close()

    if has_seed_data:
        return

    source = sqlite3.connect(settings.SQLITE_DB_PATH)
    try:
        source.row_factory = sqlite3.Row
        for table in RUNTIME_TABLES:
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue
            columns = rows[0].keys()
            placeholders = ", ".join(["?"] * len(columns))
            column_sql = ", ".join(columns)
            insert_sql = (
                f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders}) "
                "ON CONFLICT DO NOTHING"
            )
            with db_connection() as conn:
                cursor = conn.cursor()
                for row in rows:
                    cursor.execute(
                        sql_placeholders(insert_sql),
                        tuple(row[column] for column in columns),
                    )
                conn.commit()
                cursor.close()
    finally:
        source.close()
