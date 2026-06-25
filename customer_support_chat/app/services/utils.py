import os
import shutil
import sqlite3
from datetime import datetime
import pandas as pd
import requests
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from qdrant_client import QdrantClient
from typing import List, Dict, Callable

from langchain_core.messages import ToolMessage
from customer_support_chat.app.core.state import State

settings = get_settings()
_qdrant_client = None
_qdrant_client_key = None


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user. "
                        f"The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name}, "
                        "and the booking, update, or other action is not complete until after you have successfully invoked the appropriate tool. "
                        "If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control. "
                        "Do not mention who you are—just act as the proxy for the assistant."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }
    return entry_node


def download_and_prepare_db():
    settings = get_settings()
    db_file = settings.SQLITE_DB_PATH
    db_dir = os.path.dirname(db_file)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    if not os.path.exists(db_file):
        try:
            response = requests.get(db_url, timeout=30)
            response.raise_for_status()
            with open(db_file, "wb") as f:
                f.write(response.content)
            update_dates(db_file)
        except requests.RequestException as exc:
            logger.warning(
                f"Could not download travel database from {db_url}: {exc}. "
                "Creating a minimal local demo database instead."
            )
            create_demo_db(db_file)


def create_demo_db(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_no TEXT PRIMARY KEY,
            book_ref TEXT,
            passenger_id TEXT
        );

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
        );

        CREATE TABLE IF NOT EXISTS ticket_flights (
            ticket_no TEXT,
            flight_id INTEGER,
            fare_conditions TEXT
        );

        CREATE TABLE IF NOT EXISTS boarding_passes (
            ticket_no TEXT,
            flight_id INTEGER,
            seat_no TEXT
        );

        CREATE TABLE IF NOT EXISTS car_rentals (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT,
            price_tier TEXT,
            start_date TEXT,
            end_date TEXT,
            booked INTEGER
        );

        CREATE TABLE IF NOT EXISTS hotels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT,
            price_tier TEXT,
            checkin_date TEXT,
            checkout_date TEXT,
            booked INTEGER
        );

        CREATE TABLE IF NOT EXISTS trip_recommendations (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT,
            keywords TEXT,
            details TEXT,
            booked INTEGER
        );
        """
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO flights (
            flight_id, flight_no, departure_airport, arrival_airport,
            scheduled_departure, scheduled_arrival, status, aircraft_code,
            actual_departure, actual_arrival
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "LX001",
            "JFK",
            "ZRH",
            "2026-04-23 10:00:00",
            "2026-04-23 22:00:00",
            "Scheduled",
            "A320",
            None,
            None,
        ),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO tickets (ticket_no, book_ref, passenger_id) VALUES (?, ?, ?)",
        ("TICK-DEMO-001", "BOOK1", "5102 899977"),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO ticket_flights (ticket_no, flight_id, fare_conditions) VALUES (?, ?, ?)",
        ("TICK-DEMO-001", 1, "Economy"),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO boarding_passes (ticket_no, flight_id, seat_no) VALUES (?, ?, ?)",
        ("TICK-DEMO-001", 1, "12A"),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO car_rentals (id, name, location, price_tier, start_date, end_date, booked) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "Demo Airport Rental", "ZRH Airport", "standard", "2026-04-23", "2026-04-26", 0),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO hotels (id, name, location, price_tier, checkin_date, checkout_date, booked) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "Demo Central Hotel", "Zurich", "midscale", "2026-04-23", "2026-04-26", 0),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO trip_recommendations (id, name, location, keywords, details, booked) VALUES (?, ?, ?, ?, ?, ?)",
        (1, "Old Town Walking Tour", "Zurich", "history walking city", "A short guided walk through Zurich old town.", 0),
    )

    conn.commit()
    conn.close()

def update_dates(db_file):
    backup_file = db_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(db_file, backup_file)

    conn = sqlite3.connect(db_file)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "type": "tool",
                "content": f"Error: {repr(error)}\nPlease fix your mistakes.",
                "tool_call_id": tc["id"],
            }
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list):
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableLambda
    from langgraph.prebuilt import ToolNode

    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def get_qdrant_client():
    global _qdrant_client, _qdrant_client_key

    settings = get_settings()
    if _qdrant_client is not None:
        return _qdrant_client

    qdrant_url = (settings.QDRANT_URL or "").strip()
    try:
        if qdrant_url:
            # 配置并发参数
            client = QdrantClient(
                url=qdrant_url,
                timeout=30,  # 增加超时时间
                prefer_grpc=True,  # 使用gRPC提高性能
                grpc_options={
                    "grpc.max_send_message_length": 100 * 1024 * 1024,  # 100MB
                    "grpc.max_receive_message_length": 100 * 1024 * 1024,
                }
            )
            client.get_collections()
            _qdrant_client = client
            _qdrant_client_key = ("url", qdrant_url)
            return client
    except Exception as e:
        if settings.REQUIRE_QDRANT_SERVER:
            raise RuntimeError(
                f"Qdrant server is required but unavailable at {qdrant_url}: {e}"
            ) from e
        logger.warning(
            f"Failed to connect to Qdrant server at {qdrant_url}. "
            f"Falling back to local Qdrant storage at {settings.QDRANT_PATH}. Error: {str(e)}"
        )

    os.makedirs(settings.QDRANT_PATH, exist_ok=True)
    try:
        client = QdrantClient(path=settings.QDRANT_PATH)
    except RuntimeError as exc:
        logger.warning(
            f"Local Qdrant storage at {settings.QDRANT_PATH} is unavailable. "
            f"Using in-memory Qdrant for this process. Error: {str(exc)}"
        )
        client = QdrantClient(":memory:")
        _qdrant_client_key = ("memory", ":memory:")
    else:
        _qdrant_client_key = ("path", settings.QDRANT_PATH)
    client.get_collections()
    _qdrant_client = client
    return client

def flight_info_to_string(flight_info: List[Dict]) -> str:
    info_lines = [] 
    i = 0
    for flight in flight_info:
        i += 1
        line = (
            f"Ticket [{i}]:\n"
            f"Ticket Number: {flight['ticket_no']}\n"
            f"Booking Reference: {flight['book_ref']}\n"
            f"Flight ID: {flight['flight_id']}\n"
            f"Flight Number: {flight['flight_no']}\n"
            f"Departure: {flight['departure_airport']} at {flight['scheduled_departure']}\n"
            f"Arrival: {flight['arrival_airport']} at {flight['scheduled_arrival']}\n"
            f"Seat: {flight['seat_no']}\n"
            f"Fare Class: {flight['fare_conditions']}\n"
            f"\n\n"
        )
        info_lines.append(line)

    info_lines = f"User current booked flight(s) details:\n" + "\n".join(info_lines)

    return "\n".join(info_lines)
