from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.db import execute_write
from customer_support_chat.app.core.redis_controls import redis_cached
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from typing import List, Dict, Optional, Union
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH

cars_vectordb = VectorDB(table_name="car_rentals", collection_name="car_rentals_collection")

@tool
@redis_cached()
def search_car_rentals(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """Search for car rentals based on a natural language query."""
    search_results = cars_vectordb.search(query, limit=limit)

    rentals = []
    for result in search_results:
        payload = result.payload
        rentals.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "start_date": payload["start_date"],
            "end_date": payload["end_date"],
            "booked": payload["booked"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return rentals

@tool
def book_car_rental(rental_id: int) -> str:
    """Book a car rental by its ID."""
    rowcount = execute_write("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))

    if rowcount > 0:
        return f"Car rental {rental_id} successfully booked."
    else:
        return f"No car rental found with ID {rental_id}."

@tool
def update_car_rental(
    rental_id: int,
    start_date: Optional[Union[datetime, date]] = None,
    end_date: Optional[Union[datetime, date]] = None,
) -> str:
    """Update a car rental's start and end dates by its ID."""
    rowcount = 0
    if start_date:
        rowcount += execute_write(
            "UPDATE car_rentals SET start_date = ? WHERE id = ?",
            (start_date.strftime('%Y-%m-%d'), rental_id),
        )
    if end_date:
        rowcount += execute_write(
            "UPDATE car_rentals SET end_date = ? WHERE id = ?",
            (end_date.strftime('%Y-%m-%d'), rental_id),
        )

    if rowcount > 0:
        return f"Car rental {rental_id} successfully updated."
    else:
        return f"No car rental found with ID {rental_id}."

@tool
def cancel_car_rental(rental_id: int) -> str:
    """Cancel a car rental by its ID."""
    rowcount = execute_write("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))

    if rowcount > 0:
        return f"Car rental {rental_id} successfully cancelled."
    else:
        return f"No car rental found with ID {rental_id}."
