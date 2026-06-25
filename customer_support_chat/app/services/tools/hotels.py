from vectorizer.app.vectordb.vectordb import VectorDB
from customer_support_chat.app.core.db import execute_write
from customer_support_chat.app.core.redis_controls import redis_cached
from customer_support_chat.app.core.settings import get_settings
from langchain_core.tools import tool
from typing import Optional, Union, List, Dict
from datetime import datetime, date

settings = get_settings()
db = settings.SQLITE_DB_PATH
hotels_vectordb = VectorDB(table_name="hotels", collection_name="hotels_collection")

@tool
@redis_cached()
def search_hotels(
    query: str,
    limit: int = 2,
) -> List[Dict]:
    """Search for hotels based on a natural language query."""
    search_results = hotels_vectordb.search(query, limit=limit)

    hotels = []
    for result in search_results:
        payload = result.payload
        hotels.append({
            "id": payload["id"],
            "name": payload["name"],
            "location": payload["location"],
            "price_tier": payload["price_tier"],
            "checkin_date": payload["checkin_date"],
            "checkout_date": payload["checkout_date"],
            "booked": payload["booked"],
            "chunk": payload["content"],
            "similarity": result.score,
        })
    return hotels

@tool
def book_hotel(hotel_id: int) -> str:
    """Book a hotel by its ID."""
    rowcount = execute_write("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))

    if rowcount > 0:
        return f"Hotel {hotel_id} successfully booked."
    else:
        return f"No hotel found with ID {hotel_id}."

@tool
def update_hotel(
    hotel_id: int,
    checkin_date: Optional[Union[datetime, date]] = None,
    checkout_date: Optional[Union[datetime, date]] = None,
) -> str:
    """Update a hotel's check-in and check-out dates by its ID."""
    rowcount = 0
    if checkin_date:
        rowcount += execute_write(
            "UPDATE hotels SET checkin_date = ? WHERE id = ?",
            (checkin_date.strftime('%Y-%m-%d'), hotel_id),
        )
    if checkout_date:
        rowcount += execute_write(
            "UPDATE hotels SET checkout_date = ? WHERE id = ?",
            (checkout_date.strftime('%Y-%m-%d'), hotel_id),
        )

    if rowcount > 0:
        return f"Hotel {hotel_id} successfully updated."
    else:
        return f"No hotel found with ID {hotel_id}."

@tool
def cancel_hotel(hotel_id: int) -> str:
    """Cancel a hotel by its ID."""
    rowcount = execute_write("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))

    if rowcount > 0:
        return f"Hotel {hotel_id} successfully cancelled."
    else:
        return f"No hotel found with ID {hotel_id}."
