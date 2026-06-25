import os
import sqlite3
import uuid
import re
import requests
from tqdm import tqdm
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectorizer.app.core.settings import get_settings
from vectorizer.app.core.logger import logger
from customer_support_chat.app.services.utils import get_qdrant_client
from .chunkenizer import recursive_character_splitting
from vectorizer.app.embeddings.embedding_generator import generate_embedding
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
from more_itertools import chunked
import time

settings = get_settings()

class VectorDB:
    def __init__(self, table_name, collection_name, create_collection=False):
        self.table_name = table_name
        self.collection_name = collection_name
        self.connect_to_qdrant()
        if create_collection:
            self.create_or_clear_collection()
        else:
            self.ensure_collection()

    def connect_to_qdrant(self):
        self.client = get_qdrant_client()
        logger.info("Connected to Qdrant")

    def create_or_clear_collection(self):
        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection {self.collection_name} already exists. Recreating it.")
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {self.collection_name}")

    def ensure_collection(self):
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Created missing collection: {self.collection_name}")

    def format_content(self, data, collection_name):
        # Implement formatting logic for different collections
        if collection_name == 'car_rentals_collection':
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Car rental: {data['name']}, located at: {data['location']}, price tier: {data['price_tier']}. " +\
                f"Rental period starts on {data['start_date']} and ends on {data['end_date']}. " +\
                    f"Currently, the rental is: {booking_status}."

        elif collection_name == 'excursions_collection':
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Excursion: {data['name']} at {data['location']}. " +\
                f"Additional details: {data['details']}. " +\
                    f"Currently, the excursion is {booking_status}. " +\
                        f"Keywords: {data['keywords']}."

        elif collection_name == 'flights_collection':

            return f"Flight {data['flight_no']} from {data['departure_airport']} to {data['arrival_airport']} " +\
                f"was scheduled to depart at {data['scheduled_departure']} and arrive at {data['scheduled_arrival']}. " +\
                    f"The actual departure was at {data['actual_departure']} and the actual arrival was at {data['actual_arrival']}. " +\
                        f"Currently, the flight status is '{data['status']}' and it was operated with aircraft code {data['aircraft_code']}."

        elif collection_name == 'hotels_collection':
            booking_status = "booked" if data['booked'] else "not booked"
            return f"Hotel {data['name']} located in {data['location']} is categorized as {data['price_tier']} tier. " +\
                f"The check-in date is {data['checkin_date']} and the check-out date is {data['checkout_date']}. " +\
                    f"Currently, the booked status is: {booking_status}."

        elif collection_name == 'faq_collection':
            return data['page_content']  # Return the page content directly for FAQ
        else:
            return str(data)

    async def generate_embedding_async(self, content, session):
        return generate_embedding(content)

    async def process_chunk(self, chunk, metadata, session):
        embedding = await self.generate_embedding_async(chunk, session)
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "content": chunk,
                **metadata
            }
        )

    async def create_embeddings_async(self):
        if self.table_name == "faq":
            await self.index_faq_docs()
        else:
            await self.index_regular_docs()

    async def index_regular_docs(self):
        db_connection = sqlite3.connect(settings.SQLITE_DB_PATH)
        cursor = db_connection.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name}")
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        db_connection.close()

        if not rows:
            logger.warning(f"No data found in table {self.table_name}")
            return

        data = [dict(zip(column_names, row)) for row in rows]
        chunk_records = []
        for item in data:
            formatted = self.format_content(item, self.collection_name)
            for chunk in recursive_character_splitting(formatted):
                if chunk:
                    chunk_records.append((chunk, item))

        if not chunk_records:
            logger.warning(f"No valid chunks generated for {self.collection_name}")
            return

        batch_size = 100  # Adjust this value based on rate limit
        delay = 1  # Delay in seconds between batches

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(chunk_records), batch_size):
                batch = chunk_records[i:i+batch_size]
                tasks = [
                    self.process_chunk(chunk, metadata, session)
                    for chunk, metadata in batch
                ]
                
                points = []
                for task in tqdm_asyncio.as_completed(tasks, desc=f"Generating embeddings for {self.collection_name} (batch {i//batch_size + 1})", total=len(tasks)):
                    try:
                        point = await task
                        if point is not None:
                            points.append(point)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")

                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"Indexed {len(points)} documents into {self.collection_name} (batch {i//batch_size + 1})")

                if i + batch_size < len(chunk_records):
                    logger.info(f"Waiting for {delay} seconds before processing the next batch...")
                    await asyncio.sleep(delay)

        total_indexed = len(chunk_records)
        logger.info(f"Finished indexing. Total documents indexed into {self.collection_name}: {total_indexed}")

    async def index_faq_docs(self):
        faq_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(faq_url) as response:
                    response.raise_for_status()
                    faq_text = await response.text()
        except Exception as exc:
            logger.warning(f"Could not download FAQ document from {faq_url}: {exc}. Using minimal demo FAQ.")
            faq_text = (
                "## Flight changes\n"
                "Customers may request flight changes when seats are available. "
                "Check current ticket details and available replacement flights before updating a ticket.\n\n"
                "## Cancellations\n"
                "Cancellations should be confirmed with the customer before any write action is executed.\n\n"
                "## Booking support\n"
                "Hotel, car-rental, and excursion bookings should only be confirmed after the corresponding tool succeeds."
            )

        docs = []
        for txt in re.split(r"(?=\n?##)", faq_text):
            page_content = txt.strip()
            if not page_content:
                continue
            lines = page_content.splitlines()
            title = lines[0].lstrip("#").strip() if lines else "Policy"
            answer = "\n".join(lines[1:]).strip() or page_content
            docs.append(
                {
                    "page_content": page_content,
                    "question": title,
                    "answer": answer,
                    "category": "policy",
                }
            )

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.process_chunk(
                    doc["page_content"],
                    {
                        "type": "faq",
                        "question": doc["question"],
                        "answer": doc["answer"],
                        "category": doc["category"],
                    },
                    session,
                )
                for doc in docs
            ]
            points = await tqdm_asyncio.gather(*tasks, desc="Generating embeddings for FAQ documents")

        if points:
            for batch in chunked(points, 100):  # Adjust batch size as needed
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            logger.info(f"Indexed {len(points)} FAQ documents into {self.collection_name}.")
        else:
            logger.warning("No FAQ documents were successfully embedded and indexed.")

    def create_embeddings(self):
        asyncio.run(self.create_embeddings_async())

    def search(self, query, limit=2, with_payload=True):
        query_vector = generate_embedding(query)
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=with_payload
            )

        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
        )
        return getattr(query_result, "points", query_result)

if __name__ == "__main__":
    vectordb = VectorDB("example_table", "example_collection")
    vectordb.create_embeddings()
