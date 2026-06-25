from __future__ import annotations

SKILL_REGISTRY = {
    "delay_recovery": {
        "name": "delay_recovery",
        "domain": "triage",
        "description": "Recover from flight delay/cancellation and coordinate downstream itinerary impacts.",
        "recommended_agents": ["triage", "update_flight", "book_hotel", "book_car_rental", "book_excursion"],
        "tags": ["delay", "cancellation", "recovery", "coordination"],
    },
    "flight_change_coordination": {
        "name": "flight_change_coordination",
        "domain": "update_flight",
        "description": "Coordinate a flight change with downstream hotel, rental car, and excursion dependencies.",
        "recommended_agents": ["update_flight", "book_hotel", "book_car_rental", "book_excursion"],
        "tags": ["flight_change", "coordination", "itinerary"],
    },
    "hotel_rebooking": {
        "name": "hotel_rebooking",
        "domain": "book_hotel",
        "description": "Rebook or adjust hotel reservations with date/location dependencies.",
        "recommended_agents": ["book_hotel"],
        "tags": ["hotel", "rebook", "dates"],
    },
    "car_rental_alignment": {
        "name": "car_rental_alignment",
        "domain": "book_car_rental",
        "description": "Align rental pickup/dropoff with updated flight or hotel constraints.",
        "recommended_agents": ["book_car_rental"],
        "tags": ["car_rental", "pickup", "dropoff", "alignment"],
    },
    "excursion_replan": {
        "name": "excursion_replan",
        "domain": "book_excursion",
        "description": "Replan excursions after itinerary changes.",
        "recommended_agents": ["book_excursion"],
        "tags": ["excursion", "replan", "activities"],
    },
    "refund_policy_check": {
        "name": "refund_policy_check",
        "domain": "triage",
        "description": "Check refund or cancellation policy before executing cancellation/refund-related actions.",
        "recommended_agents": ["triage", "update_flight"],
        "tags": ["refund", "policy", "cancellation"],
    },
}