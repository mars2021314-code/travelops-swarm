# Car Rental Alignment

## Use when

- Rental times must align with changed flights or hotels
- Pickup/dropoff logic is the main constraint

## Objectives

- Align rental timing with itinerary changes
- Avoid unsupported assumptions about pickup location or time
- Minimize redundant follow-up questions

## Recommended sequence

1. Extract known pickup/dropoff clues from handoff and working memory
2. Search rental options if needed
3. Clarify only unresolved fields
4. Execute update/booking/cancel action when parameters are sufficient
5. Summarize alignment with flight/hotel constraints

## Common failure modes

- Assuming airport pickup without confirmation
- Forgetting downstream timing after a flight change
- Claiming confirmation without tool evidence
