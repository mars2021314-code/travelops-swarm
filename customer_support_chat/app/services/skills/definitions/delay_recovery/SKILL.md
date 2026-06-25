# Delay Recovery

## Use when

- The user's flight is delayed or cancelled
- The change may affect hotel, rental car, or excursion plans
- The user needs a recovery path, not just a single isolated booking action

## Objectives

- Confirm the flight disruption context
- Check downstream itinerary impact
- Coordinate the next best sequence of actions
- Minimize redundant user questioning

## Recommended sequence

1. Confirm the disruption using available flight context or search
2. Identify immediate user goal:
   - rebook flight
   - cancel flight
   - preserve downstream travel plans
3. Check downstream dependencies:
   - hotel dates
   - rental pickup/dropoff
   - excursions or local activities
4. Hand off to the best next domain agent if transactional execution is needed
5. Summarize what has been completed and what remains open

## Common failure modes

- Focusing only on the flight and ignoring downstream bookings
- Asking the user to repeat dates already available in handoff or working memory
- Claiming a rebooking is complete without tool evidence

## Output expectations

- Keep the plan concrete
- Separate completed actions from pending actions
- Be explicit about uncertainties
