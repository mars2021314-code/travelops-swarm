# Flight Change Coordination

## Use when

- A flight change is needed
- Other bookings may need to be aligned after the flight change

## Objectives

- Search viable replacement options
- Avoid committing to an unsupported option
- Preserve dependencies with hotel, rental car, and excursions

## Recommended sequence

1. Search available flight options
2. Confirm the user's preferred option if multiple viable options exist
3. Execute the flight change only after the choice is clear
4. Check whether hotel, rental car, or excursion dates now conflict
5. Hand off to the relevant peer agent(s) with structured context

## Common failure modes

- Updating the ticket before the user has chosen an option
- Ignoring airport/time changes that affect rental car pickup
- Ignoring arrival-date changes that affect hotel check-in

## Output expectations

- Distinguish available options from chosen option
- Confirm what changed
- Identify downstream impacts
