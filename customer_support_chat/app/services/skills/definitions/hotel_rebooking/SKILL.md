# Hotel Rebooking

## Use when

- Hotel dates or location must change
- Flight timing or destination constraints have changed

## Objectives

- Preserve date consistency
- Avoid unnecessary re-asking of known constraints
- Keep booking claims grounded in tool results

## Recommended sequence

1. Reuse known destination/date constraints from handoff and working memory
2. Search hotel options
3. Ask only for truly missing preferences
4. Execute booking/update/cancel action only when details are sufficient
5. Summarize result and remaining dependencies

## Common failure modes

- Asking again for dates already provided
- Assuming budget or hotel class without support
- Ignoring airport proximity requirements carried in handoff context
