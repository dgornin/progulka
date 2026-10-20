Model: `anthropic/claude-haiku-5.5`, queries anchored at 2026-10-20. Avoid hits count stops matching the hand-written avoid of the query (eval/queries.json); the agent's parsed one is brief_avoid in the JSON. Mean pop is the mean popularity (0..1: Wikipedia page views and language versions) of the stops that have one; top share is the share of stops with pop >= 0.7 among those whose popularity is known (scored, or without a Wikidata id); — when the lookup failed.

## Trips of several days: agent only

Run 2026-10-20T21:27+03:00, commit `7a5a8c2`. The one-prompt baseline plans a single walk by design, so trips run the agent only. Closed, impossible legs and walk km are summed over the days; repeats are pairs of stops on different days that are one place.

| query | days (planned/requested) | stops per day | closed at visit | impossible legs | days over window | repeats across days | avoid hits | interests covered | mean pop | top share | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kazan-3-days | 3/3 | 15, 14, 5 | 0 | 0 | 0 | 0 | 0 | 5/5 | 0.5 | 0.12 | 14.5 | 19 | 0.0309 | 56.4 |
| spb-5-days | 5/5 | 17, 10, 13, 9, 12 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0.51 | 0.03 | 34.4 | 27 | 0.0401 | 64.3 |
| moscow-4-days | 4/4 | 14, 8, 10, 12 | 0 | 0 | 0 | 0 | 0 | 5/5 | 0.45 | 0.02 | 31.5 | 21 | 0.0291 | 50.9 |
| paris-3-days | 3/3 | 13, 12, 6 | 0 | 1 | 0 | 0 | 0 | 4/4 | 0.6 | 0.2 | 21.7 | 23 | 0.0338 | 58.8 |
