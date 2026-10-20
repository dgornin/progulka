Model: `openai/gpt-6-luna`, queries anchored at 2026-10-20. Avoid hits count stops matching the hand-written avoid of the query (eval/queries.json); the agent's parsed one is brief_avoid in the JSON.

## Single walks: agent vs. one prompt

Run 2026-10-20T15:14+03:00, commit `fdacd35`. Agent stops are source records with coordinates, so they are located by construction; for the baseline, located means found in OpenStreetMap. The baseline has no per-stop interest labels, so its interests are not measured (—).

| query | system | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | interests covered | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | agent | 7 | 7/7 | 0 | 0 | no | 0 | 3/3 | 8.6 | 16 | 0.0034 | 44.7 |
| kazan-history | agent | 10 | 10/10 | 0 | 0 | no | 0 | 4/4 | 5.8 | 16 | 0.0038 | 57.1 |
| paris-impressionism | agent | 7 | 7/7 | 0 | 0 | no | 0 | 3/3 | 8.7 | 16 | 0.003 | 44.5 |
