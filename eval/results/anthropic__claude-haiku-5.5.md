Model: `anthropic/claude-haiku-5.5`, queries anchored at 2026-10-20. Avoid hits count stops matching the hand-written avoid of the query (eval/queries.json); the agent's parsed one is brief_avoid in the JSON. Mean pop is the mean popularity (0..1: Wikipedia page views and language versions) of the stops that have one; top share is the share of stops with pop >= 0.7 among those whose popularity is known (scored, or without a Wikidata id); — when the lookup failed.

## Single walks: agent vs. one prompt

Run 2026-10-20T19:26+03:00, commit `c620619`. Agent stops are source records with coordinates, so they are located by construction; for the baseline, located means found in OpenStreetMap. The baseline has no per-stop interest labels, so its interests are not measured (—).

| query | system | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | interests covered | mean pop | top share | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | agent | 12 | 12/12 | 0 | 0 | no | 0 | 3/3 | 0.47 | 0.0 | 7.0 | 6 | 0.0086 | 29.1 |
| spb-constructivism | baseline | 11 | 7/11 | 0 | 2 | no | 0 | — | 0.65 | 0.09 | 18.5 | 1 | 0.0005 | 35.0 |
| spb-classic-kids | agent | 8 | 8/8 | 0 | 0 | no | 0 | 3/3 | 0.63 | 0.25 | 4.2 | 6 | 0.0086 | 41.5 |
| spb-classic-kids | baseline | 7 | 6/7 | 0 | 3 | no | 0 | — | 0.6 | 0.0 | 5.6 | 1 | 0.0003 | 18.3 |
| moscow-modernism | agent | 11 | 11/11 | 0 | 0 | no | 0 | 2/2 | — | 0.0 | 4.3 | 5 | 0.0056 | 24.1 |
| moscow-modernism | baseline | 9 | 9/9 | 0 | 4 | no | 0 | — | 0.54 | 0.0 | 14.0 | 1 | 0.0004 | 33.4 |
| kazan-history | agent | 11 | 11/11 | 0 | 0 | no | 0 | 4/4 | 0.57 | 0.18 | 4.4 | 5 | 0.0068 | 28.5 |
| kazan-history | baseline | 9 | 8/9 | 0 | 5 | no | 0 | — | 0.64 | 0.11 | 17.1 | 1 | 0.0004 | 18.4 |
| kaliningrad-koenigsberg | agent | 9 | 9/9 | 0 | 0 | no | 0 | 5/5 | 0.51 | 0.0 | 9.7 | 8 | 0.0141 | 49.8 |
| kaliningrad-koenigsberg | baseline | 8 | 6/8 | 0 | 4 | no | 0 | — | 0.62 | 0.0 | 21.7 | 1 | 0.0003 | 25.5 |
| nizhny-views | agent | 11 | 11/11 | 0 | 0 | no | 0 | 4/4 | 0.44 | 0.0 | 4.9 | 6 | 0.0077 | 26.7 |
| nizhny-views | baseline | 8 | 7/8 | 0 | 3 | no | 0 | — | — | 0.0 | 9.0 | 1 | 0.0004 | 25.5 |
| ekb-avantgarde | agent | 9 | 9/9 | 0 | 0 | no | 0 | 3/3 | 0.36 | 0.0 | 4.3 | 6 | 0.0075 | 28.6 |
| ekb-avantgarde | baseline | 8 | 4/8 | 0 | 1 | no | 0 | — | 0.57 | 0.0 | 4.0 | 1 | 0.0004 | 24.5 |
| paris-impressionism | agent | 8 | 8/8 | 0 | 0 | no | 0 | 3/3 | 0.76 | 0.25 | 2.6 | 6 | 0.0083 | 29.3 |
| paris-impressionism | baseline | 8 | 8/8 | 1 | 4 | no | 0 | — | 0.78 | 0.62 | 24.8 | 1 | 0.0003 | 9.9 |

## Trips of several days: agent only

Run 2026-10-20T19:26+03:00, commit `c620619`. The one-prompt baseline plans a single walk by design, so trips run the agent only. Closed, impossible legs and walk km are summed over the days; repeats are pairs of stops on different days that are one place.

| query | days (planned/requested) | stops per day | closed at visit | impossible legs | days over window | repeats across days | avoid hits | interests covered | mean pop | top share | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kazan-3-days | 3/3 | 15, 13, 11 | 0 | 0 | 0 | 0 | 0 | 5/5 | 0.49 | 0.05 | 15.6 | 16 | 0.0249 | 36.8 |
| spb-5-days | 5/5 | 15, 11, 11, 14, 14 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0.45 | 0.02 | 41.3 | 28 | 0.0433 | 74.4 |
