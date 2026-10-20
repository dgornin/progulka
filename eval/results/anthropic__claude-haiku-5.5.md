Model: `anthropic/claude-haiku-5.5`, queries anchored at 2026-10-20. Avoid hits count stops matching the hand-written avoid of the query (eval/queries.json); the agent's parsed one is brief_avoid in the JSON.

## Single walks: agent vs. one prompt

Run 2026-10-20T14:35+03:00, commit `2e742b4`. Agent stops are source records with coordinates, so they are located by construction; for the baseline, located means found in OpenStreetMap. The baseline has no per-stop interest labels, so its interests are not measured (—).

| query | system | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | interests covered | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | agent | 13 | 13/13 | 0 | 0 | no | 0 | 3/3 | 7.1 | 6 | 0.0076 | 27.5 |
| spb-constructivism | baseline | 11 | 7/11 | 0 | 4 | no | 0 | — | 21.1 | 1 | 0.0005 | 38.2 |
| spb-classic-kids | agent | 6 | 6/6 | 0 | 0 | no | 0 | 3/3 | 3.9 | 7 | 0.0097 | 31.0 |
| spb-classic-kids | baseline | 7 | 7/7 | 0 | 4 | no | 0 | — | 9.1 | 1 | 0.0003 | 13.8 |
| moscow-modernism | agent | 9 | 9/9 | 0 | 0 | no | 0 | 2/2 | 3.6 | 5 | 0.0055 | 25.2 |
| moscow-modernism | baseline | 8 | 4/8 | 0 | 1 | no | 0 | — | 7.7 | 1 | 0.0003 | 21.1 |
| kazan-history | agent | 11 | 11/11 | 0 | 0 | no | 0 | 4/4 | 4.4 | 7 | 0.0127 | 46.2 |
| kazan-history | baseline | 9 | 8/9 | 0 | 5 | no | 0 | — | 15.8 | 1 | 0.0003 | 15.1 |
| kaliningrad-koenigsberg | agent | 9 | 9/9 | 0 | 0 | no | 0 | 5/5 | 5.3 | 5 | 0.0068 | 23.5 |
| kaliningrad-koenigsberg | baseline | 10 | 8/10 | 0 | 6 | no | 0 | — | 23.8 | 1 | 0.0004 | 31.1 |
| nizhny-views | agent | 13 | 13/13 | 0 | 0 | no | 0 | 4/4 | 4.7 | 6 | 0.0079 | 31.3 |
| nizhny-views | baseline | 8 | 4/8 | 1 | 1 | no | 0 | — | 2.7 | 1 | 0.0004 | 22.5 |
| ekb-avantgarde | agent | 8 | 8/8 | 0 | 0 | no | 0 | 3/3 | 3.7 | 7 | 0.0089 | 35.2 |
| ekb-avantgarde | baseline | 9 | 7/9 | 0 | 3 | no | 0 | — | 13.7 | 1 | 0.0004 | 25.0 |
| paris-impressionism | agent | 8 | 8/8 | 0 | 0 | no | 0 | 3/3 | 3.6 | 7 | 0.0095 | 35.2 |
| paris-impressionism | baseline | 7 | 7/7 | 2 | 2 | no | 0 | — | 10.3 | 1 | 0.0003 | 4.5 |

## Trips of several days: agent only

Run 2026-10-20T14:49+03:00, commit `d45abbd`. The one-prompt baseline plans a single walk by design, so trips run the agent only. Closed, impossible legs and walk km are summed over the days; repeats are pairs of stops on different days that are one place.

| query | days (planned/requested) | stops per day | closed at visit | impossible legs | days over window | repeats across days | avoid hits | interests covered | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kazan-3-days | 3/3 | 13, 12, 13 | 0 | 0 | 0 | 0 | 0 | 5/5 | 16.9 | 19 | 0.0294 | 59.9 |
| spb-5-days | 5/5 | 13, 13, 11, 11, 9 | 0 | 0 | 0 | 0 | 0 | 3/3 | 43.2 | 27 | 0.043 | 74.6 |
