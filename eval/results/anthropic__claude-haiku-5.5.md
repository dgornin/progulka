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

## Harness: chat with web search

`anthropic/claude-haiku-5.5`: run 2026-10-20T20:54+03:00, commit `9bb00f7+dirty`, reasoning off; `anthropic/claude-sonnet-5.5`: run 2026-10-20T21:02+03:00, commit `9bb00f7+dirty`, reasoning default. One request to a general chat model with OpenRouter's web search server tool (the provider's native search, at most 5 searches of 5 results): the baseline's prompt, today's date and an invitation to check the places on the web. The agent's model copies the prose answer into the baseline's JSON, checked by the baseline's code. Tokens, cost (search fees included) and seconds are the chat answer's, not the copy's; the baseline's seconds include the eval's OpenStreetMap lookups. Cited pages are — when the engine returns no citations; the answers are harness_answer in the JSON.

| query | model | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | mean pop | top share | walk km | web searches | cited pages | tokens | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | anthropic/claude-haiku-5.5 | 10 | 7/10 | 0 | 3 | no | 2 | 0.53 | 0.1 | 32.6 | 3 | 10 | 25499 | 0.0334 | 11.6 |
| spb-constructivism | anthropic/claude-sonnet-5.5 | 10 | 8/10 | 1 | 3 | no | 0 | — | 0.0 | 14.0 | 5 | 9 | 72189 | 0.2704 | 80.2 |
| spb-classic-kids | anthropic/claude-haiku-5.5 | 7 | 6/7 | 1 | 4 | no | 0 | 0.68 | 0.29 | 18.7 | 5 | 9 | 65303 | 0.0575 | 14.5 |
| spb-classic-kids | anthropic/claude-sonnet-5.5 | 10 | 6/10 | 0 | 1 | no | 0 | 0.7 | 0.3 | 3.0 | 5 | 6 | 68352 | 0.2293 | 48.1 |
| moscow-modernism | anthropic/claude-haiku-5.5 | 9 | 7/9 | 0 | 2 | no | 0 | 0.4 | 0.0 | 7.8 | 2 | 6 | 20279 | 0.0227 | 9.8 |
| moscow-modernism | anthropic/claude-sonnet-5.5 | 8 | 5/8 | 1 | 0 | no | 0 | — | 0.0 | 6.3 | 5 | 11 | 79036 | 0.2694 | 64.4 |
| kazan-history | anthropic/claude-haiku-5.5 | 10 | 4/10 | 1 | 4 | no | 0 | 0.54 | 0.0 | 17.4 | 4 | 8 | 32947 | 0.0441 | 11.3 |
| kazan-history | anthropic/claude-sonnet-5.5 | 10 | 6/10 | 1 | 2 | no | 0 | 0.61 | 0.0 | 20.5 | 5 | 10 | 75178 | 0.2567 | 58.5 |
| kaliningrad-koenigsberg | anthropic/claude-haiku-5.5 | 7 | 4/7 | 0 | 1 | no | 0 | 0.55 | 0.0 | 6.4 | 5 | 10 | 61881 | 0.0573 | 17.1 |
| kaliningrad-koenigsberg | anthropic/claude-sonnet-5.5 | 4 | 4/4 | 0 | 3 | no | 0 | 0.53 | 0.0 | 15.8 | 5 | 11 | 73163 | 0.262 | 66.3 |
| nizhny-views | anthropic/claude-haiku-5.5 | 11 | 4/11 | 0 | 1 | no | 0 | — | 0.0 | 7.1 | 3 | 9 | 25043 | 0.0333 | 11.1 |
| ekb-avantgarde | anthropic/claude-haiku-5.5 | 7 | 5/7 | 0 | 2 | no | 0 | 0.55 | 0.0 | 3.6 | 5 | 9 | 59985 | 0.0568 | 12.4 |
| paris-impressionism | anthropic/claude-haiku-5.5 | 9 | 5/9 | 0 | 3 | no | 0 | 0.82 | 0.56 | 3.9 | 4 | 9 | 69777 | 0.0477 | 11.7 |

## Summary: every system on the same walks

Over the single walks every system answered. Sums of stops, closed and impossible legs; per-route means of walk km, mean pop (over the routes that have one), tokens (over the routes that recorded them: — for runs before tokens were recorded), cost and seconds. Seconds are not like for like: the agent's are its whole pipeline, the one prompt's include the eval's OpenStreetMap lookups, the chat's are its answer alone.

| system | routes | real places (located) | closed at visit | impossible legs | routes with an impossible leg | walk km per route | mean pop | tokens per route | cost $ per route | s per route |
|---|---|---|---|---|---|---|---|---|---|---|
| agent, `anthropic/claude-haiku-5.5` | 5 | 51/51 | 0 | 0 | 0/5 | 5.9 | 0.54 | — | 0.0087 | 34.6 |
| one prompt, `anthropic/claude-haiku-5.5` | 5 | 36/44 | 0 | 18 | 5/5 | 15.4 | 0.61 | — | 0.0004 | 26.1 |
| chat + web search, `anthropic/claude-haiku-5.5` | 5 | 28/43 | 2 | 14 | 5/5 | 16.6 | 0.54 | 41182 | 0.043 | 12.9 |
| chat + web search, `anthropic/claude-sonnet-5.5` | 5 | 29/42 | 3 | 9 | 4/5 | 11.9 | 0.61 | 73584 | 0.2576 | 63.5 |

Without `anthropic/claude-sonnet-5.5`, which answered fewer walks: over 8 walks.

| system | routes | real places (located) | closed at visit | impossible legs | routes with an impossible leg | walk km per route | mean pop | tokens per route | cost $ per route | s per route |
|---|---|---|---|---|---|---|---|---|---|---|
| agent, `anthropic/claude-haiku-5.5` | 8 | 79/79 | 0 | 0 | 0/8 | 5.2 | 0.53 | — | 0.0084 | 32.2 |
| one prompt, `anthropic/claude-haiku-5.5` | 8 | 55/68 | 1 | 26 | 8/8 | 14.3 | 0.63 | — | 0.0004 | 23.8 |
| chat + web search, `anthropic/claude-haiku-5.5` | 8 | 42/70 | 2 | 20 | 8/8 | 12.2 | 0.58 | 45089 | 0.0441 | 12.4 |
