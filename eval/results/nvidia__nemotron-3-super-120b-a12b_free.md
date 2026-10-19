Model: `nvidia/nemotron-3-super-120b-a12b:free`, queries anchored at 2026-10-20. Agent stops are source records with coordinates, so they are located by construction; for the baseline, located means found in OpenStreetMap.

| query | system | stops | real places (located) | closed at visit | impossible legs | over window | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | agent | 10 | 10/10 | 0 | 0 | no | 7.9 | 12 | 0.0 | 112.7 |
| spb-constructivism | baseline | 11 | 8/11 | 0 | 5 | no | 13.2 | 1 | 0.0 | 41.9 |
| kazan-history | agent | 2 | 2/2 | 0 | 0 | no | 0.1 | 16 | 0.0 | 43.4 |
| kazan-history | baseline | 7 | 7/7 | 0 | 6 | no | 30.1 | 1 | 0.0 | 11.2 |
| paris-impressionism | agent | 2 | 2/2 | 0 | 0 | no | 1.0 | 16 | 0.0 | 32.2 |
| paris-impressionism | baseline | 7 | 7/7 | 2 | 3 | no | 10.4 | 1 | 0.0 | 132.6 |
