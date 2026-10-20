Model: `anthropic/claude-haiku-5.5`, queries anchored at 2026-10-20. Avoid hits count stops matching the hand-written avoid of the query (eval/queries.json); the agent's parsed one is brief_avoid in the JSON. Mean pop is the mean popularity (0..1: Wikipedia page views and language versions) of the stops that have one; top share is the share of stops with pop >= 0.7 among those whose popularity is known (scored, or without a Wikidata id); — when the lookup failed.

## Single walks: agent vs. one prompt

Run 2026-10-20T22:42+03:00, commit `55dab95+dirty`. Agent stops are source records with coordinates, so they are located by construction; for the baseline, located means found in OpenStreetMap. The baseline has no per-stop interest labels, so its interests are not measured (—).

| query | system | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | interests covered | mean pop | top share | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | agent | 13 | 13/13 | 0 | 0 | no | 0 | 3/3 | 0.43 | 0.0 | 8.5 | 6 | 0.0093 | 30.3 |
| spb-constructivism | baseline | 11 | 5/11 | 0 | 1 | no | 0 | — | 0.71 | 0.09 | 6.0 | 1 | 0.0005 | 38.7 |
| spb-classic-kids | agent | 7 | 7/7 | 0 | 0 | no | 0 | 3/3 | 0.61 | 0.29 | 3.4 | 6 | 0.007 | 24.0 |
| spb-classic-kids | baseline | 9 | 7/9 | 0 | 3 | no | 0 | — | — | 0.0 | 9.2 | 1 | 0.0004 | 25.9 |
| moscow-modernism | agent | 10 | 10/10 | 0 | 0 | no | 0 | 2/2 | — | 0.0 | 4.4 | 5 | 0.0058 | 23.8 |
| moscow-modernism | baseline | 10 | 2/10 | 1 | 0 | no | 0 | — | — | 0.0 | 1.5 | 1 | 0.0004 | 35.6 |
| kazan-history | agent | 10 | 10/10 | 0 | 0 | no | 0 | 3/3 | 0.48 | 0.1 | 3.6 | 5 | 0.0065 | 27.4 |
| kazan-history | baseline | 9 | 8/9 | 0 | 5 | no | 0 | — | 0.64 | 0.22 | 17.3 | 1 | 0.0003 | 13.3 |
| kaliningrad-koenigsberg | agent | 9 | 9/9 | 0 | 0 | no | 0 | 5/5 | 0.5 | 0.11 | 7.5 | 5 | 0.0073 | 29.1 |
| kaliningrad-koenigsberg | baseline | 8 | 2/8 | 0 | 0 | no | 0 | — | 0.61 | 0.0 | 2.9 | 1 | 0.0004 | 22.2 |
| nizhny-views | agent | 11 | 11/11 | 0 | 0 | no | 0 | 4/4 | 0.45 | 0.09 | 5.1 | 6 | 0.0091 | 31.8 |
| nizhny-views | baseline | 9 | 3/9 | 1 | 1 | no | 0 | — | — | 0.0 | 3.0 | 1 | 0.0004 | 16.0 |
| ekb-avantgarde | agent | 10 | 10/10 | 0 | 0 | no | 0 | 3/3 | 0.42 | 0.0 | 4.6 | 7 | 0.0098 | 27.6 |
| ekb-avantgarde | baseline | 9 | 5/9 | 0 | 0 | no | 0 | — | 0.57 | 0.0 | 4.0 | 1 | 0.0004 | 25.5 |
| paris-impressionism | agent | 8 | 8/8 | 0 | 0 | no | 0 | 3/3 | 0.62 | 0.12 | 2.4 | 6 | 0.0069 | 27.5 |
| paris-impressionism | baseline | 7 | 7/7 | 1 | 3 | no | 0 | — | 0.82 | 0.71 | 10.3 | 1 | 0.0003 | 5.9 |

## Trips of several days: agent only

Run 2026-10-20T21:27+03:00, commit `7a5a8c2`. The one-prompt baseline plans a single walk by design, so trips run the agent only. Closed, impossible legs and walk km are summed over the days; repeats are pairs of stops on different days that are one place.

| query | days (planned/requested) | stops per day | closed at visit | impossible legs | days over window | repeats across days | avoid hits | interests covered | mean pop | top share | walk km | LLM calls | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kazan-3-days | 3/3 | 11, 10, 3 | 0 | 0 | 0 | 0 | 0 | 5/5 | 0.53 | 0.12 | 10.6 | 31 | 0.051 | 96.2 |
| spb-5-days | 5/5 | 11, 8, 9, 11, 10 | 0 | 0 | 0 | 0 | 0 | 3/3 | 0.44 | 0.0 | 40.1 | 33 | 0.0524 | 83.0 |
| moscow-4-days | 4/4 | 12, 13, 10, 11 | 0 | 0 | 0 | 0 | 0 | 5/5 | 0.55 | 0.07 | 29.4 | 28 | 0.0442 | 71.1 |
| paris-3-days | 3/3 | 8, 15, 12 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0.47 | 0.12 | 14.2 | 22 | 0.0316 | 54.6 |

## Harness: chat with web search

`anthropic/claude-haiku-5.5`: run 2026-10-20T20:54+03:00, commit `9bb00f7+dirty`, reasoning off, rescored 2026-10-20T22:42+03:00 at commit `55dab95+dirty`; `anthropic/claude-sonnet-5.5`: run 2026-10-20T21:02+03:00, commit `9bb00f7+dirty`, reasoning default, rescored 2026-10-20T22:42+03:00 at commit `55dab95+dirty`. One request to a general chat model with OpenRouter's web search server tool (the provider's native search, at most 5 searches of 5 results): the baseline's prompt, today's date and an invitation to check the places on the web. The agent's model copies the prose answer into the baseline's JSON, checked by the baseline's code. Tokens, cost (search fees included) and seconds are the chat answer's, not the copy's; the baseline's seconds include the eval's OpenStreetMap lookups. Cited pages are — when the engine returns no citations; the answers are harness_answer in the JSON.

| query | model | stops | real places (located) | closed at visit | impossible legs | over window | avoid hits | mean pop | top share | walk km | web searches | cited pages | tokens | cost $ | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spb-constructivism | anthropic/claude-haiku-5.5 | 10 | 8/10 | 0 | 4 | no | 2 | 0.53 | 0.1 | 37.8 | 3 | 10 | 25499 | 0.0334 | 11.6 |
| spb-constructivism | anthropic/claude-sonnet-5.5 | 14 | 8/14 | 1 | 3 | no | 0 | — | 0.0 | 14.0 | 5 | 9 | 72189 | 0.2704 | 80.2 |
| spb-classic-kids | anthropic/claude-haiku-5.5 | 7 | 6/7 | 1 | 5 | no | 0 | 0.63 | 0.14 | 19.3 | 5 | 9 | 65303 | 0.0575 | 14.5 |
| spb-classic-kids | anthropic/claude-sonnet-5.5 | 10 | 6/10 | 0 | 1 | no | 0 | 0.7 | 0.3 | 3.0 | 5 | 6 | 68352 | 0.2293 | 48.1 |
| moscow-modernism | anthropic/claude-haiku-5.5 | 9 | 7/9 | 0 | 1 | no | 0 | 0.4 | 0.0 | 7.5 | 2 | 6 | 20279 | 0.0227 | 9.8 |
| moscow-modernism | anthropic/claude-sonnet-5.5 | 8 | 5/8 | 1 | 0 | no | 0 | — | 0.0 | 6.3 | 5 | 11 | 79036 | 0.2694 | 64.4 |
| kazan-history | anthropic/claude-haiku-5.5 | 10 | 4/10 | 1 | 4 | no | 0 | 0.61 | 0.0 | 20.0 | 4 | 8 | 32947 | 0.0441 | 11.3 |
| kazan-history | anthropic/claude-sonnet-5.5 | 10 | 6/10 | 1 | 2 | no | 0 | 0.61 | 0.0 | 20.5 | 5 | 10 | 75178 | 0.2567 | 58.5 |
| kaliningrad-koenigsberg | anthropic/claude-haiku-5.5 | 7 | 4/7 | 0 | 1 | no | 0 | 0.55 | 0.0 | 6.4 | 5 | 10 | 61881 | 0.0573 | 17.1 |
| kaliningrad-koenigsberg | anthropic/claude-sonnet-5.5 | 4 | 4/4 | 0 | 3 | no | 0 | 0.53 | 0.0 | 15.8 | 5 | 11 | 73163 | 0.262 | 66.3 |
| nizhny-views | anthropic/claude-haiku-5.5 | 11 | 4/11 | 0 | 1 | no | 0 | — | 0.0 | 7.1 | 3 | 9 | 25043 | 0.0333 | 11.1 |
| ekb-avantgarde | anthropic/claude-haiku-5.5 | 7 | 5/7 | 0 | 2 | no | 0 | 0.55 | 0.0 | 3.6 | 5 | 9 | 59985 | 0.0568 | 12.4 |
| paris-impressionism | anthropic/claude-haiku-5.5 | 9 | 5/9 | 0 | 3 | no | 0 | 0.82 | 0.56 | 3.9 | 4 | 9 | 69777 | 0.0477 | 11.7 |

## Summary: every system on the same walks

Over the single walks every system answered. Too short on foot: legs whose time, plus 5 minutes, is shorter than the walk (the request asks for a walking route), and the routes with one; too short even by transit: legs whose time, plus 5 minutes, is shorter than both the walk and 10 minutes plus the street distance at 20 km/h, which no metro or taxi makes. Found in OSM is over the specific places only (one call of the agent's model per route sorts the items, the same for every system; the agent's stops are source records with coordinates, found by construction); vague items are the rest: transitions, areas, activities, the route's start or end, «X or Y». Sums of legs and items; per-route means of walk km, mean pop (over the routes that have one), tokens (prompt + completion of every LLM call of the system for the route; — for runs before tokens were recorded), $ and seconds. Seconds are not like for like: the agent's are its whole pipeline, the one prompt's include the eval's OpenStreetMap lookups, the chat's are its answer alone.

| system | routes | too short on foot: legs (routes) | too short even by transit: legs (routes) | closed at visit | found in OSM, of specific places | vague items | walk km per route | mean pop | tokens per route | $ per route | s per route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent, `anthropic/claude-haiku-5.5` | 5 | 0 (0/5) | 0 (0/5) | 0 | 48/48 (100%) | 1 | 5.5 | 0.51 | 51746 | 0.0072 | 26.9 |
| one prompt, `anthropic/claude-haiku-5.5` | 5 | 9 (3/5) | 6 (3/5) | 1 | 14/29 (48%) | 18 | 7.4 | 0.65 | 1554 | 0.0004 | 27.1 |
| chat + web search, `anthropic/claude-haiku-5.5` | 5 | 15 (5/5) | 15 (5/5) | 2 | 23/26 (88%) | 17 | 18.2 | 0.54 | 41182 | 0.043 | 12.9 |
| chat + web search, `anthropic/claude-sonnet-5.5` | 5 | 9 (4/5) | 7 (4/5) | 3 | 25/29 (86%) | 17 | 11.9 | 0.61 | 73584 | 0.2576 | 63.5 |

Without `anthropic/claude-sonnet-5.5`, which answered fewer walks: over 8 walks.

| system | routes | too short on foot: legs (routes) | too short even by transit: legs (routes) | closed at visit | found in OSM, of specific places | vague items | walk km per route | mean pop | tokens per route | $ per route | s per route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent, `anthropic/claude-haiku-5.5` | 8 | 0 (0/8) | 0 (0/8) | 0 | 77/77 (100%) | 1 | 4.9 | 0.5 | 57119 | 0.0077 | 27.7 |
| one prompt, `anthropic/claude-haiku-5.5` | 8 | 13 (5/8) | 7 (4/8) | 3 | 25/49 (51%) | 23 | 6.8 | 0.67 | 1522 | 0.0004 | 22.9 |
| chat + web search, `anthropic/claude-haiku-5.5` | 8 | 21 (8/8) | 21 (8/8) | 2 | 34/39 (87%) | 31 | 13.2 | 0.58 | 45089 | 0.0441 | 12.4 |

## Chat stops not found in OpenStreetMap

Per route, the items of the chat answers that the lenient Nominatim lookup did not find, with the kind of item: a vague item is not counted against found in OSM. Each item's details are harness_detail in the JSON.

| query | model | not found (kind of item) |
|---|---|---|
| spb-constructivism | anthropic/claude-haiku-5.5 | Ленинградский ранний конструктивизм (переход) (transition); Завершение маршрута (endpoint) |
| spb-classic-kids | anthropic/claude-haiku-5.5 | Благовещенский или Тучков мост, Малая Нева (alternatives) |
| moscow-modernism | anthropic/claude-haiku-5.5 | Кофейня Surf Coffee x Lineup (place); Финал: Китай-город (endpoint) |
| kazan-history | anthropic/claude-haiku-5.5 | Переход к башне Сююмбике и обзор территории (transition); Музей истории Благовещенского собора или Музей Спасской башни (alternatives); Прогулка по улице Баумана (transition); Мечеть Марджани или Памятник Мусе Джалилю (alternatives); Набережная озера Кабан (запасной обед: ресторан «Татарская усадьба») (place); Завершение: возвращение к Кремлю или в гостиницу (endpoint) |
| kaliningrad-koenigsberg | anthropic/claude-haiku-5.5 | Кофе с выпечкой (Königsbäcker или Круассан-кафе) (alternatives); Закхаймские ворота и кофе-пауза (place); Свободное время, сувениры (activity) |
| nizhny-views | anthropic/claude-haiku-5.5 | Чкаловская лестница и смотровая площадка у памятника Чкалову (place); Обед/перерыв на Рождественской (activity); Переулки и дворы в стрит-арт-зоне (area); Рождественская церковь и кофе-пауза (place); Возврат на Большую Покровскую (transition); Сквер или Минина (ещё одна точка стрит-арта) (alternatives); Финиш на площади Минина и Пожарского (endpoint) |
| ekb-avantgarde | anthropic/claude-haiku-5.5 | Исторический сквер и набережная Исети (area); Обед / перекус в районе Ленина (кафе «Мечтатели») (activity) |
| paris-impressionism | anthropic/claude-haiku-5.5 | Lunch at a bakery near Saint-Germain (activity); Quai de la Bûcherie to the Latin Quarter bookstalls (transition); Old bookshops in the Latin Quarter (area); Wrap-up (endpoint) |
| spb-constructivism | anthropic/claude-sonnet-5.5 | Выход от Московского вокзала (endpoint); Двор Восстания, 1-3 (place); Пешком до Рубинштейна, 7 (transition); Пешком до Лиговского, 30а (transition); Дорога на Петроградскую сторону (transition); Дорога на такси (transition) |
| spb-classic-kids | anthropic/claude-sonnet-5.5 | Эрмитаж, Дворцовая площадь и Дворцовая набережная (area); Университетская набережная → Дворцовый мост (transition); Дворцовая площадь → Певческий мост через Мойку → Конюшенная площадь (transition); Конюшенная пл. → набережная канала Грибоедова (transition) |
| moscow-modernism | anthropic/claude-sonnet-5.5 | Старт у метро «Китай-город» (endpoint); Книжный «Библио-Глобус» (place); Прогулка у Калининского моста, возвращение к метро «Арбатская» (transition) |
| kazan-history | anthropic/claude-sonnet-5.5 | Спасская башня, вход в Кремль (endpoint); Башня Сююмбике и стены Кремля (place); Мечеть Кул Шариф и Музей исламской культуры (place); Апанаевская мечеть и прогулка по слободе (area) |
