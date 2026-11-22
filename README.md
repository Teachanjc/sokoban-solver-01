# Sokoban Solver — make.fun `sokoban-solver-01-run-20261121`

Push-optimal Sokoban solver in a single `main.ts` (Node 22, no dependencies,
no network, no clock). Reads `/input/<id>/level.txt`, writes
`/output/<id>/solution.txt` for all 5 instances in one run.

## Idea

The generated levels split into independent rooms: one 1-wide hallway with
1–4 rooms off it, each holding 3–4 boxes/goals behind a single door. A box
pushed out of a door into the hallway can never come back, so each room is
solved on its own and the push counts add up to the level optimum P*.

- **Search:** A\* over *pushes* per room (one search step moves a box one
  square; the player walks between pushes). A\* over pushes with an
  admissible heuristic returns a push-optimal solution — and the score pays
  for pushes only: full points need exactly P\* pushes, one extra push
  already costs about half the instance.
- **Heuristic:** cheapest one-to-one assignment of boxes to goals, where one
  box-to-goal cost is the shortest floor-only path between them. Every push
  moves one box one square, so it never overestimates.
- **Deadlock pruning** (both safe — they never remove a solvable state):
  dead squares (reverse pull-reachability from the goals) and frozen 2×2
  blocks of walls/boxes holding a box off its goal.
- **State:** sorted box squares + the player's reachable region (named by
  its lowest square), so walking without pushing never makes a new state.
- **Fallbacks:** if A\* exhausts its work budget on a room, plain
  breadth-first search over pushes is retried; if room splitting fails at
  all, the whole level is searched the same way.
- **Determinism:** budgets are counted in expanded states, never in wall
  time; no clock, no randomness, fixed direction and tie-break orders. Two
  runs write identical bytes.

## Results (local)

| Set | Score | Per-instance pushes (P\*) | Runtime |
| --- | --- | --- | --- |
| Practice (kit) | **1,000,000** | 13, 16, 29, 47, 60 | ~1.0 s |
| Fresh set s1 | **1,000,000** | 12, 16, 29, 49, 58 | ~1.6 s |
| Fresh set s2 | **1,000,000** | 12, 15, 30, 46, 62 | ~0.5 s |
| Fresh set s3 | **1,000,000** | 14, 14, 34, 44, 56 | ~0.6 s |
| Fresh set s4 | **1,000,000** | 12, 14, 32, 42, 58 | ~0.8 s |
| Fresh set s5 | **1,000,000** | 13, 14, 28, 46, 58 | ~0.5 s |

House baseline (plain-astar 1.0.0) on the practice set: **200,000** (only
i001 solved). Fresh sets were drawn with the kit's own generator from new
seeds and scored with the kit's own scorer. `grade-local --driver process`
on the practice set: score 1,000,000, passed, entry size 18,705 bytes.
Two runs on the practice set are byte-identical.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=path/to/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Files

- `main.ts` — the whole solver (the entry point).
- `LICENSE` — MIT.
