// sokoban-solver: push-optimal Sokoban solver.
//
// Approach (written for this competition):
//   * The generated levels split into independent rooms: one 1-wide hallway
//     with 1-4 rooms off it, each room holding 3-4 boxes/goals behind a single
//     door. A box pushed out of a door into the hallway can never come back,
//     so every room is solved on its own and the push counts add up.
//   * Each room is solved by A* over *pushes* (one step of the search moves a
//     box one square; the player walks between pushes). A* over pushes with an
//     admissible heuristic returns a push-optimal solution, which is what the
//     score pays for: an instance is worth full points only at P* pushes.
//   * Heuristic: the cheapest one-to-one assignment of boxes to goals, where
//     the cost of one box to one goal is the shortest floor-only path between
//     them. Every push moves one box one square, so this never overestimates.
//   * Deadlock pruning, both safe (they never remove a solvable state):
//     dead squares (a box that can never be pulled back onto any goal) and
//     frozen 2x2 blocks of walls/boxes with a box off its goal.
//   * A state is the sorted box squares plus the player's reachable region
//     (named by its lowest square), so walking without pushing never makes a
//     new state.
//   * If A* exhausts its work budget on a room, plain breadth-first search
//     over pushes is retried (the generator only keeps rooms its own BFS
//     solves, so this is a solid fallback). If room splitting fails at all,
//     the whole level is searched the same way.
//   * Determinism: budgets are counted in expanded states, never in wall
//     time; no clock, no randomness, fixed direction and tie-break orders.
//     Two runs therefore write identical bytes.
//
// Usage: node main.ts, reading /input/<id>/level.txt and writing
// /output/<id>/solution.txt (MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR override).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expanded states), never the clock: the grader times out the
// whole run at 60s, so the total work must stay far inside that on 1 CPU.
const ASTAR_BUDGET = 300_000; // per search
const BFS_BUDGET = 250_000; // per fallback search
const WHOLE_BUDGET = 400_000; // whole-level fallback search
const GLOBAL_BUDGET = 2_000_000; // all searches in one run, all instances

let spent = 0; // expansions spent so far in this run

// Directions: dx, dy, walk letter, push letter. Fixed order = tie-breaks.
const DIRS: { dx: number; dy: number; walk: string; push: string }[] = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

type Board = {
  w: number;
  h: number;
  size: number;
  wall: Uint8Array; // 1 = wall (short rows and the outer frame count as walls)
  goals: number[];
  boxes: number[]; // sorted
  player: number;
};

function parse(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const i = y * w + x;
      const ch = row[x];
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, size: w * h, wall, goals, boxes, player };
}

// Squares from which a lone box can still be pushed onto some goal: pull
// backwards from the goals (a box at y can be pushed to x when the player can
// stand at y - d, i.e. both y and y - d are open floor).
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.size);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const d of DIRS) {
      const step = d.dy * b.w + d.dx;
      const y = x - step;
      if (y < 0 || y >= b.size) continue;
      if (!b.wall[y] && !b.wall[y - step] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Shortest floor-only distances from each goal to every square (box sliding
// one square costs one push; other boxes ignored, so this underestimates).
function goalDistances(b: Board): Int32Array[] {
  return b.goals.map((g) => {
    const dist = new Int32Array(b.size).fill(-1);
    const queue: number[] = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const d of DIRS) {
        const q = p + d.dy * b.w + d.dx;
        if (!b.wall[q] && dist[q] < 0) {
          dist[q] = dist[p]! + 1;
          queue.push(q);
        }
      }
    }
    return dist;
  });
}

// Cheapest assignment of boxes to distinct goals (n <= 4 here, so trying every
// permutation is cheap). Infinity when no assignment exists: dead state.
function assignmentHeuristic(boxes: number[], dists: Int32Array[]): number {
  const n = boxes.length;
  const used = new Array<boolean>(n).fill(false);
  let best = Infinity;
  const rec = (i: number, cost: number): void => {
    if (cost >= best) return;
    if (i === n) {
      best = cost;
      return;
    }
    for (let g = 0; g < n; g++) {
      if (used[g]) continue;
      const d = dists[g]![boxes[i]!]!;
      if (d < 0) continue;
      used[g] = true;
      rec(i + 1, cost + d);
      used[g] = false;
    }
  };
  rec(0, 0);
  return best;
}

// Sum over boxes of the nearest-goal distance (admissible; used only for the
// whole-level fallback where the box count is too big for assignment).
function nearestHeuristic(boxes: number[], dists: Int32Array[]): number {
  let total = 0;
  for (const box of boxes) {
    let m = Infinity;
    for (const d of dists) {
      const v = d[box]!;
      if (v >= 0 && v < m) m = v;
    }
    if (m === Infinity) return Infinity;
    total += m;
  }
  return total;
}

// Flood-fill the player's reachable region around `occupied` boxes; returns
// the region's lowest square and marks the region in `seen` with `stamp`.
function flood(b: Board, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const d of DIRS) {
      const q = p + d.dy * b.w + d.dx;
      if (!b.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed to `t` is stuck in a 2x2 block of walls and
// boxes holding a box off its goal.
function frozen2x2(b: Board, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= b.size) continue;
    if (!(b.wall[a] || occupied[a]) || !(b.wall[a + 1] || occupied[a + 1])) continue;
    if (!(b.wall[c] || occupied[c]) || !(b.wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[a + 1] && !goal[a + 1]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes (BFS, fixed
// direction order, so ties break the same way every run).
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(b.size).fill(-1);
  const how = new Array<string>(b.size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * b.w + d.dx;
      if (!b.wall[q] && !occupied[q] && prev[q] === -1) {
        prev[q] = p;
        how[q] = d.walk;
        queue.push(q);
      }
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

// State key: region-low + sorted boxes. Numeric while it fits exactly in a
// double (rooms: <= 4 boxes), string otherwise.
function stateKey(boxes: number[], low: number, size: number): number | string {
  if (boxes.length <= 4) {
    let k = low;
    for (const box of boxes) k = k * size + box;
    return k;
  }
  return `${low}|${boxes.join(',')}`;
}

type SearchResult = { pushes: number; moves: string; endPlayer: number; expanded: number } | null;

// Push-space search over one board. `astar` picks A* (assignment/nearest
// heuristic) or plain BFS (first solved state is optimal). Returns the
// push-optimal solution within `budget` expansions, else null.
function searchPushes(b: Board, budget: number, astar: boolean): SearchResult {
  const goal = new Uint8Array(b.size);
  for (const g of b.goals) goal[g] = 1;
  const live = liveSquares(b);
  const dists = goalDistances(b);
  const useAssignment = b.boxes.length === b.goals.length && b.boxes.length <= 4;
  const heuristic = (boxes: number[]): number =>
    useAssignment ? assignmentHeuristic(boxes, dists) : nearestHeuristic(boxes, dists);
  if (b.boxes.some((x) => !live[x])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((x) => goal[x]);

  const occupied = new Uint8Array(b.size);
  const seen = new Int32Array(b.size);
  const mark = new Int32Array(b.size);
  const stack = new Int32Array(b.size);
  let stamp = 0;

  const stateBoxes: number[][] = [b.boxes.slice()];
  const statePlayer: number[] = [b.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const stateG: number[] = [0];
  const best = new Map<number | string, number>();
  for (const x of b.boxes) occupied[x] = 1;
  const startLow = flood(b, occupied, b.player, seen, ++stamp, stack);
  for (const x of b.boxes) occupied[x] = 0;
  const stateKeys: (number | string)[] = [stateKey(b.boxes, startLow, b.size)];
  best.set(stateKeys[0]!, 0);

  // Open list: a heap on (g + h, h, insertion order) for A*, a FIFO queue for BFS.
  const heap: { f: number; h: number; seq: number; s: number }[] = [];
  const queue: number[] = [];
  let seq = 0;
  const h0 = astar ? heuristic(b.boxes) : 0;
  if (h0 === Infinity) return null;
  if (astar) heap.push({ f: h0, h: h0, seq: seq++, s: 0 });
  else queue.push(0);
  const heapLess = (a: { f: number; h: number; seq: number }, c: { f: number; h: number; seq: number }): boolean =>
    a.f < c.f || (a.f === c.f && (a.h < c.h || (a.h === c.h && a.seq < c.seq)));
  const heapPush = (e: { f: number; h: number; seq: number; s: number }): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): { f: number; h: number; seq: number; s: number } => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  let found = solved(b.boxes) ? 0 : -1;
  let expanded = 0;
  let head = 0;
  while ((astar ? heap.length > 0 : head < queue.length) && found < 0) {
    const s = astar ? heapPop().s : queue[head++]!;
    const g = stateG[s]!;
    // Stale entry: a cheaper route to this same state was found since.
    if (best.get(stateKeys[s]!) !== g) continue;
    const boxes = stateBoxes[s]!;
    if (expanded >= budget || spent >= GLOBAL_BUDGET) return null;
    for (const x of boxes) occupied[x] = 1;
    const here = ++stamp;
    flood(b, occupied, statePlayer[s]!, seen, here, stack);
    expanded++;
    spent++;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di]!;
        const step = d.dy * b.w + d.dx;
        const to = box + step;
        const behind = box - step;
        if (seen[behind] !== here) continue; // player cannot reach the push square
        if (b.wall[to] || occupied[to] || !live[to]) continue;
        occupied[box] = 0;
        occupied[to] = 1;
        const frozen = frozen2x2(b, occupied, goal, to);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = to;
          next.sort((a, b2) => a - b2);
          key = stateKey(next, flood(b, occupied, box, mark, ++stamp, stack), b.size);
        }
        occupied[to] = 0;
        occupied[box] = 1;
        if (frozen || next === null) continue;
        // A cheaper route to this state supersedes the stored one; a route
        // no cheaper than the stored one is dropped. (First generation is
        // not first-best under A*, only first pop is.)
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= g + 1) continue;
        const h = astar ? heuristic(next) : 0;
        if (h === Infinity) continue;
        if (stateBoxes.length >= budget) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        best.set(key, g + 1);
        const t = stateBoxes.length;
        stateBoxes.push(next);
        stateKeys.push(key);
        statePlayer.push(box);
        parent.push(s);
        pushFrom.push(box);
        pushTo.push(to);
        pushDir.push(di);
        stateG.push(g + 1);
        if (solved(next)) {
          found = t;
          break;
        }
        if (astar) heapPush({ f: g + 1 + h, h, seq: seq++, s: t });
        else queue.push(t);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;
  // Rebuild the moves: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = b.player;
  let boxes = b.boxes.slice();
  for (const s of path) {
    const nextBoxes = stateBoxes[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const di = pushDir[s]!;
    const d = DIRS[di]!;
    const step = d.dy * b.w + d.dx;
    for (const x of boxes) occupied[x] = 1;
    const walk = walkTo(b, occupied, player, from - step);
    for (const x of boxes) occupied[x] = 0;
    if (walk === null) return null;
    moves += walk + d.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves, endPlayer: player, expanded };
}

// Split a level into rooms: cut the dead squares of the player's row (the
// hallway: a box there reaches no goal) out of the floor; each remaining part
// holding boxes is a room. Null when the level is no set of rooms.
function roomsOf(b: Board): { boxes: number[]; goals: number[] }[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.size);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  if (b.goals.some((g) => cut[g])) return null;
  const part = new Int32Array(b.size).fill(-1);
  const rooms: { boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < b.size; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const d of DIRS) {
        const q = squares[i]! + d.dy * b.w + d.dx;
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const x of b.boxes) rooms[part[x]!]!.boxes.push(x);
  for (const g of b.goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

// Solve one level: room by room (A*, BFS fallback per room), whole-level
// search if the level is no set of rooms or a room fails both searches.
function solveLevel(b: Board): string {
  const rooms = roomsOf(b);
  if (rooms !== null) {
    let player = b.player;
    let moves = '';
    let ok = true;
    const done: boolean[] = rooms.map(() => false);
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      // Other rooms' boxes stand as walls (on their goals once solved).
      const wall = b.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        for (const x of done[oj] ? rooms[oj]!.goals : rooms[oj]!.boxes) wall[x] = 1;
      }
      const sub: Board = {
        w: b.w,
        h: b.h,
        size: b.size,
        wall,
        goals: room.goals.slice().sort((x, y) => x - y),
        boxes: room.boxes.slice().sort((x, y) => x - y),
        player,
      };
      let res = searchPushes(sub, ASTAR_BUDGET, true);
      if (res === null) res = searchPushes(sub, BFS_BUDGET, false);
      if (res === null) {
        ok = false;
        break;
      }
      moves += res.moves;
      player = res.endPlayer;
      done[ri] = true;
    }
    if (ok) return moves;
  }
  const whole = searchPushes(b, WHOLE_BUDGET, true) ?? searchPushes(b, WHOLE_BUDGET, false);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(parse(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
