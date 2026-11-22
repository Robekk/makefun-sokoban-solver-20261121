// Sokoban solver: push-space A* over independent rooms, with a whole-level
// fallback. Written from the written rules and the practice levels only.
//
// A room is solved on its own: the search state is the set of box squares
// plus the region the player can reach without pushing, and each step of
// the search is one push. The guiding estimate is an optimal pairing of
// boxes to goals using single-box push distances, which never overstates
// the pushes still needed. Dead squares, frozen corners, trapped wall
// runs and immovable two-by-two blocks are cut before they can waste the
// search. Work is bounded by counting expanded states, never by the clock,
// so two runs on the same input always agree byte for byte.

import * as fs from "node:fs";
import * as path from "node:path";

const UP = 0;
const DOWN = 1;
const LEFT = 2;
const RIGHT = 3;
const ROW_STEP: number[] = [-1, 1, 0, 0];
const COL_STEP: number[] = [0, 0, -1, 1];
const STEP_LETTER: string[] = ["u", "d", "l", "r"];

type PushStep = { from: number; dir: number };
type Puzzle = {
  width: number;
  height: number;
  boxCells: number[];
  playerCells: number[];
  goals: number[];
  startBoxes: number[];
  startPlayer: number;
};
type RoomPlan = {
  puzzle: Puzzle;
  entry: number;
  pushes: PushStep[];
};

function cellAt(row: number, col: number, width: number): number {
  return row * width + col;
}
function rowOf(cell: number, width: number): number {
  return Math.floor(cell / width);
}
function colOf(cell: number, width: number): number {
  return cell - Math.floor(cell / width) * width;
}
function neighbour(cell: number, dir: number, width: number, height: number): number {
  const r = rowOf(cell, width) + ROW_STEP[dir];
  const c = colOf(cell, width) + COL_STEP[dir];
  if (r < 0 || r >= height || c < 0 || c >= width) return -1;
  return cellAt(r, c, width);
}
function sortedCopy(values: number[]): number[] {
  const out = values.slice();
  out.sort((a, b) => a - b);
  return out;
}
function sameSorted(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

type Level = {
  width: number;
  height: number;
  walls: boolean[];
  goals: number[];
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split(/\r\n|\n|\r/);
  while (rows.length > 0 && rows[rows.length - 1] === "") rows.pop();
  const height = rows.length;
  let width = 0;
  for (const row of rows) if (row.length > width) width = row.length;
  if (width === 0) width = 1;
  const walls: boolean[] = new Array(width * height).fill(true);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < height; r++) {
    const row = rows[r];
    for (let c = 0; c < width; c++) {
      const ch = c < row.length ? row[c] : "#";
      const cell = cellAt(r, c, width);
      if (ch === "#") {
        walls[cell] = true;
      } else {
        walls[cell] = false;
        if (ch === "." || ch === "*" || ch === "+") goals.push(cell);
        if (ch === "$" || ch === "*") boxes.push(cell);
        if (ch === "@" || ch === "+") player = cell;
      }
    }
  }
  return { width, height, walls, goals: sortedCopy(goals), boxes: sortedCopy(boxes), player };
}

// Split the level at the player's row: with the hallway row sealed off,
// each remaining floor component is one room, reached through exactly
// one door square. Returns null when the shape does not match that
// pattern, in which case the caller solves the whole level at once.
function splitRooms(level: Level): Puzzle[] | null {
  if (level.player < 0) return null;
  const hallRow = rowOf(level.player, level.width);
  const seen: boolean[] = new Array(level.width * level.height).fill(false);
  const rooms: Puzzle[] = [];
  for (let start = 0; start < level.width * level.height; start++) {
    if (level.walls[start] || seen[start]) continue;
    if (rowOf(start, level.width) === hallRow) continue;
    const stack: number[] = [start];
    seen[start] = true;
    const cells: number[] = [];
    while (stack.length > 0) {
      const cur = stack.pop() as number;
      cells.push(cur);
      for (let d = 0; d < 4; d++) {
        const nb = neighbour(cur, d, level.width, level.height);
        if (nb < 0 || level.walls[nb] || seen[nb]) continue;
        if (rowOf(nb, level.width) === hallRow) continue;
        seen[nb] = true;
        stack.push(nb);
      }
    }
    const roomBoxes = level.boxes.filter((b) => cells.indexOf(b) >= 0);
    const roomGoals = level.goals.filter((g) => cells.indexOf(g) >= 0);
    if (roomBoxes.length === 0 && roomGoals.length === 0) continue;
    if (roomBoxes.length === 0 || roomBoxes.length !== roomGoals.length) return null;
    const doors = cells.filter((cell) => {
      const nb = neighbour(cell, rowOf(cell, level.width) < hallRow ? DOWN : UP, level.width, level.height);
      return nb >= 0 && rowOf(nb, level.width) === hallRow && !level.walls[nb];
    });
    if (doors.length !== 1) return null;
    const entry = neighbour(doors[0], rowOf(doors[0], level.width) < hallRow ? DOWN : UP, level.width, level.height);
    rooms.push({
      width: level.width,
      height: level.height,
      boxCells: sortedCopy(cells),
      playerCells: sortedCopy(cells.concat([entry])),
      goals: roomGoals,
      startBoxes: roomBoxes,
      startPlayer: entry,
    });
  }
  if (rooms.length === 0) return null;
  rooms.sort((a, b) => {
    const da = a.playerCells[a.playerCells.length - 1];
    const db = b.playerCells[b.playerCells.length - 1];
    return a.startPlayer - b.startPlayer || da - db;
  });
  return rooms;
}

function wholePuzzle(level: Level): Puzzle {
  const floor: number[] = [];
  for (let i = 0; i < level.width * level.height; i++) if (!level.walls[i]) floor.push(i);
  return {
    width: level.width,
    height: level.height,
    boxCells: floor,
    playerCells: floor,
    goals: level.goals,
    startBoxes: level.boxes,
    startPlayer: level.player,
  };
}

// Shortest walk for the player between two squares, stepping only onto
// squares no box currently occupies. Fixed neighbour order keeps the
// walk identical on every run.
function walkPath(
  from: number,
  to: number,
  blocked: Set<number>,
  walls: boolean[],
  width: number,
  height: number,
): number[] | null {
  if (from === to) return [];
  const prev: number[] = new Array(width * height).fill(-2);
  const prevDir: number[] = new Array(width * height).fill(-1);
  prev[from] = -1;
  const queue: number[] = [from];
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    for (let d = 0; d < 4; d++) {
      const nb = neighbour(cur, d, width, height);
      if (nb < 0 || walls[nb] || blocked.has(nb) || prev[nb] !== -2) continue;
      prev[nb] = cur;
      prevDir[nb] = d;
      if (nb === to) {
        const dirs: number[] = [];
        let at = nb;
        while (at !== from) {
          dirs.push(prevDir[at]);
          at = prev[at];
        }
        dirs.reverse();
        return dirs;
      }
      queue.push(nb);
    }
  }
  return null;
}

// Replay a candidate answer on the level: every step must be legal and
// the replay must end with all boxes on goals. Returns the push count.
function replayCount(text: string, level: Level): number {
  const boxes = new Set<number>(level.boxes);
  let player = level.player;
  let pushes = 0;
  for (const ch of text) {
    const dir = STEP_LETTER.indexOf(ch.toLowerCase());
    if (dir < 0) return -1;
    const next = neighbour(player, dir, level.width, level.height);
    if (next < 0 || level.walls[next]) return -1;
    if (boxes.has(next)) {
      const beyond = neighbour(next, dir, level.width, level.height);
      if (beyond < 0 || level.walls[beyond] || boxes.has(beyond)) return -1;
      boxes.delete(next);
      boxes.add(beyond);
      pushes++;
    }
    player = next;
  }
  for (const g of level.goals) if (!boxes.has(g)) return -1;
  if (boxes.size !== level.goals.length) return -1;
  return pushes;
}

function allOrders(count: number): number[][] {
  const base: number[] = [];
  for (let i = 0; i < count; i++) base.push(i);
  const out: number[][] = [];
  const used: boolean[] = new Array(count).fill(false);
  const cur: number[] = [];
  const build = (): void => {
    if (cur.length === count) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < count; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      build();
      cur.pop();
      used[i] = false;
    }
  };
  build();
  return out;
}

// Stitch room push plans into one answer: walk to each room's hallway
// entry, then walk to each pushing square and push. Tries room orders
// until the stitched answer replays cleanly.
function assemble(
  level: Level,
  plans: RoomPlan[],
  entryOf: (plan: RoomPlan) => number,
): string | null {
  const orders = allOrders(plans.length);
  for (const order of orders) {
    const boxes = new Set<number>(level.boxes);
    let player = level.player;
    const letters: string[] = [];
    let ok = true;
    for (const idx of order) {
      const plan = plans[idx];
      const walk = walkPath(player, entryOf(plan), boxes, level.walls, level.width, level.height);
      if (walk === null) {
        ok = false;
        break;
      }
      for (const d of walk) {
        letters.push(STEP_LETTER[d]);
        player = neighbour(player, d, level.width, level.height);
      }
      for (const push of plan.pushes) {
        const pusher = neighbour(push.from, push.dir ^ 1, level.width, level.height);
        const step = walkPath(player, pusher, boxes, level.walls, level.width, level.height);
        if (step === null) {
          ok = false;
          break;
        }
        for (const d of step) {
          letters.push(STEP_LETTER[d]);
          player = neighbour(player, d, level.width, level.height);
        }
        const dest = neighbour(push.from, push.dir, level.width, level.height);
        if (dest < 0 || !boxes.has(push.from) || boxes.has(dest) || level.walls[dest]) {
          ok = false;
          break;
        }
        letters.push(STEP_LETTER[push.dir]);
        boxes.delete(push.from);
        boxes.add(dest);
        player = push.from;
      }
      if (!ok) break;
    }
    if (!ok) continue;
    const answer = letters.join("");
    if (replayCount(answer, level) >= 0) return answer;
  }
  return null;
}

const TOTAL_WORK = 2000000;

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.length === 0) return "";
  const rooms = splitRooms(level);
  if (rooms !== null) {
    let spent = 0;
    const plans: RoomPlan[] = [];
    let failed = false;
    for (let i = 0; i < rooms.length; i++) {
      const roomsLeft = rooms.length - i;
      const allowance = Math.max(20000, Math.floor((TOTAL_WORK - spent) / roomsLeft));
      const result = solvePuzzle(rooms[i], allowance);
      if (result === null) {
        failed = true;
        break;
      }
      spent += result.expansions;
      plans.push({ puzzle: rooms[i], entry: rooms[i].startPlayer, pushes: result.pushes });
    }
    if (!failed) {
      const answer = assemble(level, plans, (plan) => plan.entry);
      if (answer !== null) return answer;
    }
  }
  const whole = wholePuzzle(level);
  const result = solvePuzzle(whole, TOTAL_WORK);
  if (result !== null) {
    const plan: RoomPlan = { puzzle: whole, entry: level.player, pushes: result.pushes };
    const answer = assemble(level, [plan], (p) => p.entry);
    if (answer !== null) return answer;
  }
  return "";
}

function main(): void {
  const inputDir = process.env["MAKEFUN_INPUT_DIR"] || "/input";
  const outputDir = process.env["MAKEFUN_OUTPUT_DIR"] || "/output";
  let entries: string[] = [];
  try {
    entries = fs
      .readdirSync(inputDir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
  } catch {
    return;
  }
  for (const name of entries) {
    const inFile = path.join(inputDir, name, "level.txt");
    const outDir = path.join(outputDir, name);
    let answer = "";
    try {
      const text = fs.readFileSync(inFile, "utf8");
      answer = solveLevel(text);
    } catch {
      answer = "";
    }
    try {
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, "solution.txt"), answer, "utf8");
    } catch {
      // No output directory available; nothing further can be done here.
    }
  }
}

const INF_COST = 1000000000;
const BIG_COST = 10000000;

// Cheapest pairing of boxes to goals for one cost matrix, by the
// Hungarian method. Returns INF_COST when no pairing uses only finite
// costs. Deterministic: the scan order is fixed throughout.
function cheapestPairing(costs: number[][]): number {
  const n = costs.length;
  if (n === 0) return 0;
  const u: number[] = new Array(n + 1).fill(0);
  const v: number[] = new Array(n + 1).fill(0);
  const match: number[] = new Array(n + 1).fill(0);
  const way: number[] = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    match[0] = i;
    let col = 0;
    const best: number[] = new Array(n + 1).fill(Infinity);
    const seen: boolean[] = new Array(n + 1).fill(false);
    do {
      seen[col] = true;
      const row = match[col];
      let delta = Infinity;
      let nextCol = 0;
      for (let j = 1; j <= n; j++) {
        if (seen[j]) continue;
        const raw = costs[row - 1][j - 1];
        const cur = (raw >= INF_COST ? BIG_COST : raw) - u[row] - v[j];
        if (cur < best[j]) {
          best[j] = cur;
          way[j] = col;
        }
        if (best[j] < delta) {
          delta = best[j];
          nextCol = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (seen[j]) {
          u[match[j]] += delta;
          v[j] -= delta;
        } else {
          best[j] -= delta;
        }
      }
      col = nextCol;
    } while (match[col] !== 0);
    do {
      const prev = way[col];
      match[col] = match[prev];
      col = prev;
    } while (col !== 0);
  }
  let total = 0;
  for (let j = 1; j <= n; j++) {
    if (match[j] <= 0) return INF_COST;
    const c = costs[match[j] - 1][j - 1];
    if (c >= INF_COST) return INF_COST;
    total += c;
  }
  return total;
}

// Single-box push distances: for every box square and every goal, the
// fewest pushes that move one lone box there, with the player free to
// start anywhere and to walk anywhere the lone box does not occupy.
// Each distance is a lower bound on the real cost, so the pairing built
// on them never overestimates. Computed with a zero-one breadth search
// per starting square, over (box square, player square) pairs.
function pushDistanceTable(puzzle: Puzzle): number[][] {
  const boxSet = new Set<number>(puzzle.boxCells);
  const playerSet = new Set<number>(puzzle.playerCells);
  const boxIndex = new Map<number, number>();
  puzzle.boxCells.forEach((cell, i) => boxIndex.set(cell, i));
  const playerIndex = new Map<number, number>();
  puzzle.playerCells.forEach((cell, i) => playerIndex.set(cell, i));
  const goalIndex: number[] = [];
  for (const g of puzzle.goals) goalIndex.push(boxIndex.get(g) as number);
  const nb = puzzle.boxCells.length;
  const np = puzzle.playerCells.length;
  const table: number[][] = [];
  for (let start = 0; start < nb; start++) {
    const dist: number[] = new Array(nb * np).fill(INF_COST);
    const queue: number[] = [];
    const startCell = puzzle.boxCells[start];
    for (let pi = 0; pi < np; pi++) {
      if (puzzle.playerCells[pi] === startCell) continue;
      const state = start * np + pi;
      dist[state] = 0;
      queue.push(state);
    }
    let head = 0;
    while (head < queue.length) {
      const state = queue[head++];
      const boxAt = puzzle.boxCells[Math.floor(state / np)];
      const playerAt = puzzle.playerCells[state % np];
      const cur = dist[state];
      for (let d = 0; d < 4; d++) {
        const step = neighbour(playerAt, d, puzzle.width, puzzle.height);
        if (step >= 0 && playerSet.has(step) && step !== boxAt) {
          const ns = Math.floor(state / np) * np + (playerIndex.get(step) as number);
          if (dist[ns] > cur) {
            dist[ns] = cur;
            queue.splice(head, 0, ns);
          }
        }
        const facing = neighbour(playerAt, d, puzzle.width, puzzle.height);
        if (facing === boxAt) {
          const dest = neighbour(boxAt, d, puzzle.width, puzzle.height);
          if (dest >= 0 && boxSet.has(dest)) {
            const ns = (boxIndex.get(dest) as number) * np + (playerIndex.get(boxAt) as number);
            if (dist[ns] > cur + 1) {
              dist[ns] = cur + 1;
              queue.push(ns);
            }
          }
        }
      }
    }
    const row: number[] = [];
    for (const gi of goalIndex) {
      let best = INF_COST;
      for (let pi = 0; pi < np; pi++) {
        const d = dist[gi * np + pi];
        if (d < best) best = d;
      }
      row.push(best);
    }
    table.push(row);
  }
  return table;
}

type PuzzleContext = {
  puzzle: Puzzle;
  boxSet: Set<number>;
  playerSet: Set<number>;
  goalSet: Set<number>;
  boxIndex: Map<number, number>;
  table: number[][];
  deadCache: Map<string, boolean>;
  heurCache: Map<string, number>;
};

function makeContext(puzzle: Puzzle): PuzzleContext {
  return {
    puzzle,
    boxSet: new Set<number>(puzzle.boxCells),
    playerSet: new Set<number>(puzzle.playerCells),
    goalSet: new Set<number>(puzzle.goals),
    boxIndex: new Map<number, number>(puzzle.boxCells.map((cell, i) => [cell, i] as [number, number])),
    table: pushDistanceTable(puzzle),
    deadCache: new Map<string, boolean>(),
    heurCache: new Map<string, number>(),
  };
}

function heuristicValue(ctx: PuzzleContext, boxes: number[]): number {
  const key = boxes.join(",");
  const cached = ctx.heurCache.get(key);
  if (cached !== undefined) return cached;
  const costs: number[][] = [];
  for (const b of boxes) {
    costs.push(ctx.table[ctx.boxIndex.get(b) as number].slice());
  }
  const value = cheapestPairing(costs);
  ctx.heurCache.set(key, value);
  return value;
}

// True deadlocks only: a state is cut when some box can provably never
// reach a goal. Checks run on box squares alone, so results are cached
// per box arrangement.
function isDeadState(ctx: PuzzleContext, boxes: number[]): boolean {
  const key = boxes.join(",");
  const cached = ctx.deadCache.get(key);
  if (cached !== undefined) return cached;
  let dead = false;
  const boxSet = new Set<number>(boxes);
  const goalSet = ctx.goalSet;
  // Squares no pull sequence can ever leave, via the distance table.
  for (const b of boxes) {
    if (goalSet.has(b)) continue;
    const row = ctx.table[ctx.boxIndex.get(b) as number];
    let reachable = false;
    for (const d of row) if (d < INF_COST) reachable = true;
    if (!reachable) {
      dead = true;
      break;
    }
  }
  // Immovable two-by-two blocks of walls and boxes.
  if (!dead) {
    const blocked = (cell: number): boolean => {
      if (cell < 0) return true;
      if (boxSet.has(cell)) return true;
      return !ctx.boxSet.has(cell) && !ctx.playerSet.has(cell);
    };
    for (let r = 0; r + 1 < ctx.puzzle.height && !dead; r++) {
      for (let c = 0; c + 1 < ctx.puzzle.width; c++) {
        const cells = [
          cellAt(r, c, ctx.puzzle.width),
          cellAt(r, c + 1, ctx.puzzle.width),
          cellAt(r + 1, c, ctx.puzzle.width),
          cellAt(r + 1, c + 1, ctx.puzzle.width),
        ];
        if (cells.every(blocked)) {
          for (const cell of cells) {
            if (boxSet.has(cell) && !goalSet.has(cell)) {
              dead = true;
              break;
            }
          }
        }
      }
    }
  }
  // Corners that tighten as boxes freeze, iterated to a fixed point.
  let frozen = new Set<number>();
  if (!dead) {
    for (;;) {
      // A square blocks a push destination when a box can never occupy
      // it (a wall, the hallway entry, or a frozen box), and blocks the
      // player's footing when the player can never stand there (a wall,
      // outside the room, or a frozen box). The two roles differ exactly
      // at the door: the entry square holds the player but never a box.
      const destBlocked = (cell: number): boolean => {
        if (cell < 0) return true;
        return !ctx.boxSet.has(cell) || frozen.has(cell);
      };
      const standBlocked = (cell: number): boolean => {
        if (cell < 0) return true;
        return !ctx.playerSet.has(cell) || frozen.has(cell);
      };
      let grew = false;
      for (const b of boxes) {
        if (frozen.has(b)) continue;
        const canUp =
          !destBlocked(neighbour(b, UP, ctx.puzzle.width, ctx.puzzle.height)) &&
          !standBlocked(neighbour(b, DOWN, ctx.puzzle.width, ctx.puzzle.height));
        const canDown =
          !destBlocked(neighbour(b, DOWN, ctx.puzzle.width, ctx.puzzle.height)) &&
          !standBlocked(neighbour(b, UP, ctx.puzzle.width, ctx.puzzle.height));
        const canLeft =
          !destBlocked(neighbour(b, LEFT, ctx.puzzle.width, ctx.puzzle.height)) &&
          !standBlocked(neighbour(b, RIGHT, ctx.puzzle.width, ctx.puzzle.height));
        const canRight =
          !destBlocked(neighbour(b, RIGHT, ctx.puzzle.width, ctx.puzzle.height)) &&
          !standBlocked(neighbour(b, LEFT, ctx.puzzle.width, ctx.puzzle.height));
        if (!canUp && !canDown && !canLeft && !canRight) {
          frozen.add(b);
          grew = true;
        }
      }
      if (!grew) break;
    }
    for (const b of frozen) {
      if (!goalSet.has(b)) {
        dead = true;
        break;
      }
    }
  }
  // Wall runs: a box in a run of squares hugging a wall, with no goal
  // in the run and both ends sealed by wall or frozen boxes, can never
  // leave the run.
  if (!dead) {
    const destSealed = (cell: number): boolean => {
      if (cell < 0) return true;
      return !ctx.boxSet.has(cell) || frozen.has(cell);
    };
    const standSealed = (cell: number): boolean => {
      if (cell < 0) return true;
      return !ctx.playerSet.has(cell) || frozen.has(cell);
    };
    const scanRuns = (side: number, along: number): boolean => {
      const w = ctx.puzzle.width;
      const h = ctx.puzzle.height;
      const lines = along === RIGHT ? h : w;
      const len = along === RIGHT ? w : h;
      for (let line = 0; line < lines; line++) {
        let runStart = -1;
        const flush = (end: number): boolean => {
          if (runStart < 0) return false;
          const before =
            runStart === 0
              ? -1
              : along === RIGHT
                ? cellAt(line, runStart - 1, w)
                : cellAt(runStart - 1, line, w);
          const after =
            end === len
              ? -1
              : along === RIGHT
                ? cellAt(line, end, w)
                : cellAt(end, line, w);
          if (!destSealed(before) || !destSealed(after)) return false;
          let hasBox = false;
          let hasGoal = false;
          for (let k = runStart; k < end; k++) {
            const cell = along === RIGHT ? cellAt(line, k, w) : cellAt(k, line, w);
            if (boxSet.has(cell)) hasBox = true;
            if (goalSet.has(cell)) hasGoal = true;
          }
          return hasBox && !hasGoal;
        };
        for (let k = 0; k <= len; k++) {
          let inRun = false;
          if (k < len) {
            const cell = along === RIGHT ? cellAt(line, k, w) : cellAt(k, line, w);
            const sideCell = neighbour(cell, side, w, h);
            inRun = ctx.boxSet.has(cell) && !frozen.has(cell) && standSealed(sideCell);
          }
          if (inRun && runStart < 0) runStart = k;
          if (!inRun && runStart >= 0) {
            if (flush(k)) return true;
            runStart = -1;
          }
        }
      }
      return false;
    };
    if (scanRuns(UP, RIGHT) || scanRuns(DOWN, RIGHT) || scanRuns(LEFT, DOWN) || scanRuns(RIGHT, DOWN)) {
      dead = true;
    }
  }
  ctx.deadCache.set(key, dead);
  return dead;
}

// Flood fill over squares the player can reach without pushing.
function reachableRegion(ctx: PuzzleContext, from: number, boxes: Set<number>): Set<number> {
  const seen = new Set<number>([from]);
  const queue: number[] = [from];
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    for (let d = 0; d < 4; d++) {
      const nb = neighbour(cur, d, ctx.puzzle.width, ctx.puzzle.height);
      if (nb < 0 || !ctx.playerSet.has(nb) || boxes.has(nb) || seen.has(nb)) continue;
      seen.add(nb);
      queue.push(nb);
    }
  }
  return seen;
}

type SearchNode = {
  boxes: number[];
  player: number;
  g: number;
  f: number;
  h: number;
  parent: number;
  pushFrom: number;
  pushDir: number;
  key: string;
  seq: number;
};

// A* over pushes, with a transposition table keyed by boxes plus the
// player's reachable region. The heap order is fully determined, so the
// search (and its answer) never varies between runs.
function solvePuzzle(
  puzzle: Puzzle,
  maxExpansions: number,
): { pushes: PushStep[]; expansions: number } | null {
  const ctx = makeContext(puzzle);
  if (puzzle.startBoxes.length === 0) return { pushes: [], expansions: 0 };
  if (puzzle.startBoxes.length !== puzzle.goals.length) return null;
  const startBoxes = sortedCopy(puzzle.startBoxes);
  if (isDeadState(ctx, startBoxes)) return null;
  const startH = heuristicValue(ctx, startBoxes);
  if (startH >= INF_COST) return null;

  const nodes: SearchNode[] = [];
  const heap: number[] = [];
  const better = (a: number, b: number): boolean => {
    const x = nodes[a];
    const y = nodes[b];
    if (x.f !== y.f) return x.f < y.f;
    if (x.h !== y.h) return x.h < y.h;
    if (x.key !== y.key) return x.key < y.key;
    return x.seq < y.seq;
  };
  const heapPush = (id: number): void => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (!better(heap[i], heap[parent])) break;
      const tmp = heap[i];
      heap[i] = heap[parent];
      heap[parent] = tmp;
      i = parent;
    }
  };
  const heapPop = (): number => {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const left = i * 2 + 1;
        const right = left + 1;
        let smallest = i;
        if (left < heap.length && better(heap[left], heap[smallest])) smallest = left;
        if (right < heap.length && better(heap[right], heap[smallest])) smallest = right;
        if (smallest === i) break;
        const tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        i = smallest;
      }
    }
    return top;
  };

  const regionKey = (boxes: number[], player: number): string => {
    const region = reachableRegion(ctx, player, new Set<number>(boxes));
    let low = Infinity;
    for (const cell of region) if (cell < low) low = cell;
    return boxes.join(",") + "|" + low;
  };

  const visited = new Map<string, number>();
  const startKey = regionKey(startBoxes, puzzle.startPlayer);
  nodes.push({
    boxes: startBoxes,
    player: puzzle.startPlayer,
    g: 0,
    f: startH,
    h: startH,
    parent: -1,
    pushFrom: -1,
    pushDir: -1,
    key: startKey,
    seq: 0,
  });
  visited.set(startKey, 0);
  heapPush(0);
  let expansions = 0;
  let seqCounter = 1;

  while (heap.length > 0) {
    const id = heapPop();
    const node = nodes[id];
    const known = visited.get(node.key);
    if (known === undefined || node.g > known) continue;
    if (sameSorted(node.boxes, puzzle.goals)) {
      const pushes: PushStep[] = [];
      let at = id;
      while (nodes[at].parent >= 0) {
        pushes.push({ from: nodes[at].pushFrom, dir: nodes[at].pushDir });
        at = nodes[at].parent;
      }
      pushes.reverse();
      return { pushes, expansions };
    }
    expansions++;
    if (expansions > maxExpansions) return null;
    const boxSet = new Set<number>(node.boxes);
    const region = reachableRegion(ctx, node.player, boxSet);
    for (const box of node.boxes) {
      for (let d = 0; d < 4; d++) {
        const dest = neighbour(box, d, ctx.puzzle.width, ctx.puzzle.height);
        if (dest < 0 || !ctx.boxSet.has(dest) || boxSet.has(dest)) continue;
        const pusher = neighbour(box, d ^ 1, ctx.puzzle.width, ctx.puzzle.height);
        if (pusher < 0 || !region.has(pusher)) continue;
        const nextBoxes = node.boxes.filter((b) => b !== box);
        nextBoxes.push(dest);
        nextBoxes.sort((a, b) => a - b);
        if (isDeadState(ctx, nextBoxes)) continue;
        const h = heuristicValue(ctx, nextBoxes);
        if (h >= INF_COST) continue;
        const g2 = node.g + 1;
        const key = regionKey(nextBoxes, box);
        const prev = visited.get(key);
        if (prev !== undefined && prev <= g2) continue;
        visited.set(key, g2);
        nodes.push({
          boxes: nextBoxes,
          player: box,
          g: g2,
          f: g2 + h,
          h,
          parent: id,
          pushFrom: box,
          pushDir: d,
          key,
          seq: seqCounter++,
        });
        heapPush(nodes.length - 1);
      }
    }
  }
  return null;
}

main();
