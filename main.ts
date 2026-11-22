import * as fs from "node:fs";
import * as path from "node:path";

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = "udlr";
const INF = 1000000000;
const BIG = 1000000;
const MAX_EXP = 250000;
const IDS = ["i001", "i002", "i003", "i004", "i005"];

type Board = {
  w: number; h: number; n: number;
  wall: Uint8Array; goal: Uint8Array;
  boxes: number[]; goals: number[]; player: number;
};

function nb(c: number, d: number, w: number, h: number): number {
  const x = c % w; const y = (c / w) | 0;
  const nx = x + DX[d]; const ny = y + DY[d];
  if (nx < 0 || nx >= w || ny < 0 || ny >= h) return -1;
  return ny * w + nx;
}

function flood(start: number, blocked: Uint8Array, w: number, h: number): { rep: number; region: Uint8Array } {
  const n = w * h;
  const region = new Uint8Array(n);
  if (start < 0 || start >= n || blocked[start]) return { rep: start, region };
  const stack: number[] = [start];
  region[start] = 1;
  let rep = start;
  while (stack.length) {
    const c = stack.pop() as number;
    if (c < rep) rep = c;
    for (let d = 0; d < 4; d++) {
      const m = nb(c, d, w, h);
      if (m >= 0 && !blocked[m] && !region[m]) { region[m] = 1; stack.push(m); }
    }
  }
  return { rep, region };
}

function blockedWith(board: Board, boxes: number[], other: Uint8Array, extraBox: number): Uint8Array {
  const b = new Uint8Array(board.wall);
  for (let i = 0; i < other.length; i++) if (other[i]) b[i] = 1;
  for (const c of boxes) b[c] = 1;
  if (extraBox >= 0) b[extraBox] = 1;
  return b;
}

function parse(text: string): Board {
  const lines = text.split("\n").map((s) => s.replace(/\r$/, ""));
  while (lines.length && lines[lines.length - 1] === "") lines.pop();
  const h = lines.length;
  let w = 0;
  for (const l of lines) if (l.length > w) w = l.length;
  const n = w * h;
  const wall = new Uint8Array(n);
  const goal = new Uint8Array(n);
  const boxes: number[] = [];
  const goals: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = y * w + x;
      const ch = x < lines[y].length ? lines[y][x] : "#";
      if (ch === "#") wall[c] = 1;
      else if (ch === ".") { goal[c] = 1; goals.push(c); }
      else if (ch === "$") boxes.push(c);
      else if (ch === "*") { goal[c] = 1; goals.push(c); boxes.push(c); }
      else if (ch === "@") player = c;
      else if (ch === "+") { goal[c] = 1; goals.push(c); player = c; }
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, n, wall, goal, boxes, goals, player };
}

// Reverse pull BFS from one goal, ignoring other boxes. Returns push
// distance from every cell to this goal (INF if unreachable).
function distToGoal(board: Board, g: number): Int32Array {
  const { w, h, n, wall } = board;
  const best = new Int32Array(n).fill(INF);
  best[g] = 0;
  const seen = new Set<number>();
  const qBox: number[] = [];
  const qRep: number[] = [];
  const qDist: number[] = [];
  const blockedG = new Uint8Array(wall);
  blockedG[g] = 1;
  const seenCell = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    if (wall[c] || c === g || seenCell[c]) continue;
    const r = flood(c, blockedG, w, h);
    for (let i = 0; i < n; i++) if (r.region[i]) seenCell[i] = 1;
    const key = g * n + r.rep;
    if (!seen.has(key)) {
      seen.add(key);
      qBox.push(g); qRep.push(r.rep); qDist.push(0);
    }
  }
  let head = 0;
  while (head < qBox.length) {
    const box = qBox[head]; const rep = qRep[head]; const dist = qDist[head]; head++;
    const blocked = new Uint8Array(wall);
    blocked[box] = 1;
    const r = flood(rep, blocked, w, h);
    for (let d = 0; d < 4; d++) {
      // Pull: player stands at dest (adjacent to the box), pulls the box
      // into dest, and steps back to beyond (one more cell in the same
      // direction). dest must be reachable; beyond must be open floor.
      const dest = nb(box, d, w, h);
      if (dest < 0 || wall[dest] || !r.region[dest]) continue;
      const beyond = nb(dest, d, w, h);
      if (beyond < 0 || wall[beyond]) continue;
      const blocked2 = new Uint8Array(wall);
      blocked2[dest] = 1;
      const r2 = flood(beyond, blocked2, w, h);
      const key = dest * n + r2.rep;
      if (seen.has(key)) continue;
      seen.add(key);
      if (dist + 1 < best[dest]) best[dest] = dist + 1;
      qBox.push(dest); qRep.push(r2.rep); qDist.push(dist + 1);
    }
  }
  return best;
}

function hungarian(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const ans = new Array<number>(n + 1).fill(0);
  for (let j = 1; j <= n; j++) if (p[j] !== 0) ans[p[j]] = j;
  let total = 0;
  for (let i = 1; i <= n; i++) total += cost[i - 1][ans[i] - 1];
  return total;
}

function deadlocked(board: Board, boxes: number[], dead: Uint8Array): boolean {
  const { w, h, n, wall, goal } = board;
  const boxSet = new Uint8Array(n);
  const boxIdx = new Int32Array(n).fill(-1);
  for (let i = 0; i < boxes.length; i++) { boxSet[boxes[i]] = 1; boxIdx[boxes[i]] = i; }
  for (const b of boxes) if (!goal[b] && dead[b]) return true;
  // 2x2 frozen block: all four cells wall/box, one box off-goal.
  for (let y = 0; y + 1 < h; y++) {
    for (let x = 0; x + 1 < w; x++) {
      const a = y * w + x; const b = a + 1; const c = a + w; const d = c + 1;
      const cells = [a, b, c, d];
      let all = true; let bad = false;
      for (const cell of cells) {
        if (!wall[cell] && !boxSet[cell]) { all = false; break; }
        if (boxSet[cell] && !goal[cell]) bad = true;
      }
      if (all && bad) return true;
    }
  }
  // Iterative freeze: box blocked on both axes by walls/frozen boxes.
  const frozen = new Uint8Array(boxes.length);
  const blockedBy = (cell: number): boolean => {
    if (cell < 0) return true;
    if (wall[cell]) return true;
    const i = boxIdx[cell];
    return i >= 0 && frozen[i] === 1;
  };
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < boxes.length; i++) {
      if (frozen[i]) continue;
      const b = boxes[i];
      const horiz = blockedBy(nb(b, 2, w, h)) && blockedBy(nb(b, 3, w, h));
      const vert = blockedBy(nb(b, 0, w, h)) && blockedBy(nb(b, 1, w, h));
      if (horiz && vert) { frozen[i] = 1; changed = true; }
    }
  }
  for (let i = 0; i < boxes.length; i++) if (frozen[i] && !goal[boxes[i]]) return true;
  // Wall-segment freeze: box against a straight wall run with no goal in
  // the run and no gap to leave the run.
  for (const b of boxes) {
    if (goal[b]) continue;
    for (let wd = 0; wd < 4; wd++) {
      const wn = nb(b, wd, w, h);
      if (wn >= 0 && !wall[wn]) continue;
      const axis = wd < 2 ? 2 : 0; // perpendicular axis start dir
      const neg = axis; const pos = axis + 1;
      let lo = b;
      for (;;) {
        const m = nb(lo, neg, w, h);
        if (m < 0 || wall[m]) break;
        lo = m;
      }
      let hi = b;
      for (;;) {
        const m = nb(hi, pos, w, h);
        if (m < 0 || wall[m]) break;
        hi = m;
      }
      let trapped = true; let hasGoal = false;
      let cur = lo;
      for (;;) {
        if (goal[cur]) hasGoal = true;
        const out = nb(cur, wd, w, h);
        if (out >= 0 && !wall[out]) { trapped = false; break; }
        if (cur === hi) break;
        cur = nb(cur, pos, w, h);
      }
      if (trapped && !hasGoal) return true;
    }
  }
  return false;
}

type HeapEntry = { key: string; f: number; h: number; g: number; seq: number };
function heapPush(heap: HeapEntry[], e: HeapEntry): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    const a = heap[p]; const b = heap[i];
    if (a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.seq <= b.seq)))) break;
    heap[p] = b; heap[i] = a; i = p;
  }
}
function heapPop(heap: HeapEntry[]): HeapEntry {
  const top = heap[0];
  const last = heap.pop() as HeapEntry;
  if (heap.length) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = i * 2 + 1; const r = l + 1; let s = i;
      const less = (a: HeapEntry, b: HeapEntry): boolean =>
        a.f < b.f || (a.f === b.f && (a.h < b.h || (a.h === b.h && a.seq < b.seq)));
      if (l < heap.length && less(heap[l], heap[s])) s = l;
      if (r < heap.length && less(heap[r], heap[s])) s = r;
      if (s === i) break;
      const t = heap[s]; heap[s] = heap[i]; heap[i] = t; i = s;
    }
  }
  return top;
}

type NodeRec = {
  boxes: number[]; rep: number; playerPos: number; g: number;
  parent: string | null; pushFrom: number; pushTo: number; dir: number;
};

function findWalk(board: Board, from: number, to: number, blocked: Uint8Array): string | null {
  if (from === to) return "";
  const { w, h, n } = board;
  const prev = new Int32Array(n).fill(-1);
  const prevDir = new Int8Array(n).fill(-1);
  const q: number[] = [from];
  prev[from] = from;
  let head = 0;
  while (head < q.length) {
    const c = q[head++];
    if (c === to) break;
    for (let d = 0; d < 4; d++) {
      const m = nb(c, d, w, h);
      if (m >= 0 && !blocked[m] && prev[m] < 0) { prev[m] = c; prevDir[m] = d; q.push(m); }
    }
  }
  if (prev[to] < 0) return null;
  let out = "";
  let cur = to;
  while (cur !== from) {
    out = CH[prevDir[cur]] + out;
    cur = prev[cur];
  }
  return out;
}

function solveRoom(
  board: Board, roomBoxes: number[], roomGoals: number[], other: Uint8Array,
  playerStart: number, dists: Int32Array[], dead: Uint8Array
): { steps: string; finalPlayer: number; pushes: number } | null {
  const { w, h } = board;
  const heuristic = (boxes: number[]): number => {
    const cost: number[][] = [];
    for (const b of boxes) {
      const row: number[] = [];
      for (let j = 0; j < roomGoals.length; j++) {
        const d = dists[j][b];
        row.push(d >= INF ? BIG : d);
      }
      cost.push(row);
    }
    const total = hungarian(cost);
    return total >= BIG / 2 ? INF : total;
  };
  const isGoalState = (boxes: number[]): boolean => {
    for (const b of boxes) if (!board.goal[b]) return false;
    return true;
  };
  const startBlocked = blockedWith(board, roomBoxes, other, -1);
  const startRegion = flood(playerStart, startBlocked, w, h);
  const rootKey = roomBoxes.join(",") + "|" + startRegion.rep;
  const h0 = heuristic(roomBoxes);
  if (h0 >= INF) return null;
  const nodes = new Map<string, NodeRec>();
  nodes.set(rootKey, { boxes: roomBoxes.slice(), rep: startRegion.rep, playerPos: playerStart, g: 0, parent: null, pushFrom: -1, pushTo: -1, dir: -1 });
  const heap: HeapEntry[] = [];
  let seq = 0;
  heapPush(heap, { key: rootKey, f: h0, h: h0, g: 0, seq: seq++ });
  let expansions = 0;
  while (heap.length) {
    const e = heapPop(heap);
    const node = nodes.get(e.key);
    if (!node || node.g !== e.g) continue;
    if (expansions >= MAX_EXP) return null;
    expansions++;
    if (isGoalState(node.boxes)) {
      // Reconstruct.
      const chain: NodeRec[] = [];
      let cur: NodeRec | undefined = node;
      let curKey: string | null = e.key;
      while (cur) {
        chain.push(cur);
        curKey = cur.parent;
        cur = curKey === null ? undefined : nodes.get(curKey);
      }
      chain.reverse();
      let steps = "";
      for (let i = 1; i < chain.length; i++) {
        const nd = chain[i]; const pv = chain[i - 1];
        const stanceCell = nb(nd.pushFrom, nd.dir ^ 1, w, h);
        if (stanceCell < 0) return null;
        const blocked = blockedWith(board, pv.boxes, other, -1);
        const walk = findWalk(board, pv.playerPos, stanceCell, blocked);
        if (walk === null) return null;
        steps += walk + CH[nd.dir];
      }
      return { steps, finalPlayer: node.playerPos, pushes: node.g };
    }
    const boxSet = new Uint8Array(board.n);
    for (const b of node.boxes) boxSet[b] = 1;
    const blocked = blockedWith(board, node.boxes, other, -1);
    const region = flood(node.rep, blocked, w, h);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stanceCell = nb(b, d ^ 1, w, h);
        const dest = nb(b, d, w, h);
        if (dest < 0 || board.wall[dest] || boxSet[dest] || other[dest]) continue;
        if (stanceCell < 0 || !region.region[stanceCell]) continue;
        const nb2 = node.boxes.slice();
        nb2[bi] = dest;
        nb2.sort((a, b2) => a - b2);
        const blocked2 = blockedWith(board, nb2, other, -1);
        const r2 = flood(b, blocked2, w, h);
        if (deadlocked(board, nb2, dead)) continue;
        const hh = heuristic(nb2);
        if (hh >= INF) continue;
        const key2 = nb2.join(",") + "|" + r2.rep;
        const g2 = node.g + 1;
        const ex = nodes.get(key2);
        if (ex && ex.g <= g2) continue;
        nodes.set(key2, { boxes: nb2, rep: r2.rep, playerPos: b, g: g2, parent: e.key, pushFrom: b, pushTo: dest, dir: d });
        heapPush(heap, { key: key2, f: g2 + hh, h: hh, g: g2, seq: seq++ });
      }
    }
  }
  return null;
}

type Room = { boxes: number[]; goals: number[]; cells: number[]; minCell: number };

function decompose(board: Board, dead: Uint8Array): Room[] {
  const { w, h, n, wall } = board;
  const attempts: Uint8Array[] = [];
  const ex1 = new Uint8Array(dead);
  attempts.push(ex1);
  const ex2 = new Uint8Array(dead);
  const py = (board.player / w) | 0;
  for (let x = 0; x < w; x++) ex2[py * w + x] = 1;
  attempts.push(ex2);
  const ex3 = new Uint8Array(n);
  for (let x = 0; x < w; x++) ex3[py * w + x] = 1;
  attempts.push(ex3);
  for (const excluded of attempts) {
    const seen = new Uint8Array(n);
    const rooms: Room[] = [];
    let ok = true;
    let coveredBoxes = 0;
    for (let c = 0; c < n; c++) {
      if (wall[c] || excluded[c] || seen[c]) continue;
      const cells: number[] = [];
      const stack = [c];
      seen[c] = 1;
      while (stack.length) {
        const cur = stack.pop() as number;
        cells.push(cur);
        for (let d = 0; d < 4; d++) {
          const m = nb(cur, d, w, h);
          if (m >= 0 && !wall[m] && !excluded[m] && !seen[m]) { seen[m] = 1; stack.push(m); }
        }
      }
      const boxes = board.boxes.filter((b) => cells.indexOf(b) >= 0).sort((a, b) => a - b);
      const goals = board.goals.filter((g) => cells.indexOf(g) >= 0).sort((a, b) => a - b);
      if (boxes.length === 0 && goals.length === 0) continue;
      if (boxes.length === 0 || boxes.length !== goals.length) { ok = false; break; }
      coveredBoxes += boxes.length;
      let minCell = cells[0];
      for (const cc of cells) if (cc < minCell) minCell = cc;
      rooms.push({ boxes, goals, cells, minCell });
    }
    if (ok && coveredBoxes === board.boxes.length && rooms.length > 0) {
      // Order rooms by walking distance from the player, then min cell.
      const blocked = blockedWith(board, board.boxes, new Uint8Array(n), -1);
      const dist = new Int32Array(n).fill(-1);
      const q: number[] = [board.player];
      dist[board.player] = 0;
      let head = 0;
      while (head < q.length) {
        const cur = q[head++];
        for (let d = 0; d < 4; d++) {
          const m = nb(cur, d, w, h);
          if (m >= 0 && !blocked[m] && dist[m] < 0) { dist[m] = dist[cur] + 1; q.push(m); }
        }
      }
      for (const r of rooms) {
        let best = INF;
        for (const cc of r.cells) if (dist[cc] >= 0 && dist[cc] < best) best = dist[cc];
        (r as Room & { d: number }).d = best;
      }
      rooms.sort((a, b) => {
        const da = (a as Room & { d: number }).d; const db = (b as Room & { d: number }).d;
        if (da !== db) return da - db;
        return a.minCell - b.minCell;
      });
      return rooms;
    }
  }
  return [{ boxes: board.boxes.slice(), goals: board.goals.slice(), cells: [], minCell: 0 }];
}

function solveLevel(text: string): string {
  const board = parse(text);
  if (board.boxes.length === 0 || board.player < 0) return "";
  const dists: Int32Array[] = [];
  for (const g of board.goals) dists.push(distToGoal(board, g));
  const dead = new Uint8Array(board.n);
  for (let c = 0; c < board.n; c++) {
    if (board.wall[c]) continue;
    let alive = false;
    for (const d of dists) if (d[c] < INF) { alive = true; break; }
    dead[c] = alive ? 0 : 1;
  }
  const rooms = decompose(board, dead);
  // Map each room goal cell to its index in dists (goals are sorted).
  const goalIndex = new Map<number, number>();
  board.goals.forEach((g, i) => goalIndex.set(g, i));
  let playerPos = board.player;
  let steps = "";
  const solvedGoals: number[] = [];
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const other = new Uint8Array(board.n);
    for (let rj = 0; rj < rooms.length; rj++) {
      if (rj === ri) continue;
      const src = rj < ri ? rooms[rj].goals : rooms[rj].boxes;
      for (const c of src) other[c] = 1;
    }
    const roomDists = room.goals.map((g) => dists[goalIndex.get(g) as number]);
    const res = solveRoom(board, room.boxes, room.goals, other, playerPos, roomDists, dead);
    if (!res) return "";
    steps += res.steps;
    playerPos = res.finalPlayer;
    for (const g of room.goals) solvedGoals.push(g);
  }
  return steps;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  for (const id of IDS) {
    let steps = "";
    try {
      const text = fs.readFileSync(path.join(inputDir, id, "level.txt"), "utf8");
      steps = solveLevel(text);
    } catch {
      steps = "";
    }
    const dir = path.join(outputDir, id);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, "solution.txt"), steps, "utf8");
  }
}

main();
