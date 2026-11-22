import * as fs from "node:fs";
import * as path from "node:path";

// Solver written from the prose specification alone.
// Push-space A* per room, with an assignment heuristic built from
// single-box push distances, plus deadlock pruning.

const DIRS = [
  { dr: -1, dc: 0, ch: "u" },
  { dr: 1, dc: 0, ch: "d" },
  { dr: 0, dc: -1, ch: "l" },
  { dr: 0, dc: 1, ch: "r" },
];
const INF = 1000000;
let workDone = 0;
const WORK_LIMIT = 1200000;
let seqCounter = 0;

type SearchNode = {
  boxes: number[];
  player: number;
  g: number;
  f: number;
  seq: number;
  parent: SearchNode | null;
  from: number;
  dir: number;
};

type RoomInfo = {
  n: number;
  width: number;
  height: number;
  neigh: number[][];
  localOfGlobal: number[];
  globalOfLocal: number[];
  goals: number[];
  goalSet: Set<number>;
  rows: number[];
  cols: number[];
  pushDist: number[][];
  deadMin: number[];
  startBoxes: number[];
  startPlayer: number;
  door: number;
  porch: number;
  porchGlobal: number;
};

function boxesKey(boxes: number[]): string {
  return boxes.join(",");
}

function floodFrom(
  start: number,
  boxSet: Set<number>,
  neigh: number[][],
  n: number,
): { visited: boolean[]; rep: number } {
  const visited: boolean[] = new Array(n).fill(false);
  const stack: number[] = [start];
  visited[start] = true;
  let rep = start;
  while (stack.length > 0) {
    const cur = stack.pop();
    if (cur === undefined) break;
    if (cur < rep) rep = cur;
    for (let d = 0; d < 4; d++) {
      const nb = neigh[cur][d];
      if (nb === -1) continue;
      if (boxSet.has(nb)) continue;
      if (visited[nb]) continue;
      visited[nb] = true;
      stack.push(nb);
    }
  }
  return { visited, rep };
}

function heapLess(a: SearchNode, b: SearchNode): boolean {
  if (a.f !== b.f) return a.f < b.f;
  if (a.g !== b.g) return a.g > b.g;
  return a.seq < b.seq;
}

function heapPush(heap: SearchNode[], node: SearchNode): void {
  heap.push(node);
  let i = heap.length - 1;
  while (i > 0) {
    const p = Math.floor((i - 1) / 2);
    if (!heapLess(heap[i], heap[p])) break;
    const tmp = heap[i];
    heap[i] = heap[p];
    heap[p] = tmp;
    i = p;
  }
}

function heapPop(heap: SearchNode[]): SearchNode | null {
  if (heap.length === 0) return null;
  const top = heap[0];
  const last = heap.pop();
  if (heap.length > 0 && last !== undefined) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      let best = i;
      if (left < heap.length && heapLess(heap[left], heap[best])) best = left;
      if (right < heap.length && heapLess(heap[right], heap[best])) best = right;
      if (best === i) break;
      const tmp = heap[i];
      heap[i] = heap[best];
      heap[best] = tmp;
      i = best;
    }
  }
  return top;
}

function nextPermutation(a: number[]): boolean {
  let i = a.length - 2;
  while (i >= 0 && a[i] >= a[i + 1]) i--;
  if (i < 0) return false;
  let j = a.length - 1;
  while (a[j] <= a[i]) j--;
  const t = a[i];
  a[i] = a[j];
  a[j] = t;
  let lo = i + 1;
  let hi = a.length - 1;
  while (lo < hi) {
    const u = a[lo];
    a[lo] = a[hi];
    a[hi] = u;
    lo++;
    hi--;
  }
  return true;
}

function buildPushDist(
  n: number,
  neigh: number[][],
  goals: number[],
): number[][] {
  const dist: number[][] = [];
  for (let s = 0; s < n; s++) {
    dist.push(new Array(goals.length).fill(INF));
  }
  for (let s = 0; s < n; s++) {
    const sd: number[] = new Array(n * n).fill(INF);
    const dq: number[] = [];
    for (let p = 0; p < n; p++) {
      if (p === s) continue;
      sd[s * n + p] = 0;
      dq.push(s * n + p);
    }
    while (dq.length > 0) {
      const code = dq.shift();
      if (code === undefined) break;
      const box = Math.floor(code / n);
      const player = code % n;
      const d = sd[code];
      for (let dir = 0; dir < 4; dir++) {
        const np = neigh[player][dir];
        if (np === -1 || np === box) continue;
        const nc = box * n + np;
        if (d < sd[nc]) {
          sd[nc] = d;
          dq.unshift(nc);
        }
      }
      for (let dir = 0; dir < 4; dir++) {
        const opp = dir ^ 1;
        const stance = neigh[box][opp];
        if (stance !== player) continue;
        const dest = neigh[box][dir];
        if (dest === -1) continue;
        const nc = dest * n + box;
        if (d + 1 < sd[nc]) {
          sd[nc] = d + 1;
          dq.push(nc);
        }
      }
    }
    for (let gi = 0; gi < goals.length; gi++) {
      let best = INF;
      for (let p = 0; p < n; p++) {
        const v = sd[goals[gi] * n + p];
        if (v < best) best = v;
      }
      dist[s][gi] = best;
    }
  }
  return dist;
}

function assignmentCost(
  boxes: number[],
  pushDist: number[][],
  goalCount: number,
): number {
  const perm: number[] = [];
  for (let i = 0; i < goalCount; i++) perm.push(i);
  let best = INF;
  for (;;) {
    let sum = 0;
    for (let i = 0; i < boxes.length; i++) {
      const c = pushDist[boxes[i]][perm[i]];
      if (c >= INF) {
        sum = INF;
        break;
      }
      sum += c;
      if (sum >= best) break;
    }
    if (sum < best) best = sum;
    if (!nextPermutation(perm)) break;
  }
  return best;
}

function frozenBoxes(
  boxes: number[],
  neigh: number[][],
): Set<number> {
  const frozen = new Set<number>();
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (frozen.has(b)) continue;
      const left = neigh[b][2];
      const right = neigh[b][3];
      const up = neigh[b][0];
      const down = neigh[b][1];
      const hBlocked =
        (left === -1 || frozen.has(left)) &&
        (right === -1 || frozen.has(right));
      const vBlocked =
        (up === -1 || frozen.has(up)) && (down === -1 || frozen.has(down));
      if (hBlocked && vBlocked) {
        frozen.add(b);
        changed = true;
      }
    }
  }
  return frozen;
}

function isTrapped(
  boxes: number[],
  room: RoomInfo,
  trapCache: Map<string, boolean>,
): boolean {
  const key = boxesKey(boxes);
  const cached = trapCache.get(key);
  if (cached !== undefined) return cached;
  let trapped = false;
  const boxSet = new Set<number>(boxes);
  for (const b of boxes) {
    if (room.deadMin[b] >= INF) {
      trapped = true;
      break;
    }
  }
  if (!trapped) {
    const frozen = frozenBoxes(boxes, room.neigh);
    for (const b of frozen) {
      if (!room.goalSet.has(b)) {
        trapped = true;
        break;
      }
    }
    if (!trapped) {
      const width = room.width;
      const height = room.height;
      outer: for (let r = 0; r < height - 1; r++) {
        for (let c = 0; c < width - 1; c++) {
          const cells = [
            r * width + c,
            r * width + c + 1,
            (r + 1) * width + c,
            (r + 1) * width + c + 1,
          ];
          if (room.porchGlobal >= 0 && cells.includes(room.porchGlobal)) {
            continue;
          }
          let anyBox = false;
          let anyGoal = false;
          for (const g of cells) {
            const local = room.localOfGlobal[g];
            if (local === -1) continue;
            if (room.goalSet.has(local)) anyGoal = true;
            if (boxSet.has(local)) anyBox = true;
          }
          // Cells outside the room are walls; cells inside that are
          // neither box nor goal break the block only when they are
          // empty floor. The loop above marks allBlocked false for
          // any in-room cell that is not a box, including a goal cell,
          // so re-check: a 2x2 is a trap only when every cell is a
          // wall or a box and no cell is a goal.
          let everyWallOrBox = true;
          for (const g of cells) {
            const local = room.localOfGlobal[g];
            if (local === -1) continue;
            if (!boxSet.has(local)) {
              everyWallOrBox = false;
              break;
            }
          }
          if (everyWallOrBox && anyBox && !anyGoal) {
            trapped = true;
            break outer;
          }
        }
      }
    }
    if (!trapped) {
      const frozen2 = frozenBoxes(boxes, room.neigh);
      for (const b of boxes) {
        if (room.goalSet.has(b)) continue;
        for (let s = 0; s < 4; s++) {
          const side = room.neigh[b][s];
          if (!(side === -1 || frozen2.has(side))) continue;
          const perps = s === 0 || s === 1 ? [2, 3] : [0, 1];
          const seg: number[] = [b];
          let endA = b;
          let cur = b;
          for (;;) {
            const nxt = room.neigh[cur][perps[0]];
            if (nxt === -1) break;
            const sn = room.neigh[nxt][s];
            if (!(sn === -1 || frozen2.has(sn))) break;
            seg.push(nxt);
            cur = nxt;
            endA = nxt;
          }
          let endB = b;
          cur = b;
          for (;;) {
            const nxt = room.neigh[cur][perps[1]];
            if (nxt === -1) break;
            const sn = room.neigh[nxt][s];
            if (!(sn === -1 || frozen2.has(sn))) break;
            seg.push(nxt);
            cur = nxt;
            endB = nxt;
          }
          const beyondA = room.neigh[endA][perps[0]];
          const beyondB = room.neigh[endB][perps[1]];
          const endABlocked =
            beyondA === -1 || frozen2.has(beyondA);
          const endBBlocked =
            beyondB === -1 || frozen2.has(beyondB);
          if (!endABlocked || !endBBlocked) continue;
          let segGoal = false;
          for (const cell of seg) {
            if (room.goalSet.has(cell)) {
              segGoal = true;
              break;
            }
          }
          if (!segGoal) {
            trapped = true;
            break;
          }
        }
        if (trapped) break;
      }
    }
  }
  trapCache.set(key, trapped);
  return trapped;
}

function searchRoom(
  room: RoomInfo,
  needExit: boolean,
): { from: number; dir: number }[] | null {
  const startBoxes = [...room.startBoxes].sort((a, b) => a - b);
  const goalsSorted = [...room.goals].sort((a, b) => a - b);
  const isGoalState = (boxes: number[]): boolean => {
    for (let i = 0; i < boxes.length; i++) {
      if (boxes[i] !== goalsSorted[i]) return false;
    }
    return true;
  };
  const trapCache = new Map<string, boolean>();
  const heurCache = new Map<string, number>();
  const heur = (boxes: number[]): number => {
    const k = boxesKey(boxes);
    const hit = heurCache.get(k);
    if (hit !== undefined) return hit;
    const v = assignmentCost(boxes, room.pushDist, room.goals.length);
    heurCache.set(k, v);
    return v;
  };
  const startTrapped = isTrapped(startBoxes, room, trapCache);
  if (startTrapped) return null;
  const startH = heur(startBoxes);
  if (startH >= INF) return null;
  const startSet = new Set<number>(startBoxes);
  const startRep = floodFrom(room.startPlayer, startSet, room.neigh, room.n).rep;
  const visited = new Map<string, number>();
  visited.set(boxesKey(startBoxes) + "|" + startRep, 0);
  const heap: SearchNode[] = [];
  seqCounter = 0;
  heapPush(heap, {
    boxes: startBoxes,
    player: room.startPlayer,
    g: 0,
    f: startH,
    seq: seqCounter++,
    parent: null,
    from: -1,
    dir: -1,
  });
  while (heap.length > 0) {
    const node = heapPop(heap);
    if (node === null) break;
    const nodeSet = new Set<number>(node.boxes);
    const nodeRep = floodFrom(node.player, nodeSet, room.neigh, room.n).rep;
    const nodeKey = boxesKey(node.boxes) + "|" + nodeRep;
    const bestG = visited.get(nodeKey);
    if (bestG !== undefined && node.g > bestG) continue;
    if (isGoalState(node.boxes)) {
      if (needExit) {
        const exitCell = room.porch >= 0 ? room.porch : room.door;
        const reach = floodFrom(node.player, nodeSet, room.neigh, room.n);
        if (!reach.visited[exitCell]) continue;
      }
      const pushes: { from: number; dir: number }[] = [];
      let cur: SearchNode | null = node;
      while (cur !== null && cur.parent !== null) {
        pushes.push({ from: cur.from, dir: cur.dir });
        cur = cur.parent;
      }
      pushes.reverse();
      return pushes;
    }
    workDone++;
    if (workDone > WORK_LIMIT) return null;
    const reach = floodFrom(node.player, nodeSet, room.neigh, room.n);
    for (const box of node.boxes) {
      for (let dir = 0; dir < 4; dir++) {
        const opp = dir ^ 1;
        const stance = room.neigh[box][opp];
        if (stance === -1) continue;
        if (!reach.visited[stance]) continue;
        const dest = room.neigh[box][dir];
        if (dest === -1) continue;
        if (dest === room.porch) continue;
        if (nodeSet.has(dest)) continue;
        const nextBoxes = node.boxes.map((b) => (b === box ? dest : b));
        nextBoxes.sort((a, b) => a - b);
        if (isTrapped(nextBoxes, room, trapCache)) continue;
        const h = heur(nextBoxes);
        if (h >= INF) continue;
        const nextSet = new Set<number>(nextBoxes);
        const rep = floodFrom(box, nextSet, room.neigh, room.n).rep;
        const k = boxesKey(nextBoxes) + "|" + rep;
        const ng = node.g + 1;
        const prev = visited.get(k);
        if (prev !== undefined && prev <= ng) continue;
        visited.set(k, ng);
        heapPush(heap, {
          boxes: nextBoxes,
          player: box,
          g: ng,
          f: ng + h,
          seq: seqCounter++,
          parent: node,
          from: box,
          dir,
        });
      }
    }
  }
  return null;
}

type ParsedLevel = {
  width: number;
  height: number;
  wall: boolean[];
  goal: boolean[];
  boxes: number[];
  player: number;
  hallRow: number;
};

function parseLevel(text: string): ParsedLevel {
  const rawLines = text.split("\n").map((l) => l.replace(/\r$/, ""));
  while (rawLines.length > 0 && rawLines[rawLines.length - 1] === "") {
    rawLines.pop();
  }
  const height = rawLines.length;
  let width = 0;
  for (const l of rawLines) if (l.length > width) width = l.length;
  const wall: boolean[] = new Array(width * height).fill(true);
  const goal: boolean[] = new Array(width * height).fill(false);
  const boxes: number[] = [];
  let player = -1;
  let hallRow = -1;
  for (let r = 0; r < height; r++) {
    const line = rawLines[r];
    for (let c = 0; c < width; c++) {
      const ch = c < line.length ? line[c] : "#";
      const idx = r * width + c;
      if (ch === "#") {
        wall[idx] = true;
      } else {
        wall[idx] = false;
        if (ch === "." || ch === "*" || ch === "+") goal[idx] = true;
        if (ch === "$" || ch === "*") boxes.push(idx);
        if (ch === "@" || ch === "+") {
          player = idx;
          hallRow = r;
        }
      }
    }
  }
  boxes.sort((a, b) => a - b);
  return { width, height, wall, goal, boxes, player, hallRow };
}

function buildRoom(
  cells: number[],
  parsed: ParsedLevel,
  doorGlobal: number,
  startPlayerGlobal: number,
): RoomInfo {
  const width = parsed.width;
  const height = parsed.height;
  const localOfGlobal: number[] = new Array(width * height).fill(-1);
  const globalOfLocal: number[] = [];
  const rows: number[] = [];
  const cols: number[] = [];
  cells.forEach((g, i) => {
    localOfGlobal[g] = i;
    globalOfLocal.push(g);
    rows.push(Math.floor(g / width));
    cols.push(g % width);
  });
  const n = cells.length;
  const neigh: number[][] = [];
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    const c = cols[i];
    const rowNeigh: number[] = [];
    for (let d = 0; d < 4; d++) {
      const nr = r + DIRS[d].dr;
      const nc = c + DIRS[d].dc;
      if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
        rowNeigh.push(-1);
      } else {
        rowNeigh.push(localOfGlobal[nr * width + nc]);
      }
    }
    neigh.push(rowNeigh);
  }
  const goals: number[] = [];
  const goalSet = new Set<number>();
  const startBoxes: number[] = [];
  for (const g of cells) {
    const local = localOfGlobal[g];
    if (parsed.goal[g]) {
      goals.push(local);
      goalSet.add(local);
    }
    if (parsed.boxes.includes(g)) startBoxes.push(local);
  }
  goals.sort((a, b) => a - b);
  startBoxes.sort((a, b) => a - b);
  const hasHall = cells.some(
    (g) => Math.floor(g / width) === parsed.hallRow,
  );
  let porch = -1;
  let porchGlobal = -1;
  let totalN = n;
  let startPlayer = localOfGlobal[startPlayerGlobal];
  if (!hasHall) {
    const doorLocal = localOfGlobal[doorGlobal];
    const doorRow = Math.floor(doorGlobal / width);
    const hallDir = doorRow < parsed.hallRow ? 1 : 0;
    porch = n;
    porchGlobal = parsed.hallRow * width + (doorGlobal % width);
    const porchNeigh: number[] = [-1, -1, -1, -1];
    porchNeigh[hallDir ^ 1] = doorLocal;
    neigh.push(porchNeigh);
    neigh[doorLocal][hallDir] = porch;
    rows.push(parsed.hallRow);
    cols.push(doorGlobal % width);
    globalOfLocal.push(porchGlobal);
    totalN = n + 1;
    startPlayer = porch;
  }
  const pushDist = buildPushDist(totalN, neigh, goals);
  const deadMin: number[] = new Array(totalN).fill(INF);
  for (let i = 0; i < totalN; i++) {
    let best = INF;
    for (let gi = 0; gi < goals.length; gi++) {
      if (pushDist[i][gi] < best) best = pushDist[i][gi];
    }
    deadMin[i] = best;
  }
  return {
    n: totalN,
    width,
    height,
    neigh,
    localOfGlobal,
    globalOfLocal,
    goals,
    goalSet,
    rows,
    cols,
    pushDist,
    deadMin,
    startBoxes,
    startPlayer,
    door: localOfGlobal[doorGlobal],
    porch,
    porchGlobal,
  };
}

function findRooms(parsed: ParsedLevel): { cells: number[]; door: number }[] | null {
  if (parsed.player === -1 || parsed.hallRow === -1) return null;
  const width = parsed.width;
  const height = parsed.height;
  const seen: boolean[] = new Array(width * height).fill(false);
  const rooms: { cells: number[]; door: number }[] = [];
  for (let r = 0; r < height; r++) {
    if (r === parsed.hallRow) continue;
    for (let c = 0; c < width; c++) {
      const idx = r * width + c;
      if (parsed.wall[idx] || seen[idx]) continue;
      const cells: number[] = [];
      const queue: number[] = [idx];
      seen[idx] = true;
      let qi = 0;
      while (qi < queue.length) {
        const cur = queue[qi++];
        cells.push(cur);
        const cr = Math.floor(cur / width);
        const cc = cur % width;
        for (let d = 0; d < 4; d++) {
          const nr = cr + DIRS[d].dr;
          const nc = cc + DIRS[d].dc;
          if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
          if (nr === parsed.hallRow) continue;
          const ni = nr * width + nc;
          if (parsed.wall[ni] || seen[ni]) continue;
          seen[ni] = true;
          queue.push(ni);
        }
      }
      let door = -1;
      const sortedCells = [...cells].sort((a, b) => a - b);
      for (const cell of sortedCells) {
        const cr = Math.floor(cell / width);
        const cc = cell % width;
        const hr = parsed.hallRow;
        if (Math.abs(cr - hr) === 1) {
          const hallIdx = hr * width + cc;
          if (!parsed.wall[hallIdx]) {
            door = cell;
            break;
          }
        }
      }
      if (door === -1) return null;
      rooms.push({ cells: sortedCells, door });
    }
  }
  return rooms;
}

function bfsGlobal(
  start: number,
  target: number,
  boxSet: Set<number>,
  parsed: ParsedLevel,
): number[] | null {
  if (start === target) return [];
  const width = parsed.width;
  const height = parsed.height;
  const prev: number[] = new Array(width * height).fill(-1);
  const prevDir: number[] = new Array(width * height).fill(-1);
  const queue: number[] = [start];
  prev[start] = start;
  let qi = 0;
  while (qi < queue.length) {
    const cur = queue[qi++];
    const cr = Math.floor(cur / width);
    const cc = cur % width;
    for (let d = 0; d < 4; d++) {
      const nr = cr + DIRS[d].dr;
      const nc = cc + DIRS[d].dc;
      if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
      const ni = nr * width + nc;
      if (parsed.wall[ni]) continue;
      if (boxSet.has(ni)) continue;
      if (prev[ni] !== -1) continue;
      prev[ni] = cur;
      prevDir[ni] = d;
      if (ni === target) {
        const dirs: number[] = [];
        let walk = ni;
        while (walk !== start) {
          dirs.push(prevDir[walk]);
          walk = prev[walk];
        }
        dirs.reverse();
        return dirs;
      }
      queue.push(ni);
    }
  }
  return null;
}

function bfsLocal(
  start: number,
  target: number,
  boxSet: Set<number>,
  room: RoomInfo,
): number[] | null {
  if (start === target) return [];
  const prev: number[] = new Array(room.n).fill(-1);
  const prevDir: number[] = new Array(room.n).fill(-1);
  const queue: number[] = [start];
  prev[start] = start;
  let qi = 0;
  while (qi < queue.length) {
    const cur = queue[qi++];
    for (let d = 0; d < 4; d++) {
      const nb = room.neigh[cur][d];
      if (nb === -1) continue;
      if (boxSet.has(nb)) continue;
      if (prev[nb] !== -1) continue;
      prev[nb] = cur;
      prevDir[nb] = d;
      if (nb === target) {
        const dirs: number[] = [];
        let walk = nb;
        while (walk !== start) {
          dirs.push(prevDir[walk]);
          walk = prev[walk];
        }
        dirs.reverse();
        return dirs;
      }
      queue.push(nb);
    }
  }
  return null;
}

function applyDirGlobal(pos: number, dir: number, width: number): number {
  const r = Math.floor(pos / width);
  const c = pos % width;
  return (r + DIRS[dir].dr) * width + (c + DIRS[dir].dc);
}

function solveLevel(text: string): string {
  const parsed = parseLevel(text);
  if (parsed.player === -1) return "";
  const roomsRaw = findRooms(parsed);
  if (roomsRaw === null) {
    return solveWhole(parsed);
  }
  let totalBoxes = 0;
  for (const room of roomsRaw) totalBoxes += room.cells.filter((g) => parsed.boxes.includes(g)).length;
  if (totalBoxes !== parsed.boxes.length) return solveWhole(parsed);
  for (const room of roomsRaw) {
    const goalsIn = room.cells.filter((g) => parsed.goal[g]).length;
    const boxesIn = room.cells.filter((g) => parsed.boxes.includes(g)).length;
    if (goalsIn !== boxesIn || boxesIn === 0) return solveWhole(parsed);
  }
  const moves: string[] = [];
  let globalPlayer = parsed.player;
  let globalBoxes: number[] = [...parsed.boxes];
  for (let ri = 0; ri < roomsRaw.length; ri++) {
    const raw = roomsRaw[ri];
    const room = buildRoom(raw.cells, parsed, raw.door, raw.door);
    const needExit = ri < roomsRaw.length - 1;
    const pushes = searchRoom(room, needExit);
    if (pushes === null) return solveWhole(parsed);
    const porchGlobal =
      parsed.hallRow * parsed.width + (raw.door % parsed.width);
    const boxSetGlobal = new Set<number>(globalBoxes);
    const travel = bfsGlobal(globalPlayer, porchGlobal, boxSetGlobal, parsed);
    if (travel === null) return solveWhole(parsed);
    for (const d of travel) {
      moves.push(DIRS[d].ch);
      globalPlayer = applyDirGlobal(globalPlayer, d, parsed.width);
    }
    let localPlayer = room.startPlayer;
    let localBoxes: number[] = [...room.startBoxes];
    for (const push of pushes) {
      const fromLocal = push.from;
      const dir = push.dir;
      const opp = dir ^ 1;
      const stance = room.neigh[fromLocal][opp];
      if (stance === -1) return solveWhole(parsed);
      const localSet = new Set<number>(localBoxes);
      const walk = bfsLocal(localPlayer, stance, localSet, room);
      if (walk === null) return solveWhole(parsed);
      for (const d of walk) {
        moves.push(DIRS[d].ch);
        globalPlayer = applyDirGlobal(globalPlayer, d, parsed.width);
        localPlayer = room.neigh[localPlayer][d];
      }
      const fromGlobal = room.globalOfLocal[fromLocal];
      const destLocal = room.neigh[fromLocal][dir];
      const destGlobal = room.globalOfLocal[destLocal];
      moves.push(DIRS[dir].ch);
      globalPlayer = fromGlobal;
      globalBoxes = globalBoxes.map((b) => (b === fromGlobal ? destGlobal : b));
      localBoxes = localBoxes.map((b) => (b === fromLocal ? destLocal : b));
      localPlayer = fromLocal;
    }
  }
  return moves.join("");
}

function solveWhole(parsed: ParsedLevel): string {
  const allCells: number[] = [];
  for (let i = 0; i < parsed.width * parsed.height; i++) {
    if (!parsed.wall[i]) allCells.push(i);
  }
  const room = buildRoom(allCells, parsed, parsed.player, parsed.player);
  const pushes = searchRoom(room, false);
  if (pushes === null) return "";
  const moves: string[] = [];
  let globalPlayer = parsed.player;
  let localPlayer = room.startPlayer;
  let localBoxes: number[] = [...room.startBoxes];
  for (const push of pushes) {
    const fromLocal = push.from;
    const dir = push.dir;
    const opp = dir ^ 1;
    const stance = room.neigh[fromLocal][opp];
    if (stance === -1) return "";
    const localSet = new Set<number>(localBoxes);
    const walk = bfsLocal(localPlayer, stance, localSet, room);
    if (walk === null) return "";
    for (const d of walk) {
      moves.push(DIRS[d].ch);
      globalPlayer = applyDirGlobal(globalPlayer, d, parsed.width);
      localPlayer = room.neigh[localPlayer][d];
    }
    const destLocal = room.neigh[fromLocal][dir];
    moves.push(DIRS[dir].ch);
    globalPlayer = room.globalOfLocal[fromLocal];
    localBoxes = localBoxes.map((b) => (b === fromLocal ? destLocal : b));
    localPlayer = fromLocal;
  }
  return moves.join("");
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  const ids = ["i001", "i002", "i003", "i004", "i005"];
  for (const id of ids) {
    const levelPath = path.join(inDir, id, "level.txt");
    let answer = "";
    try {
      const text = fs.readFileSync(levelPath, "utf8");
      answer = solveLevel(text);
    } catch {
      answer = "";
    }
    const outSub = path.join(outDir, id);
    fs.mkdirSync(outSub, { recursive: true });
    fs.writeFileSync(path.join(outSub, "solution.txt"), answer, "utf8");
  }
}

main();
