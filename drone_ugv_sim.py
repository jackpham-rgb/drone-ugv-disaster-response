"""
================================================================================
drone_ugv_sim.py
Area Disaster Response — Drone–UGV Cooperative Rescue Simulation Engine

Author: Trung Hieu Pham (Jack Pham)
Advisors: Dr. Adam Thorpe, Dr. Ufuk Topcu — UT Austin
Funding: NSF REU Site: CI Research 4 Social Change, Award #2150390

This module provides:
  - TerrainWorld: procedural terrain generation (mountain / city biomes)
  - DroneSwarm:   MAPPO / Boids / Random UAV exploration policies
  - UGVFleet:     QMIX / Greedy / Random ground rescue policies
  - FireTruckFleet: greedy fire suppression
  - run_episode():         run a single simulation episode
  - compare_configurations(): run multiple seeds across algorithm combos
  - fairness_reward():    the core fairness eq from the original REU paper

Algorithm constants (pass to run_episode):
  MAPPO_DRONE, BOIDS_DRONE, RANDOM_DRONE
  QMIX_UGV, GREEDY_UGV, RANDOM_UGV
================================================================================
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ── Algorithm constants ──────────────────────────────────────────────────────
MAPPO_DRONE  = 'mappo'
BOIDS_DRONE  = 'boids'
RANDOM_DRONE = 'random'
QMIX_UGV     = 'qmix'
GREEDY_UGV   = 'greedy'
RANDOM_UGV   = 'random'

# Belief map cell states
B_UNKNOWN  = 0
B_CLEAR    = 1
B_SURVIVOR = 2
B_FIRE     = 3
B_RESCUED  = 4
B_WALL     = 5


# ── Seeded RNG ────────────────────────────────────────────────────────────────
class RNG:
    """Deterministic pseudo-random number generator (LCG)."""
    def __init__(self, seed: int = 42):
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0

    def int(self, a: int, b: int) -> int:
        return a + int(self.next() * (b - a))

    def float(self, a: float = 0.0, b: float = 1.0) -> float:
        return a + self.next() * (b - a)

    def choose(self, arr):
        return arr[self.int(0, len(arr))]

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.int(0, i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr


# ── Noise functions ───────────────────────────────────────────────────────────
def _hash(n: int, seed: int = 0) -> float:
    h = (int(n) ^ seed) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    return (h & 0xFFFFFFFF) / 4294967296.0

def noise2(x: float, y: float, seed: int = 0) -> float:
    """2D smooth value noise in [0, 1]."""
    xi, yi = int(np.floor(x)), int(np.floor(y))
    xf, yf = x - xi, y - yi
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    a = _hash(xi + yi * 57, seed)
    b = _hash(xi + 1 + yi * 57, seed)
    c = _hash(xi + (yi + 1) * 57, seed)
    d = _hash(xi + 1 + (yi + 1) * 57, seed)
    return a + (b - a) * u + (c - a) * v + (d - c - b + a) * u * v

def fbm(x: float, y: float, seed: int = 0, octaves: int = 4) -> float:
    """Fractional Brownian Motion — layered noise for realistic terrain."""
    v, amp, freq, total = 0.0, 0.5, 1.0, 0.0
    for i in range(octaves):
        v += noise2(x * freq, y * freq, seed + i * 997) * amp
        total += amp
        amp *= 0.5
        freq *= 2.2
    return v / total


# ── Terrain world ─────────────────────────────────────────────────────────────
class TerrainWorld:
    """
    Procedural terrain grid with fire spread and belief map.

    Attributes
    ----------
    grid_size : int
    biome     : 'mountain' or 'city'
    seed      : int
    height    : 2D array, elevation
    t_type    : 2D object array, terrain type string
    passable  : 2D bool array
    fuel      : 2D float array, fire fuel load (0–1)
    fire      : 2D float array, burning cells (0 or 1)
    belief    : 2D int array, drone belief map
    survivors : list of Survivor dicts
    wind_dir  : (wx, wz) unit vector
    wind_spd  : float
    """

    def __init__(self, grid_size: int = 24, biome: str = 'mountain',
                 seed: int = 42, n_survivors: int = None, fire_rate: int = 4):
        self.GRID      = grid_size
        self.biome     = biome
        self.seed      = seed
        self.fire_rate = fire_rate
        self.rng       = RNG(seed)

        self.height   = np.zeros((grid_size, grid_size))
        self.t_type   = np.empty((grid_size, grid_size), dtype=object)
        self.passable = np.ones((grid_size, grid_size), dtype=bool)
        self.fuel     = np.zeros((grid_size, grid_size))
        self.fire     = np.zeros((grid_size, grid_size))
        self.belief   = np.zeros((grid_size, grid_size), dtype=int)

        self.survivors: List[Dict] = []
        self.wind_dir  = [1.0, 0.4]
        self.wind_spd  = 1.0

        self._generate(n_survivors)

    def _generate(self, n_survivors):
        G, s = self.GRID, self.seed
        for r in range(G):
            for c in range(G):
                if self.biome == 'mountain':
                    h = max(0, fbm(c / G * 3.5, r / G * 3.5, s) * 8
                              + fbm(c / G * 8,   r / G * 8,   s + 51) * 2)
                    self.height[r, c] = h
                    if   h < 2: t, f, p = 'water',  0.05, False
                    elif h < 4: t, f, p = 'grass',  0.40, True
                    elif h < 6: t, f, p = 'forest', 0.80, True
                    elif h < 8: t, f, p = 'rock',   0.02, True
                    else:       t, f, p = 'peak',   0.01, False
                else:  # city
                    is_street = (c % 4 == 0 or r % 4 == 0)
                    bval = noise2(int(c/4)*0.7, int(r/4)*0.7, s)
                    if not is_street and bval > 0.25:
                        h = 3 + noise2(c * 0.3, r * 0.3, s + 7) * 5
                        t, f, p = 'building', 0.0, False
                    else:
                        h = 0
                        t, f, p = ('street', 0.3, True) if is_street else ('plaza', 0.35, True)
                    self.height[r, c] = h
                self.t_type[r, c]   = t
                self.passable[r, c] = p
                self.fuel[r, c]     = f
                # Border impassable
                if r == 0 or r == G-1 or c == 0 or c == G-1:
                    self.passable[r, c] = False

        # Clear cross for guaranteed pathfinding
        mid = G // 2
        for c in range(1, G-1):
            if not self.passable[mid, c] and self.t_type[mid, c] != 'building':
                self.passable[mid, c] = True
                self.t_type[mid, c]   = 'grass'

        # Wind
        wa = self.rng.float(0, 2 * np.pi)
        self.wind_dir = [np.cos(wa), np.sin(wa)]
        self.wind_spd = self.rng.float(0.6, 1.8)

        # Initial fires
        occupied = set()
        for _ in range(self.rng.int(2, 4)):
            for _ in range(100):
                r2, c2 = self.rng.int(4, G-4), self.rng.int(4, G-4)
                if self.passable[r2, c2] and self.fuel[r2, c2] > 0.1:
                    self.fire[r2, c2] = 1.0
                    self.belief[r2, c2] = B_FIRE
                    occupied.add((r2, c2))
                    break

        # Survivors
        ns = n_survivors if n_survivors else self.rng.int(5, 9)
        for i in range(ns):
            for _ in range(200):
                r2, c2 = self.rng.int(2, G-2), self.rng.int(2, G-2)
                if self.passable[r2, c2] and self.fire[r2, c2] == 0 and (r2, c2) not in occupied:
                    occupied.add((r2, c2))
                    self.survivors.append({'id': i, 'row': r2, 'col': c2,
                                           'status': 'unknown', 'assigned_to': None})
                    break

    def spread_fire(self):
        """One step of cellular automaton fire spread."""
        if np.random.random() > 0.35:
            return
        new_fire = self.fire.copy()
        rate = self.fire_rate / 120.0
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for r in range(1, self.GRID-1):
            for c in range(1, self.GRID-1):
                if self.fire[r, c] > 0.5 and self.fuel[r, c] > 0:
                    for dc, dr in [(d[1], d[0]) for d in dirs]:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.GRID and 0 <= nc < self.GRID): continue
                        if self.fire[nr, nc] > 0.5 or not self.passable[nr, nc]: continue
                        wind_align = dc * self.wind_dir[0] + dr * self.wind_dir[1]
                        wf = 1 + wind_align * self.wind_spd * 0.4
                        p = rate * self.fuel[nr, nc] * wf
                        if np.random.random() < p:
                            new_fire[nr, nc] = 1.0
                            if self.belief[nr, nc] != B_UNKNOWN:
                                self.belief[nr, nc] = B_FIRE
                            # Mark survivor as lost
                            for sv in self.survivors:
                                if sv['row'] == nr and sv['col'] == nc and sv['status'] not in ('rescued', 'lost'):
                                    sv['status'] = 'lost'
        self.fire = new_fire

    @property
    def fire_pct(self) -> float:
        return float(np.sum(self.fire > 0.5) / self.GRID**2 * 100)

    @property
    def explore_pct(self) -> float:
        return float(np.sum(self.belief != B_UNKNOWN) / self.GRID**2 * 100)


# ── A* pathfinder ─────────────────────────────────────────────────────────────
def astar(world: TerrainWorld, sc: int, sr: int, gc: int, gr: int,
          avoid_fire: bool = True) -> List[Tuple[int, int]]:
    """A* on terrain grid. Returns list of (col, row) steps."""
    G = world.GRID
    h = lambda c, r: abs(c - gc) + abs(r - gr)
    open_list = [(h(sc, sr), 0, sc, sr, [(sc, sr)])]
    visited = set()
    while open_list:
        _, g, c, r, path = heapq.heappop(open_list)
        if (c, r) in visited: continue
        visited.add((c, r))
        if c == gc and r == gr: return path
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < G and 0 <= nr < G): continue
            if not world.passable[nr, nc]: continue
            if avoid_fire and world.fire[nr, nc] > 0.5: continue
            if (nc, nr) in visited: continue
            ng = g + 1
            heapq.heappush(open_list, (ng + h(nc, nr), ng, nc, nr, path + [(nc, nr)]))
    return [(sc, sr)]


# ── Fairness reward (from original REU paper) ─────────────────────────────────
def fairness_reward(performances: List[float], eps: float = 1e-7) -> List[float]:
    """
    r_t^i = (ε + |e_t^i / ē_t − 1|) / ē_t

    Parameters
    ----------
    performances : list of float — e_t^i for each agent
    eps          : numerical stability constant ε

    Returns
    -------
    rewards : list of float — lower = fairer
    """
    e_bar = max(np.mean(performances), eps)
    return [(eps + abs(e / e_bar - 1)) / e_bar for e in performances]


# ── Drone swarm (UAV exploration) ─────────────────────────────────────────────
class DroneSwarm:
    """
    Fleet of UAV scouts. Policies: MAPPO, Boids+RL, Random.

    Key method: step(world) — advances all drones one timestep,
    scanning terrain and updating world.belief.
    """
    SCAN_RADIUS = 4

    def __init__(self, n: int, world: TerrainWorld, policy: str = MAPPO_DRONE,
                 spawn: str = 'random'):
        self.policy = policy
        self.drones = []
        G = world.GRID
        mid = G // 2
        occupied = {(sv['col'], sv['row']) for sv in world.survivors}

        for i in range(n):
            if spawn == 'station':
                c, r = mid + i - n // 2, mid - 3
                c = max(1, min(G-2, c))
            else:
                for _ in range(200):
                    c, r = world.rng.int(2, G-2), world.rng.int(2, G-2)
                    if world.passable[r, c] and (c, r) not in occupied:
                        occupied.add((c, r))
                        break
            self.drones.append({
                'id': i, 'col': c, 'row': r,
                'tc': c, 'tr': r,
                'explored': 0, 'handoffs': 0,
                'visit_mask': np.zeros((G, G), dtype=bool),
                'vx': 0.0, 'vz': 0.0,  # for boids
            })

    def step(self, world: TerrainWorld) -> int:
        """Move all drones one step. Returns total new cells discovered."""
        total_new = 0
        for d in self.drones:
            if d['col'] == d['tc'] and d['row'] == d['tr']:
                total_new += self._scan(d, world)
                nc, nr = self._action(d, world)
                d['tc'], d['tr'] = nc, nr
            else:
                dc = int(np.sign(d['tc'] - d['col']))
                dr = int(np.sign(d['tr'] - d['row']))
                nc, nr = d['col'] + dc, d['row'] + dr
                G = world.GRID
                if 0 < nc < G-1 and 0 < nr < G-1 and world.fire[nr, nc] < 0.5:
                    d['col'], d['row'] = nc, nr
        return total_new

    def _scan(self, d: dict, world: TerrainWorld) -> int:
        """Drone scans area, updates belief map. Returns new cell count."""
        G, R = world.GRID, self.SCAN_RADIUS
        new = 0
        for dr in range(-R, R+1):
            for dc in range(-R, R+1):
                if dr**2 + dc**2 > R**2 + 1: continue
                nr, nc = d['row'] + dr, d['col'] + dc
                if not (0 <= nr < G and 0 <= nc < G): continue
                if world.belief[nr, nc] == B_UNKNOWN:
                    if world.fire[nr, nc] > 0.5:
                        world.belief[nr, nc] = B_FIRE
                    else:
                        sv = next((s for s in world.survivors
                                   if s['col'] == nc and s['row'] == nr
                                   and s['status'] == 'unknown'), None)
                        if sv:
                            world.belief[nr, nc] = B_SURVIVOR
                            sv['status'] = 'discovered'
                            d['handoffs'] += 1
                        else:
                            world.belief[nr, nc] = B_CLEAR
                    new += 1
                if not d['visit_mask'][nr, nc]:
                    d['visit_mask'][nr, nc] = True
                    d['explored'] += 1
        return new

    def _action(self, d: dict, world: TerrainWorld) -> Tuple[int, int]:
        G = world.GRID
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        if self.policy == RANDOM_DRONE:
            opts = [(d['col']+dc, d['row']+dr) for dc,dr in dirs
                    if 0 < d['col']+dc < G-1 and 0 < d['row']+dr < G-1
                    and world.fire[d['row']+dr, d['col']+dc] < 0.5]
            return world.rng.choose(opts) if opts else (d['col'], d['row'])

        if self.policy == BOIDS_DRONE:
            return self._boids_action(d, world)

        # MAPPO: score each candidate position
        best, best_score = (d['col'], d['row']), -np.inf
        for dc, dr in dirs:
            nc, nr = d['col'] + dc, d['row'] + dr
            if not (0 < nc < G-1 and 0 < nr < G-1): continue
            if world.fire[nr, nc] > 0.5: continue
            if not world.passable[nr, nc] and world.t_type[nr, nc] == 'building': continue

            score = 0.0
            R = self.SCAN_RADIUS
            for sr in range(-R, R+1):
                for sc2 in range(-R, R+1):
                    if sr**2 + sc2**2 > R**2 + 1: continue
                    tnr, tnc = nr + sr, nc + sc2
                    if not (0 <= tnr < G and 0 <= tnc < G): continue
                    if world.belief[tnr, tnc] == B_UNKNOWN: score += 1.0
                    if d['visit_mask'][tnr, tnc]:            score -= 0.25

            # Coordination penalty (drives drones apart without communication)
            for od in self.drones:
                if od['id'] == d['id']: continue
                dist = abs(od['col'] - nc) + abs(od['row'] - nr)
                if dist < 4: score -= 1.5 * (4 - dist)

            score += world.rng.float(-0.1, 0.1)  # small exploration noise
            if score > best_score:
                best_score = score
                best = (nc, nr)
        return best

    def _boids_action(self, d: dict, world: TerrainWorld) -> Tuple[int, int]:
        G = world.GRID
        sep, ali, coh, n = [0,0], [0,0], [0,0], 0
        R = 5
        for od in self.drones:
            if od['id'] == d['id']: continue
            dd = abs(od['col']-d['col']) + abs(od['row']-d['row'])
            if dd < R:
                sep[0] -= (od['col']-d['col']) / (dd + 1)
                sep[1] -= (od['row']-d['row']) / (dd + 1)
                ali[0] += od['vx']; ali[1] += od['vz']
                coh[0] += od['col']; coh[1] += od['row']
                n += 1
        if n:
            ali[0] /= n; ali[1] /= n
            coh[0] = coh[0]/n - d['col']; coh[1] = coh[1]/n - d['row']

        # Coverage gradient
        best_dir, best_e = [0, 0], -1
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
            nc, nr = d['col']+dc, d['row']+dr
            if not (0 < nc < G-1 and 0 < nr < G-1): continue
            if world.fire[nr, nc] > 0.5: continue
            e = sum(world.belief[nr+sr, nc+sc] == B_UNKNOWN
                    for sr in range(-3,4) for sc in range(-3,4)
                    if 0<=nr+sr<G and 0<=nc+sc<G)
            if e > best_e: best_e = e; best_dir = [dc, dr]

        vx = sep[0]*1.5 + ali[0]*0.5 + coh[0]*0.3 + best_dir[0]*2
        vz = sep[1]*1.5 + ali[1]*0.5 + coh[1]*0.3 + best_dir[1]*2
        d['vx'], d['vz'] = vx * 0.5, vz * 0.5
        nc = d['col'] + int(np.sign(vx))
        nr = d['row'] + int(np.sign(vz))
        if 0 < nc < G-1 and 0 < nr < G-1 and world.fire[nr, nc] < 0.5:
            return nc, nr
        return d['col'], d['row']


# ── UGV fleet (ground rescue robots) ─────────────────────────────────────────
class UGVFleet:
    """
    Fleet of ground rescue robots. Policies: QMIX, Greedy, Random.
    """
    def __init__(self, n: int, world: TerrainWorld, policy: str = QMIX_UGV,
                 spawn: str = 'random'):
        self.policy = policy
        self.ugvs   = []
        G = world.GRID
        mid = G // 2
        occupied = {(sv['col'], sv['row']) for sv in world.survivors}

        for i in range(n):
            if spawn == 'station':
                c, r = mid + i - n//2, mid
            else:
                for _ in range(200):
                    c, r = world.rng.int(2, G-2), world.rng.int(2, G-2)
                    if world.passable[r, c] and (c, r) not in occupied:
                        occupied.add((c, r)); break
            self.ugvs.append({
                'id': i, 'col': c, 'row': r,
                'target': None, 'path': [], 'path_idx': 0,
                'rescued': 0, 'dist': 0,
            })

    def step(self, world: TerrainWorld) -> int:
        """Move all UGVs one step. Returns survivors rescued this step."""
        rescued_this_step = 0
        for u in self.ugvs:
            # Check arrival / rescue
            if u['target']:
                sv = u['target']
                if u['col'] == sv['col'] and u['row'] == sv['row'] and sv['status'] == 'discovered':
                    sv['status'] = 'rescued'
                    sv['assigned_to'] = None
                    u['target'] = None
                    u['rescued'] += 1
                    rescued_this_step += 1
                    world.belief[sv['row'], sv['col']] = B_RESCUED
                    u['path'] = []
            # Assign new target
            if not u['target']:
                t = self._pick_target(u, world)
                if t:
                    t['assigned_to'] = u['id']
                    u['target'] = t
                    u['path'] = astar(world, u['col'], u['row'], t['col'], t['row'])
                    u['path_idx'] = 0
            # Move along path
            if u['path'] and u['path_idx'] < len(u['path']) - 1:
                u['path_idx'] += 1
                nc, nr = u['path'][u['path_idx']]
                if world.fire[nr, nc] > 0.5:
                    if u['target']: u['target']['assigned_to'] = None
                    u['target'] = None; u['path'] = []
                else:
                    u['dist'] += abs(nc - u['col']) + abs(nr - u['row'])
                    u['col'], u['row'] = nc, nr
        return rescued_this_step

    def _pick_target(self, u: dict, world: TerrainWorld) -> Optional[dict]:
        avail = [sv for sv in world.survivors
                 if sv['status'] == 'discovered' and sv['assigned_to'] is None]
        if not avail: return None
        if self.policy == RANDOM_UGV:
            return world.rng.choose(avail)
        if self.policy == GREEDY_UGV:
            return min(avail, key=lambda sv: abs(sv['col']-u['col']) + abs(sv['row']-u['row']))
        # QMIX with fairness term
        eps = 1e-7
        loads = [uu['rescued'] for uu in self.ugvs]
        avg   = max(np.mean(loads), eps)
        e_i   = 1.0 / (u['rescued'] + eps)
        e_bar = 1.0 / (avg + eps)
        fair_pen = (eps + abs(e_i / e_bar - 1)) / e_bar
        def q(sv):
            dist = abs(sv['col']-u['col']) + abs(sv['row']-u['row'])
            overlap = sum(1 for uu in self.ugvs if uu['target'] == sv)
            idle_boost = sum(1 for uu in self.ugvs if not uu['target']) * 0.4
            return -dist * 0.5 - 1.5 * fair_pen - overlap * 10 + idle_boost
        return max(avail, key=q)


# ── Fire truck fleet ──────────────────────────────────────────────────────────
class FireTruckFleet:
    """
    Fleet of fire suppression vehicles. Greedy nearest-fire targeting.
    Suppresses a radius around target when arrived.
    """
    SUPPRESS_RADIUS = 2

    def __init__(self, n: int, world: TerrainWorld, spawn: str = 'random'):
        self.trucks = []
        G = world.GRID
        mid = G // 2
        occupied = {(sv['col'], sv['row']) for sv in world.survivors}

        for i in range(n):
            if spawn == 'station':
                c, r = mid + i - n//2, mid + 3
            else:
                for _ in range(200):
                    c, r = world.rng.int(2, G-2), world.rng.int(2, G-2)
                    if world.passable[r, c] and (c, r) not in occupied:
                        occupied.add((c, r)); break
            self.trucks.append({
                'id': i, 'col': c, 'row': r,
                'target_fire': None, 'path': [], 'path_idx': 0,
                'suppressed': 0, 'dist': 0,
            })
        self.total_suppressed = 0

    def step(self, world: TerrainWorld) -> int:
        """Move all trucks one step. Returns cells suppressed this step."""
        suppressed = 0
        for ft in self.trucks:
            # Check arrival
            if ft['target_fire']:
                tr, tc = ft['target_fire']
                if ft['col'] == tc and ft['row'] == tr:
                    G = world.GRID
                    R = self.SUPPRESS_RADIUS
                    for dr in range(-R, R+1):
                        for dc in range(-R, R+1):
                            nr, nc = tr+dr, tc+dc
                            if 0 <= nr < G and 0 <= nc < G and world.fire[nr, nc] > 0.5:
                                world.fire[nr, nc] = 0.0
                                world.fuel[nr, nc] *= 0.1
                                if world.belief[nr, nc] == B_FIRE:
                                    world.belief[nr, nc] = B_CLEAR
                                ft['suppressed'] += 1
                                suppressed += 1
                    ft['target_fire'] = None; ft['path'] = []
            # Assign new target
            if not ft['target_fire']:
                t = self._pick_fire(ft, world)
                if t:
                    ft['target_fire'] = t
                    ft['path'] = astar(world, ft['col'], ft['row'], t[1], t[0], avoid_fire=False)
                    ft['path_idx'] = 0
            # Move
            if ft['path'] and ft['path_idx'] < len(ft['path']) - 1:
                ft['path_idx'] += 1
                nc, nr = ft['path'][ft['path_idx']]
                ft['dist'] += abs(nc - ft['col']) + abs(nr - ft['row'])
                ft['col'], ft['row'] = nc, nr
        self.total_suppressed += suppressed
        return suppressed

    def _pick_fire(self, ft: dict, world: TerrainWorld) -> Optional[Tuple[int,int]]:
        G = world.GRID
        best, best_d = None, float('inf')
        for r in range(G):
            for c in range(G):
                if world.fire[r, c] > 0.5:
                    already = any(t['target_fire'] == (r, c) and t['id'] != ft['id']
                                  for t in self.trucks)
                    if already: continue
                    d = abs(r - ft['row']) + abs(c - ft['col'])
                    if d < best_d: best_d = d; best = (r, c)
        return best


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(seed: int = 42, biome: str = 'mountain',
                n_drones: int = 4, n_ugv: int = 3, n_firetruck: int = 2,
                drone_algo: str = MAPPO_DRONE, ugv_algo: str = QMIX_UGV,
                fire_rate: int = 4, max_steps: int = 500,
                spawn: str = 'random',
                snapshot_steps: List[int] = None) -> Dict:
    """
    Run a single simulation episode.

    Returns
    -------
    dict with keys:
      steps, rescued, lost, total_survivors,
      explored_pct, fire_pct, collabs, gini, score, grade,
      explore_history, fire_history, rescue_times, ugv_loads,
      belief_snapshots (if snapshot_steps provided)
    """
    world  = TerrainWorld(grid_size=24, biome=biome, seed=seed, fire_rate=fire_rate)
    drones = DroneSwarm(n_drones, world, drone_algo, spawn)
    ugvs   = UGVFleet(n_ugv, world, ugv_algo, spawn)
    trucks = FireTruckFleet(n_firetruck, world, spawn)

    total_sv      = len(world.survivors)
    explore_hist  = []
    fire_hist     = []
    rescue_times  = []
    belief_snaps  = {}
    step          = 0

    while step < max_steps:
        step += 1
        drones.step(world)
        rescued_now = ugvs.step(world)
        trucks.step(world)
        world.spread_fire()

        explore_hist.append(world.explore_pct)
        fire_hist.append(world.fire_pct)

        if rescued_now > 0:
            rescue_times.extend([step] * rescued_now)

        if snapshot_steps and step in snapshot_steps:
            belief_snaps[step] = world.belief.copy()

        sv_done  = sum(1 for sv in world.survivors if sv['status'] == 'rescued')
        sv_lost  = sum(1 for sv in world.survivors if sv['status'] == 'lost')
        if sv_done + sv_lost >= total_sv or world.fire_pct > 75:
            break

    rescued = sum(1 for sv in world.survivors if sv['status'] == 'rescued')
    lost    = sum(1 for sv in world.survivors if sv['status'] == 'lost')
    collabs = sum(d['handoffs'] for d in drones.drones)
    loads   = [u['rescued'] for u in ugvs.ugvs]
    g       = _gini(loads)

    score = (rescued / total_sv * 50) + (world.explore_pct / 100 * 25) + (1 - world.fire_pct / 100) * 25
    grade = 'S' if score >= 90 else 'A' if score >= 75 else 'B' if score >= 60 else 'C' if score >= 45 else 'D'

    result = {
        'steps':           step,
        'rescued':         rescued,
        'lost':            lost,
        'total_survivors': total_sv,
        'explored_pct':    world.explore_pct,
        'fire_pct':        world.fire_pct,
        'collabs':         collabs,
        'gini':            g,
        'score':           score,
        'grade':           grade,
        'explore_history': explore_hist,
        'fire_history':    fire_hist,
        'rescue_times':    rescue_times,
        'ugv_loads':       loads,
        'drone_loads':     [d['explored'] for d in drones.drones],
    }
    if snapshot_steps:
        result['belief_snapshots'] = belief_snaps
    return result


def compare_configurations(seeds: List[int], **kwargs) -> List[Dict]:
    """
    Run run_episode() across multiple seeds with the same kwargs.
    Returns list of result dicts.
    """
    return [run_episode(seed=s, **kwargs) for s in seeds]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _gini(values: List[float]) -> float:
    if not values or not any(values): return 0.0
    a = sorted(values)
    n = len(a)
    s = sum(a)
    if s == 0: return 0.0
    num = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(a))
    return max(0.0, num / (n * s))


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Running self-test...')
    for biome in ['mountain', 'city']:
        r = run_episode(seed=42, biome=biome, n_drones=4, n_ugv=3,
                        n_firetruck=2, drone_algo=MAPPO_DRONE,
                        ugv_algo=QMIX_UGV, max_steps=100)
        print(f'  {biome:10s}: rescued={r["rescued"]}/{r["total_survivors"]} '
              f'explored={r["explored_pct"]:.1f}% fire={r["fire_pct"]:.1f}% '
              f'grade={r["grade"]}')
    print('Self-test passed ✅')
