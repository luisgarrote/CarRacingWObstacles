import math
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d import car_racing as base_cr
from car_racing import CarRacing as BaseCarRacing
from car_dynamics import Car

try:
    import Box2D
    from Box2D.b2 import fixtureDef, polygonShape
except ImportError as e:
    from gymnasium.error import DependencyNotInstalled

    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` '
        'followed by `pip install "gymnasium[box2d]"`'
    ) from e


# Reuse constants from the base implementation
PLAYFIELD = base_cr.PLAYFIELD
TRACK_WIDTH = base_cr.TRACK_WIDTH
FPS = base_cr.FPS


class CarRacingObstacles(BaseCarRacing):
    """
    CarRacingObstacles-v3

    Extension of Gymnasium's CarRacing-v3 environment with:

      * Static obstacles on or near the road
      * Dynamic obstacles moving across the track
      * Large "mountains" as off-road terrain near the track
      * Ghost car ("shadow") that follows the track, offset behind the player
      * Reward shaping + termination rules for collisions / being stopped / off-road

    Notes
    -----
    - The RL agent still controls ONLY the main car (steer, gas, brake).
    - Obstacles and ghost car are just environment features; the base CarRacing
      reward is adjusted but the original tile reward mechanism is still there.
    """

    metadata = BaseCarRacing.metadata.copy()

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
        n_static_obstacles: int = 0,
        n_dynamic_obstacles: int = 0,
        n_mountains: int = 0,
        use_ghost: bool = False,
        obstacle_scale: float = 0.1,
    ):
        super().__init__(
            render_mode=render_mode,
            verbose=verbose,
            lap_complete_percent=lap_complete_percent,
            domain_randomize=domain_randomize,
            continuous=continuous,
        )

        # Configuration
        self.n_static_obstacles = n_static_obstacles
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.n_mountains = n_mountains
        self.use_ghost = use_ghost
        self.obstacle_scale = float(obstacle_scale)

        # Extra Box2D bodies
        self.static_obstacles: List[Box2D.b2Body] = []
        self.dynamic_obstacles: List[Dict[str, Any]] = []  # dict with body + motion params
        self.mountains: List[Box2D.b2Body] = []
        self.trees: List[Box2D.b2Body] = []

        # Ghost car
        self.ghost_car: Optional[Car] = None
        self.ghost_progress: float = 0.0  # progress in "tiles"
        self.ghost_speed_tiles_per_sec: float = 3.0
        self.ghost_offset = -5.0  # ghost starts a few meters behind the car
        self._ghost_wheel_local: List[Tuple[float, float, Box2D.b2Body]] = []
        # Cached track centerline for distance computations
        self._track_x = None
        self._track_y = None

        # ---- rule / reward parameters ----
        # Static obstacle collision: big negative + episode end
        self.static_collision_penalty = 200.0
        self.car_collision_radius = 2.5  # approx radius of the car in world units

        # "Stopped" logic (anywhere)
        self.still_speed_threshold = 0.5       # below this, car considered "stopped"
        self.still_penalty_start = 3.0         # seconds before we start penalizing
        self.still_terminate_time = 5.0        # seconds before auto-terminate
        self.still_penalty_per_sec = 1.0       # penalty per second after 3 s
        self.still_terminate_penalty = 50.0    # extra penalty when we kill the episode

        # Off-road (green or mountain) logic
        self.offroad_penalty_start = 3.0       # seconds off-road before penalty
        self.offroad_terminate_time = 5.0      # seconds off-road before termination
        self.offroad_penalty_per_sec = 1.0     # per-second penalty after 3 s
        self.offroad_terminate_penalty = 50.0  # extra negative when terminating

        # reward when moving on track
        self.moving_reward_scale = 5.0         # tune this
        self.min_moving_speed = 1.0            # only reward if speed above this

        # internal state
        self.still_time = 0.0                  # how long we've been stopped
        self.offroad_time = 0.0                # how long we've been off-road (green or mountain)
        self.was_on_start = False


        self.verbose=True
    # --------------------------------------------------------------------------
    # Clean-up helpers
    # --------------------------------------------------------------------------
    def _destroy_extras(self) -> None:
        if getattr(self, "world", None) is not None:
            for b in self.static_obstacles:
                try:
                    self.world.DestroyBody(b)
                except Exception:
                    pass
            for d in self.dynamic_obstacles:
                body = d.get("body", None)
                if body is not None:
                    try:
                        self.world.DestroyBody(body)
                    except Exception:
                        pass
            for b in self.mountains:
                try:
                    self.world.DestroyBody(b)
                except Exception:
                    pass


        for b in self.trees:
            try:
                self.world.DestroyBody(b)
            except:
                pass
        self.trees.clear()
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        self.mountains.clear()

        if self.ghost_car is not None:
            try:
                self.ghost_car.destroy()
            except Exception:
                pass
            self.ghost_car = None

    # --------------------------------------------------------------------------
    # Reset: call parent, then add mountains, obstacles, ghost
    # --------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        # Destroy previous extras first
        self._destroy_extras()

        obs, info = super().reset(seed=seed, options=options)

        # Build cached track centerline arrays
        if hasattr(self, "track") and len(self.track) > 0:
            self._track_x = np.array([t[2] for t in self.track], dtype=np.float32)
            self._track_y = np.array([t[3] for t in self.track], dtype=np.float32)
        else:
            self._track_x = None
            self._track_y = None

        # Reset timers
        self.still_time = 0.0
        self.offroad_time = 0.0

        # Create extras
        self._create_mountains()
        self._create_trees()

        self._create_road_obstacles()
        if self.use_ghost:
            self._create_ghost()

        return obs, info




    def _create_tree_body(self, x: float, y: float, size: float) -> Box2D.b2Body:
        """
        Create a simple X-shaped tree:
           green leaves in an X pattern
           yellow dot center
        """
        half = size

        # Two diagonal rectangles making an 'X'
        verts1 = [
            (-half, -half*0.3),
            (half, half*0.3),
            (half, half*0.3 + 0.3),
            (-half, -half*0.3 - 0.3),
        ]
        verts2 = [
            (-half, half*0.3),
            (half, -half*0.3),
            (half, -half*0.3 - 0.3),
            (-half, half*0.3 + 0.3),
        ]

        body = self.world.CreateStaticBody(position=(x, y))
        body.is_tree = True
        body.collision_radius = size * 1.2

        body.CreateFixture(fixtureDef(shape=polygonShape(vertices=verts1), density=0.1))
        body.CreateFixture(fixtureDef(shape=polygonShape(vertices=verts2), density=0.1))

        self.trees.append(body)
        return body


    def _create_trees(self) -> None:
        if not hasattr(self, "track") or len(self.track) == 0:
            return

        num = max(3, self.n_mountains)  # same scale as mountains
        indices = np.linspace(0, len(self.track) - 1, num=num, endpoint=False, dtype=int)

        for idx in indices:
            _, beta, x, y = self.track[idx]

            for _ in range(15):
                perp = beta + np.pi / 2.0
                dist = TRACK_WIDTH * 3.5
                tx = x + dist * math.cos(perp)
                ty = y + dist * math.sin(perp)
                size = TRACK_WIDTH * self.np_random.uniform(0.2, 0.3)

                if abs(tx) > PLAYFIELD or abs(ty) > PLAYFIELD:
                    continue

                # ensure trees don't overlap road
                clear = size + 0.15 * TRACK_WIDTH
                d = self._min_distance_to_track_center(tx, ty)
                if d >= clear:
                    self._create_tree_body(tx, ty, size)
                    break


    # --------------------------------------------------------------------------
    # Track helper
    # --------------------------------------------------------------------------
    def _track_heading(self, idx):
        prev_idx = (idx - 1) % len(self.track)
        next_idx = (idx + 1) % len(self.track)

        _, _, x_prev, y_prev = self.track[prev_idx]
        _, _, x_next, y_next = self.track[next_idx]

        return math.atan2(y_next - y_prev, x_next - x_prev)

    # --------------------------------------------------------------------------
    # Collision / on-track helpers
    # --------------------------------------------------------------------------
    def _check_static_collision(self) -> bool:
        """Check collision with static obstacles ONLY (not mountains)."""
        if self.car is None:
            return False

        cx, cy = self.car.hull.position
        car_r = self.car_collision_radius

        # static obstacles count as "hard" collisions
        for b in self.static_obstacles:
            bx, by = b.position
            br = getattr(b, "collision_radius", 0.0)
            dx = cx - bx
            dy = cy - by
            if dx * dx + dy * dy <= (car_r + br) ** 2:
                return True
        return False

    def _closest_track_point(self, x: float, y: float):
        if self._track_x is None or self._track_y is None or len(self._track_x) == 0:
            return None
        dx = self._track_x - x
        dy = self._track_y - y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        dist = float(math.sqrt(d2[idx]))
        return idx, dist

    def _on_track(self, x: float, y: float, margin: float = TRACK_WIDTH) -> bool:
        res = self._closest_track_point(x, y)
        if res is None:
            return False
        _, dist = res
        return dist <= margin

    def _on_hazard(self, x: float, y: float) -> bool:
        """Check if car is on top of (or inside) a mountain region."""
        car_r = self.car_collision_radius

        # mountains
        for m in self.mountains:
            mx, my = m.position
            mr = getattr(m, "collision_radius", 0.0)
            if (x - mx)**2 + (y - my)**2 <= (mr + car_r)**2:
                return True

        # trees behave like mountains
        for t in self.trees:
            tx, ty = t.position
            tr = getattr(t, "collision_radius", 0.0)
            if (x - tx)**2 + (y - ty)**2 <= (tr + car_r)**2:
                return True

        return False

    def _regenerate_track_tiles(self):
        """Allow tiles to give +1000/N again, for multiple laps."""
        for t in self.road:
            t.road_visited = False
        self.tile_visited_count = 0
        self.new_lap = False  # reset flag set by contact listener

    # --------------------------------------------------------------------------
    # Step override: add rules & extra rewards
    # --------------------------------------------------------------------------
    def step(self, action):
        # Run original CarRacing step (physics + base reward logic)
        obs, reward, terminated, truncated, info = super().step(action)

        dt = 1.0 / FPS

        # ---- 1) Static collisions: big penalty + terminate ----
        hit_static = self._check_static_collision()
        if hit_static:
            reward -= self.static_collision_penalty
            terminated = True
            info["collision_static"] = True

        # Car state
        x, y = self.car.hull.position
        speed = float(np.linalg.norm(self.car.hull.linearVelocity))

        # ---- 2) Stuck detection (3s penalty, 5s terminate) ----
        if speed < self.still_speed_threshold:
            self.still_time += dt
        else:
            self.still_time = 0.0

        if self.still_time > self.still_penalty_start:
            # linear penalty in time after 3 seconds
            reward -= self.still_penalty_per_sec * dt

        if self.still_time >= self.still_terminate_time:
            terminated = True
            reward -= self.still_terminate_penalty
            info["stuck"] = True

        # ---- 3) Off-road (green or mountain) timer: 3–5s logic ----
        on_track = self._on_track(x, y)
        hazard = self._on_hazard(x, y)
        offroad = (not on_track) or hazard

        if offroad:
            self.offroad_time += dt
        else:
            self.offroad_time = 0.0

        if self.offroad_time > self.offroad_penalty_start:
            # penalize staying on green / mountains even if car is stopped
            reward -= self.offroad_penalty_per_sec * dt

        if self.offroad_time >= self.offroad_terminate_time:
            terminated = True
            reward -= self.offroad_terminate_penalty
            info["offroad_too_long"] = True

        # ---- 4) Per-step reward when on track and moving ----
        if speed > self.min_moving_speed and on_track:
            # reward ~ speed * time, tune scale
            reward += self.moving_reward_scale * speed * dt
            info["moving_on_track"] = True
        else:
            info["moving_on_track"] = False

        # ---- 5) Handle laps: regenerate tiles but do NOT terminate ----
        tile_idx = self.tile_idx
        if (tile_idx==0):
            if not self.was_on_start and self.tile_visited_count > 5:

                # We've completed a lap: regenerate tiles
                self._regenerate_track_tiles()
                # keep episode running
                terminated = False
                info["lap_finished"] = True  # keep flag for students / logging
        else:
            self.was_on_start = False
 
        # ---- 6) Update dynamic obstacles and ghost car after physics ----
        self._update_dynamic_obstacles()
        self._update_ghost()

        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------------
    # Mountain creation (off-road, near track, visible)
    # --------------------------------------------------------------------------
    def _create_mountain_body(self, x: float, y: float, radius: float) -> Box2D.b2Body:
        """
        Create a mountain as a static polygonal blob. We use <= 12 vertices to
        obey Box2D's vertex limit.
        """
        num_vertices = 10  # <= 16 for Box2D
        verts = []
        for k in range(num_vertices):
            ang = 2.0 * math.pi * k / num_vertices
            r = radius * self.np_random.uniform(0.8, 1.15)
            verts.append((x + r * math.cos(ang), y + r * math.sin(ang)))

        body = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=verts),
                density=0.1,
            )
        )
        body.is_mountain = True
        body.road_friction = 0.5
        body.collision_radius = radius
        self.mountains.append(body)
        return body

    def _min_distance_to_track_center(self, x: float, y: float) -> float:
        """
        Min Euclidean distance from (x,y) to the track centerline points.
        """
        if self._track_x is None or self._track_y is None or len(self._track_x) == 0:
            return float("inf")
        dx = self._track_x - x
        dy = self._track_y - y
        d2 = dx * dx + dy * dy
        return float(np.sqrt(d2.min()))

    def _create_mountains(self) -> None:
        """
        Place large mountain blobs off the track, but close enough to see.
        """
        if not hasattr(self, "track") or len(self.track) == 0:
            return

        #print("here1")
        num = self.n_mountains
        if num <= 0:
            return

        #print("here2")
        indices = np.linspace(0, len(self.track) - 1, num=num, endpoint=False, dtype=int)

        for idx in indices:
            _, beta, x, y = self.track[idx]

            placed = False
            for _ in range(30):
                # perpendicular direction to track
                perp_angle = beta + math.pi / 2.0

                # Bring mountains closer to the road edge so they're visible
                # and not crazy huge
                dist = TRACK_WIDTH * self.np_random.uniform(8, 10)
                mx = x + dist * math.cos(perp_angle)
                my = y + dist * math.sin(perp_angle)
                radius = TRACK_WIDTH * self.np_random.uniform(0.8, 1.4)

                # keep within playfield
                if abs(mx) > PLAYFIELD or abs(my) > PLAYFIELD:
                    continue

                # require some separation from centerline, but less strict
                clearance_needed = radius + 0.2 * TRACK_WIDTH
                d = self._min_distance_to_track_center(mx, my)
                #print(clearance_needed, d)
                if d >= clearance_needed:
                    self._create_mountain_body(mx, my, radius)
                    placed = True
                    break

            if not placed and self.verbose:
                print(f"[CarRacingObstacles] Could not place mountain around tile {idx}")
            else:
                print(f"[CarRacingObstacles] Placed mountain around tile {idx}")


    # --------------------------------------------------------------------------
    # Obstacles on/near road
    # --------------------------------------------------------------------------
    def _create_static_obstacle(self, x: float, y: float, size: float):
        half = size
        verts = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ]
        body = self.world.CreateStaticBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=verts),
                density=0.5,
            ),
        )
        body.is_obstacle_static = True
        # approximate bounding radius for collision checks
        body.collision_radius = 0.25*size
        self.static_obstacles.append(body)
        return body

    def _create_dynamic_obstacle(self, x: float, y: float, size: float, beta: float) -> Dict[str, Any]:
        """
        Kinematic obstacle moving back-and-forth across the road.
        We place it with a short motion along a direction perpendicular to the track.
        """
        half = size
        verts = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ]

        body = self.world.CreateKinematicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=verts),
                density=0.5,
                friction=0.5,
            ),
        )
        body.is_obstacle_dynamic = True

        # Motion parameters
        # Movement along perpendicular direction to heading
        perp_angle = beta + math.pi / 2.0
        amp = TRACK_WIDTH * self.np_random.uniform(0.8, 1.5)
        speed = self.np_random.uniform(0.5, 1.2)  # cycles / second
        phase = self.np_random.uniform(0.0, 2.0 * math.pi)

        dyn = {
            "body": body,
            "center": np.array([x, y], dtype=np.float32),
            "perp_angle": float(perp_angle),
            "amp": float(amp),
            "speed": float(speed),
            "phase": float(phase),
        }
        self.dynamic_obstacles.append(dyn)
        return dyn

    def _create_road_obstacles(self) -> None:
        """
        Sample a set of tiles along the track and place static/dynamic obstacles.
        We ensure they are not too close to the start.
        """
        if not hasattr(self, "track") or len(self.track) < 10:
            return

        total_tiles = len(self.track)

        # Static obstacles
        n_static = max(0, min(self.n_static_obstacles, total_tiles // 15))
        n_dynamic = max(0, min(self.n_dynamic_obstacles, total_tiles // 20))

        # Skip the first part of the lap so that the car can start safely
        valid_indices = np.arange(total_tiles // 6, total_tiles - total_tiles // 6)
        if len(valid_indices) == 0:
            return

        # Sample static and dynamic indices without replacement if possible
        self.np_random.shuffle(valid_indices)
        static_indices = valid_indices[:n_static]
        dynamic_indices = valid_indices[n_static:n_static + n_dynamic]

        base_size = TRACK_WIDTH * 0.7 * self.obstacle_scale

        # Place static obstacles
        for idx in static_indices:
            _, beta, x, y = self.track[int(idx)]
            # Shift slightly towards left or right side of the road
            side = self.np_random.choice([-1.0, 1.0])
            offset = TRACK_WIDTH * self.np_random.uniform(0, 1) * side
            nx = math.cos(beta)
            ny = math.sin(beta)
            ox = x + offset * nx
            oy = y + offset * ny
            self._create_static_obstacle(ox, oy, base_size)
  #ver isto
        # Place dynamic obstacles
        for idx in dynamic_indices:
            _, beta, x, y = self.track[int(idx)]
            # Put them roughly in center of road
            self._create_dynamic_obstacle(x, y, base_size * 0.8, beta)

    # --------------------------------------------------------------------------
    # Dynamic obstacle motion
    # --------------------------------------------------------------------------
    def _update_dynamic_obstacles(self) -> None:
        """
        Move kinematic dynamic obstacles along their sinusoidal paths.
        """
        if not self.dynamic_obstacles:
            return

        t = getattr(self, "t", 0.0)  # simulation time from base env
        for dyn in self.dynamic_obstacles:
            body = dyn["body"]
            cx, cy = dyn["center"]
            amp = dyn["amp"]
            speed = dyn["speed"]
            phase = dyn["phase"]
            ang = dyn["perp_angle"]

            # s(t) in [-1,1]
            s = math.sin(2.0 * math.pi * speed * t + phase)
            dx = amp * s * math.cos(ang)
            dy = amp * s * math.sin(ang)

            body.position = (cx + dx, cy + dy)
            body.linearVelocity = (0.0, 0.0)
            body.angularVelocity = 0.0

    # --------------------------------------------------------------------------
    # Ghost car ("shadow")
    # --------------------------------------------------------------------------
    def _create_ghost(self) -> None:
        """
        Create a ghost car that follows the track centerline, purely for visuals.
        It starts a few meters *behind* the player along the track direction,
        and we cache wheel offsets so wheels move with the hull.
        """
        if not hasattr(self, "track") or len(self.track) == 0:
            return

        _, beta0, x0, y0 = self.track[0]
        offset = self.ghost_offset  # negative = behind

        # create ghost car slightly behind the start tile center
        self.ghost_car = Car(
            self.world,
            beta0,
            x0 + offset * math.cos(beta0),
            y0 + offset * math.sin(beta0),
        )

        # Adjust ghost colors for better contrast
        self.ghost_car.hull.color = (0.25, 0.25, 0.9)
        for w in self.ghost_car.wheels:
            w.color = (0.6, 0.6, 1.0)

        # Cache wheel positions in the hull's local frame so we can move them
        self._ghost_wheel_local = []
        hx, hy = self.ghost_car.hull.position
        ha = self.ghost_car.hull.angle
        ca, sa = math.cos(-ha), math.sin(-ha)  # inverse rotation

        for w in self.ghost_car.wheels:
            wx, wy = w.position
            dx, dy = wx - hx, wy - hy
            # rotate into hull local frame
            lx = dx
            ly = dy
            self._ghost_wheel_local.append((lx, ly, w))

        self.ghost_progress = 0.0


    def _update_ghost(self) -> None:
            if self.ghost_car is None or not hasattr(self, "track") or len(self.track) == 0:
                return

            # Advance ghost along tiles at constant "tile velocity"
            self.ghost_progress += self.ghost_speed_tiles_per_sec * (2.0 / FPS)
            idx = int(self.ghost_progress) % len(self.track)
            _, beta, x, y = self.track[idx]

            offset = self.ghost_offset  # keep ghost behind the centerline point
            hx = x + offset * math.cos(beta)
            hy = y + offset * math.sin(beta)
            ha = beta

            # move hull
            self.ghost_car.hull.position = (hx, hy)
            self.ghost_car.hull.angle = ha

            # move wheels according to cached local offsets
            if self._ghost_wheel_local:
                ca, sa = math.cos(ha), math.sin(ha)
                for (lx, ly, w) in self._ghost_wheel_local:
                    wx = hx + ca * lx - sa * ly
                    wy = hy + sa * lx + ca * ly
                    w.position = (wx, wy)
                    w.angle = ha
    # --------------------------------------------------------------------------
    # Rendering helpers
    # --------------------------------------------------------------------------
    def _world_poly_from_body(self, body: Box2D.b2Body) -> List[Tuple[float, float]]:
        """
        Convert first fixture polygon of a Box2D body to a list of world coordinates.
        """
        if body is None or len(body.fixtures) == 0:
            return []
        fixture = body.fixtures[0]
        shape = fixture.shape
        verts = []
        for v in shape.vertices:
            wv = body.transform * v
            verts.append((float(wv[0]), float(wv[1])))
        return verts
    def _render_trees(self, zoom, translation, angle) -> None:
        for t in self.trees:
            for fixture in t.fixtures:
                shape = fixture.shape
                verts = []
                for v in shape.vertices:
                    wv = t.transform * v
                    verts.append((float(wv[0]), float(wv[1])))
                # Tree leaves = green
                color = (20, 160, 20)
                self._draw_colored_polygon(self.surf, verts, color, zoom, translation, angle)

            # Yellow center dot
            cx, cy = t.position
            r = t.collision_radius * 0.2
            center_poly = [
                (cx + r, cy),
                (cx, cy + r),
                (cx - r, cy),
                (cx, cy - r),
            ]
            self._draw_colored_polygon(self.surf, center_poly, (240, 200, 40), zoom, translation, angle)



    def _render_mountains(self, zoom, translation, angle) -> None:
        for m in self.mountains:
            poly = self._world_poly_from_body(m)
            if not poly:
                continue
            # Warm brown-ish color
            color = (130, 100, 70)
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_obstacles(self, zoom, translation, angle) -> None:
        # Static obstacles: concrete gray blocks
        for b in self.static_obstacles:
            poly = self._world_poly_from_body(b)
            if not poly:
                continue
            color = (200, 200, 200)
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

        # Dynamic obstacles: orange / red blocks with gentle pulsing via alpha
        t = getattr(self, "t", 0.0)
        pulse = 0.5 * (math.sin(4.0 * t) + 1.0)  # [0,1]
        base = np.array([230, 150, 70], dtype=float)
        dark = np.array([140, 60, 30], dtype=float)
        mix = dark + pulse * (base - dark)
        color = tuple(int(c) for c in mix)

        for dyn in self.dynamic_obstacles:
            b = dyn["body"]
            poly = self._world_poly_from_body(b)
            if not poly:
                continue
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    # --------------------------------------------------------------------------
    # Start/finish line overlay
    # --------------------------------------------------------------------------
    def _render_start_finish_line(self, zoom, translation, angle):
        if not self.road_poly:
            return

        # road_poly[0] corresponds to tile idx 0
        poly, _color = self.road_poly[0]
        verts = [(p[0], p[1]) for p in poly]  # just (x,y)
        line_color = (255, 255, 255)          # white stripe

        # Draw a bright overlay on the first tile
        self._draw_colored_polygon(self.surf, verts, line_color, zoom, translation, angle)

    # --------------------------------------------------------------------------
    # Override _render_road to inject extra layers
    # --------------------------------------------------------------------------
    def _render_road(self, zoom, translation, angle):
        """
        First let the base environment draw background + grass + road.
        Then overlay mountains, obstacles, ghost car, and start/finish line.
        """
        # Draw default background, grass, road tiles and borders
        super()._render_road(zoom, translation, angle)

        # Start/finish line
        self._render_start_finish_line(zoom, translation, angle)

        # Our extras on top of the road layer
        self._render_mountains(zoom, translation, angle)
        self._render_trees(zoom, translation, angle)

        self._render_obstacles(zoom, translation, angle)

        # Finally draw ghost car, including wheels (no skid particles)
        if self.ghost_car is not None:
            self.ghost_car.draw(
                self.surf,
                zoom,
                translation,
                angle,
                draw_particles=False,
            )


# --------------------------------------------------------------------------
# Manual keyboard test (like original car_racing.py main)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import pygame

    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def register_input():
        global quit_game, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit_game = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0.0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0.0
                if event.key == pygame.K_UP:
                    a[1] = 0.0
                if event.key == pygame.K_DOWN:
                    a[2] = 0.0

            if event.type == pygame.QUIT:
                quit_game = True

    env = CarRacingObstacles(render_mode="human")

    quit_game = False
    while not quit_game:
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            obs, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit_game:
                break
    env.close()
