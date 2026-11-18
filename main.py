"""
Budget VAR  -  two-tier version
===============================
LOW  mode (default): ball-only, detector runs every few frames, no player tracking.
HIGH mode: switches on when the ball is in/near the penalty area or goal and stays
           on for ALERT_HOLD_S after it leaves. Ball detected every frame, players
           tracked (ByteTrack) and split into teams by shirt colour.
EVENT: ball inside a trigger zone for CONFIRM_FRAMES -> all cameras hand over
       PRE_ROLL_S + POST_ROLL_S of video. Pre-roll frames recorded in LOW mode have
       no players yet, so they are analysed retroactively in a background thread
       before the replay is shown (heavy compute only when it matters).

Usage
-----
    python var_system.py --calibrate sample_match.mp4 --cam cam1   # mark zones once
    python var_system.py
"""
"import statements"
import argparse
import json
import os
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from ball_tracker import BallTracker

# ----------------------------------------------------------------- config ---
CAMERAS = {
    # name: source (file path, RTSP url, or webcam index)
    "cam1": "sample_match.mp4",
    # "cam2": "sample_match_angle2.mp4",
}
ZONES_FILE = "zones.json"
TRIGGER_ZONES = ("penalty", "goal")

# --- tiering ---
ALERT_MARGIN_PX = 200            # amp up when ball is within this many px of a zone (full-res px)
ALERT_HOLD_S = 3.0               # stay in HIGH this long after the ball leaves
LOW_DETECT_EVERY = 3             # LOW: ball detector runs every Nth frame (Kalman coasts between)
LOW_SCAN_EVERY = 8               # LOW: full-frame tile scan every Nth frame while ball is lost
HIGH_SCAN_EVERY = 2              # HIGH: same, but more often
PLAYER_IMGSZ = 960

# --- event capture ---
PRE_ROLL_S = 5
POST_ROLL_S = 3
CONFIRM_FRAMES = 3
COOLDOWN_S = 10
STORE_WIDTH = 960
REPLAY_SLOWDOWN = 2.0
PACE_FILES = True
CLIP_DIR = "var_clips"

# --- model ---
MODEL_PATH = "yolov8n.pt"
BALL_CLS, PERSON_CLS = 32, 0
BALL_CONF = 0.05
TILE = 640
TILE_OVERLAP = 0.25

TEAM_COL = {0: (0, 0, 255), 1: (255, 120, 0), -1: (170, 170, 170)}


# ---------------------------------------------------------------- zones ---
def load_zones():
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE) as f:
            raw = json.load(f)
        return {cam: {z: np.array(p, np.int32) for z, p in zs.items()}
                for cam, zs in raw.items()}
    return {}


def zone_hit(zones, point):
    """Returns (zone_name, signed_distance_px). Distance >= 0 means inside."""
    best = (None, -1e9)
    for name in TRIGGER_ZONES:
        poly = zones.get(name)
        if poly is None:
            continue
        d = cv2.pointPolygonTest(poly, point, True)
        if d > best[1]:
            best = (name, d)
    return best


def calibrate(source, cam):
    """Click polygon corners. Enter = finish zone, s = save, q = quit."""
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read {source}")
    data = {}
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE) as f:
            data = json.load(f)
    data.setdefault(cam, {})
    todo, pts = list(TRIGGER_ZONES), []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while True:
        view = frame.copy()
        for poly in data[cam].values():
            cv2.polylines(view, [np.array(poly, np.int32)], True, (0, 200, 255), 2)
        for p in pts:
            cv2.circle(view, tuple(p), 4, (0, 255, 0), -1)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], False, (0, 255, 0), 2)
        label = todo[0] if todo else "done - press s"
        cv2.putText(view, f"Click corners for: {label}  (Enter=finish, s=save, q=quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(pts) >= 3 and todo:
            data[cam][todo.pop(0)] = pts.copy()
            pts.clear()
        elif k == ord("s"):
            with open(ZONES_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {ZONES_FILE}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


# ------------------------------------------------------------- detection ---
def _starts(length, size, overlap):
    if length <= size:
        return [0]
    step = int(size * (1 - overlap))
    xs = list(range(0, length - size + 1, step))
    if xs[-1] + size < length:
        xs.append(length - size)
    return xs


def detect_ball(model, frame, hint=None):
    """hint=(x,y): scan one TILE crop around it. hint=None: tiled full-frame scan."""
    h, w = frame.shape[:2]
    if hint is not None:
        half = TILE // 2
        x0 = int(min(max(hint[0] - half, 0), max(w - TILE, 0)))
        y0 = int(min(max(hint[1] - half, 0), max(h - TILE, 0)))
        origins = [(x0, y0)]
    else:
        origins = [(x, y) for y in _starts(h, TILE, TILE_OVERLAP)
                   for x in _starts(w, TILE, TILE_OVERLAP)]
    cands = []
    for x0, y0 in origins:
        crop = frame[y0:y0 + TILE, x0:x0 + TILE]
        r = model.predict(crop, imgsz=TILE, conf=BALL_CONF,
                          classes=[BALL_CLS], verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        for b, cf in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            cands.append((b[0] + x0, b[1] + y0, b[2], b[3], float(cf)))
    return cands


class TeamClassifier:
    """Splits tracked people into 2 teams by torso colour (k-means on running averages).
    Budget heuristic: referees and keepers will sometimes be put in a team."""

    def __init__(self):
        self.colors = {}        # id -> [mean_lab(3), n]
        self.centers = None
        self.calls = 0

    def update(self, frame, xyxy, ids):
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            w, h = x2 - x1, y2 - y1
            crop = frame[int(y1 + .2 * h):int(y1 + .55 * h), int(x1 + .25 * w):int(x2 - .25 * w)]
            if crop.size == 0:
                continue
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)
            m, n = self.colors.get(int(tid), [np.zeros(3), 0])
            self.colors[int(tid)] = [(m * n + lab) / (n + 1), n + 1]
        self.calls += 1
        if self.calls % 15 == 1 and len(self.colors) >= 4:
            data = np.float32([v[0] for v in self.colors.values()])
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, _, self.centers = cv2.kmeans(data, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    def team_of(self, tid):
        if self.centers is None or int(tid) not in self.colors:
            return -1
        d = np.linalg.norm(self.centers - self.colors[int(tid)][0], axis=1)
        return int(d.argmin())


def analyze_players(model, frame, persist, teams):
    res = model.track(frame, persist=persist, tracker="bytetrack.yaml",
                      imgsz=PLAYER_IMGSZ, conf=0.25,
                      classes=[PERSON_CLS], verbose=False)[0]
    out = []
    if res.boxes is not None and len(res.boxes) and res.boxes.id is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int)
        teams.update(frame, xyxy, ids)
        out = [(*map(float, b), int(i), teams.team_of(i)) for b, i in zip(xyxy, ids)]
    return out


def render(frame, meta, zones, cam):
    vis = frame.copy()
    for poly in zones.values():
        cv2.polylines(vis, [poly], True, (0, 200, 255), 2)
    for x1, y1, x2, y2, tid, team in (meta["players"] or []):
        col = TEAM_COL.get(team, TEAM_COL[-1])
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        tag = ("~" if meta["retro"] else "") + f"#{tid}"
        cv2.putText(vis, tag, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    if meta["ball"] is not None:
        bx, by, measured = meta["ball"]
        cv2.circle(vis, (int(bx), int(by)), 6, (0, 255, 0) if measured else (0, 165, 255), -1)
    if meta["zone"]:
        cv2.putText(vis, f"BALL IN {meta['zone'].upper()}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    badge = (0, 0, 255) if meta["mode"] == "HIGH" else (200, 200, 200)
    cv2.putText(vis, f"{cam} [{meta['mode']}]", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge, 2)
    return vis


# ----------------------------------------------------------- event manager ---
class EventManager:
    """Collects clips from all cameras, fills in missing player analysis in the
    background, writes mp4s, and hands finished replays to the main thread."""

    def __init__(self, cameras):
        self.cameras = cameras
        self.lock = threading.Lock()
        self.next_id = 1
        self.last_fire = 0.0
        self.collected = {}
        self.analysis_queue = queue.Queue()
        self.replay_queue = queue.Queue()
        threading.Thread(target=self._analysis_loop, daemon=True).start()

    def trigger(self, cam_name, zone):
        with self.lock:
            now = time.time()
            if now - self.last_fire < COOLDOWN_S:
                return
            self.last_fire = now
            eid = self.next_id
            self.next_id += 1
            self.collected[eid] = {}
        print(f"[VAR] Event #{eid}: ball in '{zone}' on {cam_name} - collecting angles...")
        for cam in self.cameras:
            cam.start_capture(eid)

    def camera_done(self, eid, cam_name, items, fps, zones):
        with self.lock:
            self.collected[eid][cam_name] = (items, fps, zones)
            if len(self.collected[eid]) < len(self.cameras):
                return
            clips = self.collected.pop(eid)
        self.analysis_queue.put((eid, clips))

    def _analysis_loop(self):
        os.makedirs(CLIP_DIR, exist_ok=True)
        model = None
        while True:
            eid, clips = self.analysis_queue.get()
            if model is None:
                model = YOLO(MODEL_PATH)     # loaded lazily; only used when an event happens
            rendered = {}
            for cam, (items, fps, zones) in clips.items():
                teams, prev_gap, filled = TeamClassifier(), False, 0
                for frame, meta in items:
                    if meta["players"] is None:       # recorded in LOW mode -> analyse now
                        meta["players"] = analyze_players(model, frame, prev_gap, teams)
                        meta["retro"] = True
                        prev_gap, filled = True, filled + 1
                    else:
                        prev_gap = False
                frames = [render(f, m, zones, cam) for f, m in items]
                rendered[cam] = (frames, fps)
                if frames:
                    h, w = frames[0].shape[:2]
                    path = os.path.join(CLIP_DIR, f"event{eid:03d}_{cam}.mp4")
                    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    for fr in frames:
                        vw.write(fr)
                    vw.release()
                print(f"[VAR] Event #{eid} {cam}: {len(frames)} frames "
                      f"({filled} analysed retroactively) -> {CLIP_DIR}/")
            self.replay_queue.put((eid, rendered))


# ------------------------------------------------------------- camera worker ---
class CameraWorker(threading.Thread):
    def __init__(self, name, source, zones):
        super().__init__(daemon=True)
        self.name, self.source, self.zones_full = name, source, zones
        self.manager = None
        self.stop_flag = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.captures = {}
        self.stats = {"LOW": 0, "HIGH": 0, "ball_calls": 0, "player_calls": 0}

        cap = cv2.VideoCapture(source)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self.buffer = deque(maxlen=int(PRE_ROLL_S * self.fps))

    def start_capture(self, eid):
        with self.lock:
            self.captures[eid] = {
                "items": [(f, dict(m)) for f, m in self.buffer],   # copy meta: retro pass mutates it
                "remaining": int(POST_ROLL_S * self.fps)}

    def run(self):
        track_model = YOLO(MODEL_PATH)       # persons + ByteTrack (only used in HIGH)
        ball_model = YOLO(MODEL_PATH)        # separate instance so predict() can't disturb the tracker
        ball, teams = BallTracker(), TeamClassifier()
        cap = cv2.VideoCapture(self.source)
        is_file = isinstance(self.source, str) and os.path.exists(self.source)
        frame_time = 1.0 / self.fps
        frame_idx, streak, hold, prev_mode = 0, 0, 0, "LOW"
        sc, zones_small = None, None

        while not self.stop_flag.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            if sc is None:
                sc = STORE_WIDTH / w
                zones_small = {n: (p * sc).astype(np.int32) for n, p in self.zones_full.items()}

            mode = "HIGH" if hold > 0 else "LOW"
            self.stats[mode] += 1

            # ---- ball (cheap path in LOW, every-frame in HIGH) ----
            detect_every = 1 if mode == "HIGH" else LOW_DETECT_EVERY
            scan_every = HIGH_SCAN_EVERY if mode == "HIGH" else LOW_SCAN_EVERY
            if ball.initialized:
                if frame_idx % detect_every == 0:
                    hint = (float(ball.kf.x[0, 0]), float(ball.kf.x[1, 0]))
                    est = ball.update(detect_ball(ball_model, frame, hint))
                    self.stats["ball_calls"] += 1
                else:
                    est = ball.coast()
            elif frame_idx % scan_every == 0:
                est = ball.update(detect_ball(ball_model, frame))
                self.stats["ball_calls"] += 1
            else:
                est = None

            # ---- area of interest -> tier + trigger ----
            zone_name, in_trigger, alert = None, False, False
            if est is not None:
                zone_name, d = zone_hit(self.zones_full, (est[0], est[1]))
                in_trigger = d >= 0
                alert = d >= -ALERT_MARGIN_PX
            if alert:
                hold = int(ALERT_HOLD_S * self.fps)
            elif hold > 0:
                hold -= 1

            # ---- players (HIGH only) ----
            players = None
            if mode == "HIGH":
                persist = prev_mode == "HIGH"     # fresh tracker when re-entering HIGH
                p = analyze_players(track_model, frame, persist, teams)
                players = [(x1 * sc, y1 * sc, x2 * sc, y2 * sc, i, t) for x1, y1, x2, y2, i, t in p]
                self.stats["player_calls"] += 1
            prev_mode = mode

            streak = streak + 1 if in_trigger else 0
            if streak == CONFIRM_FRAMES:
                self.manager.trigger(self.name, zone_name)

            # ---- buffer raw frame + metadata (overlays are rendered later) ----
            small = cv2.resize(frame, (STORE_WIDTH, int(h * sc)))
            meta = {"mode": mode, "retro": False, "players": players,
                    "ball": None if est is None else (est[0] * sc, est[1] * sc, est[2]),
                    "zone": zone_name if in_trigger else None}
            self.buffer.append((small, meta))

            done = []
            with self.lock:
                for eid, st in self.captures.items():
                    st["items"].append((small, dict(meta)))
                    st["remaining"] -= 1
                    if st["remaining"] <= 0:
                        done.append((eid, st["items"]))
                for eid, _ in done:
                    del self.captures[eid]
            for eid, items in done:
                self.manager.camera_done(eid, self.name, items, self.fps, zones_small)

            if is_file and PACE_FILES:
                time.sleep(max(0.0, frame_time - (time.time() - t0)))

        cap.release()
        self.finished.set()

    def report(self):
        s = self.stats
        total = max(s["LOW"] + s["HIGH"], 1)
        print(f"[{self.name}] {total} frames: {100 * s['HIGH'] / total:.1f}% HIGH | "
              f"ball-detector calls {s['ball_calls']} | player-tracker calls {s['player_calls']} "
              f"(vs {total} if always-on)")


# ------------------------------------------------------------------ replay ---
def play_replay(eid, clips):
    names = sorted(clips)
    seqs = [clips[n][0] for n in names]
    fps = clips[names[0]][1]
    n = max(len(s) for s in seqs)
    delay = max(1, int(1000 / fps * REPLAY_SLOWDOWN))
    win = f"VAR REVIEW - event {eid}  (q to skip)"
    for i in range(n):
        row = [s[min(i, len(s) - 1)] for s in seqs]
        tile = np.hstack(row) if len(row) > 1 else row[0].copy()
        cv2.putText(tile, f"REPLAY {1 / REPLAY_SLOWDOWN:.1f}x", (10, tile.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, tile)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    cv2.destroyWindow(win)


# -------------------------------------------------------------------- main ---
def main():
    all_zones = load_zones()
    if not all_zones:
        print(f"No {ZONES_FILE} found - run with --calibrate first.")
        return
    workers = [CameraWorker(n, src, all_zones[n]) for n, src in CAMERAS.items() if n in all_zones]
    if not workers:
        print("No cameras have zones. Run --calibrate.")
        return

    manager = EventManager(workers)
    for w in workers:
        w.manager = manager
        w.start()
    print("VAR running (LOW mode until the ball nears a zone). Ctrl+C to quit.")

    try:
        while True:
            try:
                eid, clips = manager.replay_queue.get(timeout=0.2)
                play_replay(eid, clips)
            except queue.Empty:
                if all(w.finished.is_set() for w in workers) and manager.analysis_queue.empty():
                    time.sleep(1.0)
                    if manager.replay_queue.empty():
                        break
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.stop_flag.set()
            w.report()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", metavar="VIDEO")
    ap.add_argument("--cam", default="cam1")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.calibrate, args.cam)
    else:
        main()