
# overlay_video.py
# ------------------------------------------------------------
# Climbing / pose video analyzer (1 person, indoor)
# Outputs:
#   1) overlay mp4 (skeleton + COM + stability/effort text)
#   2) per-frame CSV metrics
#   3) summary TXT
#
# Designed to be imported from app.py (Streamlit 등)
#   from overlay_video import analyze_video
# ------------------------------------------------------------

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# -----------------------------
# Utils
# -----------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a / (b + eps))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    angle ABC in degrees (2D/3D OK)
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosv = float(np.dot(ba, bc) / denom)
    cosv = max(-1.0, min(1.0, cosv))
    return float(np.degrees(np.arccos(cosv)))


def put_text_block(
    img: np.ndarray,
    lines: List[str],
    org: Tuple[int, int] = (20, 30),
    line_h: int = 26,
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    x, y = org
    for i, t in enumerate(lines):
        yy = y + i * line_h
        # outline
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# -----------------------------
# Pose / Metrics
# -----------------------------
POSE = mp.solutions.pose
DRAW = mp.solutions.drawing_utils
DRAW_STYLES = mp.solutions.drawing_styles


@dataclass
class FrameMetrics:
    frame_idx: int
    t_sec: float
    has_pose: int

    # COM (center of mass) related
    com_x: float
    com_y: float
    com_speed: float
    com_acc: float
    com_jerk: float

    # Angles (deg)
    l_elbow: float
    r_elbow: float
    l_knee: float
    r_knee: float
    l_hip: float
    r_hip: float

    # Stability / effort (0-100)
    stability_0_100: float
    effort_0_100: float

    # Extra (debug)
    visibility_mean: float
    bos_margin_px: float
    in_bos: int


def _lm_xyv(
    lms,
    idx: int,
    w: int,
    h: int,
) -> Tuple[Optional[np.ndarray], float]:
    """
    return (xy pixel np.array([x,y]) or None, visibility)
    """
    lm = lms.landmark[idx]
    v = float(getattr(lm, "visibility", 0.0))
    if np.isnan(lm.x) or np.isnan(lm.y):
        return None, v
    x = float(lm.x) * w
    y = float(lm.y) * h
    return np.array([x, y], dtype=np.float32), v


def _compute_com_2d(points: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Simple COM approximation by weighted average of key points.
    (No true force estimation; this is a practical heuristic)
    """
    # Rough body mass distribution weights
    # torso heavy, then hips, then legs/arms
    weights = {
        "l_sh": 0.10,
        "r_sh": 0.10,
        "l_hip": 0.18,
        "r_hip": 0.18,
        "l_knee": 0.08,
        "r_knee": 0.08,
        "l_ank": 0.05,
        "r_ank": 0.05,
        "l_wri": 0.04,
        "r_wri": 0.04,
        "nose": 0.10,
    }

    acc = np.zeros(2, dtype=np.float32)
    wsum = 0.0
    for k, w in weights.items():
        if k in points:
            acc += points[k] * float(w)
            wsum += float(w)

    if wsum <= 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float32)

    return acc / float(wsum)


def _polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    return float(cv2.contourArea(poly.astype(np.float32)))


def _point_in_poly(poly: np.ndarray, p: np.ndarray) -> Tuple[int, float]:
    """
    returns (in_poly 0/1, signed distance)
    signed distance: + inside margin, - outside
    """
    if poly is None or len(poly) < 3:
        return 0, float("nan")
    signed = float(cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True))
    return (1 if signed >= 0 else 0), signed


def _make_bos_polygon(points: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Base of support polygon:
    For climbing, "support" can be hands/feet.
    Practical heuristic:
      - Use available ankles + wrists (if present) as support candidates.
      - If too few, fallback to ankles only.
    """
    candidates = []
    for k in ("l_ank", "r_ank", "l_wri", "r_wri"):
        if k in points and np.all(np.isfinite(points[k])):
            candidates.append(points[k])

    if len(candidates) < 3:
        # fallback: try ankles + knees/hips to make a polygon
        candidates = []
        for k in ("l_ank", "r_ank", "l_knee", "r_knee", "l_hip", "r_hip"):
            if k in points and np.all(np.isfinite(points[k])):
                candidates.append(points[k])

    if len(candidates) < 3:
        return None

    pts = np.stack(candidates, axis=0).astype(np.float32)
    hull = cv2.convexHull(pts)
    hull = hull.reshape(-1, 2)  # (N,2)
    if len(hull) < 3:
        return None
    return hull


def _smooth_ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or (not np.all(np.isfinite(prev))):
        return cur
    return (1.0 - alpha) * prev + alpha * cur


# -----------------------------
# Main API: analyze_video
# -----------------------------
def analyze_video(
    input_path: str,
    out_video_path: str,
    out_csv_path: str,
    out_txt_path: str,
    *,
    target_fps: float = 0.0,
    resize_width: int = 0,
    model_complexity: int = 1,
    min_det_conf: float = 0.35,
    min_track_conf: float = 0.35,
    draw_landmarks: bool = True,
    draw_bos: bool = True,
    draw_com: bool = True,
    fast_mode: bool = True,
) -> int:
    """
    input_path: input mp4
    out_video_path: overlay mp4 output path
    out_csv_path: per-frame metrics csv path
    out_txt_path: summary txt path

    target_fps:
      - 0: keep original
      - >0: skip frames to approximate target fps (faster)
    resize_width:
      - 0: keep original size
      - >0: resize frame to this width (keeps aspect ratio) (faster)

    model_complexity:
      - 0 fastest, 2 most accurate

    fast_mode:
      - True: prefer speed (more skipping/resizing friendly)
      - False: prefer accuracy (less smoothing, optional)
    """
    in_path = Path(input_path)
    out_v = Path(out_video_path)
    out_c = Path(out_csv_path)
    out_t = Path(out_txt_path)

    out_v.parent.mkdir(parents=True, exist_ok=True)
    out_c.parent.mkdir(parents=True, exist_ok=True)
    out_t.parent.mkdir(parents=True, exist_ok=True)

    cap = None
    writer = None

    # CSV header
    csv_fields = [f.name for f in FrameMetrics.__dataclass_fields__.values()]  # type: ignore

    # motion history
    prev_com = None           # np.ndarray(2)
    prev_com_v = None         # velocity
    prev_com_a = None         # acceleration
    prev_t = None

    # smoothing
    # COM smoothing helps with jitter
    ema_alpha = 0.20 if fast_mode else 0.12

    metrics_rows: List[FrameMetrics] = []

    try:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            print("[ERROR] Video open failed:", input_path)
            return 1

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

        # decide output size
        if resize_width and resize_width > 0 and resize_width < src_w:
            out_w = int(resize_width)
            out_h = int(round(src_h * (out_w / float(src_w))))
        else:
            out_w, out_h = src_w, src_h

        # decide frame skipping for target fps
        if target_fps and target_fps > 0:
            # skip ratio
            step = max(1, int(round(src_fps / float(target_fps))))
            out_fps = float(target_fps)
        else:
            step = 1
            out_fps = float(src_fps)

        # video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # most compatible without extra setup
        writer = cv2.VideoWriter(str(out_v), fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            print("[ERROR] VideoWriter open failed:", str(out_v))
            return 2

        # Mediapipe pose
        pose = POSE.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_track_conf),
        )

        # CSV open
        with open(out_c, "w", newline="", encoding="utf-8") as fcsv:
            wcsv = csv.DictWriter(fcsv, fieldnames=csv_fields)
            wcsv.writeheader()

            frame_idx = 0
            kept_idx = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # skip frames
                if (frame_idx - 1) % step != 0:
                    continue
                kept_idx += 1

                # resize if needed
                if (out_w != src_w) or (out_h != src_h):
                    frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # timestamp (approx)
                # If cv2 provides POS_MSEC, prefer it
                t_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                t_sec = t_msec / 1000.0

                # mediapipe needs RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                overlay = frame_bgr.copy()

                has_pose = 0
                vis_mean = 0.0

                # defaults
                com = np.array([np.nan, np.nan], dtype=np.float32)
                com_speed = 0.0
                com_acc = 0.0
                com_jerk = 0.0

                l_elbow = r_elbow = l_knee = r_knee = l_hip = r_hip = float("nan")

                in_bos = 0
                bos_margin = float("nan")
                stability_0_100 = 0.0
                effort_0_100 = 0.0

                if results.pose_landmarks is not None:
                    has_pose = 1
                    lms = results.pose_landmarks

                    # collect key points
                    pts: Dict[str, np.ndarray] = {}
                    vis: List[float] = []

                    # indices
                    idx = POSE.PoseLandmark
                    keymap = {
                        "nose": idx.NOSE,
                        "l_sh": idx.LEFT_SHOULDER,
                        "r_sh": idx.RIGHT_SHOULDER,
                        "l_el": idx.LEFT_ELBOW,
                        "r_el": idx.RIGHT_ELBOW,
                        "l_wri": idx.LEFT_WRIST,
                        "r_wri": idx.RIGHT_WRIST,
                        "l_hip": idx.LEFT_HIP,
                        "r_hip": idx.RIGHT_HIP,
                        "l_knee": idx.LEFT_KNEE,
                        "r_knee": idx.RIGHT_KNEE,
                        "l_ank": idx.LEFT_ANKLE,
                        "r_ank": idx.RIGHT_ANKLE,
                    }

                    for k, landmark_idx in keymap.items():
                        p, v = _lm_xyv(lms, int(landmark_idx), out_w, out_h)
                        if p is not None:
                            pts[k] = p
                        vis.append(v)

                    vis_mean = float(np.mean(vis)) if len(vis) else 0.0

                    # angles (2D)
                    # elbow: shoulder - elbow - wrist
                    if "l_sh" in pts and "l_el" in pts and "l_wri" in pts:
                        l_elbow = angle_3pts(pts["l_sh"], pts["l_el"], pts["l_wri"])
                    if "r_sh" in pts and "r_el" in pts and "r_wri" in pts:
                        r_elbow = angle_3pts(pts["r_sh"], pts["r_el"], pts["r_wri"])

                    # knee: hip - knee - ankle
                    if "l_hip" in pts and "l_knee" in pts and "l_ank" in pts:
                        l_knee = angle_3pts(pts["l_hip"], pts["l_knee"], pts["l_ank"])
                    if "r_hip" in pts and "r_knee" in pts and "r_ank" in pts:
                        r_knee = angle_3pts(pts["r_hip"], pts["r_knee"], pts["r_ank"])

                    # hip: shoulder - hip - knee
                    if "l_sh" in pts and "l_hip" in pts and "l_knee" in pts:
                        l_hip = angle_3pts(pts["l_sh"], pts["l_hip"], pts["l_knee"])
                    if "r_sh" in pts and "r_hip" in pts and "r_knee" in pts:
                        r_hip = angle_3pts(pts["r_sh"], pts["r_hip"], pts["r_knee"])

                    # COM
                    raw_com = _compute_com_2d(pts)
                    com = _smooth_ema(prev_com, raw_com, ema_alpha)
                    prev_com = com

                    # BOS polygon and margin
                    bos = _make_bos_polygon(pts)
                    if bos is not None and np.all(np.isfinite(com)):
                        in_bos, bos_margin = _point_in_poly(bos, com)

                        # draw BOS
                        if draw_bos:
                            cv2.polylines(
                                overlay,
                                [bos.astype(np.int32)],
                                isClosed=True,
                                color=(0, 255, 0),
                                thickness=2,
                            )

                    # COM motion derivatives
                    # Use dt based on output fps if POS_MSEC not reliable
                    if prev_t is None:
                        dt = 1.0 / max(1.0, out_fps)
                    else:
                        dt = max(1e-3, float(t_sec - prev_t))
                    prev_t = float(t_sec)

                    if np.all(np.isfinite(com)):
                        if prev_com_v is None:
                            com_v = np.array([0.0, 0.0], dtype=np.float32)
                        else:
                            com_v = (com - prev_com_prev) / float(dt)  # placeholder overwritten below
                    # Compute properly with stored last COM for derivatives
                    # We'll keep an additional variable:
                    # Use local static variables via closure style is messy; do it explicit:
                    # We'll store last COM for derivative separately:
                    # (Below: com_prev_for_deriv is kept in function scope as prev_com_deriv)

                # ----- We need derivative state that doesn't fight EMA state -----
                # We'll implement derivative using separate vars
                # (To keep code simple, compute after pose block with separate variables.)
                # ---------------------------------------------------------

                # draw pose landmarks
                if has_pose and draw_landmarks:
                    DRAW.draw_landmarks(
                        overlay,
                        results.pose_landmarks,
                        POSE.POSE_CONNECTIONS,
                        landmark_drawing_spec=DRAW_STYLES.get_default_pose_landmarks_style(),
                        connection_drawing_spec=DRAW.DrawingSpec(color=(0, 255, 255), thickness=2),
                    )

                # draw COM
                if has_pose and draw_com and np.all(np.isfinite(com)):
                    cv2.circle(overlay, (int(com[0]), int(com[1])), 7, (255, 0, 255), -1)
                    cv2.circle(overlay, (int(com[0]), int(com[1])), 11, (0, 0, 0), 2)

                # -----------------------------
                # Derivatives + scores
                # -----------------------------
                # Use separate derivative states
                # We'll store them in function scope via local dict:
                if not hasattr(analyze_video, "_state"):
                    analyze_video._state = {}  # type: ignore
                stt = analyze_video._state  # type: ignore

                com_prev = stt.get("com_prev", None)
                v_prev = stt.get("v_prev", None)
                a_prev = stt.get("a_prev", None)
                t_prev = stt.get("t_prev", None)

                if t_prev is None:
                    dt = 1.0 / max(1.0, out_fps)
                else:
                    dt = max(1e-3, float(t_sec - float(t_prev)))

                if has_pose and np.all(np.isfinite(com)):
                    if com_prev is None:
                        v = np.array([0.0, 0.0], dtype=np.float32)
                        a = np.array([0.0, 0.0], dtype=np.float32)
                        j = np.array([0.0, 0.0], dtype=np.float32)
                    else:
                        v = (com - com_prev) / float(dt)
                        if v_prev is None:
                            a = np.array([0.0, 0.0], dtype=np.float32)
                        else:
                            a = (v - v_prev) / float(dt)
                        if a_prev is None:
                            j = np.array([0.0, 0.0], dtype=np.float32)
                        else:
                            j = (a - a_prev) / float(dt)

                    # magnitudes
                    com_speed = float(np.linalg.norm(v))
                    com_acc = float(np.linalg.norm(a))
                    com_jerk = float(np.linalg.norm(j))

                    # update state
                    stt["com_prev"] = com.copy()
                    stt["v_prev"] = v.copy()
                    stt["a_prev"] = a.copy()
                    stt["t_prev"] = float(t_sec)
                else:
                    # decay state slowly if pose missing
                    stt["t_prev"] = float(t_sec)

                # ---- Stability score heuristic ----
                # 1) BOS margin (inside is good)
                # 2) COM speed/acc/jerk (lower is steadier)
                # Normalize by video size
                norm = float(max(out_w, out_h))

                # BOS contribution
                if np.isnan(bos_margin):
                    bos_n = 0.45  # unknown -> medium
                else:
                    # inside: margin positive, outside: negative
                    # map [-120..+80] px roughly
                    if in_bos == 1:
                        bos_n = clamp01(bos_margin / 80.0)
                    else:
                        bos_n = 1.0 - clamp01((-bos_margin) / 120.0)

                # motion contributions (lower motion -> higher stability)
                sp_n = 1.0 - clamp01((com_speed / (0.020 * norm + 1e-9)))  # ~2% of frame per sec
                ac_n = 1.0 - clamp01((com_acc / (0.060 * norm + 1e-9)))    # ~6% per sec^2
                jk_n = 1.0 - clamp01((com_jerk / (0.250 * norm + 1e-9)))   # ~25% per sec^3

                # visibility penalty
                vis_n = clamp01(vis_mean)

                # weighted blend
                stability = (
                    0.45 * bos_n +
                    0.25 * sp_n +
                    0.20 * ac_n +
                    0.10 * jk_n
                )
                stability *= (0.50 + 0.50 * vis_n)  # if visibility poor, reduce confidence
                stability_0_100 = float(round(100.0 * clamp01(stability), 2))

                # ---- Effort score heuristic ----
                # Effort ~ movement intensity + joint flexion extremes
                # More speed/acc + smaller knee angles (deep flex) + smaller elbow angles -> higher effort
                flex_pen = 0.0
                def flex_component(a_deg: float, min_deg: float, max_deg: float) -> float:
                    if np.isnan(a_deg):
                        return 0.0
                    # map angle into effort: more bent -> higher effort
                    # If angle below min_deg => strong flex -> 1
                    # If angle above max_deg => straight -> 0
                    return clamp01((max_deg - a_deg) / (max_deg - min_deg + 1e-9))

                flex_pen += 0.5 * flex_component(l_knee, 60.0, 170.0)
                flex_pen += 0.5 * flex_component(r_knee, 60.0, 170.0)
                flex_pen += 0.4 * flex_component(l_elbow, 50.0, 170.0)
                flex_pen += 0.4 * flex_component(r_elbow, 50.0, 170.0)

                # motion energy
                motion_e = clamp01((com_speed / (0.030 * norm + 1e-9))) * 0.6 + clamp01((com_acc / (0.080 * norm + 1e-9))) * 0.4
                effort = clamp01(0.55 * motion_e + 0.45 * clamp01(flex_pen / 1.8))
                effort_0_100 = float(round(100.0 * effort, 2))

                # ---- overlay text ----
                lines = [
                    f"frame: {kept_idx}",
                    f"pose: {'OK' if has_pose else 'NO'}  vis={vis_mean:.2f}",
                    f"stability: {stability_0_100:.1f} / 100",
                    f"effort:    {effort_0_100:.1f} / 100",
                ]
                if has_pose and np.all(np.isfinite(com)):
                    lines.append(f"COM v/a/j: {com_speed:.1f} / {com_acc:.1f} / {com_jerk:.1f}")
                if not np.isnan(bos_margin):
                    lines.append(f"BOS margin(px): {bos_margin:.1f}  in={in_bos}")

                put_text_block(overlay, lines, org=(20, 32), line_h=24, scale=0.65)

                # write frame
                writer.write(overlay)

                # save metrics row
                row = FrameMetrics(
                    frame_idx=kept_idx,
                    t_sec=float(t_sec),
                    has_pose=int(has_pose),
                    com_x=float(com[0]) if np.all(np.isfinite(com)) else float("nan"),
                    com_y=float(com[1]) if np.all(np.isfinite(com)) else float("nan"),
                    com_speed=float(com_speed),
                    com_acc=float(com_acc),
                    com_jerk=float(com_jerk),
                    l_elbow=float(l_elbow),
                    r_elbow=float(r_elbow),
                    l_knee=float(l_knee),
                    r_knee=float(r_knee),
                    l_hip=float(l_hip),
                    r_hip=float(r_hip),
                    stability_0_100=float(stability_0_100),
                    effort_0_100=float(effort_0_100),
                    visibility_mean=float(vis_mean),
                    bos_margin_px=float(bos_margin),
                    in_bos=int(in_bos),
                )
                metrics_rows.append(row)
                wcsv.writerow(row.__dict__)

                if kept_idx % 200 == 0:
                    print(f"...processing {kept_idx} frames")

        pose.close()

    except Exception as e:
        print("[ERROR]", e)
        return 3

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

    # -----------------------------
    # Summary TXT
    # -----------------------------
    try:
        if len(metrics_rows) == 0:
            out_t.write_text("분석 실패: 프레임이 없습니다.\n", encoding="utf-8")
            return 4

        arr_stab = np.array([m.stability_0_100 for m in metrics_rows if m.has_pose == 1], dtype=np.float32)
        arr_eff = np.array([m.effort_0_100 for m in metrics_rows if m.has_pose == 1], dtype=np.float32)
        pose_ratio = float(np.mean([m.has_pose for m in metrics_rows]))

        def stat_block(name: str, arr: np.ndarray) -> str:
            if arr.size == 0:
                return f"{name}: (no pose)\n"
            return (
                f"{name} 평균={float(np.mean(arr)):.1f}, "
                f"중앙값={float(np.median(arr)):.1f}, "
                f"최소={float(np.min(arr)):.1f}, "
                f"최대={float(np.max(arr)):.1f}, "
                f"p10={float(np.percentile(arr, 10)):.1f}, "
                f"p90={float(np.percentile(arr, 90)):.1f}\n"
            )

        txt = []
        txt.append("클라이밍 포즈 분석 요약\n")
        txt.append("----------------------------------------\n")
        txt.append(f"입력: {str(in_path)}\n")
        txt.append(f"출력(영상): {str(out_v)}\n")
        txt.append(f"출력(CSV): {str(out_c)}\n")
        txt.append(f"출력(요약): {str(out_t)}\n\n")

        txt.append(f"총 처리 프레임: {len(metrics_rows)}\n")
        txt.append(f"포즈 인식 성공 비율: {pose_ratio*100.0:.1f}%\n\n")

        txt.append(stat_block("안정성(stability)", arr_stab))
        txt.append(stat_block("노력(effort)", arr_eff))
        txt.append("\n")

        # quick coaching text (very simple)
        if arr_stab.size > 0:
            mean_stab = float(np.mean(arr_stab))
            if mean_stab >= 75:
                txt.append("코멘트: 전체적으로 중심이 안정적입니다. 더 빠른 루트에서도 흔들림(가속/저크)을 낮추면 좋습니다.\n")
            elif mean_stab >= 55:
                txt.append("코멘트: 보통 수준의 안정성입니다. 발/손 지지(지지다각형) 밖으로 COM이 자주 벗어나면 흔들림이 커집니다.\n")
            else:
                txt.append("코멘트: 안정성이 낮은 편입니다. 발/손 지지점이 줄어드는 구간에서 COM 이동(속도/가속)이 급해질 가능성이 큽니다.\n")

        if arr_eff.size > 0:
            mean_eff = float(np.mean(arr_eff))
            if mean_eff >= 70:
                txt.append("코멘트: 노력 점수가 높습니다(움직임 강도/관절 굴곡 큼). 루트가 어렵거나 불필요한 동작이 많을 수 있습니다.\n")
            elif mean_eff >= 45:
                txt.append("코멘트: 적당한 노력 수준입니다. 불필요한 몸통 흔들림이 줄면 안정성이 더 올라갑니다.\n")
            else:
                txt.append("코멘트: 노력 점수가 낮습니다. 동작이 작고 안정적이거나, 반대로 포즈 인식이 부족했을 수도 있습니다.\n")

        out_t.write_text("".join(txt), encoding="utf-8")

    except Exception as e:
        print("[ERROR] summary write failed:", e)
        return 5

    print("✅ 완료:", str(out_v))
    return 0
