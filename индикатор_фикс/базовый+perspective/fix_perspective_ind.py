import argparse
import math
import os
import sys
import time

import cv2
import numpy as np


def _ellipse_extreme_points(ellipse, samples=720):
    (cx, cy), (w, h), ang = ellipse
    a = max(float(w) * 0.5, 1.0)
    b = max(float(h) * 0.5, 1.0)
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)

    c = math.cos(math.radians(ang))
    s = math.sin(math.radians(ang))

    x = cx + a * np.cos(t) * c - b * np.sin(t) * s
    y = cy + a * np.cos(t) * s + b * np.sin(t) * c
    pts = np.column_stack((x, y)).astype(np.float32)

    top = pts[np.argmin(pts[:, 1])]
    right = pts[np.argmax(pts[:, 0])]
    bottom = pts[np.argmax(pts[:, 1])]
    left = pts[np.argmin(pts[:, 0])]
    return top, right, bottom, left


def _score_contour(cnt, h, w, short, prev_ellipse):
    area = cv2.contourArea(cnt)
    if area < max(7000.0, 0.015 * h * w) or len(cnt) < 5:
        return None, -1.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    if solidity < 0.66:
        return None, -1.0

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (ew, eh), _ = ellipse

    major = max(float(ew), float(eh))
    minor = min(float(ew), float(eh))
    ratio = minor / major if major > 0 else 0.0

    if ratio < 0.48:
        return None, -1.0
    if major < short * 0.28:
        return None, -1.0
    if cx < w * 0.02 or cx > w * 0.98:
        return None, -1.0
    if cy < h * 0.02 or cy > h * 0.98:
        return None, -1.0

    ell_area = math.pi * (ew * 0.5) * (eh * 0.5)
    fill = area / ell_area if ell_area > 0 else 0.0
    if fill < 0.40 or fill > 1.55:
        return None, -1.0

    score = 0.0
    score += 2.0 * ratio
    score += 1.3 * solidity
    score += 0.9 * max(0.0, 1.0 - abs(fill - 1.0))
    score += 0.9 * (area / (h * w))

    if prev_ellipse is not None:
        (pcx, pcy), (pw, ph), _ = prev_ellipse
        dist_c = math.hypot(cx - pcx, cy - pcy)
        dist_a = abs(major - max(pw, ph))
        score += max(0.0, 1.0 - dist_c / 100.0) * 0.8
        score += max(0.0, 1.0 - dist_a / 100.0) * 0.5

    return ellipse, score


def find_gauge_ellipse(gray, prev_ellipse=None):
    h, w = gray.shape
    short = min(h, w)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    best_ellipse = None
    best_score = -1.0
    masks = []

    for thresh in range(25, 121, 6):
        _, m = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern, iterations=1)
        masks.append(m)

    for thresh in range(125, 246, 6):
        _, m = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern, iterations=1)
        masks.append(m)

    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kern, iterations=1)
    masks.append(edges)

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ell, sc = _score_contour(cnt, h, w, short, prev_ellipse)
            if sc > best_score:
                best_score = sc
                best_ellipse = ell

    return best_ellipse


def _stabilize_ellipse(curr_ellipse, prev_ellipse):
    if curr_ellipse is None:
        return None
    if prev_ellipse is None:
        return curr_ellipse

    (cx, cy), (cw, ch), ca = curr_ellipse
    (px, py), (pw, ph), pa = prev_ellipse

    c_major = max(float(cw), float(ch))
    c_minor = min(float(cw), float(ch))
    p_major = max(float(pw), float(ph))
    p_minor = min(float(pw), float(ph))

    center_jump = math.hypot(cx - px, cy - py)
    major_rel = abs(c_major - p_major) / max(p_major, 1.0)
    minor_rel = abs(c_minor - p_minor) / max(p_minor, 1.0)

    if center_jump > 85.0 or major_rel > 0.25 or minor_rel > 0.25:
        return prev_ellipse

    alpha = 0.70
    sx = alpha * px + (1.0 - alpha) * cx
    sy = alpha * py + (1.0 - alpha) * cy
    sw = alpha * pw + (1.0 - alpha) * cw
    sh = alpha * ph + (1.0 - alpha) * ch

    d_ang = ((ca - pa + 90.0) % 180.0) - 90.0
    sa = pa + (1.0 - alpha) * d_ang

    return ((float(sx), float(sy)), (float(sw), float(sh)), float(sa))


def warp_ellipse_to_circle(frame, ellipse):
    top, right, bottom, left = _ellipse_extreme_points(ellipse)
    src = np.float32([top, right, bottom, left])

    est_d1 = np.linalg.norm(right - left)
    est_d2 = np.linalg.norm(bottom - top)
    r_est = max(20.0, 0.5 * max(est_d1, est_d2))

    margin = int(r_est * 0.30)
    radius = int(round(r_est))
    c = margin + radius
    size = int(2 * (margin + radius))

    dst = np.float32([
        [c, c - radius],
        [c + radius, c],
        [c, c + radius],
        [c - radius, c],
    ])

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, H, (size, size))
    return warped, int(c), int(c), int(radius), H


def find_needle_angle(frame, cx, cy, r):
    h, w = frame.shape[:2]
    pad = int(r * 1.05)
    x1, y1 = max(cx - pad, 0), max(cy - pad, 0)
    x2, y2 = min(cx + pad, w), min(cy + pad, h)
    crop = frame[y1:y2, x1:x2]
    lx, ly = cx - x1, cy - y1

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ring_mask = np.zeros_like(binary)
    cv2.circle(ring_mask, (lx, ly), int(r * 0.60), 255, -1)
    cv2.circle(ring_mask, (lx, ly), int(r * 0.11), 0, -1)
    binary = cv2.bitwise_and(binary, ring_mask)

    mask_pixels = cv2.countNonZero(ring_mask)
    white_pixels = cv2.countNonZero(binary)
    if mask_pixels > 0 and white_pixels / mask_pixels > 0.50:
        binary = cv2.bitwise_and(cv2.bitwise_not(binary), ring_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt, best_score = None, -1.0
    for cnt in contours:
        if cv2.contourArea(cnt) < 35 or len(cnt) < 5:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        radial = np.hypot(pts[:, 0] - lx, pts[:, 1] - ly)
        near_r, far_r = float(np.min(radial)), float(np.max(radial))
        if near_r > r * 0.23 or far_r < r * 0.40:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        long_s, short_s = max(rw, rh), max(min(rw, rh), 1)
        aspect = long_s / short_s
        if aspect < 3.0:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        dist = math.hypot(M["m10"]/M["m00"] - lx, M["m01"]/M["m00"] - ly)
        score = far_r * aspect / max(8.0, dist + 0.35 * near_r)
        if score > best_score:
            best_cnt, best_score = cnt, score

    if best_cnt is None:
        return float('nan')

    vx, vy, x0, y0 = cv2.fitLine(best_cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    t = 300
    if math.hypot(x0+vx*t - lx, y0+vy*t - ly) < math.hypot(x0-vx*t - lx, y0-vy*t - ly):
        vx, vy = -vx, -vy
    return math.degrees(math.atan2(-vy, vx))



def angle_to_value(angle, start_a, end_a, min_v, max_v):
    if math.isnan(angle):
        return float("nan")

    norm = lambda a: (a % 360) - 360 if (a % 360) > 180 else (a % 360)

    start_n = norm(start_a)
    end_n = norm(end_a)
    ang_n = norm(angle)

    span = (start_n - end_n) % 360
    if span == 0:
        return float("nan")

    if span < 10:
        span = 360 - span

    offset = (start_n - ang_n) % 360
    frac = offset / span

    if frac < -0.05 or frac > 1.05:
        return float("nan")
    frac = max(0.0, min(1.0, frac))
    return min_v + frac * (max_v - min_v)


def process(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {args.input}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Видео: {w}x{h}, {fps:.1f} кадр/с, {total} кадров, {total / fps:.1f} с")
    print(f"Шкала: {args.min_val}-{args.max_val} {args.units}")
    print(f"Углы : начало={args.start_angle}, конец={args.end_angle}")
    print()

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    log_file = open(args.log, "w") if args.log else None
    if log_file:
        log_file.write("frame\ttime/sec\tvalue\n")

    prev_ellipse = None
    last_val = float("nan")
    idx = 0
    algo_time_sum = 0.0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % args.skip == 0:
            t_alg_start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ellipse = find_gauge_ellipse(gray, prev_ellipse=prev_ellipse)

            if ellipse is not None or prev_ellipse is not None:
                if ellipse is None:
                    stable_ellipse = prev_ellipse
                else:
                    stable_ellipse = _stabilize_ellipse(ellipse, prev_ellipse)
                    prev_ellipse = stable_ellipse

                warped, wcx, wcy, wr, _ = warp_ellipse_to_circle(frame, stable_ellipse)
                angle = find_needle_angle(warped, wcx, wcy, wr)
                last_val = angle_to_value(angle, args.start_angle, args.end_angle, args.min_val, args.max_val)
            else:
                last_val = float("nan")

            algo_time_sum += time.perf_counter() - t_alg_start
            processed_frames += 1

            t = idx / fps
            if math.isnan(last_val):
                print(f"  кадр {idx:<5d}  {t:6.2f} с  --- не распознано")
                if log_file:
                    log_file.write(f"{idx}\t\t{t:.2f}\t\t-10\n")
            else:
                print(f"  кадр {idx:<5d}  {t:6.2f} с  {last_val:6.1f} {args.units}")
                if log_file:
                    log_file.write(f"{idx}\t\t{t:.2f}\t\t{last_val:.1f}\n")

        if writer and not math.isnan(last_val):
            out = frame.copy()
            text = f"{last_val:.1f} {args.units}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(text, font, 0.9, 2)
            tx, ty = w - tw - 20, h - 20
            cv2.rectangle(out, (tx - 8, ty - th - 8), (tx + tw + 8, ty + base + 8), (0, 0, 0), -1)
            cv2.putText(out, text, (tx, ty), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(out)
        elif writer:
            writer.write(frame)

        idx += 1

    algo_fps = processed_frames / algo_time_sum if algo_time_sum > 0 else 0.0
    print(f"\nFPS: {algo_fps:.2f}")

    cap.release()
    if writer:
        writer.release()
        print(f"\nВидео сохранено: {args.output}")
    if log_file:
        log_file.close()
        print(f"Лог сохранен: {args.log}")


def main():
    p = argparse.ArgumentParser(description="Распознавание показаний индикатора по видео (с коррекцией перспективы)")
    p.add_argument("input", help="Путь к видеофайлу")
    p.add_argument("-o", "--output", help="Путь для аннотированного видео")
    p.add_argument("--log", help="Путь для лог-файла (кадр, время, показание)")
    p.add_argument("--skip", type=int, default=3, help="Каждый N-й кадр [3]")

    g = p.add_argument_group("Калибровка (все обязательные)")
    g.add_argument("--min-val", type=float, required=True)
    g.add_argument("--max-val", type=float, required=True)
    g.add_argument("--start-angle", type=float, required=True)
    g.add_argument("--end-angle", type=float, required=True)
    g.add_argument("--units", required=True)
    g.add_argument("--radius", type=int, default=None)

    args = p.parse_args()
    if not os.path.isfile(args.input):
        print(f"Ошибка: файл не найден - {args.input}", file=sys.stderr)
        sys.exit(1)

    process(args)


if __name__ == "__main__":
    main()
