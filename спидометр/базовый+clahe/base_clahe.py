import argparse
import math
import os
import sys
import time

import cv2
import numpy as np


def find_gauge(gray, radius=None, prev=None):
    h, w = gray.shape
    short = min(h, w)
    r_min = int(radius * 0.85) if radius else int(short * 0.20)
    r_max = int(radius * 1.15) if radius else int(short * 0.55)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    for dp, p2 in [(1.2, 50), (1.5, 30)]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=r_min * 2,
            param1=100, param2=p2,
            minRadius=r_min, maxRadius=r_max)
        if circles is not None:
            break
    else:
        return None

    circles = np.round(circles[0]).astype(int)
    cx_img, cy_img = w / 2, h / 2

    best_i, best_s = 0, -1e9
    for i, (x, y, r) in enumerate(circles):
        s = r - 0.3 * math.hypot(x - cx_img, y - cy_img)
        if prev:
            s += 100 - 0.5 * math.hypot(x - prev[0], y - prev[1])
        if s > best_s:
            best_i, best_s = i, s

    return tuple(int(v) for v in circles[best_i])


def find_needle_angle(frame, cx, cy, r):
    h, w = frame.shape[:2]
    pad = int(r * 1.05)
    x1, y1 = max(cx - pad, 0), max(cy - pad, 0)
    x2, y2 = min(cx + pad, w), min(cy + pad, h)
    crop = frame[y1:y2, x1:x2]
    lx, ly = cx - x1, cy - y1

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ring_mask = np.zeros_like(binary)
    cv2.circle(ring_mask, (lx, ly), int(r * 0.78), 255, -1)
    cv2.circle(ring_mask, (lx, ly), int(r * 0.12), 0, -1)
    binary = cv2.bitwise_and(binary, ring_mask)

    mask_pixels = cv2.countNonZero(ring_mask)
    white_pixels = cv2.countNonZero(binary)
    if mask_pixels > 0 and white_pixels / mask_pixels > 0.5:
        binary = cv2.bitwise_and(cv2.bitwise_not(binary), ring_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt, best_score = None, -1
    for cnt in contours:
        if cv2.contourArea(cnt) < 80 or len(cnt) < 5:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        long_s, short_s = max(rw, rh), max(min(rw, rh), 1)
        aspect = long_s / short_s
        if aspect < 3.0 or long_s < r * 0.25:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        dist = math.hypot(M["m10"]/M["m00"] - lx, M["m01"]/M["m00"] - ly)
        score = long_s * aspect / max(dist, 1)
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

    print(f"Видео: {w}x{h}, {fps:.1f} кадр/с, {total} кадров, {total/fps:.1f} с")
    print(f"Шкала: {args.min_val}-{args.max_val} {args.units}")
    print(f"Углы : начало={args.start_angle}, конец={args.end_angle}")
    print()

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    log_file = open(args.log, 'w') if args.log else None
    if log_file:
        log_file.write("frame\ttime/sec\tvalue\n")

    prev_gauge = None
    last_val = float('nan')
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
            gauge = find_gauge(gray, radius=args.radius, prev=prev_gauge)

            if gauge:
                prev_gauge = gauge
                cx, cy, r = gauge
                angle = find_needle_angle(frame, cx, cy, r)
                last_val = angle_to_value(angle, args.start_angle, args.end_angle, args.min_val, args.max_val)
            else:
                last_val = float('nan')

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
            cv2.rectangle(out, (tx-8, ty-th-8), (tx+tw+8, ty+base+8), (0,0,0), -1)
            cv2.putText(out, text, (tx, ty), font, 0.9, (255,255,255), 2, cv2.LINE_AA)
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
    p = argparse.ArgumentParser(
        description="Распознавание показаний спидометра по видео (CLAHE)")
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
        print(f"Ошибка: файл не найден — {args.input}", file=sys.stderr)
        sys.exit(1)

    process(args)


if __name__ == "__main__":
    main()
