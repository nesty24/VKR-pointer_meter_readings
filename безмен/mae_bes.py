import math
from pathlib import Path
from typing import List, Tuple

# ==========================
# Настройки (меняйте только эти 2 строки)
# ==========================
GT_FILE = r"C:\Users\79213\PycharmProjects\python\диплом\протестированное\безмен\ручная разметка\glare_hand.csv"
P15_FILE = r"C:\Users\79213\PycharmProjects\python\диплом\протестированное\безмен\базовый+smooth\glare_smooth (vs human).csv"

# Диапазон шкалы прибора для nMAE
SCALE_RANGE_R = 22

# Порог ошибки для Accuracy
ACC_THRESHOLD = 0.2


def read_all_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as f:
        return f.readlines()


def parse_data_line(line: str, line_no: int, file_path: Path) -> Tuple[int, float]:
    parts = line.strip().split()
    if len(parts) < 3:
        raise ValueError(
            f"Некорректный формат в файле '{file_path}' на строке {line_no}: "
            f"ожидалось минимум 3 столбца, получено {len(parts)}."
        )
    frame = int(parts[0])
    value = float(parts[2])
    return frame, value


def format_frame_list(frames: List[int]) -> str:
    if not frames:
        return "нет"
    return ",".join(str(x) for x in frames)


def circular_error(pred_value: float, gt_value: float, scale_range: float) -> float:
    abs_diff = abs(pred_value - gt_value)
    return min(abs_diff, scale_range - abs_diff)


def calculate_metrics(gt_path: Path, pred_path: Path) -> int:
    gt_lines = read_all_lines(gt_path)
    pred_lines = read_all_lines(pred_path)

    print(f"GT файл:   {gt_path}")
    print(f"P15 файл:  {pred_path}")
    print(f"Строк в GT:  {len(gt_lines)}")
    print(f"Строк в P15: {len(pred_lines)}")

    if len(gt_lines) != len(pred_lines):
        print("Ошибка: количество строк в файлах не совпадает. Расчеты остановлены.")
        return 1

    if len(gt_lines) <= 1:
        print("Ошибка: в файлах нет данных для расчета (только заголовок или пусто).")
        return 1

    used_rows: List[Tuple[int, float, float, float]] = []
    dropped_gt_minus10_frames: List[int] = []
    dropped_pred_minus10_frames: List[int] = []
    mismatched_frames: List[Tuple[int, int]] = []

    for line_no, (gt_line, pred_line) in enumerate(
        zip(gt_lines[1:], pred_lines[1:]), start=2
    ):
        if not gt_line.strip() and not pred_line.strip():
            continue

        gt_frame, gt_value_raw = parse_data_line(gt_line, line_no, gt_path)
        pred_frame, pred_value_raw = parse_data_line(pred_line, line_no, pred_path)

        if gt_frame != pred_frame:
            mismatched_frames.append((gt_frame, pred_frame))
            continue

        # N учитывает только валидные GT кадры (GT != -10).
        if gt_value_raw == -10:
            dropped_gt_minus10_frames.append(gt_frame)
            continue

        # n учитывает только валидные кадры, где алгоритм дал ответ (P15 != -10).
        if pred_value_raw == -10:
            dropped_pred_minus10_frames.append(gt_frame)
            continue

        # Для индикатора оставляем значения с 1 знаком после запятой.
        gt_value = round(gt_value_raw, 1)
        pred_value = round(pred_value_raw, 1)
        delta_i = round(circular_error(pred_value, gt_value, SCALE_RANGE_R), 1)

        used_rows.append((gt_frame, pred_value, gt_value, delta_i))

    all_matched_gt_frames = [row[0] for row in used_rows] + dropped_pred_minus10_frames
    N = len(all_matched_gt_frames)
    n = len(used_rows)

    dropped_union_sorted = sorted(
        set(dropped_gt_minus10_frames + dropped_pred_minus10_frames)
    )

    print()
    print(
        "Delta_i = min(|p_i - g_i|, R - |p_i - g_i|) "
        "по каждому использованному кадру:"
    )
    for frame, pred_value, gt_value, delta_i in used_rows:
        abs_diff = abs(pred_value - gt_value)
        print(
            f"  кадр {frame:5d}: "
            f"p_i={pred_value:6.1f}, g_i={gt_value:6.1f}, "
            f"|p_i-g_i|={abs_diff:.1f}, "
            f"Delta_i=min({abs_diff:.1f}, {SCALE_RANGE_R}-{abs_diff:.1f})={delta_i:.1f}"
        )

    print()
    print(f"n (кадры, где алгоритм дал ответ): {n}")
    print(f"N (все кадры с GT, где GT != -10): {N}")
    print(f"Кадры исключены из N (GT = -10): {len(dropped_gt_minus10_frames)}")
    print(
        "Номера исключенных из N кадров: "
        f"{format_frame_list(dropped_gt_minus10_frames)}"
    )
    print(
        "Кадры, где алгоритм не дал ответ (P15 = -10): "
        f"{len(dropped_pred_minus10_frames)}"
    )
    print(
        "Номера кадров без ответа алгоритма: "
        f"{format_frame_list(dropped_pred_minus10_frames)}"
    )
    print(
        "Всего отброшенных кадров (-10 в GT или P15): "
        f"{len(dropped_union_sorted)}"
    )
    print(f"Номера всех отброшенных кадров: {format_frame_list(dropped_union_sorted)}")

    if mismatched_frames:
        print()
        print(
            "Внимание: обнаружены несовпадения номера кадра между файлами "
            f"({len(mismatched_frames)} строк). Эти строки тоже пропущены."
        )
        print(
            "Примеры (GT_frame, P15_frame): "
            + ", ".join(f"({a},{b})" for a, b in mismatched_frames[:10])
        )

    if N == 0:
        print()
        print("SR не может быть рассчитан: N = 0 (нет валидных GT кадров).")
        return 1

    sr = (n / N) * 100
    print()
    print(
        f"SR = n / N * 100% = {n} / {N} * 100% = {sr:.4f}%"
    )

    if n == 0:
        print()
        print(
            "Метрики ошибок (MAE/Median AE/nMAE/P95/Acc) не могут быть рассчитаны: "
            "нет валидных кадров, где алгоритм дал ответ."
        )
        return 1

    errors = [row[3] for row in used_rows]
    sorted_errors = sorted(errors)
    m = len(sorted_errors)

    mae = sum(errors) / n

    if m % 2 == 1:
        median_idx = m // 2
        median_ae = float(sorted_errors[median_idx])
        median_comment = (
            f"нечетное число ошибок, берем элемент #{median_idx + 1} "
            f"(0-based индекс {median_idx})"
        )
    else:
        right_idx = m // 2
        left_idx = right_idx - 1
        median_ae = (sorted_errors[left_idx] + sorted_errors[right_idx]) / 2
        median_comment = (
            "четное число ошибок, берем среднее двух центральных элементов: "
            f"#{left_idx + 1} и #{right_idx + 1}"
        )

    if SCALE_RANGE_R <= 0:
        print()
        print("Ошибка: SCALE_RANGE_R должен быть больше 0.")
        return 1
    nmae = (mae / SCALE_RANGE_R) * 100

    p95_rank = math.ceil(0.95 * m)
    p95_idx = max(0, p95_rank - 1)
    p95 = float(sorted_errors[p95_idx])

    correct_frames = [row for row in used_rows if row[3] <= ACC_THRESHOLD]
    acc = (len(correct_frames) / n) * 100

    print()
    print(f"MAE = (1/n) * sum(Delta_i) = {mae:.4f}")
    print()
    print(f"Отсортированные Delta_i для Median AE и P95: {sorted_errors}")
    print(f"Median AE: {median_comment}")
    print(f"Median AE = {median_ae:.4f}")
    print()
    print(
        "nMAE = MAE / R * 100% = "
        f"{mae:.4f} / {SCALE_RANGE_R} * 100% = {nmae:.4f}%"
    )
    print()
    print(
        "P95 (nearest-rank) = значение на позиции ceil(0.95 * m): "
        f"ceil(0.95 * {m}) = {p95_rank}, индекс {p95_idx}, P95 = {p95:.4f}"
    )

    print()
    print(f"Сравнение Delta_i с threshold={ACC_THRESHOLD}:")
    for frame, pred_value, gt_value, delta_i in used_rows:
        status = "OK" if delta_i <= ACC_THRESHOLD else "FAIL"
        sign = "<=" if delta_i <= ACC_THRESHOLD else ">"
        abs_diff = abs(pred_value - gt_value)
        print(
            f"  кадр {frame:5d}: Delta_i=min({abs_diff:.1f}, {SCALE_RANGE_R}-{abs_diff:.1f})="
            f"{delta_i:.1f} {sign} {ACC_THRESHOLD} -> {status}"
        )

    print()
    print(
        "Acc = (кадры где Delta_i <= threshold) / n * 100% = "
        f"{len(correct_frames)} / {n} * 100% = {acc:.4f}%"
    )

    # Единый итоговый блок, чтобы все метрики были видны сразу.
    print()
    print("=" * 70)
    print("ИТОГ ПО ВСЕМ МЕТРИКАМ")
    print("=" * 70)
    print(f"SR      = n / N * 100%                         = {n} / {N} * 100% = {sr:.4f}%")
    print(f"MAE     = (1/n) * sum(Delta_i)                 = (1/{n}) * {sum(errors):.4f} = {mae:.4f}")
    print(f"MedianAE= median(sorted(Delta_i))              = {median_ae:.4f}")
    print(f"nMAE    = MAE / R * 100%                       = {mae:.4f} / {SCALE_RANGE_R} * 100% = {nmae:.4f}%")
    print(f"P95     = percentile_95(sorted(Delta_i))       = {p95:.4f}")
    print(
        f"Acc     = count(Delta_i <= threshold) / n * 100% "
        f"= {len(correct_frames)} / {n} * 100% = {acc:.4f}% (threshold={ACC_THRESHOLD})"
    )
    print("=" * 70)
    return 0


def build_input_paths() -> Tuple[Path, Path]:
    return Path(GT_FILE), Path(P15_FILE)


def main() -> int:
    gt_path, pred_path = build_input_paths()

    if not gt_path.exists():
        print(f"Ошибка: GT файл не найден: {gt_path}")
        return 1

    if not pred_path.exists():
        print(f"Ошибка: P15 файл не найден: {pred_path}")
        return 1

    return calculate_metrics(gt_path, pred_path)


if __name__ == "__main__":
    raise SystemExit(main())
