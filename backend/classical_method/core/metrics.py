import numpy as np
import cv2

def calculate_complexity(cluster_labels, config):
    h, w = cluster_labels.shape[:2]
    canvas_w_mm = config['canvas_width_mm']
    canvas_h_mm = config['canvas_height_mm']
    min_diam_mm = config['min_diameter_mm']

    px_per_mm = max(w, h) / max(canvas_w_mm, canvas_h_mm)
    brush_radius_px = (min_diam_mm / 2.0) * px_per_mm
    brush_area_px = np.pi * (brush_radius_px ** 2)

    count_small = 0
    count_medium = 0
    count_large = 0

    area_small = 0
    area_medium = 0
    area_large = 0

    unpaintable_max_area = 0
    unpaintable_median_area = 0

    total_perimeter = 0
    total_area = 0

    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        mask = (cluster_labels == label).astype(np.uint8)
        num_comp, comp_labels = cv2.connectedComponents(mask, connectivity=8)

        for comp_id in range(1, num_comp):
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            area = comp_mask.sum()
            if area == 0:
                continue

            total_area += area

            # Contours
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
            total_perimeter += perimeter

            # Distance transform
            dt = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            dt_values = dt[comp_mask > 0]
            
            if len(dt_values) > 0:
                max_dt = dt_values.max()
                median_dt = np.median(dt_values)
            else:
                max_dt = 0
                median_dt = 0

            if max_dt < brush_radius_px:
                unpaintable_max_area += area
            if median_dt < brush_radius_px:
                unpaintable_median_area += area

            # Bins
            if area < 2 * brush_area_px:
                count_small += 1
                area_small += area
            elif area <= 5 * brush_area_px:
                count_medium += 1
                area_medium += area
            else:
                count_large += 1
                area_large += area

    if total_area == 0:
        total_area = 1

    pct_small = (area_small / total_area) * 100
    pct_medium = (area_medium / total_area) * 100
    pct_large = (area_large / total_area) * 100

    pct_unpaintable_max = (unpaintable_max_area / total_area) * 100
    pct_unpaintable_median = (unpaintable_median_area / total_area) * 100

    border_density = total_perimeter / total_area

    # overall complexity score
    complexity_score = pct_small * 2 + pct_unpaintable_max * 5 + border_density * 10

    return {
        'count_small': count_small,
        'count_medium': count_medium,
        'count_large': count_large,
        'area_small': area_small,
        'area_medium': area_medium,
        'area_large': area_large,
        'pct_small': pct_small,
        'pct_medium': pct_medium,
        'pct_large': pct_large,
        'pct_unpaintable_max': pct_unpaintable_max,
        'pct_unpaintable_median': pct_unpaintable_median,
        'border_density': border_density,
        'complexity_score': complexity_score
    }
