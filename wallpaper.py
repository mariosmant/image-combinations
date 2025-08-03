#!/usr/bin/env python3
"""
wallpaper.py

Generate a gapless 4K mosaic (landscape or portrait), with:
  • minimal overlaps, no gaps
  • random rotation, slight & big enlargements
  • clamp loops to shrink into cell bounds
  • center overlapped images
  • if visible < 50%, move to another row/column
  • at most one big per column (landscape)
    at most one big per row    (portrait)
"""

import os
import sys
import random
import argparse
import math
import logging
from collections import defaultdict
from PIL import Image

# constants
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("mode", choices=("landscape", "portrait"),
                   help="Layout orientation")
    p.add_argument("--source_dir", required=True,
                   help="Folder with input images")
    p.add_argument("--output", required=True,
                   help="Output PNG filename")
    p.add_argument("--width", type=int, required=True,
                   help="Canvas width in px")
    p.add_argument("--height", type=int, required=True,
                   help="Canvas height in px")
    p.add_argument("--angle_pct", type=float, default=0.05,
                   help="Fraction of images to rotate")
    p.add_argument("--rotation_range", nargs=2, type=float,
                   default=[-5.0, 5.0], metavar=('MIN', 'MAX'),
                   help="Rotation degrees range")
    p.add_argument("--slight_pct", type=float, default=0.01,
                   help="Fraction for slight enlargement")
    p.add_argument("--slight_range", nargs=2, type=float,
                   default=[1.5, 2.0], metavar=('MIN', 'MAX'),
                   help="Slight enlarge factor range")
    p.add_argument("--big_pct", type=float, default=0.02,
                   help="Fraction for big enlargement")
    p.add_argument("--big_range", nargs=2, type=float,
                   default=[3.0, 4.0], metavar=('MIN', 'MAX'),
                   help="Big enlarge factor range")
    p.add_argument("--opt_iters", type=int, default=200,
                   help="Max clamp iterations")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed")
    return p.parse_args()


def load_images(folder):
    if not os.path.isdir(folder):
        logging.error("Source directory not found: %s", folder)
        sys.exit(1)

    images = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXT:
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert("RGBA")
            images.append(img)
        except Exception as e:
            logging.warning("Skipping %s: %s", fname, e)
    if not images:
        logging.error("No valid images in %s", folder)
        sys.exit(1)
    return images


def assign_indices(n, pct):
    count = max(1, int(n * pct))
    return set(random.sample(range(n), count))


def two_stage_layout(weights, W, H):
    """Divide canvas into rows and then columns per weight."""
    n = len(weights)
    # approximate number of rows
    rows = max(1, int(math.sqrt(n * (H / W))))
    base, rem = divmod(n, rows)
    sizes = [base + 1 if i < rem else base for i in range(rows)]
    total_weight = sum(weights)

    cells, idx = [], 0
    y0 = 0.0
    for sz in sizes:
        group = list(range(idx, idx + sz))
        row_weight = sum(weights[i] for i in group)
        row_h = H * (row_weight / total_weight)
        x0 = 0.0
        for i in group:
            w = W * (weights[i] / row_weight)
            cells.append((i, x0, y0, w, row_h))
            x0 += w
        y0 += row_h
        idx += sz

    # sort back by image index
    cells.sort(key=lambda c: c[0])
    return cells


def clamp_scales(items, cells):
    """
    Clamp each item's factor so its rotated bounding box
    exactly covers its cell (no gaps, no overlap).
    Returns True if any factor changed.
    """
    changed = False
    for (i, x, y, cw, ch), it in zip(cells, items):
        ow, oh = it["ow"], it["oh"]
        θ = math.radians(abs(it["angle"]))
        c, s = math.cos(θ), math.sin(θ)

        # rotated bounding box dims
        bw = ow * c + oh * s
        bh = ow * s + oh * c
        # base scale to fill cell
        base = max(cw / ow, ch / oh)
        # allowed factor range without gaps
        fmin = max(cw / bw, ch / bh) / base
        fmax = min(cw / bw, ch / bh) / base

        old = it["factor"]
        if fmin <= fmax:
            it["factor"] = min(max(old, fmin), fmax)
        else:
            # if no valid range, at least avoid gaps
            it["factor"] = max(old, fmin)

        if abs(it["factor"] - old) > 1e-6:
            changed = True
    return changed


def enforce_orientation(cells, items, big_idxs, mode):
    """
    Ensure at most one big image per column (landscape)
    or per row (portrait).
    Swap bigs with nearest non-big in next col/row.
    """
    group_map = defaultdict(list)
    key_idx = 1 if mode == "landscape" else 2  # x or y
    for idx, cell in enumerate(cells):
        grp_coord = int(round(cell[key_idx]))
        group_map[grp_coord].append(idx)

    for coord, idxs in group_map.items():
        bigs = [i for i in idxs if i in big_idxs]
        for extra in bigs[1:]:
            # look for swap target in larger coord
            for c2, idxs2 in group_map.items():
                if c2 <= coord:
                    continue
                for j in idxs2:
                    if j not in big_idxs:
                        cells[extra], cells[j] = cells[j], cells[extra]
                        big_idxs.remove(extra)
                        big_idxs.add(j)
                        break
                else:
                    continue
                break


def handle_visibility(items, cells, big_idxs, slight_idxs, mode):
    """
    For each big/slight item, compute visible ratio.
    If < 0.5, swap its cell with a non-problematic in
    another row/col. Repeat until all ≥ 0.5 or no fix.
    """
    max_loops = len(items)
    for _ in range(max_loops):
        moved = False
        for idx in list(big_idxs | slight_idxs):
            cell = cells[idx]
            vis_ratio = compute_visible_ratio(items[idx], cell)
            if vis_ratio < 0.5:
                target = find_swap_target(cells, idx, big_idxs, mode)
                if target is not None:
                    cells[idx], cells[target] = cells[target], cells[idx]
                    moved = True
        if not moved:
            break


def compute_visible_ratio(item, cell):
    """
    Compute area of (cell ∩ rotated_thumb) / thumb_area.
    """
    _, x, y, cw, ch = cell
    ow, oh, f, θ = item["ow"], item["oh"], item["factor"], item["angle"]
    # create a bounding thumb in memory
    scale = max(cw / ow, ch / oh) * f
    tw, th = int(ow * scale), int(oh * scale)
    θrad = θ
    thumb = item["img"].resize((tw, th), Image.LANCZOS)
    if θ != 0:
        thumb = thumb.rotate(θ, resample=Image.BICUBIC, expand=True)
    tw, th = thumb.size
    x0 = x - (tw - cw) / 2
    y0 = y - (th - ch) / 2
    # overlap region
    left = max(x0, x)
    top = max(y0, y)
    right = min(x0 + tw, x + cw)
    bottom = min(y0 + th, y + ch)
    if right <= left or bottom <= top:
        return 0.0
    vis_area = (right - left) * (bottom - top)
    return vis_area / (tw * th)


def find_swap_target(cells, idx, big_idxs, mode):
    """
    Find a non-big/slight index to swap into a different row/col.
    """
    key_idx = 2 if mode == "portrait" else 1
    my_coord = int(round(cells[idx][key_idx]))
    for j, cell in enumerate(cells):
        if j in big_idxs:
            continue
        coord = int(round(cell[key_idx]))
        if coord != my_coord:
            return j
    return None


def render_canvas(args):
    random.seed(args.seed)
    imgs = load_images(args.source_dir)

    n = len(imgs)
    angle_idxs = assign_indices(n, args.angle_pct)
    big_idxs   = assign_indices(n, args.big_pct)
    # safe sample from list
    rem_list    = [i for i in range(n) if i not in big_idxs]
    slight_idxs = set(random.sample(rem_list,
                                    max(1, int(n * args.slight_pct))))

    # prepare item dicts
    items = []
    for i, img in enumerate(imgs):
        ow, oh = img.size
        if i in big_idxs:
            f = random.uniform(*args.big_range)
        elif i in slight_idxs:
            f = random.uniform(*args.slight_range)
        else:
            f = 1.0
        a = (random.uniform(*args.rotation_range)
             if i in angle_idxs else 0.0)
        items.append({
            "img": img, "ow": ow, "oh": oh,
            "factor": f, "angle": a
        })

    # initial layout
    weights = [it["factor"] for it in items]
    cells = two_stage_layout(weights, args.width, args.height)

    # clamp to avoid gaps/overlaps
    for _ in range(args.opt_iters):
        if not clamp_scales(items, cells):
            break

    # enforce orientation rule
    enforce_orientation(cells, items, big_idxs, args.mode)

    # re-clamp after swaps
    clamp_scales(items, cells)

    # handle <50% visibility
    handle_visibility(items, cells, big_idxs, slight_idxs, args.mode)

    # final render
    canvas = Image.new("RGBA", (args.width, args.height), (0, 0, 0, 255))
    # draw small→large to layer big on top
    for idx in sorted(range(n), key=lambda i: items[i]["factor"]):
        it = items[idx]
        cell = cells[idx]
        _, x, y, cw, ch = cell
        ow, oh, f, θ = it["ow"], it["oh"], it["factor"], it["angle"]

        scale = max(cw / ow, ch / oh) * f
        thumb = it["img"].resize(
            (int(ow * scale), int(oh * scale)), Image.LANCZOS)
        if θ != 0.0:
            thumb = thumb.rotate(θ, resample=Image.BICUBIC, expand=True)

        tw, th = thumb.size
        x0 = int(round(x - (tw - cw) / 2))
        y0 = int(round(y - (th - ch) / 2))

        # crop to cell
        left = max(x, x0); top = max(y, y0)
        right = min(x + cw, x0 + tw)
        bottom = min(y + ch, y0 + th)
        if left < right and top < bottom:
            crop = (
                int(left - x0), int(top - y0),
                int(right - x0), int(bottom - y0)
            )
            region = thumb.crop(crop)
            canvas.paste(region, (int(left), int(top)), region)

    canvas.convert("RGB").save(args.output, "PNG")
    logging.info("Saved %s", args.output)


def main():
    args = parse_args()
    try:
        render_canvas(args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
