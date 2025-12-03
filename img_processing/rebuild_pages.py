import os
import re
from glob import glob
from collections import defaultdict

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ---------- generic helpers ----------

def extract_line_number(filename: str):
    """
    Extract numeric line index from filenames like '..._line_0005_...'.
    Returns int or +inf if no match (so it sorts last).
    """
    m = re.search(r'line[_\-]?(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


def _localname(tag: str) -> str:
    """Strip namespace from an XML tag name."""
    return tag.split('}')[-1] if '}' in tag else tag


def _detect_flavour(root) -> str:
    """
    Heuristically detect PAGE vs ALTO based on namespace/root tags.
    Returns 'PAGE' or 'ALTO'.
    """
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[1:].split("}")[0].lower()
    tag0 = _localname(root.tag).lower()

    if "primaresearch.org/page" in ns or tag0 in ("pcgts", "page"):
        return "PAGE"
    if "alto" in ns or tag0 == "alto":
        return "ALTO"

    # Fallback: scan for typical PAGE nodes
    for el in root.iter():
        ln = _localname(el.tag)
        if ln in ("PcGts", "TextRegion"):
            return "PAGE"

    return "ALTO"


def _register_default_and_common_ns(root):
    """
    Register default namespace and a few common prefixes
    to avoid ns0/ns1 in the output.
    """
    if root.tag.startswith('{'):
        uri = root.tag[1:].split('}')[0]
        ET.register_namespace('', uri)

    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


# ---------- page-key helpers ----------

_PAGE_KEY_RE = re.compile(r'^(?P<key>.+?)_?line[_\-]?\d+', re.IGNORECASE)


def infer_page_key_from_filename(fname: str) -> str | None:
    """
    From a crop filename like 'XXVI.14_1_line_0005_aug2.png' return 'XXVI.14_1'.
    Fallback: stem before '_line' when present.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = _PAGE_KEY_RE.match(stem)
    return m.group('key') if m else None


def find_source_xml_and_image(data_root: str, page_key: str):
    """
    Recursively search under data_root for:
      - an XML whose basename starts with / contains page_key
      - an image whose basename starts with / contains page_key
        (excluding filenames containing '_line_' to avoid augmented line crops)

    Returns:
        (xml_path, img_path) or (None, None) if not found.
    """
    # XML candidates
    xml_candidates = glob(os.path.join(data_root, '**', '*.xml'), recursive=True)
    xml_matches = [
        p for p in xml_candidates
        if os.path.splitext(os.path.basename(p))[0].startswith(page_key)
    ]
    if not xml_matches:
        xml_matches = [p for p in xml_candidates if page_key in os.path.basename(p)]
    xml_path = xml_matches[0] if xml_matches else None

    # Image candidates
    img_candidates = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'):
        img_candidates += glob(os.path.join(data_root, '**', ext), recursive=True)

    # Exclude cropped line images
    img_candidates = [
        p for p in img_candidates
        if "_line_" not in os.path.basename(p).lower()
    ]

    img_matches = [
        p for p in img_candidates
        if os.path.splitext(os.path.basename(p))[0] == page_key
    ]
    if not img_matches:
        img_matches = [
            p for p in img_candidates
            if os.path.splitext(os.path.basename(p))[0].startswith(page_key)
        ]
    if not img_matches:
        img_matches = [p for p in img_candidates if page_key in os.path.basename(p)]

    img_path = img_matches[0] if img_matches else None

    return xml_path, img_path


# ---------- XML parsers ----------

def parse_alto_xml(xml_path):
    """
    Parse ALTO XML and return (tree, root, lines) where:
        lines = list of (x, y, w, h, element)

    It prefers TextLine->Shape->Polygon as tight bbox.
    If no polygon is found, it falls back to HPOS/VPOS/WIDTH/HEIGHT.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for tl in root.iter():
        if _localname(tl.tag) != "TextLine":
            continue

        poly_el = None
        # Try TextLine/Shape/Polygon
        for ch in tl:
            if _localname(ch.tag) == "Shape":
                for ch2 in ch:
                    if _localname(ch2.tag) == "Polygon":
                        poly_el = ch2
                        break
            if poly_el is not None:
                break

        # Or direct TextLine/Polygon
        if poly_el is None:
            for ch in tl:
                if _localname(ch.tag) == "Polygon":
                    poly_el = ch
                    break

        if poly_el is not None and (poly_el.get("POINTS") or poly_el.get("points")):
            pts_attr = poly_el.get("POINTS") or poly_el.get("points")
            s = pts_attr.replace(",", " ")
            toks = s.split()
            pts = []
            for i in range(0, len(toks) - 1, 2):
                try:
                    x = int(float(toks[i]))
                    y = int(float(toks[i + 1]))
                    pts.append((x, y))
                except ValueError:
                    continue

            if len(pts) >= 3:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                lines.append((x, y, w, h, tl))
                continue

        # Fallback: rectangular attributes on TextLine
        hpos = int(float(tl.attrib.get('HPOS', 0)))
        vpos = int(float(tl.attrib.get('VPOS', 0)))
        height = int(float(tl.attrib.get('HEIGHT', 0)))
        width = int(float(tl.attrib.get('WIDTH', 0)))
        lines.append((hpos, vpos, width, height, tl))

    return tree, root, lines


def parse_page_xml(xml_path):
    """
    Parse PAGE XML and return (tree, root, lines) where:
        lines = list of (x, y, w, h, element)

    It uses TextLine/Coords@points and computes a tight bbox.
    Baseline-only lines are skipped.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for tl in root.iter():
        if _localname(tl.tag) != "TextLine":
            continue

        coords = None
        for ch in tl:
            if _localname(ch.tag) == "Coords" and ch.get("points"):
                coords = ch.get("points")
                break
        if not coords:
            continue

        s = coords.replace(",", " ")
        toks = s.split()
        pts = []
        for i in range(0, len(toks) - 1, 2):
            try:
                x = int(float(toks[i]))
                y = int(float(toks[i + 1]))
                pts.append((x, y))
            except ValueError:
                continue

        if len(pts) < 3:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        lines.append((x, y, w, h, tl))

    return tree, root, lines


# ---------- paste helper (no mask) ----------

def _paste_line_safe(canvas, x, y, w, h, line_img):
    """
    Paste a line (line_img) into canvas at (x, y, w, h) with no masking.

    Returns:
        True if something was pasted, False otherwise.
    """
    if line_img is None:
        return False

    H, W = canvas.shape[:2]
    if w <= 0 or h <= 0:
        return False

    # Clip target ROI to canvas bounds
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return False

    # Resize to requested bbox, then crop to clipped ROI
    src = cv2.resize(line_img, (w, h), interpolation=cv2.INTER_LINEAR)
    off_x = x0 - x
    off_y = y0 - y
    src = src[off_y:off_y + rh, off_x:off_x + rw]

    if src.size == 0:
        return False

    canvas[y0:y0 + rh, x0:x0 + rw] = src
    return True


# ---------- metadata updates ----------

def update_alto_with_augmented(tree, root, subdir):
    """
    Update ALTO XML TextLine elements with deterministic IDs and a STYLE marker
    to track reconstruction source.
    """
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        tl.set("ID", f"aug_{subdir}_line_{i:04d}")
        tl.set("STYLE", f"reconstructed_from_{subdir}")
    return tree


def update_page_with_augmented(tree, root, subdir):
    """
    Update PAGE XML TextLine elements, appending a marker into @custom
    to track reconstruction source and index.
    """
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        prev = tl.get("custom", "")
        marker = f"reconstructed_from:{subdir};index:{i:04d}"
        tl.set("custom", (prev + ";" if prev else "") + marker)
    return tree


# ---------- main rebuild ----------

def rebuild_pages_by_method(base_folder="augmented_output",
                            data_root="data_root_with_many_manuscripts",
                            output_folder="rebuilt_pages",
                            augmentations=None):
    """
    Rebuild full pages from augmented line crops, without mixing manuscripts.

    - base_folder is expected to contain method subfolders (e.g. bezier, L2A, affine...),
      or directly images.
    - If the structure is <base>/<method>/<page_key>/*, the rebuild is done per page_key.
    - Otherwise, images inside each method subfolder are partitioned by page_key and rebuilt.

    Parameters
    ----------
    base_folder : str
        Root folder containing augmented line images (grouped by method).
    data_root : str
        Root folder where original XML and page images are stored.
    output_folder : str
        Folder where reconstructed pages (PNG + XML) will be written.
    augmentations : int or None
        Optional upper bound on the number of augmentation passes per page.
        If None, use all available augmentations inferred from the filenames.

    Logging is minimal:
    - only two messages per successful augment:
        [OK] Image saved: <path>
        [OK] XML saved:   <path>
    """
    os.makedirs(output_folder, exist_ok=True)

    # Level-1: methods (bezier, L2A, affine, perspective...) or leaf dirs with images
    for level1 in os.listdir(base_folder):
        level1_path = os.path.join(base_folder, level1)
        if not os.path.isdir(level1_path):
            continue

        # Detect whether page_key subfolders exist under level1 (CASE A)
        subfolders = [
            d for d in os.listdir(level1_path)
            if os.path.isdir(os.path.join(level1_path, d))
        ]

        # --- inner helper: rebuild a single page from all its line images ---

        def _rebuild_for_group(image_files, page_key_label):
            if not image_files:
                return

            # First, find XML (and a candidate image) via filesystem search
            xml_path, original_image_path = find_source_xml_and_image(data_root, page_key_label)
            if not xml_path:
                # Hard failure: cannot rebuild this page
                return

            # Parse XML once; we will reuse it for:
            # - reading ALTO/PAGE layout
            # - retrieving the correct page image from <fileName>
            tree0 = ET.parse(xml_path)
            root0 = tree0.getroot()
            flavour = _detect_flavour(root0)

            # Try to override original_image_path using <fileName> in ALTO/PAGE Description
            file_name_from_xml = None
            for el in root0.iter():
                if _localname(el.tag) == "fileName":
                    if el.text:
                        file_name_from_xml = el.text.strip()
                    break

            if file_name_from_xml:
                candidate = file_name_from_xml
                if not os.path.isabs(candidate):
                    candidate = os.path.join(os.path.dirname(xml_path), candidate)
                if os.path.exists(candidate):
                    original_image_path = candidate

            if not original_image_path:
                # No page image found: cannot rebuild
                return

            original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
            if original_image is None:
                return

            H, W = original_image.shape[:2]

            # Parse line bounding boxes from XML
            if flavour == "PAGE":
                _, _, boxes = parse_page_xml(xml_path)
            else:
                _, _, boxes = parse_alto_xml(xml_path)

            if not boxes:
                return

            # Group files by line number inferred from filename
            grouped_by_line = defaultdict(list)
            for f in image_files:
                ln = extract_line_number(os.path.basename(f))
                grouped_by_line[ln].append(f)

            for ln in grouped_by_line:
                grouped_by_line[ln].sort()
            sorted_line_numbers = sorted(grouped_by_line.keys())

            # Compute actual maximum number of augmentations available for this page
            max_aug = max(len(files_for_line) for files_for_line in grouped_by_line.values())
            if max_aug == 0:
                return

            # Respect optional upper bound if provided
            if augmentations is not None:
                max_aug = min(max_aug, int(augmentations))

            # Prepare output: one directory per method/page_key
            out_dir = os.path.join(output_folder, f"{level1}_{page_key_label}")
            os.makedirs(out_dir, exist_ok=True)

            # Map “line_xxxx” to bbox index when plausible
            def _bbox_index_for_line(ln):
                if ln != float('inf') and 0 <= ln < len(boxes):
                    return ln
                return None

            # Loop over actual augmentation indices available
            for augmentation_index in range(max_aug):
                canvas = np.full((H, W, 3), 255, dtype=np.uint8)
                something_pasted = False

                used_boxes = set()

                # 1) Direct placement: use same index as in filename if plausible
                for ln in sorted_line_numbers:
                    idx = _bbox_index_for_line(ln)
                    if idx is None or idx in used_boxes:
                        continue

                    files_for_line = grouped_by_line[ln]
                    if augmentation_index >= len(files_for_line):
                        continue

                    path_img = files_for_line[augmentation_index]
                    line_img = cv2.imread(path_img, cv2.IMREAD_COLOR)
                    if line_img is None:
                        continue

                    x, y, w, h, _ = boxes[idx]
                    ok = _paste_line_safe(canvas, x, y, w, h, line_img)
                    if ok:
                        used_boxes.add(idx)
                        something_pasted = True

                # 2) Fallback: lines without valid index go into remaining boxes sequentially
                seq_boxes = [i for i in range(len(boxes)) if i not in used_boxes]
                seq_ptr = 0
                for ln in sorted_line_numbers:
                    if _bbox_index_for_line(ln) is not None:
                        continue
                    if seq_ptr >= len(seq_boxes):
                        break

                    files_for_line = grouped_by_line[ln]
                    if augmentation_index >= len(files_for_line):
                        continue

                    path_img = files_for_line[augmentation_index]
                    line_img = cv2.imread(path_img, cv2.IMREAD_COLOR)
                    if line_img is None:
                        continue

                    idx = seq_boxes[seq_ptr]
                    seq_ptr += 1

                    x, y, w, h, _ = boxes[idx]
                    ok = _paste_line_safe(canvas, x, y, w, h, line_img)
                    if ok:
                        something_pasted = True

                # If nothing was pasted, skip saving a white page
                if not something_pasted:
                    continue

                # Save rebuilt image
                img_out = os.path.join(
                    out_dir,
                    f"reconstructed_{page_key_label}_{augmentation_index + 1}.png"
                )
                cv2.imwrite(img_out, canvas)

                # Save XML annotated with the reconstruction marker
                xml_tree = ET.parse(xml_path)
                xml_root = xml_tree.getroot()
                _register_default_and_common_ns(xml_root)
                if flavour == "PAGE":
                    xml_tree = update_page_with_augmented(xml_tree, xml_root, page_key_label)
                else:
                    xml_tree = update_alto_with_augmented(xml_tree, xml_root, page_key_label)

                xml_out = os.path.join(
                    out_dir,
                    f"reconstructed_{page_key_label}_{augmentation_index + 1}.xml"
                )
                xml_tree.write(xml_out, encoding="utf-8", xml_declaration=True)

                # Only these two messages:
                print(f"[OK] Image saved: {img_out}")
                print(f"[OK] XML saved:   {xml_out}")

        # ===== CASE A: there are page_key subfolders =====
        if subfolders:
            for page_key in subfolders:
                page_dir = os.path.join(level1_path, page_key)
                if not os.path.isdir(page_dir):
                    continue
                imgs = [
                    f for f in glob(os.path.join(page_dir, "*.jpg"))
                    if "debug overlay" not in f.lower()
                ]
                imgs += [
                    f for f in glob(os.path.join(page_dir, "*.png"))
                    if "debug overlay" not in f.lower()
                ]
                imgs.sort()
                _rebuild_for_group(imgs, page_key)
            continue

        # ===== CASE B: no subfolders → partition files by page_key =====
        flat_imgs = [
            f for f in glob(os.path.join(level1_path, "*.jpg"))
            if "debug overlay" not in f.lower()
        ]
        flat_imgs += [
            f for f in glob(os.path.join(level1_path, "*.png"))
            if "debug overlay" not in f.lower()
        ]
        if not flat_imgs:
            continue

        buckets = defaultdict(list)
        for f in flat_imgs:
            key = infer_page_key_from_filename(os.path.basename(f)) or "unknown_page"
            buckets[key].append(f)

        for key, files in buckets.items():
            files.sort()
            _rebuild_for_group(files, key)


# Example usage (adjust paths to your setup):
# if __name__ == "__main__":
#     rebuild_pages_by_method(
#         base_folder="data/augmented_lines",
#         data_root="data",
#         output_folder="data/rebuilt_pages",
#         augmentations=4,
#     )
