import os
import cv2
import numpy as np
import lxml.etree as ET
import shutil
from tqdm import tqdm
from copy import deepcopy

# =========================
# Global settings / speed
# =========================
RNG = np.random.default_rng(12345)  # deterministic randomness
KERNEL_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
PNG_PARAMS_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]
USE_JPG = False
JPG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(os.cpu_count() or 1)
except:
    pass


# =========================
# Color-space helpers
# =========================
def _to_lab(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

def _from_lab(image_lab):
    return cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)


# =========================
# Augmentations
# =========================
def apply_morphological_transforms(image_bgr):
    lab = _to_lab(image_bgr)
    L, A, B = cv2.split(lab)

    if RNG.random() < 0.5:
        L = cv2.dilate(L, KERNEL_2, iterations=int(RNG.integers(1, 4)))
    if RNG.random() < 0.5:
        L = cv2.erode(L, KERNEL_2, iterations=int(RNG.integers(1, 3)))
    if RNG.random() < 0.3:
        L = cv2.GaussianBlur(L, (3, 3), 0)
    if RNG.random() < 0.5:
        L = cv2.morphologyEx(L, cv2.MORPH_OPEN, KERNEL_2)

    return _from_lab(cv2.merge([L, A, B]))


def apply_random_shadow(image_bgr):
    h, w = image_bgr.shape[:2]
    out = image_bgr.astype(np.float32)
    for _ in range(int(RNG.integers(1, 3))):
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = int(RNG.integers(0, w)), int(RNG.integers(0, h))
        axes = (int(RNG.integers(12, 55)), int(RNG.integers(16, 60)))
        angle = int(RNG.integers(0, 180))
        density = int(RNG.integers(10, 90))
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, density, -1)
        alpha = float(RNG.uniform(0.2, 0.55))
        out -= (mask.astype(np.float32) * alpha)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# ALTO helpers
# =========================
def _load_alto_with_ns(xml_path):
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(xml_path, parser)
    root = tree.getroot()
    nsmap = root.nsmap.copy() if hasattr(root, "nsmap") else {}
    if None in nsmap:
        nsmap['alto'] = nsmap.pop(None)
    return tree, root, nsmap

def _iter_textlines_and_points(root, ns):
    if ns and 'alto' in ns:
        lines = root.findall(".//alto:TextLine", namespaces=ns)
        bl_tag = "alto:Baseline"
    else:
        lines = root.findall(".//TextLine")
        bl_tag = "Baseline"
    for tl in lines:
        bl = tl.find(bl_tag, namespaces=ns if ns else None)
        if bl is None:
            continue
        pts = bl.attrib.get("points", "")
        if not pts:
            continue
        coords = np.array([list(map(int, p.split(","))) for p in pts.split()], dtype=np.int32)
        yield tl, bl, coords

def _write_points(bl_elem, coords):
    bl_elem.attrib["points"] = " ".join(f"{int(x)},{int(y)}" for x,y in coords)


# =========================
# Text Region Mask (ALTO + fallback)
# =========================
def text_mask_from_alto(xml_path, shape, expand_px=18):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(xml_path):
        return mask

    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(xml_path, parser)
    root = tree.getroot()
    ns = root.nsmap.copy() if hasattr(root, "nsmap") else {}
    if None in ns:
        ns["alto"] = ns.pop(None)
    def find(path): return root.findall(path, namespaces=ns if ns else None)

    polys = []
    try:
        lines = find(".//alto:TextLine") if "alto" in ns else find(".//TextLine")
        btag = "alto:Baseline" if "alto" in ns else "Baseline"
        for tl in lines:
            bl = tl.find(btag, namespaces=ns if ns else None)
            if bl is None: continue
            pts = bl.attrib.get("points","").strip()
            if not pts: continue
            arr = np.array([list(map(int,p.split(","))) for p in pts.split()], dtype=np.int32)
            polys.append(arr)
    except:
        polys = []

    if polys:
        all_pts = np.vstack(polys)
        hull = cv2.convexHull(all_pts.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
    else:
        blocks = find(".//alto:TextBlock") if "alto" in ns else find(".//TextBlock")
        for b in blocks:
            try:
                x,y = int(float(b.attrib["HPOS"])), int(float(b.attrib["VPOS"]))
                ww,hh = int(float(b.attrib["WIDTH"])), int(float(b.attrib["HEIGHT"]))
                cv2.rectangle(mask, (x,y), (x+ww,y+hh), 255, -1)
            except:
                pass

    if expand_px>0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(expand_px,expand_px))
        mask = cv2.dilate(mask,k,1)
    return mask


def auto_text_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,41,15)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(15,7))
    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,k,2)
    cnts,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if cnts:
        c = max(cnts,key=cv2.contourArea)
        hull = cv2.convexHull(c)
        cv2.fillConvexPoly(mask,hull,255)
        mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(25,15)),1)
    return mask


def ink_bleed_in_mask(image_bgr, mask, blur_strength=15, stretch_factor=1.08, alpha=0.35):
    if mask.sum()==0:
        return image_bgr
    out = image_bgr.copy()
    x,y,w,h = cv2.boundingRect(mask)
    roi = out[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    flipped = cv2.flip(roi,1)
    k=max(3,blur_strength|1)
    blurred=cv2.GaussianBlur(flipped,(k,k),k//3)
    H=max(1,int(h*stretch_factor))
    stretched=cv2.resize(blurred,(w,H))
    resized=cv2.resize(stretched,(w,h))
    blend=cv2.addWeighted(roi.astype(np.float32),1-alpha,
                          resized.astype(np.float32),alpha,0).astype(np.uint8)
    out[y:y+h,x:x+w]=np.where(mask_roi[...,None]>0,blend,roi)
    return out


def apply_ink_bleed_page(image_bgr, xml_path):
    mask = text_mask_from_alto(xml_path, image_bgr.shape, expand_px=18) \
            if os.path.exists(xml_path) else auto_text_mask(image_bgr)
    return ink_bleed_in_mask(image_bgr,mask)


# =========================
# Per-image processing
# =========================
def process_image(file_name, args):
    img_path = os.path.join(args.src,file_name)
    xml_path = os.path.join(args.src,os.path.splitext(file_name)[0]+".xml")
    if not os.path.exists(xml_path):
        return

    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    if img is None:
        return

    tree, root, ns = _load_alto_with_ns(xml_path)
    lines = list(_iter_textlines_and_points(root,ns))
    h,w = img.shape[:2]

    shutil.copy2(img_path, os.path.join(args.augmented_pages,file_name))
    shutil.copy2(xml_path, os.path.join(args.augmented_pages,os.path.splitext(file_name)[0]+".xml"))

    # ---- FIXED: ALWAYS 2 AUGMENTATIONS ----
    for i in range(2):
        aug = img.copy()
        aug = apply_morphological_transforms(aug)
        aug = apply_random_shadow(aug)
        if RNG.random() < 0.5:
            aug = apply_ink_bleed_page(aug, xml_path)

        tree_i = deepcopy(tree)
        root_i = tree_i.getroot()
        lines_i = list(_iter_textlines_and_points(root_i,ns))
        noise = getattr(args,"baseline_noise",1)
        for (_,_,pts0),(_,bl_i,pts_i) in zip(lines,lines_i):
            delta = RNG.integers(-noise,noise+1,pts_i.shape)
            new_pts = np.clip(pts_i+delta,[0,0],[w-1,h-1])
            _write_points(bl_i,new_pts)

        base = os.path.splitext(file_name)[0]
        img_name = f"{base}_aug{i}.jpg" if USE_JPG else f"{base}_aug{i}.png"
        cv2.imwrite(os.path.join(args.augmented_pages,img_name),aug,
                    JPG_PARAMS if USE_JPG else PNG_PARAMS_FAST)
        tree_i.write(os.path.join(args.augmented_pages,f"{base}_aug{i}.xml"),
                     pretty_print=True,encoding="UTF-8",xml_declaration=True)


# =========================
# Batch entry point
# =========================
def augmentation(args):
    os.makedirs(args.augmented_pages,exist_ok=True)
    files=[f for f in os.listdir(args.src)
           if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
    for f in tqdm(files):
        process_image(f,args)
    print("Done.")
