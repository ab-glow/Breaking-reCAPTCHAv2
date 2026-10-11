import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import os
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import cv2
import time
import csv
from datetime import datetime
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import (
    WebDriverException,
    NoSuchElementException,
    StaleElementReferenceException
)
import traceback
import sys, os, json, time
from datetime import datetime

SESSION_DIR = os.path.join("logs", datetime.now().strftime("session_%Y%m%d_%H%M%S"))
os.makedirs(SESSION_DIR, exist_ok=True)
_console_file = open(os.path.join(SESSION_DIR, "console.log"), "a", encoding="utf-8", buffering=1)

class _Tee:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        _console_file.write(data)
    def flush(self):
        self.stream.flush()
        _console_file.flush()
    def __getattr__(self, name):      # keep any other stream attributes working
        return getattr(self.stream, name)

sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)

_events_file = open(os.path.join(SESSION_DIR, "events.jsonl"), "a", encoding="utf-8", buffering=1)

def event(kind, **data):
    """One JSON object per line, easy to read back or load into pandas."""
    _events_file.write(json.dumps({"t": datetime.now().isoformat(timespec="milliseconds"),
                                   "kind": kind, **data}, default=str) + "\n")
    
# Import YOLO models (PyTorch)
from models.YOLO_Classification import predict
from models.YOLO_Segment import predict as predict_segment

# Handle optional VPN import safely for Windows
try:
    from IP import vpn
    HAS_VPN = True
except ImportError:
    HAS_VPN = False

# Configuration and Constants
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo?hl=en"  # Forces English reCAPTCHA
THRESHOLD = 0.2
USE_TOP_N_STRATEGY = False
N = 3
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'stairs', 'traffic']
MODEL = "YOLO"
TYPE1 = True  # 3x3 single-selection
TYPE2 = True  # 4x4 segmentation
TYPE3 = True  # 3x3 dynamic captcha
ENABLE_LOGS = True
ENABLE_VPN = False
ENABLE_MOUSE_MOVEMENT = True
ENABLE_COOKIES = False  # Set to True if using a specific Firefox profile

# Pause settings for reCAPTCHA animations (in seconds)
DYNAMIC_TILE_PAUSE = 2.5

# Windows profile path fallback
PATH_TO_FIREFOX_PROFILE = os.path.expanduser('~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles\\default-release')

# Create data directory for saved screenshots
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)


def click_element(driver, element):
    """
    Reliably moves to and clicks an element using Selenium ActionChains.
    """
    try:
        if ENABLE_MOUSE_MOVEMENT:
            actions = ActionChains(driver)
            actions.move_to_element(element).pause(0.1).click().perform()
        else:
            element.click()
        sleep(0.2)
    except Exception:
        driver.execute_script("arguments[0].click();", element)
        sleep(0.2)



###############################################
# FADE-ANIMATION SAFE TILE WAITING
###############################################

def _tile_state(driver, i):
    """
    Take a fresh screenshot of tile i.

    Returns:
        (src, grayscale_image)

    The image is reduced to 32x32 so comparing frames is cheap.
    """
    tile = driver.find_elements(
        By.CLASS_NAME, "rc-imageselect-tile"
    )[i]

    img = tile.find_element(By.TAG_NAME, "img")

    src = img.get_attribute("src")

    png = tile.screenshot_as_png

    arr = np.asarray(
        Image.open(BytesIO(png))
        .convert("L")
        .resize((32, 32)),
        dtype=np.float32
    )

    return src, arr


def wait_tile_settled(
    driver,
    i,
    before,
    timeout=12.0,
    sample_interval=0.10,
    stable_window=0.40,
    change_threshold=8.0,
    stable_threshold=1.5,
    minimum_wait=0.25
):
    """
    Wait for a dynamically replaced reCAPTCHA tile to finish changing.

    The function does NOT wait for a fixed animation duration.

    Instead it:

        1. remembers the image before the click
        2. waits until the tile actually changes
        3. waits until the new image remains visually stable
        4. returns once the replacement has settled

    Returns:
        ("settled", elapsed)
        ("timeout", elapsed)
    """

    _, before_arr = before

    start = time.time()

    changed = False
    stable_since = None
    frame_difference = None

    trace = []

    while time.time() - start < timeout:

        try:
            _, current = _tile_state(driver, i)

        except (
            StaleElementReferenceException,
            IndexError,
            NoSuchElementException
        ):
            time.sleep(sample_interval)
            continue

        elapsed = time.time() - start

        # How different is the current image from the image
        # that existed before we clicked?
        change_from_before = float(
            np.mean(np.abs(current - before_arr))
        )

        # Once the old image has disappeared, we compare
        # consecutive frames to determine whether the new
        # image has stopped changing.
        if not changed:

            if change_from_before >= change_threshold:
                changed = True
                stable_since = None

        else:

            # Difference from the previous sampled frame.
            if 'previous' in locals():
                frame_difference = float(
                    np.mean(np.abs(current - previous))
                )
            else:
                frame_difference = None

            if frame_difference is not None:

                if frame_difference <= stable_threshold:

                    if stable_since is None:
                        stable_since = time.time()

                    stable_duration = time.time() - stable_since

                    if (
                        stable_duration >= stable_window
                        and elapsed >= minimum_wait
                    ):
                        trace.append({
                            "time": round(elapsed, 3),
                            "change_from_before": round(
                                change_from_before, 2
                            ),
                            "frame_difference": round(
                                frame_difference, 2
                            ),
                            "state": "SETTLED"
                        })

                        event(
                            "tile_wait",
                            tile=i + 1,
                            status="settled",
                            elapsed=round(elapsed, 3),
                            trace=trace
                        )

                        return "settled", elapsed

                else:
                    # Image is still changing.
                    stable_since = None

            else:
                frame_difference = None

        trace.append({
            "time": round(elapsed, 3),
            "change_from_before": round(
                change_from_before, 2
            ),
            "frame_difference": (
                None
                if frame_difference is None
                else round(frame_difference, 2)
            ),
            "state": (
                "WAITING_FOR_CHANGE"
                if not changed
                else "WAITING_FOR_STABILITY"
            )
        })

        previous = current

        time.sleep(sample_interval)

    elapsed = time.time() - start

    event(
        "tile_wait",
        tile=i + 1,
        status="timeout",
        elapsed=round(elapsed, 3),
        trace=trace
    )

    return "timeout", elapsed




def draw_frame_tags(image_path, label_text, prob_score):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        confidence_pct = prob_score * 100
        is_match = prob_score > THRESHOLD

        border_color = "#00FF00" if is_match else "#FF0000"
        draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=4)

        banner_text = f"{label_text.upper()}: {confidence_pct:.1f}%"
        draw.rectangle([0, 0, width, 26], fill=border_color)
        draw.text((6, 5), banner_text, fill="#000000")

        img.save(image_path)
    except Exception as e:
        print(f"⚠️ Could not draw frame tags on image: {e}")


COUNT = 0

###############################################
# LIVE YOLO PREVIEW
###############################################

PREVIEW_ON = True
PREVIEW_CELL = 220
_preview_cells = {}
_preview_header = ""
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _preview_draw():
    global PREVIEW_ON
    if not PREVIEW_ON:
        return
    try:
        cell, head = PREVIEW_CELL, 36
        canvas = np.full((head + 3 * cell, 3 * cell, 3), 30, dtype=np.uint8)
        cv2.putText(canvas, _preview_header, (8, 24), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        for i, d in _preview_cells.items():
            r, c = divmod(i, 3)
            x, y = c * cell, head + r * cell
            canvas[y:y + cell, x:x + cell] = d["img"]
            color = (0, 170, 0) if d["target_p"] > THRESHOLD else (0, 0, 200)
            cv2.rectangle(canvas, (x, y), (x + cell - 1, y + cell - 1), color, 4)
            if d["clicked"]:
                cv2.rectangle(canvas, (x + 6, y + 6), (x + cell - 7, y + cell - 7), (255, 140, 0), 3)
            cv2.rectangle(canvas, (x, y), (x + cell - 1, y + 44), color, -1)
            cv2.putText(canvas, f"sees: {d['top']} {d['top_p'] * 100:.0f}%", (x + 6, y + 17),
                        _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{d['target']}: {d['target_p'] * 100:.1f}%", (x + 6, y + 37),
                        _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("YOLO view (3x3)", canvas)
        cv2.waitKey(1)
    except Exception as e:
        print(f"⚠️ Preview disabled: {e}")
        PREVIEW_ON = False


def preview_reset(header):
    global _preview_header
    _preview_cells.clear()
    _preview_header = header
    _preview_draw()


def preview_update(i, img_path, top_label, top_p, target_label, target_p):
    if not PREVIEW_ON:
        return
    try:
        img = np.asarray(Image.open(img_path).convert("RGB")
                         .resize((PREVIEW_CELL, PREVIEW_CELL)))[:, :, ::-1]
        _preview_cells[i] = {"img": img, "top": top_label, "top_p": top_p,
                             "target": target_label, "target_p": target_p, "clicked": False}
        _preview_draw()
    except Exception as e:
        print(f"⚠️ Preview update failed: {e}")


def preview_clicked(i):
    if PREVIEW_ON and i in _preview_cells:
        _preview_cells[i]["clicked"] = True
        _preview_draw()


def preview_4x4(annotated, grid):
    if not PREVIEW_ON or annotated is None:
        return
    try:
        img = annotated.copy()
        h, w = img.shape[:2]
        for r in range(4):
            for c in range(4):
                p0 = (int(c * w / 4), int(r * h / 4))
                p1 = (int((c + 1) * w / 4), int((r + 1) * h / 4))
                if grid[r][c] == 1:
                    cv2.rectangle(img, p0, p1, (0, 200, 0), 3)
                else:
                    cv2.rectangle(img, p0, p1, (90, 90, 90), 1)
        cv2.imshow("YOLO view (4x4)", img)
        cv2.waitKey(1)
    except Exception as e:
        print(f"⚠️ 4x4 preview failed: {e}")

def process_tile(i, captcha_text, class_index, driver, pause_after_click=False, defer_click=False):
    global COUNT
    print(f"Processing tile {i+1}/9 (Target: '{YOLO_CLASSES[class_index]}', Class Index: {class_index})")

    try:
        tiles = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "rc-imageselect-tile"))
        )
    except Exception as e:
        print(f"⚠️ Could not locate challenge tiles in DOM: {e}")
        return False

    if i >= len(tiles):
        print(f"⚠️ Tile index {i+1} out of range (Found {len(tiles)} tiles)")
        return False

    matched_tile = tiles[i]
    filename = f"tile_{COUNT}.png"
    tile_path = os.path.join(data_dir, filename)

    try:
        matched_tile.screenshot(tile_path)
    except Exception as e:
        print(f"⚠️ Failed to capture screenshot for tile {i+1}: {e}")
        return False

    try:
        result = predict.predict_tile(tile_path)
        current_object_probability = result[0][class_index]
        top_predicted_index = result[2]
        object_name = YOLO_CLASSES[top_predicted_index]
    except Exception as e:
        print(f"⚠️ Error during YOLO prediction on tile {i+1}: {e}")
        traceback.print_exc()
        try:
            import shutil
            shutil.copy(tile_path, os.path.join(data_dir, f"ERR_{int(time.time())}_{i+1}.png"))
        except Exception:
            pass
        return False

    preview_update(i, tile_path, object_name, float(result[0][top_predicted_index]),
                   YOLO_CLASSES[class_index], float(current_object_probability))    

    draw_frame_tags(tile_path, object_name, current_object_probability)

    new_path = os.path.join(data_dir, f"{object_name}_{filename}")
    if os.path.exists(tile_path):
        try:
            os.rename(tile_path, new_path)
        except Exception:
            pass

    confidence_pct = current_object_probability * 100
    print(f"Tile {COUNT}: YOLO sees '{object_name}' (Target Confidence: {confidence_pct:.1f}%)")
    COUNT += 1
    if defer_click:
        return float(current_object_probability)  # score only, caller decides what to click

    should_click = False
    if USE_TOP_N_STRATEGY:
        top_n_indices = sorted(range(len(result[0])), key=lambda k: result[0][k], reverse=True)[:N]
        should_click = class_index in top_n_indices
    else:
        should_click = current_object_probability > THRESHOLD
        if should_click:
            print(f" -> Match found! ({confidence_pct:.1f}% > {THRESHOLD*100:.0f}%)")

    if should_click:
        # Snapshot BEFORE the click so we can tell when the new image arrives
        before = _tile_state(driver, i) if pause_after_click else None
        click_element(driver, matched_tile)
        if pause_after_click:
            status, elapsed = wait_tile_settled(driver, i, before)
            print(f" -> tile {i+1}: {status} in {elapsed:.2f}s")
        return True

    return False


def solve_type2(driver):
    save_path = "temp"
    os.makedirs(save_path, exist_ok=True)

    try:
        captcha_text_elem = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[@id='rc-imageselect']//strong"))
        )
        captcha_text = captcha_text_elem.text
    except Exception as e:
        print("⚠️ Could not locate instruction text for Type 2 captcha:", e)
        return

    log("Type2", captcha_text)
    class_index = get_class_index(captcha_text)
    if class_index is None:
        print(f"'{captcha_text}' is not a class this project knows. Reloading...")
        _reload_challenge(driver)
        return    
    print(f"4x4 target: '{captcha_text}' -> index {class_index} ({YOLO_CLASSES[class_index]})")
    if not predict_segment.supports(class_index):
        print(f"4x4: '{captcha_text}' can't be detected by the segmentation model. Reloading...")
        _reload_challenge(driver)
        return

    clicked = 0
    try:
        img = driver.find_element(By.XPATH, "//div[@id='rc-imageselect']//img")
        img_url = img.get_attribute("src")
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            timestamp = str(time.time())
            filename = f"image_{captcha_text}_{timestamp}.png"
            full_path = os.path.join(save_path, filename)
            with open(full_path, 'wb') as f:
                f.write(response.content)

            success, grid = predict_segment.predict(class_index, full_path)
            print(f"Segmentation grid (success={success}):")
            for row in grid:
                print(row)
            preview_4x4(predict_segment.LAST_PLOT, grid)
            tiles = driver.find_elements(By.CLASS_NAME, "rc-imageselect-tile")
            for r in range(4):
                for c in range(4):
                    idx = r * 4 + c
                    if grid[r][c] == 1 and idx < len(tiles):
                        click_element(driver, tiles[idx])
                        clicked += 1
                        sleep(0.5)
    except Exception as e:
        print("⚠️ Error solving Type 2 captcha:", e)

    if clicked == 0:
        print("No 4x4 tiles selected. Reloading instead of skipping...")
        try:
            driver.find_element(By.ID, "recaptcha-reload-button").click()
        except Exception as e:
            print("⚠️ Could not click reload:", e)
        sleep(2)
        return

    try:
        verify_btn = driver.find_element(By.ID, "recaptcha-verify-button")
        click_element(driver, verify_btn)
    except Exception as e:
        print("⚠️ Could not click verify button in Type 2:", e)

    sleep(0.5)


def open_browser_with_captcha():
    if ENABLE_VPN and HAS_VPN:
        try:
            vpn.connect()
            print("VPN connected")
        except Exception as e:
            print("VPN connection failed:", e)

    options = Options()
    options.set_preference("intl.accept_languages", "en-US, en")
    if ENABLE_COOKIES and os.path.exists(PATH_TO_FIREFOX_PROFILE):
        options.profile = PATH_TO_FIREFOX_PROFILE
        print("Initializing Firefox with profile cookies")

    driver = webdriver.Firefox(options=options)

    for _ in range(10):
        try:
            driver.get(CAPTCHA_URL)
            WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
            sleep(1)

            recapcha_frame = driver.find_element(By.XPATH, "//iframe[@title='reCAPTCHA']")
            driver.switch_to.frame(recapcha_frame)
            driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border").click()
            sleep(4)
            driver.switch_to.default_content()

            WebDriverWait(driver, 5).until(
                EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='recaptcha challenge expires in two minutes']"))
            )
            print("Opened browser with reCAPTCHA challenge")
            return driver
        except WebDriverException:
            print("Error loading page. Retrying...")
            try:
                driver.refresh()
            except Exception:
                pass

    print("Failed to open browser with captcha after 10 attempts.")
    return None


def get_class_index(captcha_text):
    text = captcha_text.lower()
    for idx, name in enumerate(YOLO_CLASSES):
        if re.search(rf"\b{name}(?:e?s)?\b", text):
            return idx
    return None   # prompt is not a class this project knows


def handle_dynamic_captcha(driver, captcha_text, class_index, to_check):
    if len(to_check) < 1:
        try:
            verify_btn = driver.find_element(By.ID, "recaptcha-verify-button")
            click_element(driver, verify_btn)
        except Exception:
            pass
        sleep(1)
    else:
        while True:
            for i in list(to_check):
                if process_tile(i, captcha_text, class_index, driver, pause_after_click=True):
                    if i not in to_check:
                        to_check.append(i)
                else:
                    if i in to_check:
                        to_check.remove(i)

            if len(to_check) < 1:
                try:
                    verify_btn = driver.find_element(By.ID, "recaptcha-verify-button")
                    click_element(driver, verify_btn)
                except Exception:
                    pass
                sleep(2)

                try:
                    error_message = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "rc-imageselect-error-select-more"))
                    )
                    if 'none' in error_message.get_attribute('style'):
                        print("No additional images required.")
                    else:
                        print("'Select more images' message appeared. Reloading...")
                        driver.find_element(By.ID, "recaptcha-reload-button").click()
                except Exception:
                    print("Selection submitted without errors.")
                break


def captcha_is_solved(driver):
    sleep(1)
    try:
        driver.switch_to.default_content()
        recapcha_frame = driver.find_element(By.XPATH, "//iframe[@title='reCAPTCHA']")
        driver.switch_to.frame(recapcha_frame)
        checkbox = driver.find_element(By.XPATH, '//*[@id="recaptcha-anchor"]')
        return checkbox.get_attribute('aria-checked') == 'true'
    except Exception:
        return False
    finally:
        try:
            driver.switch_to.default_content()
            WebDriverWait(driver, 5).until(
                EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='recaptcha challenge expires in two minutes']"))
            )
        except Exception:
            pass

###############################################
# DYNAMIC 3x3 SOLVER
###############################################

RETRY_FLOOR = 0.05     # GUESS: a skipped tile must score at least this to be tried
MAX_FALLBACKS = 2      # guessed clicks allowed per challenge
MAX_ROUNDS = 12        # safety cap on click / re-scan rounds
GATE_STILL_S = 1.0     # GUESS, was 0.6
GATE_THRESH = 0.6      # GUESS, was 1.0 (your log passed at 0.87 and 0.92)


def _grid_element(driver):
    for sel in ("table[class*='rc-imageselect-table']", "#rc-imageselect"):
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            return els[0]
    return None


def _wait_grid_still(driver, window_s=GATE_STILL_S, thresh=GATE_THRESH,
                     timeout=12.0, white_cap_s=8.0):
    """Wait until nothing on the 3x3 grid has changed for window_s seconds
    and no tile is still a mostly-white blank."""
    t0 = time.time()
    history = []
    while time.time() - t0 < timeout:
        try:
            png = _grid_element(driver).screenshot_as_png
        except Exception:
            time.sleep(0.1)
            continue

        arr = np.asarray(Image.open(BytesIO(png)).convert("L").resize((192, 192)),
                         dtype=np.float32)
        now = time.time()
        history.append((now, arr))

        ref = None
        for ts, a in reversed(history):
            if now - ts >= window_s:
                ref = a
                break
        history = [h for h in history if now - h[0] <= window_s * 3]

        if ref is not None:
            worst = float(np.abs(arr - ref).reshape(3, 64, 3, 64).mean(axis=(1, 3)).max())
            white = float((arr.reshape(3, 64, 3, 64) > 240).mean(axis=(1, 3)).max())
            white_block = white >= 0.9 and (now - t0) < white_cap_s
            if worst < thresh and not white_block:
                print(f" -> grid still after {now - t0:.2f}s (diff {worst:.2f}, white {white:.2f})")
                event("grid_gate", status="still", elapsed=round(now - t0, 3),
                      worst=round(worst, 2), white=round(white, 2))
                return True

    print("⚠️ Grid never went still, continuing anyway")
    event("grid_gate", status="timeout", elapsed=round(time.time() - t0, 3))
    return False


def _score_tile(i, captcha_text, class_index, driver, phase):
    """Target score for tile i, or None if YOLO failed twice (unknown, not 'no match')."""
    for _ in range(2):
        p = process_tile(i, captcha_text, class_index, driver, defer_click=True)
        if p is not False:
            event("tile_scored", tile=i + 1, phase=phase, prob=round(p, 4))
            return p
        time.sleep(0.3)
    event("tile_error", tile=i + 1, phase=phase)
    return None

PROBE_FADE = True

def _probe_fade(driver, i, seconds=5.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            tile = driver.find_elements(By.CLASS_NAME, "rc-imageselect-tile")[i]
            img = tile.find_element(By.TAG_NAME, "img")
            overlays = tile.find_elements(By.CLASS_NAME, "rc-image-tile-overlay")
            ov = overlays[0].value_of_css_property("opacity") if overlays else "none"
            print(f"[probe {time.time() - t0:.2f}s] tile={tile.value_of_css_property('opacity')} "
                  f"img={img.value_of_css_property('opacity')} overlay={ov} "
                  f"class='{tile.get_attribute('class')}'")
        except Exception as e:
            print("[probe] error:", e)
        time.sleep(0.25)

def _click_tiles_and_wait(driver, indices):
    """Click all given tiles first, then wait for every one to finish fading."""
    befores, clicked = {}, []
    for i in indices:
        try:
            befores[i] = _tile_state(driver, i)       # snapshot just before the click
            tiles = driver.find_elements(By.CLASS_NAME, "rc-imageselect-tile")
            click_element(driver, tiles[i])
            clicked.append(i)
            preview_clicked(i)
        except Exception as e:
            print(f"⚠️ Could not click tile {i + 1}: {e}")

    if PROBE_FADE and clicked:
        _probe_fade(driver, clicked[0])
    t0 = time.time()
    for i in clicked:
        status, elapsed = wait_tile_settled(driver, i, befores[i], timeout=10.0)
        print(f" -> tile {i + 1}: {status} in {elapsed:.2f}s")
    print(f" -> all {len(clicked)} clicked tiles done after {time.time() - t0:.2f}s of waiting")
    return clicked


def _click_verify(driver):
    try:
        click_element(driver, driver.find_element(By.ID, "recaptcha-verify-button"))
        return True
    except Exception as e:
        print("⚠️ Could not click verify button:", e)
        return False


def _reload_challenge(driver):
    try:
        driver.find_element(By.ID, "recaptcha-reload-button").click()
    except Exception as e:
        print("⚠️ Could not click reload:", e)
    sleep(2)


def _verify_outcome(driver):
    """'gone' | 'dynamic_more' | 'select_more' | 'other'"""
    try:
        if not driver.find_elements(By.ID, "rc-imageselect"):
            return "gone"

        def shown(cls):
            return any(e.is_displayed() for e in driver.find_elements(By.CLASS_NAME, cls))

        if shown("rc-imageselect-error-dynamic-more"):   # class name from memory, unverified
            return "dynamic_more"
        if shown("rc-imageselect-error-select-more"):
            return "select_more"
        return "other"
    except WebDriverException:
        return "gone"


def _best_skipped(scores, tried):
    cands = [k for k, p in scores.items()
             if p is not None and k not in tried and p >= RETRY_FLOOR]
    return max(cands, key=lambda k: scores[k]) if cands else None


def solve_dynamic(driver, captcha_text, class_index):
    scores, tried = {}, set()
    fallbacks = dyn_more = rounds = total_clicks = 0

    # 1. Score all 9 tiles before clicking anything
    for i in range(9):
        scores[i] = _score_tile(i, captcha_text, class_index, driver, "first_scan")
    pending = [i for i, p in scores.items() if p is not None and p > THRESHOLD]

    MIN_FIRST_CLICKS = 3
    if len(pending) < MIN_FIRST_CLICKS:
        extras = sorted((k for k, p in scores.items() if p is not None and k not in pending),
                        key=lambda k: scores[k], reverse=True)[:MIN_FIRST_CLICKS - len(pending)]
        print(f"Only {len(pending)} above threshold; adding best-scoring tiles: "
              f"{[(k + 1, round(scores[k], 3)) for k in extras]}")
        pending += extras
        tried.update(extras)   # so the fallback never picks them again    

    while True:
        # 2-4. Click matches, wait for ALL fades, re-scan only the clicked tiles, repeat
        while pending and rounds < MAX_ROUNDS:
            rounds += 1
            clicked = _click_tiles_and_wait(driver, pending)
            sleep(0.8)
            _wait_grid_still(driver)    # re-scan only once nothing on the grid is moving            
            total_clicks += len(clicked)
            pending = []
            for i in clicked:
                scores[i] = _score_tile(i, captcha_text, class_index, driver, "rescan")
                if scores[i] is not None and scores[i] > THRESHOLD:
                    pending.append(i)

        # Nothing clicked at all: guess the best skipped tile, or reload
        if total_clicks == 0:
            best = _best_skipped(scores, tried)
            if best is None or fallbacks >= MAX_FALLBACKS:
                print("No tile matched. Reloading for a new challenge...")
                _reload_challenge(driver)
                return
            fallbacks += 1
            tried.add(best)
            pending = [best]
            print(f"Nothing matched. Guessing tile {best + 1} (score {scores[best]:.3f})")
            continue

        # 5. Verify only when the whole grid has stopped moving
        if not _wait_grid_still(driver):
            _wait_grid_still(driver)
        if not _click_verify(driver):
            return
        sleep(1.5)

        outcome = _verify_outcome(driver)
        print(f"Verify outcome: {outcome}")
        event("verify", outcome=outcome, fallbacks=fallbacks, rounds=rounds,
              scores={k + 1: (None if v is None else round(v, 3)) for k, v in scores.items()})

        if outcome in ("gone", "other"):
            return

        if outcome == "dynamic_more" and dyn_more < 2:
            dyn_more += 1
            print("New images still need checking. Re-scanning all 9 tiles...")
            _wait_grid_still(driver)
            for i in range(9):
                scores[i] = _score_tile(i, captcha_text, class_index, driver, "recheck")
            pending = [i for i, p in scores.items() if p is not None and p > THRESHOLD]
            continue

        # "select more": something was missed. Try the best skipped tile, reload last
        best = _best_skipped(scores, tried)
        if best is None or fallbacks >= MAX_FALLBACKS:
            print("No fallback left. Reloading...")
            _reload_challenge(driver)
            return
        fallbacks += 1
        tried.add(best)
        pending = [best]
        print(f"Fallback {fallbacks}: trying tile {best + 1} (score {scores[best]:.3f})")

def solve_classification_type(driver, dynamic_captcha):
    try:
        captcha_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'rc-imageselect'))
        )
        strong_element = captcha_element.find_element(By.TAG_NAME, 'strong')
        captcha_text = strong_element.text
    except Exception as e:
        print("⚠️ Could not retrieve captcha instruction text:", e)
        return

    class_index = get_class_index(captcha_text)
    if class_index is None:
        print(f"'{captcha_text}' is not a class this project knows. Reloading...")
        _reload_challenge(driver)
        return    
    print(f"Target object: '{captcha_text}' -> Class Index {class_index} ({YOLO_CLASSES[class_index]})")
    log("dynamic" if dynamic_captcha else "Type1", captcha_text)
    preview_reset(f"Target: {captcha_text}")    

    # ---------- Static 3x3: score everything first, then click ----------
    if not dynamic_captcha:
        scores = {}
        for i in range(9):
            p = process_tile(i, captcha_text, class_index, driver, defer_click=True)
            if p is not False:          # False means the tile errored out
                scores[i] = p

        ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
        selected = ranked[:3]   # always the 3 best, never more, never fewer
        print(f"Static 3x3: picking top 3 -> "
              f"{[(i + 1, round(scores[i], 3)) for i in selected]}")
        event("static_pick", picks={i + 1: round(scores[i], 3) for i in selected},
              all_scores={i + 1: round(p, 3) for i, p in scores.items()})

        tiles = driver.find_elements(By.CLASS_NAME, "rc-imageselect-tile")
        for i in selected:
            if i < len(tiles):
                click_element(driver, tiles[i])
                preview_clicked(i)

        try:
            verify_btn = driver.find_element(By.ID, "recaptcha-verify-button")
            click_element(driver, verify_btn)
        except Exception as e:
            print("⚠️ Could not click verify button:", e)
        return
    # ---------- Dynamic 3x3 ----------
    solve_dynamic(driver, captcha_text, class_index)


log_filename = None
session_folder = None


def log(captcha_type, captcha_object):
    global log_filename, session_folder

    if not ENABLE_LOGS:
        return

    if session_folder is None:
        highest_session = 0
        for dirname in os.listdir('.'):
            if dirname.startswith('Session'):
                try:
                    highest_session = max(highest_session, int(dirname[7:]))
                except ValueError:
                    pass
        session_folder = f'Session{highest_session + 1:02}'
        os.makedirs(session_folder, exist_ok=True)

    if log_filename is None:
        highest_log = 0
        for filename in os.listdir(session_folder):
            if filename.startswith('logs_'):
                try:
                    highest_log = max(highest_log, int(filename[5:7]))
                except ValueError:
                    pass
        log_filename = os.path.join(session_folder, f'logs_{highest_log + 1:02}.csv')

    with open(log_filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, captcha_type, captcha_object])


def run():
    driver = None
    try:
        driver = open_browser_with_captcha()
    except Exception as e:
        print("Failed to start browser:", e)
        if ENABLE_VPN and HAS_VPN:
            try:
                vpn.disconnect()
            except Exception:
                pass
        return False

    if not driver:
        print("Driver failed to initialize.")
        return False

    while True:
        try:
            try:
                rc_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'rc-imageselect'))
                )
                rc_text = rc_element.text
            except Exception:
                if captcha_is_solved(driver):
                    print("🎉 Captcha solved successfully!")
                    log("SOLVED", "captcha solved")
                    driver.quit()
                    break
                else:
                    try:
                        driver.find_element(By.ID, "recaptcha-reload-button").click()
                    except Exception:
                        pass
                    continue

            if "squares" in rc_text and TYPE2:
                print("Found 4x4 segmentation captcha")
                solve_type2(driver)
            elif "none" in rc_text and TYPE3:
                print("Found 3x3 dynamic captcha")
                solve_classification_type(driver, dynamic_captcha=True)
            elif TYPE1:
                print("Found 3x3 single-selection captcha")
                solve_classification_type(driver, dynamic_captcha=False)
            else:
                try:
                    driver.find_element(By.ID, "recaptcha-reload-button").click()
                except Exception:
                    pass
                continue

            if captcha_is_solved(driver):
                log("SOLVED", "captcha solved")
                print("🎉 Captcha solved successfully!")
                try:
                    driver.quit()
                except Exception:
                    pass
                if ENABLE_VPN and HAS_VPN:
                    try:
                        vpn.disconnect()
                    except Exception:
                        pass
                break

        except WebDriverException as e:
            print(f"⚠️ Browser session disconnected or crashed: {e}")
            print("Restarting browser instance...")
            try:
                driver.quit()
            except Exception:
                pass

            driver = open_browser_with_captcha()
            if not driver:
                print("Could not restart browser. Exiting.")
                break

        except Exception as e:
            print("An error occurred during execution:", e)
            traceback.print_exc()

            if captcha_is_solved(driver):
                log("SOLVED", "captcha solved")
                try:
                    driver.quit()
                except Exception:
                    pass
                break

            try:
                driver.find_element(By.ID, "recaptcha-reload-button").click()
            except Exception:
                pass
            continue


if __name__ == "__main__":
    run()
