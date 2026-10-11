from selenium import webdriver 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import os
import sys
import requests
from PIL import Image
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from selenium.webdriver.firefox.options import Options

# Ensure parent directory is in sys.path for models import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import YOLO model (PyTorch)
from models.YOLO_Classification import predict

# Constants
CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo?hl=en"  # Forces English reCAPTCHA
THRESHOLD = 0.2
COLLECT_TYPE1 = True
COLLECT_TYPE2 = True
CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
EXCLUDED_CLASSES = []
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic', 'stairs']

image_count = 0
total_images = 100

# Ensure raw data directories exist
raw_data_dir = os.path.join(os.getcwd(), "data", "RawData")
raw_data_type2_dir = os.path.join(os.getcwd(), "data", "RawDataType2")
os.makedirs(raw_data_dir, exist_ok=True) 
os.makedirs(raw_data_type2_dir, exist_ok=True) 

COUNT = 0

def process_tile(i, captcha_text, class_index, driver):
    global COUNT, image_count
    try:
        tiles = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "rc-imageselect-tile"))
        )
    except Exception:
        return False

    if i >= len(tiles):
        return False

    matched_tile = tiles[i]
    timestamp = str(time.time())

    # 1. Save screenshot as .png first to satisfy Selenium
    png_filename = f"tile_{COUNT}_{timestamp}.png"
    png_path = os.path.join(raw_data_dir, png_filename)
    try:
        matched_tile.screenshot(png_path)

        # 2. Convert and resize with Pillow to .jpg
        img = Image.open(png_path).convert('RGB').resize((224, 224))
        jpg_filename = png_filename.replace('.png', '.jpg')
        jpg_path = os.path.join(raw_data_dir, jpg_filename)
        img.save(jpg_path)

        # 3. Clean up the initial .png screenshot
        if os.path.exists(png_path):
            os.remove(png_path)

        image_count += 1
    except Exception as e:
        print(f"Error processing tile screenshot: {e}")
        return False

    # Predict class using YOLOv8
    try:
        result = predict.predict_tile(jpg_path)
        current_object_probability = result[class_index]
        object_name = YOLO_CLASSES[result]
    except Exception:
        object_name = "unknown"
        current_object_probability = 0.0

    # Rename image with predicted class prefix
    new_filename = f"{object_name}_{jpg_filename}"
    new_path = os.path.join(raw_data_dir, new_filename)
    if os.path.exists(jpg_path):
        try:
            os.rename(jpg_path, new_path)
        except Exception:
            pass

    COUNT += 1
    return True

def process_type2(driver):
    global image_count
    try:
        xpath_image = "//div[@id='rc-imageselect']//img"
        xpath_text = "//div[@id='rc-imageselect']//strong"

        captcha_text = driver.find_element(By.XPATH, xpath_text).text
        img = driver.find_element(By.XPATH, xpath_image)
        img_url = img.get_attribute("src")

        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            timestamp = str(time.time())
            save_file = os.path.join(raw_data_type2_dir, f"image_{captcha_text}_{timestamp}.png")
            with open(save_file, 'wb') as f:
                f.write(response.content)
            image_count += 1
        return True
    except Exception as e:
        print(f"Error saving Type 2 image: {e}")
        return False

def get_class_index(captcha_text):
    text_lower = captcha_text.lower()
    for i in YOLO_CLASSES:
        if i in text_lower or (i.endswith('s') and i[:-1] in text_lower) or (i + 's' in text_lower):
            return YOLO_CLASSES.index(i)
    return 0

def run():
    options = Options()
    options.set_preference("intl.accept_languages", "en-US, en")
    driver = webdriver.Firefox(options=options)
    driver.set_window_size(600, 700)

    try:
        driver.get(CAPTCHA_URL)
        recapcha_frame = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//iframe[@title='reCAPTCHA']"))
        )
        driver.switch_to.frame(recapcha_frame)
        driver.find_element(By.CLASS_NAME, "recaptcha-checkbox-border").click()
        sleep(3)

        driver.switch_to.default_content()
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@title='recaptcha challenge expires in two minutes']"))
        )

        title_wrapper = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, 'rc-imageselect'))
        )
        
        if "squares" in title_wrapper.text and COLLECT_TYPE2:
            process_type2(driver)
        else:
            captcha_text = title_wrapper.find_element(By.TAG_NAME, 'strong').text
            class_index = get_class_index(captcha_text)
            
            for i in range(9):
                process_tile(i, captcha_text, class_index, driver)

    except Exception as e:
        print(f"Run iteration error: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# Main collection loop
pbar = tqdm(total=total_images)
while image_count < total_images:
    before = image_count
    run()
    after = image_count
    pbar.update(after - before)

pbar.close()

