## About this fork

This repository is a fork of [aplesner/Breaking-reCAPTCHAv2](https://github.com/aplesner/Breaking-reCAPTCHAv2), the code accompanying the paper **"Breaking reCAPTCHAv2"** (COMPSAC 2024, [arXiv:2409.08831](https://arxiv.org/abs/2409.08831)). All credit for the original idea, models and training data goes to the original authors.

The original project is no longer maintained and no longer ran as-is (for example, it relied on hard-coded macOS file paths). This fork was made for a university course project to get it running again and to improve it. It is for research and educational use only.

### Changes in this fork

- **Dynamic 3x3 challenges:** replaced the fixed `sleep` with a state-based wait for the tile fade, observed only through Selenium (no JavaScript injected into the page). Tiles are scanned, matches clicked, only clicked tiles are re-scanned, and Verify is pressed only once the grid is still.
- **Static 3x3 challenges:** always picks the top 3 tiles by confidence.
- **4x4 segmentation:** tile selection by mask coverage with tiered, more forgiving passes, and a live YOLO preview window.
- **Class-order bug fixed:** `stairs` was missing from the class list, which caused an `IndexError` and mislabeled stairs and traffic lights.
- **Unknown prompts:** prompts for classes the model does not know no longer default to "bicycle".
- **Logging:** per-session `console.log` and `events.jsonl` under `logs/`.
- **Housekeeping:** updated `requirements.txt` (added `torch`, `torchvision`, `numpy`; removed unused `pynput` and `shap`), removed the unused `shap` import, and fixed stale imports in `utils/label_tool.py`.

### Setup (this fork)

- Python 3.12, Firefox with Geckodriver
- `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
- VPN support (`IP/vpn.py`) is optional and off by default (`ENABLE_VPN = False`)
- Run: `python solve_recaptcha.py`

### Citation

If you use this work, please cite the original paper:

> Plesner, A., Vontobel, T., Wattenhofer, R. "Breaking reCAPTCHAv2." COMPSAC 2024. arXiv:2409.08831

# Breaking reCAPTCHAv2
Code for the paper `Breaking reCAPTCHAv2` accepted at COMPSAC 2024 (https://arxiv.org/abs/2409.08831).

## Introduction
CAPTCHAs (Completely Automated Public Turing Tests to Tell Computers and Humans Apart) have been a vital security measure on the internet, protecting websites from automated bots and malicious activities. However, with the rapid advancements in machine learning and artificial intelligence, the effectiveness of CAPTCHAs in distinguishing between humans and machines has come into question.

This project focuses on Google's reCAPTCHAv2 system, which is widely used across the web. We aim to analyze the effectiveness of reCAPTCHAv2 in rejecting bots using advanced deep learning models such as YOLO (You Only Look Once). Our research explores the vulnerabilities of image-based CAPTCHAs and develops efficient methods to solve them using state-of-the-art machine learning techniques.

Our main findings include:

- We can solve 100% of the CAPTCHAs presented by reCAPTCHAv2, surpassing the success rates of previous works, which range from 68% to 71%.
- There is no significant difference in the number of challenges required by humans and bots to solve CAPTCHAs in reCAPTCHAv2, suggesting that current AI technologies can effectively exploit advanced image-based CAPTCHAs.
- reCAPTCHAv2 heavily relies on cookie and browser history data when evaluating whether a user is human or not, highlighting the importance of considering user-specific data in CAPTCHA design.


| CAPTCHA Example | CAPTCHA Result |
|:---:|:---:|
| <img src="assets/type2_example.gif" width="200" /> | <img src="assets/type2_example_result.png" width="200" /> |

## Requirements
- Python 3.9
- Firefox (Geckodriver)
- Required libraries (see `requirements.txt`)

## Installation
1. Clone the repository

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

3. Download and set up Geckodriver:
   - Download Geckodriver from the official website: [Geckodriver Releases](https://github.com/mozilla/geckodriver/releases)
   - Extract the downloaded archive and add the path to the `geckodriver` executable to your system's PATH environment variable.

## Usage
- Run `solve_recaptcha.py` to solve a single reCAPTCHA challenge:
  ```
  python solve_recaptcha.py
  ```

- Run `test_environment.py` to solve multiple reCAPTCHA challenges and create a log file:
  ```
  python test_environment.py
  ```

## Data
- The training data for the classification task can be found [here](https://drive.google.com/drive/folders/19kET6PFXHaHZqzr9DU_ZsgX-n13Ef4sj?usp=sharing).
- The validation data for the classification task can be found [here](https://drive.google.com/drive/folders/19kG2NQls2iH1sUq0js0MOArBiJiOmVGM?usp=sharing).

## Project Structure
This project has the following directory structure:

- `assets/`: Contains the images used in the README.
- `IP/`: Contains the script for changing IP address.
  - `vpn.py`: Script for changing IP address (only works on macOS).
- `models/`: Contains all the models used in the project.
  - `YOLO_Classification/`: Contains the YOLO model for classification.
  - `YOLO_Segment/`: Contains the YOLO model for segmentation.
- `utils/`: Contains utility scripts.
  - `collect_data.py`: Script for collecting data for the classification model.
  - `label_tool.py`: A UI tool for labeling CAPTCHA images.
  - `visualize_log_files.py`: Analyzes log files, providing insights into task attempts, their distribution, and key statistical measures.
- `README.md`: This file, providing an overview of the project.
- `solve_recaptcha.py`: Demo script for solving a single reCAPTCHA challenge using Selenium.
- `test_environment.py`: Demo script for solving multiple reCAPTCHA challenges using Selenium and creating a log file.
- `requirements.txt`: Contains the required libraries for the project.

## System Compatibility
Please note that the code in this project was developed and tested on macOS M1. While most of the code should be compatible with other operating systems, certain functionalities, such as the VPN script (`vpn.py`), may not work on non-macOS systems. If you encounter any issues running the code on a different operating system, please refer to the documentation of the specific libraries or tools used in the project for guidance on how to set them up for your system.

