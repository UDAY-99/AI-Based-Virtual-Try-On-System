# 🎽 AI-Based Virtual Try-On System with Variable Fitting

This project implements a deep-learning–based virtual try-on system capable of transferring garment images onto human model images with high visual quality. A key enhancement of this project is the introduction of a **variable fitting mechanism**, which allows garments to appear **tight, normal, or loose** by adjusting the garment mask before synthesis. The system provides both CLI-based inference and an interactive Gradio interface for real-time experimentation.

---

## 📌 Features
- High-quality virtual try-on output using pretrained diffusion-based modules  
- Variable garment fitting — **tight (0.7)**, **normal (1.0)**, **loose (1.3)**  
- Full preprocessing pipeline (parsing, pose extraction, mask generation)  
- Clean, modular Python implementation with extendable architecture  
- GPU-accelerated inference for faster execution  
- Gradio-based UI for user-friendly visualization  
- Organized output folders for comparing different fit scales  

---

## 📁 Folder Structure

```text
AI-Based-Virtual-Try-On-System/
│
├── checkpoints/               # Pretrained try-on model checkpoints
├── examples/
│   ├── model/                 # Sample human images
│   └── garment/               # Sample garment images
│
├── run/
│   ├── run_ootd.py            # Base inference script
│   ├── run_ootd_fit.py        # Modified script with variable fitting
│   ├── gradio_ootd.py         # Original interface
│   ├── gradio_ootd_fit.py     # Variable-fitting enabled interface
│   └── utils_ootd.py          # Preprocessing + helper utilities
│
├── images_output/             # Output folder for generated results
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


##⚙️ Installation
1️⃣ Create Conda Environment
conda create -n ootd python=3.11 -y
conda activate ootd

2️⃣ Install Dependencies
pip install -r requirements.txt

🚀 How to Run the System
Option 1 — CLI Inference

Run a single try-on operation:

python run_ootd_fit.py --model_path ./examples/model/model_2.png --cloth_path ./examples/garment/00055_00.jpg --fit_scale 1.3 --n_steps 12 --out_dir ./images_output/fit_1_3

Option 2 — Run Gradio Interface
python gradio_ootd_fit.py


Open the browser at:

http://localhost:7865


This interface allows:

Uploading model + garment images

Adjusting fit-scale

Previewing outputs immediately

🎛️ Variable Fitting Module

The proposed enhancement enables garment fitting control using a fit-scale parameter:

Fit Scale	Fitting Style
0.7	Tight Fit
1.0	Standard Fit
1.3	Loose Fit

The mask is scaled accordingly before inference:

mask_cond = scale_mask(mask, fit_scale)
mask_cond = mask_cond.filter(ImageFilter.GaussianBlur(radius=2))


Outputs for each fit scale are automatically saved in separate directories such as:

images_output/
    fit_0.7/
    fit_1.0/
    fit_1.3/

🖼️ Recommended Output Samples

(Add these images after generating your results)

tight_fit.png

normal_fit.png

loose_fit.png

🔧 Tech Stack Used

Python 3.11

PyTorch (for model inference)

OpenCV & PIL (image processing)

Gradio (UI interface)

ONNX Runtime (for parsing models)

NumPy / Matplotlib
