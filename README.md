# FPNSTproject
## Region-Aware Neural Style Transfer

This repository contains the code for my Final Year Project: a **region-aware neural style transfer system**.  
It applies artistic styles to batches of images while **preserving important regions** like faces, logos, and text.

---

## Requirements

- Python 3.9+
- GPU recommended (for faster processing, but CPU also works)

---

## Installation

Clone the repo and install dependencies:
pip install -r requirements.txt

---

## Pretrained Weights

Some pretrained models are too large to upload to GitHub.
Please download them from the official sources and place them in the correct folders:

Holistically-Nested Edge Detection (HED)
→ Place hed_pretrained_bsds.caffemodel in edge/weights/

GroundingDINO
→ Place groundingdino_swint_ogc.pth in logo/weights/

Adaptive Instance Normalization (AdaIN)
→ Place vgg_normalised.pth and decoder.pth in models/

---

## Running the App

Start the Streamlit interface:

streamlit run ./ui/batch_update.py
Then open the link shown in the terminal (http://localhost:8501).

---

## Usage

Upload one or more content images.

Upload a style image.

Adjust the stylization strength slider.

Download results individually or as a ZIP.

---

## Acknowledgements

This project builds on the following open-source works:

HED (https://github.com/s9xie/hed)
GroundingDINO (https://github.com/IDEA-Research/GroundingDINO)
AdaIN (https://github.com/naoto0804/pytorch-AdaIN)
