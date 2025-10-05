# Hecate Super-Resolution (HSR)

A lightweight ×4 image super-resolution repository implemented on top of **BasicSR (1.4.2)** for reproducible inference. The default example uses `test_options/SMFANet_DIV2K_x4SR.yml`.

## Requirements

- Ubuntu **20.04**
- Python **3.8.0**
- PyTorch **2.1.0**
- CUDA **11.8**
- BasicSR **1.4.2**

## Installation
```bash
# Clone
git clone "https://github.com/AutoML-NASA/hecate-super-resolution.git"
cd hecate-super-resolution

# Conda env
conda create -n hsr python==3.8 -y
conda activate hsr

# Python deps (adjust filename if yours is requirements.txt)
pip install -r requirments.txt
```

## Testing
```bash
python infer.py -opt test_options/SMFANet_DIV2K_x4SR.yml
```

- Input/Output example (shape):  
  - **Input**: `(111, 87, 3)`  
  - **Output**: `(444, 348, 3)` (×4 upscaling)

## Repository Structure (example)

- `infer.py`: inference entry script  
- `basicsr/`: BasicSR core and extensions  
- `test_options/SMFANet_DIV2K_x4SR.yml`: ×4 inference config  
- `requirements.txt` (or `requirments.txt`): dependencies

## Notes

- PyTorch 2.1.0 with CUDA 11.8 is recommended; verify local driver/runtime compatibility.  
- With BasicSR 1.4.2, dataset loader keys (e.g., `dataroot_*`, `io_backend`) must match your data layout.

## Citation
```bibtex
@inproceedings{smfanet,
    title={SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution},
    author={Zheng, Mingjun and Sun, Long and Dong, Jiangxin and Pan, Jinshan},
    booktitle={ECCV},
    year={2024}
}
```

## Acknowledgement

This code is based on the excellent **SMFANet** toolbox. Thanks for the awesome work.  
- GitHub: https://github.com/Zheng-MJ/SMFANet.git

## License & Third-Party Notices

- **Model/Code Usage**: This repository references the paper in *Citation* and the GitHub code in *Acknowledgement* to build the training/inference pipelines. Please **comply with the original LICENSE** terms of the upstream works.

- **Dataset**: Trained/evaluated with **Apollo Pan Images (LROC)**.  
  - Source: **[Apollo Pan Images (LROC)](https://data.im-ldi.com)**  
  - Follow the provider’s terms (copyright, citation rules, redistribution policy, etc.).

