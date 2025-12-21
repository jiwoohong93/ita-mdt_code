<p align="center">
  <h2 align="center"><strong>ITA-MDT:<br> Image-Timestep-Adaptive Masked Diffusion Transformer Framework<br>for Image-Based Virtual Try-On</strong></h2>

<p align="center">
    <a href="http://sanctusfactory.com/family_02.php">Ji Woo Hong</a>,
    <a href="https://triton99.github.io/">Tri Ton</a>,
    <a href="https://trungpx.github.io/">Pham X. Trung</a>,
    <a href="https://kookie12.github.io/">Gwanhyeong Koo</a>,
    <a href="https://dbstjswo505.github.io/">Sunjae Yoon</a>,
    <a href="http://sanctusfactory.com/family.php">Chang D. Yoo</a>
    <br>
    <b>Korea Advanced Institute of Science and Technology (KAIST)</b>
</p>

<div align="center">

<a href='https://arxiv.org/abs/2503.20418'><img src='https://img.shields.io/badge/arXiv-2503.20418-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://jiwoohong93.github.io/ita-mdt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;

</div>

<p align="center">
    <img src="images_github/teaser.png" alt="ITA-MDT" width="700" height="auto">
</p>

---
<!-- ## TO-DO
We're currently in the process of cleaning up and organizing the code. 

Thank you for your patience, everything will be up soon.
- [ ] Train, Inference, and Evaluation code release.
- [ ] Model release. -->

## Requirements

```bash
git clone https://github.com/jiwoohong93/ita-mdt_code.git
cd ita-mdt_code

bash environment.sh
conda activate ITA-MDT
```

The above commands will create and activate the conda environment with all core dependencies for ITA-MDT.

**(optional)** We recommend utilizing [Adan](https://github.com/sail-sg/Adan) and [xFormers](https://github.com/facebookresearch/xformers) for improved training and generation efficiency.  

### Pre-trained Models Required
Two pre-trained components are required and will be automatically downloaded on the first run of training or generation:

- **[DINOv2](https://huggingface.co/facebook/dinov2-large)** — Vision Transformer backbone for garment feature extraction.  
- **[Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse)** — Variational Autoencoder for image encoding/decoding in latent space.  

Once downloaded, they will be cached locally for subsequent runs. 

---

## Datasets Preparation

Download **VITON-HD** from **[HERE](https://github.com/shadow2496/VITON-HD)**


Download **DressCode** from **[HERE](https://github.com/aimagelab/dress-code)**

Place both datasets inside the `DATA/` folder:

```
DATA/
  ├── zalando-hd-resized/
  ├── DressCode/
```

### 1. Additional Images Required for DressCode

To generate the agnostic images and corresponding masks, we adopt the dataset preparation of [CAT-DM](https://github.com/zengjianhao/CAT-DM). For DensePose, we utilize new part-based color map DensePose images provided by [IDM-VTON](https://github.com/yisol/IDM-VTON) to ensure consistency with the VITON-HD dataset.

These images are required for proper training and generation.
They can all be downloaded from **[HERE](https://huggingface.co/datasets/jiwoohong93/dresscode_agnostic_and_densepose)**.

After downloading, place each garment category’s folder and its images into the corresponding directory of the original DressCode dataset.


### 2. Pre-process SRE (Salient Region Extraction)


This code pre-processes SRE and saves salient region images in advance for faster and more efficient training and generation.

Run the following command:

```bash
python preprocess_salient_region_extraction.py --path_to_datasets ./DATA
```

- `--path_to_datasets` should point to the folder containing `zalando-hd-resized` and `DressCode` directories.  
- This script will process **both datasets** and save salient region images into the `cloth_sr` folder for each category.

or, you can download the pre-processed salient region images from **[HERE](https://huggingface.co/datasets/jiwoohong93/ita-mdt_sre)**.

After downloading, place each garment category’s folder and its images into the corresponding directory.

### Expected Data Structure

```
zalando-hd-resized/
  ├── test/
  │   ├── agnostic-mask
  │   ├── agnostic-v3.2
  │   ├── cloth
  │   ├── cloth_sr
  │   ├── image
  │   └── image-densepose
  ├── train/
  │   ├── agnostic-mask
  │   ├── agnostic-v3.2
  │   ├── cloth
  │   ├── cloth_sr
  │   ├── image
  │   └── image-densepose
  ├── test_pairs.txt
  └── train_pairs.txt

DressCode/
  ├── dresses/
  │   ├── agnostic
  │   ├── cloth_sr
  │   ├── image-densepose
  │   ├── images
  │   ├── mask
  │   ├── test_pairs_paired.txt
  │   ├── test_pairs_unpaired.txt
  │   └── train_pairs.txt
  ├── lower_body/
  │   ├── agnostic
  │   ├── cloth_sr
  │   ├── image-densepose
  │   ├── images
  │   ├── mask
  │   ├── test_pairs_paired.txt
  │   ├── test_pairs_unpaired.txt
  │   └── train_pairs.txt
  └── upper_body/
      ├── agnostic
      ├── cloth_sr
      ├── image-densepose
      ├── images
      ├── mask
      ├── test_pairs_paired.txt
      ├── test_pairs_unpaired.txt
      └── train_pairs.txt
```
---
## Training

Run:  

```bash
bash train.sh
```

#### Variables to Edit in `train.sh`

- **`export CUDA_VISIBLE_DEVICES=`** → GPU IDs to use for training (comma-separated).  
- **`NUM_GPUS=`** → Number of GPUs to use.  
- **`export OPENAI_LOGDIR=`** → Directory to save training logs and checkpoints.  
- **`LR=`** → Learning rate.  
- **`BATCH_SIZE=`** → Batch size.  
- **`SAVE_INTERVAL=`** → Save model checkpoint every this many steps.  
- **`MASTER_PORT=`** → Port used for inter-process communication in distributed training (change if conflict occurs).   
- *(Optional)* **`--resume_checkpoint`** → Uncomment and set a path if resuming from a saved checkpoint.  

---

## Generation

You can download the checkpoint of our ITA-MDT from **[HERE](https://huggingface.co/jiwoohong93/ita-mdt_weights)**.

[2025-10-08] Reuploaded with the correct model weights.

#### **VITON-HD**
Run:  
```bash
bash generate_vitonhd.sh
```

#### **DressCode**
Run:  
```bash
bash generate_dc.sh
```

#### Variables to Edit in Generation Scripts

Common for both `generate_vitonhd.sh` and `generate_dc.sh`:

- **`export CUDA_VISIBLE_DEVICES=`** → GPU ID to use for generation.
- **`OUTPUT_DIR=`** → Path where generated images will be saved.  
- **`MODEL_PATH=`** → Path to trained weights (ema).  
- **`BATCH_SIZE=`** → Images generated per batch.  
- **`NUM_SAMPLING_STEPS=`** → Diffusion sampling steps.  
- **`UNPAIR=false`** → Whether to use unpaired garment-person combinations.

For `generate_dc.sh`:
- **`SUBDATA=`** → Category of DressCode dataset (`dresses`, `upper_body`, or `lower_body`).  

---

## Evaluation

The evaluation code is adapted from [LaDI-VTON](https://github.com/miccunifi/ladi-vton). Please refer to the original repository for the environment required to run the evaluation.

Run:  
```bash
bash eval.sh
```

#### Variables to Edit in `eval.sh`

- **`CUDA_VISIBLE_DEVICES=`** → GPU ID to use for evaluation.  
- **`--batch_size=`** → Batch size for evaluation.    
- **`--gen_folder=`** → Path to generated images to be evaluated.  
- **`--dataset=`** → Dataset to evaluate on (`vitonhd` or `dresscode`).  
- **`--test_order=`** → Paired/unpaired evaluation (`paired` or `unpaired`). For unpaired, only FID is valid.
- **`--category=`** → Category for DressCode dataset (`upper_body`, `lower_body`, `dresses`).  


---
## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{hong2025ita,
  title={ITA-MDT: Image-Timestep-Adaptive Masked Diffusion Transformer Framework for Image-Based Virtual Try-On},
  author={Hong, Ji Woo and Ton, Tri and Pham, Trung X and Koo, Gwanhyeong and Yoon, Sunjae and Yoo, Chang D},
  journal={arXiv preprint arXiv:2503.20418},
  year={2025}
}
```

---
## License

The codes in this repository are released under the **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** license.


---

## Acknowledgement
```bibtex
This work was supported by Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)
(No. RS-2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments),
and partly supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT)
(No. RS-2022-II220184, Development and Study of AI Technologies to Inexpensively Conform to Evolving Policy on Ethics).
```

