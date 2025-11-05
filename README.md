# FAST: Foreground‑aware Diffusion with Accelerated Sampling Trajectory for Segmentation‑oriented Anomaly Synthesis
✨ Update (2025 11/05):


Added comprehensive code annotations across key module and files to clarify inputs/outputs, mask formats, and sampling steps.

✨ Update (2025 10/21): 
1. Our paper FAST has been accepted at NeurIPS 2025
2. We have thoroughly refined our implementation, addressing several limitations — including the introduction of an Adaptor (called FineTune module in ddpm.py) to incorporate mask information into the network for better performance.
3. For reproducibility, you can download the **official checkpoint** of the Latent Diffusion Model (LDM) from the following link:  
🔗 [Official LDM Checkpoint – txt2img-f8-large](https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt)  

> **Note:** After loading the pretrained LDM weights, you only need to train the **FARM** and the newly added **Adaptor** modules, which are lightweight compared to the full LDM checkpoint and can be efficiently trained on standard GPUs.

## 📉 Introduction

**FAST** (**Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis**) is a novel and efficient framework for generating synthetic anomalies tailored for **anomaly segmentation** tasks, specifically designed to improve both the **efficiency** and **fidelity** of synthesized anomalies, 

### 🔹 Key Innovations of FAST:

- **Anomaly-Informed Accelerated Sampling (AIAS):** A training-free sampling algorithm that divides the reverse diffusion into coarse-to-fine segments. By analytically aggregating multiple DDPM transitions, AIAS achieves up to 100× acceleration while maintaining structural alignment under anomaly mask guidance.
- **Foreground-Aware Reconstruction Module (FARM):** Ensures fine-grained and well-aligned anomaly injection by enhancing the synthesis of critical foreground regions.

FAST achieves impressive results on benchmark datasets including **MVTec-AD**, **BTAD**, outperforming previous anomaly synthesis models in segmentation accuracy and synthesis realism .

---

## 📊 Repository Structure

```
├── configs/ 
├── logs/                             
├── ldm/                   
├── taming/               
├── scripts/               
├── utils/                 
├── main.py                
├── generate_with_mask.py  
├── requirements.txt       
├── README.md             
```

---

## 🚀 Training FAST

To train FAST on **MVTec-AD**:

```bash
python main.py --base configs/fast_mvtec.yaml -t \
--actual_resume pretrained_models/ldm_base.ckpt \
-n mvtec_run --gpus 0 \
--init_word "screw" \
--mvtec_path='path_to_mvtec_data/' \
--log_folder 'logs/mvtec_fast/'
```

For **BTAD** dataset:

```bash
python main.py --base configs/fast_btad.yaml -t \
--actual_resume pretrained_models/ldm_base.ckpt \
-n btad_run --gpus 0 \
--init_word "capsule" \
--mvtec_path='path_to_btad_data/' \
--log_folder 'logs/btad_fast/'
```

---

## 🤔 Evaluation & Inference

To synthesize anomaly samples for evaluation:

```bash
python generate_with_mask.py \
--data_root='path_to_normal_images/' \
--weight_idx 10000 \
--sample_name='samples/fast_output/' \
--init_word "bottle" \
--anomaly_name='bottle_test' \
--pt_path='checkpoints/fast/' \
--mask_path='masks/bottle_masks/'
```

Generated samples will be saved under `samples/fast_output/`.

---

## 📣 Acknowledgments

This project builds upon **Latent Diffusion Models** and benefits from insights in works like **DRAEM**, and **Anomaly Diffusion**. We thank the original authors of these methods and the community contributions to **industrial anomaly segmentation** research.

For bug reports or feature requests, please open an **Issue** or submit a **Pull Request**.

---

## 📜 License

This project is licensed under the **MIT License**.
