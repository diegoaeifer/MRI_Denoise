# MRI Denoising & Super-Resolution: Promising Models with Pretrained Availability

Analyzed **66** relevant papers from `research-organized/`.  
Showing top 25 models ranked by promise score + approach + code availability.

## Summary Table

| Rank | Model | Task | Promise | GitHub | HuggingFace | Pretrained |
|------|-------|------|---------|--------|-------------|------------|
| 1 | **Universal** | Denoising | 5/5 | — | — | ❓ |
| 2 | **Innovative** | Denoisation | 5/5 | — | — | ❓ |
| 3 | **Patch2Self** | Denoise | 5/5 | — | — | ❓ |
| 4 | **Deep** | Denoising | 4/5 | — | — | ❓ |
| 5 | **Low** | Denoising | 5/5 | — | — | ❓ |
| 6 | **Noise2Average** | Denoising | 5/5 | — | — | ❌ |
| 7 | **Unsupervised** | Denoising | 5/5 | — | — | ❓ |
| 8 | **Self-Prior** | Other | 4/5 | — | — | ❓ |
| 9 | **MRI** | Denoisation | 4/5 | — | — | ❓ |
| 10 | **Dual-stage, Supervised & Zero-Shot Learning** | Denoise | 4/5 | — | — | ❓ |
| 11 | **Dual** | Denoise | 4/5 | — | — | ❓ |
| 12 | **Coil** | Denoising | 5/5 | — | — | ❓ |
| 13 | **Denoising** | Denoising | 4/5 | — | — | ❓ |
| 14 | **Hybrid** | Denoising | 4/5 | — | — | ❓ |
| 15 | **Noise2Noise** | Denoising | 5/5 | — | — | ❓ |
| 16 | **Non-local** | Denoising | 4/5 | — | — | ❓ |
| 17 | **Self-supervised** | Denoising | 5/5 | — | — | ❓ |
| 18 | **Self-Supervised** | Denoising | 5/5 | — | — | ❓ |
| 19 | **Adaptive** | Enhancement | 5/5 | — | — | ❓ |
| 20 | **Multispectral** | Filtering, Denoising | 4/5 | — | — | ❓ |
| 21 | **Deep** | Other | 4/5 | — | — | ❓ |
| 22 | **Deform-Mamba** | Other | 4/5 | — | — | ❓ |
| 23 | **SCORE-BASED** | Other | 5/5 | — | — | ❓ |
| 24 | **AgentMRI** | Reconstruction | 5/5 | — | — | ❓ |
| 25 | **Variational Diffusion Model with guided k-space sampling** | Super-Resolution | 4/5 | — | — | ❓ |

---

## Model Details

### Universal

**Task:** Denoising | **Approach:** FoundationModel | **Promise:** 5/5 | **Score:** 8.0

**Key method:** Random-matrix-theory-based noise removal pipeline

**Summary:** The paper proposes a universal denoising pipeline based on random-matrix-theory for non-Cartesian MRI data, demonstrating its effectiveness in both numerical phantoms and ex vivo mouse brain data.

**Pretrained:** ❓ Not searched

---

### Innovative

**Task:** Denoisation | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 7.0

**Key method:** Federated Learning with Transfer Learning

**Summary:** This study presents a hybrid model combining transfer learning and federated learning for MRI denoising, enhancing image quality while ensuring patient data privacy.

**Pretrained:** ❓ Not searched

---

### Patch2Self

**Task:** Denoise | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 7.0

**Key method:** J-invariance regression framework

**Summary:** Patch2Self, a self-supervised learning-based denoiser, improves the repeatability and conspicuity of pathology in spinal cord dMRI scans by removing assumptions on signal structure.

**Pretrained:** ❓ Not searched

---

### Deep

**Task:** Denoising | **Approach:** FoundationModel | **Promise:** 4/5 | **Score:** 7.0

**Key method:** Deep-CNN with soft shrinkage activation

**Summary:** The paper proposes a deep learning-based approach for MRI image denoising that adapts to the input noise power, outperforming state-of-the-art methods like BM3D.

**Pretrained:** ❓ Not searched

---

### Low

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 7.0

**Key method:** Non-local robust PCA

**Summary:** The paper proposes a non-local robust PCA-based method to jointly denoise multi-contrast images acquired on a low-field MRI system, demonstrating improved image quality compared to BM3D and BM4D methods.

**Pretrained:** ❓ Not searched

---

### Noise2Average

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 7.0

**Key method:** Supervised residual learning, Transfer learning

**Summary:** A novel approach called Noise2Average improves upon the Noise2Noise method for MRI denoising by employing supervised residual learning to preserve image sharpness and transfer learning for subject-specific training.

**Pretrained:** ❌ No weights found

---

### Unsupervised

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 7.0

**Key method:** Noise transformation network and null-space diffusion sampling

**Summary:** The paper presents an unsupervised zero-shot MRI denoising framework that combines a noise transformation network with a pretrained diffusion model to effectively remove complex real-world noise patterns while preserving diagnostic details.

**Pretrained:** ❓ Not searched

---

### Self-Prior

**Task:** Other | **Approach:** FoundationModel | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Self-prior guided Mamba-UNet with improved 2D-selective-scan (ISS2D)

**Summary:** The paper introduces a Mamba-based UNet architecture for medical image super-resolution, leveraging a self-prior mechanism and an improved 2D-selective-scan module. This approach efficiently models long-range dependencies with linear computational complexity, outperforming existing CNN and Transformer-based methods.

**Pretrained:** ❓ Not searched

---

### MRI

**Task:** Denoisation | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Native noise denoising network

**Summary:** This work presents a method for MRI denoising using 'native noise denoising network' (NNDnet), leveraging the inherent noise in the data rather than simulations. It achieves good performance on T1-weighted, T2-weighted images from TMRF and low-field brain imaging.

**Pretrained:** ❓ Not searched

---

### Dual-stage, Supervised & Zero-Shot Learning

**Task:** Denoise | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Dual-stage, Supervised & Zero-Shot Learning

**Summary:** This paper proposes a dual-stage denoising method for low-field MRI that combines supervised and zero-shot learning to improve image quality. It demonstrates consistent performance improvements over traditional supervised methods.

**Pretrained:** ❓ Not searched

---

### Dual

**Task:** Denoise | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Global mask mapper, perceptual loss, adaptive hybrid attention, GANs

**Summary:** This paper proposes a dual-stage model (NRAE) for MRI image restoration, focusing on blind spot denoising and hybrid attention mechanisms to improve image quality while preserving important anatomical details.

**Pretrained:** ❓ Not searched

---

### Coil

**Task:** Denoising | **Approach:** Self-supervised | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Phased-array coil data processing with N2N algorithm

**Summary:** The Coil to Coil (C2C) method generates two noise-corrupted images from single phased-array coil data for self-supervised denoising, requiring no clean images or paired noisy images. It outperforms existing methods in both real and synthetic noised images.

**Pretrained:** ❓ Not searched

---

### Denoising

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Improved 2-step non-local PCA

**Summary:** The authors propose an improved 2-step non-local PCA method for denoising diffusion MRI, showing substantial improvement in image quality and DTI metric estimation compared to existing local-PCA-based methods.

**Pretrained:** ❓ Not searched

---

### Hybrid

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Hybrid-PCA, Marchenko-Pastur distribution, random matrix theory

**Summary:** This paper introduces Hybrid-PCA denoising, a method that combines a-priori noise variance estimation with random matrix theory to improve PCA denoising in MRI data corrupted by spatially correlated noise.

**Pretrained:** ❓ Not searched

---

### Noise2Noise

**Task:** Denoising | **Approach:** Other | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Noise2Noise (N2N) theory for denoising

**Summary:** A deep learning denoising method called N2N-MRI is proposed for high-resolution diffusion-weighted imaging of the brain. Unlike conventional methods that require highly averaged ground-truth images, N2N-MRI uses noise2noise theory to achieve comparable performance without clean targets. The method outperformed traditional approaches in terms of maximum peak SNRs during training.

**Pretrained:** ❓ Not searched

---

### Non-local

**Task:** Denoising | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Two-step non-local low-rank denoising

**Summary:** A two-step non-local low-rank denoising method is proposed to improve the quality of complex-valued diffusion-weighted MRI images. The method outperforms existing approaches in simulation and in vivo data, demonstrating improvements in PSNR and SSIM.

**Pretrained:** ❓ Not searched

---

### Self-supervised

**Task:** Denoising | **Approach:** Self-supervised | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Self-supervised model trained on noisy images

**Summary:** The paper presents a self-supervised deep-learning framework for fast denoising of multidimensional MRI data, which does not require ground truth clean images. The method exploits redundancy in multidimensional MRI data and achieves significant improvement over previous methods in noise reduction and quantification accuracy.

**Pretrained:** ❓ Not searched

---

### Self-Supervised

**Task:** Denoising | **Approach:** Self-supervised Learning | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Repetition to Repetition (Rep2Rep) learning

**Summary:** This paper introduces Repetition to Repetition (Rep2Rep) learning, a novel self-supervised framework for MRI denoising at low-field (<1T). It trains on repeated MRI acquisitions without ground truth data, enabling noise-adaptive training and improving generalization across varying noise levels. The approach outperforms existing methods like MC-SURE in terms of preserving structural details and reducing residual noise.

**Pretrained:** ❓ Not searched

---

### Adaptive

**Task:** Enhancement | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Proximal Gradient Descent, Unrolled Network, Property Constraints

**Summary:** PGDNet, a hybrid approach combining proximal gradient descent and unrolling networks, is proposed for adaptive MRI image denoising and deblurring. It uses property constraints on the degradation kernel to improve performance.

**Pretrained:** ❓ Not searched

---

### Multispectral

**Task:** Filtering, Denoising | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Multispectral NLM with RR

**Summary:** The paper presents a multispectral nonlocal means filter (MS-NLM) incorporating rotations and reflections to improve noise reduction while preserving edges in MRI images. The approach compares local neighborhoods of voxels after rotation and reflection, enhancing the use of image redundancy.

**Pretrained:** ❓ Not searched

---

### Deep

**Task:** Other | **Approach:** FoundationModel | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Deep Image Prior

**Summary:** The study demonstrates an unsupervised learning approach for MR denoising and super-resolution by leveraging the architectural bias of convolutional neural networks (Deep Image Prior) as an image constraint. This method eliminates the need for paired ground-truth images, relying instead on network parameter optimization.

**Pretrained:** ❓ Not searched

---

### Deform-Mamba

**Task:** Other | **Approach:** FoundationModel | **Promise:** 4/5 | **Score:** 6.0

**Key method:** Deformable convolution and vision Mamba block encoder

**Summary:** The paper introduces Deform-Mamba, a novel architecture combining deformable convolution and State Space Models for efficient MRI super-resolution. It addresses the computational limitations of Transformers while effectively capturing both local and global image features.

**Pretrained:** ❓ Not searched

---

### SCORE-BASED

**Task:** Other | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Generalized denoising score matching loss

**Summary:** Corruption2Self (C2S) is a self-supervised MRI denoising framework that uses generalized denoising score matching to learn from noisy observations without clean ground truth. It effectively balances noise reduction and feature preservation using a reparameterized noise-level strategy.

**Pretrained:** ❓ Not searched

---

### AgentMRI

**Task:** Reconstruction | **Approach:** Hybrid | **Promise:** 5/5 | **Score:** 6.0

**Key method:** Multi-query VLM strategy for robust corruption detection and automatic selection of deep learning models for MRI reconstruction

**Summary:** AgentMRI is an AI-driven system leveraging vision language models to perform fully autonomous MRI reconstruction in the presence of multiple degradations, dynamically selecting the best correction model without manual intervention.

**Pretrained:** ❓ Not searched

---

### Variational Diffusion Model with guided k-space sampling

**Task:** Super-Resolution | **Approach:** Hybrid | **Promise:** 4/5 | **Score:** 5.5

**Key method:** Variational Diffusion Model with guided k-space sampling

**Summary:** This paper proposes a new MR image super-resolution method using a variational diffusion model, which leverages lower-resolution K-space measurements to guide the generation process without retraining.

**Pretrained:** ❓ Not searched

---
