# Automatic Music Transcription (AMT) with Ensemble of Deep Neural Networks

This repository contains the implementation of a supervised deep learning pipeline for **Multi-Instrument Polyphonic Automatic Music Transcription (AMT)**. Developed as part of the **CS415** course at **Sabancı University**, this project explores instrument-aware transcription using the **Slakh2100** dataset.

---

## Project Overview

Traditional AMT systems often struggle with **timbre disentanglement**—the ability to distinguish between different instruments (e.g., a violin vs. a guitar) playing the same frequency simultaneously in a mixed audio track. 

Our approach addresses this limitation by:
* **Decomposing the problem:** Instead of a single monolithic model, we employ an ensemble of instrument-specific models. 
* **Targeted Prediction:** Each network is trained exclusively to predict a binary piano-roll activation matrix for a single target instrument from the full polyphonic mixture.
* **Architectural Comparison:** We conduct a controlled comparison between an **encoder-only Transformer** and an **xLSTM-based** model, both utilizing a shared **CNN feature extractor**.

---

## Key Features

* **Instrument-Specific Models:** Specialized independent models were developed for **Piano, Bass, Strings,** and **Guitar** based on dataset coverage and note activity.
* **Advanced Sequence Modeling:** Implementation of **xLSTM (Matrix LSTM)** blocks, which utilize exponential gating and 2D matrix memory for linear scaling $O(L)$ compared to the quadratic complexity $O(L^{2})$ of standard Transformers.
* **Robust Data Pipeline:** Features an offline preprocessing stage that serializes raw FLAC audio and MIDI labels into standardized **PyTorch tensors (.pt files)** to eliminate I/O bottlenecks during training.
* **Novel Evaluation Metrics:** In addition to frame-level **F1 scores**, we utilize **Chamfer Distance** to capture the geometric and temporal proximity of predicted notes, providing a more musically relevant error metric.

---

## System Architecture

The pipeline consists of three main stages:

### 1. Feature Extraction (CNN Backbone)

Both architectures share a foundational CNN front-end that functions as a "feature refiner" to perform frequency compression.
* **Input:** Log-Mel Spectrograms scaled via $S_{log}=log(S_{mel}+\epsilon)$ and processed with Instance Normalization.
* **Structure:** Successive layers of **Conv2d, BatchNorm, and ELU**.

### 2. Sequence Modeling (Backends)
* **Transformer Encoder:** A 3-layer stack using **8-head multi-head self-attention** with a **Pre-Norm** configuration for improved training stability.
* **xLSTM Stack:** Utilizes **mLSTM blocks** that are fully parallelizable across the temporal dimension, matching Transformer training speeds while retaining a recurrent inductive bias.

### 3. Classification Head
A standardized head consisting of **Layer Normalization**, a fully connected network with **ELU** and **Dropout** ($p=0.3$), projecting to **88 output units** (MIDI indices 21-109).

---

## Dataset & Preprocessing

The project utilizes the **Slakh2100** dataset, providing synthesized polyphonic audio aligned with MIDI labels.
* **Audio Preparation:** Audio stems are resampled and down-mixed to mono by averaging channel intensities.
* **Labels:** MIDI events are rasterized into a **Piano Roll** representation synchronized exactly with the STFT hop length.
* **Chunking:** Tracks are sliced into deterministic **4-second segments** (128 frames at a 512 hop length) for consistent batch processing.

---

## Results & Comparison

| Metric | Transformer | xLSTM |
| :--- | :--- | :--- |
| **Complexity** | Quadratic $O(L^{2})$ | Linear $O(L)$ |
| **Data Efficiency** | High demand ("data-hungry") | High efficiency via recurrent bias |
| **Convergence** | Slower; prone to early overfitting | Faster; reached optimal performance in fewer epochs |

**Key Finding:** The **xLSTM** architecture generally outperformed the Transformer in this domain, particularly in handling data scarcity and capturing musically meaningful temporal patterns.

---

Contributors: Yeşim Tosun, Umut Yunusoğlu, Damla Uçar, Teoman Arabul
