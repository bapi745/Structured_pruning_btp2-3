# Brainstorm: Novel Data-Free Structured Pruning for CV
## Target: **IJCNN 2027** | Cape Town, South Africa | June 14–18, 2027

---

## 🎯 Primary Target: IJCNN 2027

| Conference | Date | Location | Submission Deadline | Status |
|---|---|---|---|---|
| **IJCNN 2027** | June 14–18, 2027 | Cape Town, South Africa | **Jan 31, 2027** | ✅ ~5.5 months away |
| Acceptance Notification | — | — | Mar 15, 2027 | — |

**Why IJCNN is a great fit:**
- IJCNN is the flagship conference of INNS (International Neural Networks Society) — well-regarded for neural network compression, efficient learning, and hardware-aware methods
- More accessible than CVPR/WACV in terms of reviewer bar, yet respected in the community
- A 5.5-month runway allows for thorough experiments, ablations, and solid writing
- Neural network pruning is a core topic area, making FREQ-PRUNE a natural fit

---

## 💡 Core Novel Idea: **FREQ-PRUNE**
### *Frequency-Domain Filter Redundancy Pruning with Cross-Layer Balanced Masks*

> *"Redundant filters in a CNN are not just small in magnitude — they are spectrally homogeneous. Prune by measuring frequency diversity, not weight size."*

---

### 🧠 The Central Insight

When CNNs are trained, each convolutional filter learns to detect a specific pattern. From the neuroscience literature, we know that early CNN layers learn **Gabor-like, frequency-selective filters** (edge detectors, orientation detectors). A well-trained CNN should have a **diverse filter bank** — each filter tuned to different spatial frequencies and orientations.

**Key observation**: Redundant filters are spectrally homogeneous — they respond to the same frequency bands. We can detect this without any data by analyzing the filter's **2D Discrete Cosine Transform (DCT) energy spectrum**.

This gives us a completely data-free, interpretable, and CV-native saliency score.

---

## 🏗️ Method: Step-by-Step

### Phase 1 — Per-Filter Frequency Diversity Score (FDS)

For each filter `f_i` of shape `(C_in, k, k)` in a convolutional layer:

1. Average across input channels: `F_i = mean_c(|f_i[c, :, :]|)` → shape `(k, k)`
2. Apply 2D DCT: `D_i = DCT2D(F_i)`
3. Flatten and normalize: `p_i = |D_i|² / Σ|D_i|²` → a probability distribution over frequencies
4. Compute **Spectral Entropy** as the saliency score:
   ```
   FDS(i) = -Σ p_i(k) log p_i(k)
   ```

**Interpretation**:
- **High FDS** → energy spread across many frequencies → filter is **spectrally rich** → **KEEP**
- **Low FDS** → energy concentrated in a few coefficients → filter detects a very narrow band → **likely redundant if multiple such filters exist**

This is entirely **data-free**: uses only the weights of the trained model.

---

### Phase 2 — Cross-Layer Filter Redundancy via NMF Fingerprinting

Inspired by **YOPO's cross-layer NMF**:

1. For each filter `f_i`, compute its DCT spectrum vector `d_i ∈ R^(k²)` (flattened 2D-DCT coefficients)
2. Stack all filter spectra across ALL layers into matrix `A ∈ R^(N_total × k²)` where `N_total` = total number of filters
3. Fit a **rank-r NMF**: `A ≈ WH` where `W ∈ R^(N × r)`, `H ∈ R^(r × k²)`
4. `W[i, :]` = "NMF fingerprint" of filter `i` — how much it participates in each spectral basis
5. Filters with **low L1 row norm in W** are globally redundant across all layers

**Why this beats per-layer L1 pruning**:
- A filter may have decent magnitude per-layer but be globally redundant if 5 other filters across the network already capture that frequency pattern
- NMF captures **cross-layer spectral redundancy** that single-layer methods miss
- NMF fingerprints are **transferable**: same scores, re-threshold for different sparsity targets (no recomputation)

---

### Phase 3 — Spatial-Aware Saliency Correction (SPSRC Insight)

Inspired by **SPSRC's convolution reorganization**: zero-padding biases make central kernel positions more frequently used than edge positions. Apply a **position-importance weight mask** to the filter before computing DCT:

```
W_pos[i,j] = (number of times position (i,j) participates in a valid convolution) / total_convolutions
```

For a k×k filter with p padding on an H×W feature map:
```
W_pos[i,j] = min(i+p+1, k, H) × min(j+p+1, k, W) / (H × W)
```

Multiply `F_i` by `W_pos` before DCT → spatially-corrected frequency saliency.

---

### Phase 4 — Hardware-Balanced Pruning (Cross-Filter Insight)

Inspired by **Cross-Filter Structured Pruning**: enforce balanced sparsity across filter groups to eliminate load imbalance:

1. Group filters by their NMF fingerprint similarity (cluster by `W[i, :]`)
2. Within each cluster, keep only the **centroid filter** (or the one with highest FDS)
3. Remaining filters in the cluster are pruned
4. This naturally produces **uniform pruning patterns** across groups → all processing elements handle equal workloads on hardware accelerators

---

## 🎯 What Makes FREQ-PRUNE Novel

| Property | YOPO | Cross-Filter | SPSRC | RED++ | **FREQ-PRUNE** |
|---|---|---|---|---|---|
| Data-free | ✅ | ❌ | ❌ | ✅ | ✅ |
| Structured | Partial | ✅ | ✅ | ✅ | ✅ |
| Frequency-domain | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cross-layer global | ✅ | ❌ | ❌ | ❌ | ✅ |
| Spatially-corrected | ❌ | ❌ | ✅ | ❌ | ✅ |
| Hardware-balanced | ❌ | ✅ | ❌ | ❌ | ✅ |
| Transferable masks | ✅ | ❌ | ❌ | ❌ | ✅ |
| Interpretable | ❌ | ❌ | Partial | Partial | ✅ (spectral) |

**Novelty gap over existing data-free methods:**
- **SNIP/GraSP**: Gradient-based, data-dependent
- **YOPO**: Uses NMF on raw weights, not frequency domain; targets CNNs at initialization
- **RED/RED++**: Similarity-based clustering on raw weights, no spectral analysis
- **FPGM (Filter Pruning via Geometric Median)**: Geometric-median distance, no cross-layer, no frequency

**FREQ-PRUNE uniquely combines**: spectral diversity scoring + cross-layer NMF fingerprinting + spatial correction + hardware-balanced clustering — none of the above does all four.

---

## 🔬 Compelling Research Questions

1. **RQ1**: Does spectral entropy of DCT-transformed filters correlate with filter redundancy across diverse CNN architectures (ResNet, VGG, MobileNet, EfficientNet)?
2. **RQ2**: Does cross-layer NMF fingerprinting discover more globally redundant filters than per-layer L1 magnitude?
3. **RQ3**: Does spatial-position correction (SPSRC-style) improve the quality of frequency-domain saliency scores?
4. **RQ4**: How well do FREQ-PRUNE masks transfer across (a) sparsity budgets — re-threshold NMF scores, (b) domains — compute on ImageNet model, apply to COCO fine-tuned version?

---

## 📊 Evaluation Plan

### Models (CNN + Hybrid)
- ResNet-50, ResNet-101 (backbone standard)
- VGG-16 (classic benchmark)
- MobileNetV2 / MobileNetV3 (edge deployment)
- EfficientNet-B0 (modern efficient baseline)
- ConvNeXt-T (modern CNN — shows generalization beyond classic CNNs)

### Benchmarks
- **Imagenette** — 10-class subset of ImageNet, fast to train, strong proxy for full ImageNet; used for main Pareto curves
- **CIFAR-10 / CIFAR-100** — ablation studies and sensitivity analysis (fastest iteration)
- **Tiny-ImageNet** (200 classes, 64×64) — medium-scale benchmark between CIFAR and full ImageNet; shows scalability beyond toy datasets
- **Oxford 102 Flowers** — medium-scale fine-grained recognition; demonstrates transfer of pruned model to domain shift

### Baselines to Beat
| Method | Data-Free? | Structured? |
|---|---|---|
| L1-norm filter pruning | ✅ | ✅ |
| FPGM | ✅ | ✅ |
| YOPO (adapted to post-training) | ✅ | Partial |
| RED / RED++ | ✅ | ✅ |
| Taylor expansion (with calibration data) | ❌ | ✅ |

### Metrics
- **Accuracy** @ 30%, 50%, 70% sparsity
- **FLOPs reduction** (structured → direct FLOP savings)
- **Wall-clock inference speedup** (PyTorch, ONNX Runtime)
- **Mask transfer quality**: compute mask on ImageNet-pretrained ResNet-50, evaluate directly on CIFAR fine-tuned model
- **Frequency diversity visualization** (qualitative: show pruned vs. kept filters in spectral space)

---

## 📐 Theoretical Hook

**Claim**: The spectral entropy of a convolutional filter's DCT representation is a lower bound on the filter's unique contribution to the network's receptive field basis. If multiple filters share low spectral entropy concentrated in the same frequency bands, at most one needs to be retained.

We can support this with:
- Correlation: Show FDS score vs. Taylor importance score (data-dependent oracle) on held-out set — demonstrate FDS is a strong proxy
- Visualization: t-SNE of filter DCT spectra — clusters correspond to redundant groups, FREQ-PRUNE correctly identifies and prunes within clusters

---

## 🌍 CVPR Industry / Workshop Angle

Frame the paper with these industry pain points:

1. **Privacy-preserving compression**: "Healthcare and finance companies cannot send patient/customer data to model compression services. FREQ-PRUNE compresses using only the model weights."
2. **Deployment to new domains**: "A model trained on ImageNet can be pruned and deployed to a specialized robotics domain without collecting domain data for compression."
3. **Reusable compression**: "Compute FREQ-PRUNE scores once, deploy at multiple sparsity levels by re-thresholding — no recomputation per deployment target."
4. **Hardware-ready**: "Clustered filter pruning ensures load-balanced execution on multi-PE accelerators, directly translating to measured inference speedup."

---

## 🔀 Alternative / Backup Ideas

### Alt-A: Gabor Projection Score (more interpretable)
For each filter, compute its projection onto a pre-built Gabor wavelet dictionary (multiple scales × orientations). Measure `coverage = ||G^T f||_0` (how many distinct Gabor atoms are activated). Low-coverage filters are narrow-band and likely redundant. Very interpretable, directly tied to neuroscience.

### Alt-B: Persistent Homology Filter Diversity
Use topological data analysis: compute 0D persistent homology of the filter's weight surface. Filters with low topological complexity (few connected components, simple structure) are structurally simpler and more redundant. Very novel, but high implementation complexity.

### Alt-C: Weight Covariance Eigenspectrum Pruning
For each layer, compute the sample covariance matrix of filter weights: `Cov = F^T F / N`. Use eigenvalue Participation Ratio of Cov as saliency. Low PR means filters are nearly collinear → prune collinear ones. This is the IRSP idea adapted from attention heads to conv filters.

---

## 📅 Timeline to IJCNN 2027 (Deadline: Jan 31, 2027)

> [!TIP]
> ~5.5 months is a comfortable and realistic window to implement all 4 phases, run full experiments, and write a polished paper.

| Month | Phase | Tasks |
|---|---|---|
| **Aug 2026** | Implementation | Implement Phase 1 (FDS scoring); CIFAR-10/100 baseline experiments; compare vs. L1-norm & FPGM |
| **Sep 2026** | Implementation | Implement Phase 2 (Cross-layer NMF); run ablations on CIFAR-100 for NMF rank sensitivity |
| **Oct 2026** | Experiments | Full experiments on Imagenette + Tiny-ImageNet; Phases 3 & 4 (spatial correction + balanced clustering) |
| **Nov 2026** | Experiments | Oxford Flowers transfer experiments; hardware inference speedup benchmarks (PyTorch + ONNX) |
| **Dec 2026** | Writing | Draft paper, figures, Pareto plots, and ablation tables |
| **Jan 1–25, 2027** | Polish & Submit | Revision, proofreading, final submission before Jan 31 deadline |

---

## 🔑 One-Line Paper Pitch

> "FREQ-PRUNE identifies and removes spectrally redundant convolutional filters — using only their DCT energy distribution and a cross-layer NMF fingerprint — producing hardware-balanced structured masks with no data, no gradients, and transferable across sparsity budgets and domains."
