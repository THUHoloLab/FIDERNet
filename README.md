# FIDER-Net: imaging through scattering media based on multi-core fiber

Turbid fluids are commonly present in biological cavities and can severely impair visualization of luminal surfaces during procedures such as cystoscopy. A Fiber Imaging Descattering and Enhancement Restoration Network (FIDER-Net) is developed by combining statistical-prior-guided preprocessing with multi-stage training. This strategy decouples scattering degradation from turbid liquids and discrete sampling degradation from the multi-core fiber (MCF), enabling restoration under previously unseen scattering concentrations.

## Model and training stages

Inference follows this sequence:

```text
Raw grayscale scattering image
    -> Local-GLCME preprocessing (CPU/GPU)
    -> Descatter network
    -> Dehoney network with spectral gating
    -> Restored grayscale image
```

There are three training stages and two inference networks:

1. **Stage 1:** train the descatter network to map GLCME-processed scattering images to GLCME-processed fiber-only images, using L1 loss.
2. **Stage 2:** freeze Stage 1 and train the dehoney network against clear GT. For each training sample, use the bridge branch with probability 0.5 and the real fiber input otherwise. The bridge input is `(1 - gamma) * real_fiber + gamma * Stage1_prediction`; gamma ramps from 0.1 to 1.0 over the first 10 epochs. Stage 2 checkpoints are selected using validation loss on Stage 1 predictions.
3. **Stage 3:** fine-tune the cascade. Stage 1 is frozen for the first 10 epochs and only the final L1 loss is active. After unfreezing, intermediate and final L1 losses have weight 1.0 each, and the descatter learning rate is scaled by 0.1.

The reference run used 50 epochs per stage, learning rates `1e-3 / 1e-3 / 1e-4`, batch size 8, crop size 1024, and seed 42. Local-GLCME uses an 8-by-8 tile grid and clip-limit ratio 0.002. Spectral-gate settings are core spacing 4.0 pixels at the input scale, bandwidth 0.06, and harmonics `(1, 2)`.

## Repository layout

```text
FIDERNet/
  README.md
  requirements.txt
  .gitignore
  combine_train/
    arch_util.py                         # Spectral gating and network helpers
    descatternet_model_3stage.py          # Subnetwork architecture
    glcme_preprocess_3stage.py            # CPU Local-GLCME and image cropping
    train_pipeline_common_3stage.py      # Datasets, losses, and training loops
    train_3stage.py                      # Three-stage training entry point
    train_stage1_3stage.py
    train_stage2_3stage.py               # Bridge training
    train_stage3_3stage.py               # Cascade fine-tuning
    prepare_unseen_scattering_split.py   # Seen/unseen concentration folders
    demo_3stage.py                       # Single-image restoration
    val_3stage.py                        # Paired validation and saved outputs
    eval_3stage_table1.py                # Split-based quality evaluation
```

Run the commands below from the repository root. Paths use forward slashes so they can also be used on Windows.

## Requirements and installation

The packaged implementation was checked with Python 3.8.20, PyTorch 2.0.0+cu117, torchvision 0.15.1+cu117, NumPy 1.24.3, Pillow 10.4.0, scikit-image 0.21.0, and Matplotlib 3.7.2 on Windows. The dependency versions are pinned in `requirements.txt`.

Create a Python 3.8 or 3.9 environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For the reference CUDA 11.7 build, install the matching PyTorch wheels before the remaining dependencies:

```bash
python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Use `--device cpu` for CPU inference. LPIPS is used only by `eval_3stage_table1.py`; its first evaluation may download the pretrained perceptual-network weights. The basic validation script reports MSE, PSNR, and SSIM without LPIPS.

## Dataset organization

- Click [**here**](https://drive.google.com/drive/folders/17M0TCEe1HDCundUIhVlBbj6f6OyZifpG?usp=sharing) to download the official datasets.

- Use paired grayscale images with matching filename stems across scattering conditions, fiber-only references, and clear GT:

```text
datasets/
  train_scatter/
    fat_0.05/0.png, 1.png, ...
    fat_0.1/0.png, 1.png, ...
    fat_0.20/0.png, 1.png, ...
    fat_0.25/0.png, 1.png, ...
    fat_0.35/0.png, 1.png, ...
    fat_0.40/0.png, 1.png, ...
  train_fiber/0.png, 1.png, ...
  train_GT/0.png, 1.png, ...
  val_scatter/
    capture_f0.05_test/0.png, 1.png, ...
    capture_f0.1_test/0.png, 1.png, ...
    ...
  val_fiber/0.png, 1.png, ...
  val_GT/0.png, 1.png, ...
```

Training splits are created by image basename before expansion across concentration folders. The reference split uses 80% training, 10% validation, and 10% testing with seed 42. With 2000 common basenames, this produces 1600/200/200 basenames. Keep the generated `split_basenames.json` with the run; using the original split is necessary for exact evaluation of a historical experiment.

Prepare the reference seen/unseen concentration folders:

```bash
python combine_train/prepare_unseen_scattering_split.py --source_scatter_root datasets/train_scatter --output_root dataset_unseen_split --mode copy
```

The training concentrations are `fat_0.05`, `fat_0.1`, `fat_0.25`, and `fat_0.40`. The held-out concentrations are `fat_0.20` and `fat_0.35`. The folder-preparation script also recognizes numeric aliases such as `fat_0.2`.

## Training

Train all three stages with the reference bridge settings:

```bash
python combine_train/train_3stage.py --scattering_root dataset_unseen_split/train_scatter_seen --fiber_root datasets/train_fiber --clear_root datasets/train_GT --output_root runs_fidernet_bridge --crop_size 1024 --batch_size 8 --seed 42 --device cuda
```

The remaining reference settings described above are the defaults in `train_3stage.py`. On machines with less GPU memory, lower `--batch_size`. On Windows, `--num_workers 0` and `--cache_workers 1` can simplify loading, without changing the network definition.

The script creates:

```text
runs_fidernet_bridge/train_<timestamp>/
  split_basenames.json
  stage1_descatter/weights/best.pt
  stage2_dehoney/weights/best.pt
  stage3_joint/weights/best.pt
```

Each stage also writes its configuration, training history, and curves. The final inference checkpoint is `stage3_joint/weights/best.pt`, which contains `descatter_state_dict`, `dehoney_state_dict`, and `config`. A third network is not used at inference.

## Single-image inference

Supply your own trained joint checkpoint:

```bash
python combine_train/demo_3stage.py --input_image path/to/input.png --weights_joint path/to/stage3_joint/weights/best.pt --output_dir outputs/demo --inference_mode sliding --crop_size 1024 --device cuda --save_glcme_input --save_mid_prediction --save_panel
```

The script saves `<input_stem>_restored.png`, and the requested intermediate images. A 1024-by-1024 input uses one 1024-by-1024 inference window; larger images use overlapping windows with averaging. Images are not resized. Keep Local-GLCME enabled for the published full staged bridge model; `--no_glcme` is for separately trained raw-input models.



