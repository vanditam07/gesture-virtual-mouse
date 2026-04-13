# Prototypical Network Gesture Personalisation

This project includes a few-shot meta-learning system that lets you record a handful of gesture samples and get a classifier personalised to your hands. When confident, the personalised classifier overrides the default rule-based system. When not confident, it silently falls back -- so nothing breaks.

## Quick Start

### 1. Run gesture mode

```bash
python main.py --mode gesture
```

### 2. Press `P` to launch the Personalisation Wizard

The wizard walks you through recording all 8 gestures:

| Step | Gesture            | What to do with your hand                        |
|------|--------------------|--------------------------------------------------|
| 1    | PALM               | Open hand, all fingers spread                    |
| 2    | V_GEST             | Index + middle finger in a V shape               |
| 3    | FIST               | Closed fist                                      |
| 4    | MID                | Only middle finger extended                      |
| 5    | INDEX              | Only index finger pointing                       |
| 6    | TWO_FINGER_CLOSED  | Index + middle finger together (not spread)      |
| 7    | PINCH_MINOR        | Thumb + index pinch, other fingers open          |
| 8    | PINCH_MAJOR        | Thumb + index pinch, other fingers curled        |

For each gesture:

1. Read the on-screen prompt.
2. Press **SPACE** when ready.
3. A 3-second countdown begins.
4. Hold the pose steadily for 4 seconds while frames are captured.
5. Repeat for all 8 gestures.
6. Press **ESC** at any point to cancel enrolment.

### 3. After enrolment

The system immediately loads your personalised prototypes. You will see the HUD in the camera window change from `[rule]` to `[proto (XX%)]` when the prototypical network is confident about a gesture.

## How It Works

1. **Feature extraction** -- Each frame produces a 77-dimensional vector from 21 MediaPipe hand landmarks (63 raw coordinates + 5 fingertip distances + 5 inter-finger angles + 4 curl ratios).
2. **Encoder** -- A small MLP (77 → 256 → 128 → 64) maps the feature vector to a 64-dim L2-normalised embedding.
3. **Prototypes** -- During enrolment, the mean embedding per gesture class becomes that class's prototype.
4. **Inference** -- Each frame's embedding is compared to all prototypes via squared Euclidean distance. Softmax over negative distances gives class probabilities.
5. **Confidence gate** -- If the top probability is ≥ 65%, the personalised prediction is used. Otherwise, the system falls back to the original rule-based classifier.
6. **Majority vote** -- A sliding window of 5 frames smooths predictions to avoid jitter.

## Files

| File                                  | Purpose                                      |
|---------------------------------------|----------------------------------------------|
| `src/proto_net/feature_extraction.py` | 77-dim feature vector from landmarks         |
| `src/proto_net/encoder.py`            | ProtoEncoder MLP definition                  |
| `src/proto_net/meta_train.py`         | Episodic training loop + synthetic data      |
| `src/proto_net/classifier.py`         | Real-time inference with confidence gate      |
| `src/proto_net/wizard.py`             | Personalisation Wizard UI                    |
| `src/proto_net/gesture_templates.py`  | Canonical hand-pose templates (8 classes)    |
| `src/models/pretrained_encoder.pth`   | Pre-trained encoder checkpoint               |
| `user_prototypes.npy`                 | Your personalised prototypes (after enrolment)|

## Tips

- **Lighting matters** -- Record in the same lighting conditions you normally use the app in.
- **Hand distance** -- Keep your hand at a comfortable distance from the camera, roughly 30-60 cm.
- **Re-enrol any time** -- Press `P` again to re-run the wizard. The old prototypes are overwritten.
- **Persistence** -- `user_prototypes.npy` is saved at the project root and persists across sessions. You only need to enrol once unless you want to re-record.
- **HUD feedback** -- The camera window shows `[proto (XX%)]` or `[rule]` so you always know which classifier is active.

## Re-training the Encoder

The encoder ships pre-trained on synthetic data. To re-train with real data or different parameters:

```bash
cd src
python -m proto_net.meta_train --episodes 10000 --n-way 5 --k-shot 5 --q-query 10 --lr 0.001
```

The best checkpoint is saved to `src/models/pretrained_encoder.pth`.
