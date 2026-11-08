# Facial Expression Classification

A learning project that combines a PyTorch Convolutional Neural Network (CNN) with OpenCV to classify facial expressions from a webcam. Detected faces are highlighted with a bounding box and a predicted expression label. The application also announces changes in the predicted label using text-to-speech.

## Features

- Classifies seven expression categories: **angry, disgust, fear, happy, neutral, sad, and surprise**.
- Detects faces using OpenCV's pretrained Haar cascade.
- Processes grayscale face images at **64 × 64 pixels**.
- Uses a custom CNN with three convolutional blocks, batch normalization, max pooling, and dropout.
- Provides spoken feedback with `pyttsx3` when the predicted expression changes.

## Repository Files

| File | Purpose |
| --- | --- |
| `comp_visi.py` | Loads the image dataset, trains the CNN, evaluates it on the validation folder, and saves the model weights. |
| `cam_test.py` | Loads the saved weights and runs webcam classification with text and speech output. |

The dataset and trained weights are **not included** in this repository. Training creates `face_emote_recog.pth`, which is required by the webcam script.

## Setup

Clone the repository and install the dependencies in your Python environment:

```bash
git clone https://github.com/Doddlet/open_cv_project.git
cd open_cv_project
python -m pip install torch torchvision opencv-python Pillow matplotlib pyttsx3
```

A working webcam and an available system text-to-speech engine are required for the webcam application. The code uses camera index `0` by default.

## Prepare the Dataset

Place labeled face images in these two directories:

- `data/images/train/` — training images.
- `data/images/validation/` — validation images.

Inside **each** directory, create the same seven class folders: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. Place each image in its corresponding class folder. Keep the training and validation images separate.

## Run

Run these commands from the repository's root directory.

**1. Train and evaluate the model**

```bash
python comp_visi.py
```

The script trains for 50 epochs, applies horizontal flips and small rotations to training images, prints validation accuracy, and saves the final model weights as `face_emote_recog.pth`.

**2. Start webcam classification**

```bash
python cam_test.py
```

Keep `face_emote_recog.pth` in the repository's root directory. Press **Q** while the webcam window is active to exit.

## Limitations

- Predictions may vary with lighting, face angle, occlusion, and differences from the training images.
- Expression labels describe the model's visual classification; they do not establish a person's actual emotional state.
- The repository currently provides no recorded accuracy results or separate test-set evaluation. The training script evaluates on the `validation` folder.
- Speech is triggered by label changes, so rapidly changing predictions may produce repeated announcements.

## Author

[Doddlet](https://github.com/Doddlet)
