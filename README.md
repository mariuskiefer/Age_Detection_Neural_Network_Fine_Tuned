# Age Classification with Transfer Learning

This project implements an age classification model using deep learning techniques, leveraging **transfer learning** with `EfficientNetB0` and advanced training strategies such as **callbacks** to optimize model performance. It is designed for real-world applications like verifying minimum legal age for alcohol sales, using facial image inputs.

---

## Project Highlights

- **Transfer Learning**: Utilizes a pretrained `EfficientNetB0` model (trained on ImageNet) to extract rich visual features. Only the top layers are fine-tuned, significantly reducing training time and improving generalization on the UTKFace dataset.
- **Callbacks for Efficient Training**:
  - `ModelCheckpoint` is used to save the model with the best validation performance.
  - `ReduceLROnPlateau` automatically lowers the learning rate when training stalls, helping the model escape local minima.
- **Data Augmentation**: Employs `ImageDataGenerator` to simulate real-world variability, improving robustness.
- **Performance Metrics**: Evaluates precision and recall in addition to accuracy to ensure fairness and model reliability.

---

## Dataset

This model uses the **UTKFace** dataset, which includes over 20,000 images labeled with age, gender, and ethnicity. Due to the dataset size (~1.39GB), files must be manually downloaded.

### Download Instructions

Please download the following tar files from the [UTKFace Dataset Website](https://susanqq.github.io/UTKFace/):

- `part1.tar.gz`
- `part2.tar.gz`
- `part3.tar.gz`

---

## Data Setup

1. Place the downloaded tar files in the root directory.
2. **No extraction is needed** — the script handles it automatically.

---

## Running the Code

Ensure all dependencies are installed (see below), then run the Jupyter notebook:  
`code.ipynb`

Make sure file paths point to the correct dataset directory.

---

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- PIL
- scikit-learn

You can install the required packages via:

```bash
pip install -r requirements.txt
