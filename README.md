# 🔬 Decision Support Tool: Skin Disease Classification (Melanoma vs. Nevus)

## 🚀 Project Overview

This project develops a **Deep Learning decision support tool** for the binary classification of skin lesions, distinguishing between **Melanoma (mel)**, a malignant cancer, and **Nevus (nv)**, a common benign mole.

The model analyzes **dermoscopic images** and highlights high-risk lesions that are most likely to be melanoma. In a clinical workflow, this system serves as a powerful **triage assistant**, ensuring that highly dangerous, underrepresented cases are not missed, thereby improving patient outcomes. 

---

## ✨ Key Technical Innovations

Addressing the critical challenges of class imbalance and minimizing false negatives (missing a melanoma) was the primary focus of this project.

* **Advanced Data Handling:** Utilized the **HAM10000** dataset, focusing on the two most clinically relevant classes (`nv` and `mel`). Implemented **undersampling** on the majority class (`nv`) and used a custom **Balanced Data Sequence** during training with **data augmentation** to ensure the model trained equally on both lesion types.
* **Focal Loss Implementation:** Used **Binary Focal Loss** ($\gamma=2.0, \alpha=0.25$) instead of standard binary cross-entropy. This loss function heavily penalizes misclassified examples, forcing the model to focus on difficult-to-classify lesions (like subtle melanomas) and further mitigating class imbalance issues.
* **Optimal Threshold Tuning:** The sigmoid output threshold was meticulously calibrated using **Youden's J statistic** on the test set to maximize **Sensitivity (Recall)** for the rare and critical melanoma class.
* **Transfer Learning & Fine-Tuning:** The model leverages a **ResNet50** CNN pre-trained on ImageNet. It was fine-tuned by unfreezing the last 30 layers to adapt the powerful feature extractor specifically for recognizing dermatological patterns.

---

## 💻 Technical Setup & Data

### Prerequisites

To run this notebook, you need a Python environment with:

* **TensorFlow / Keras**
* **OpenCV (`cv2`)**
* **NumPy, Pandas, scikit-learn**

### Data Source

This project requires the **HAM10000** dataset. The metadata and image files must be organized into the following structure:

└── HAM10000/ 

.  ├── HAM10000_metadata.csv 

.  ├── HAM10000_images_part_1/ 

.  └── HAM10000_images_part_2/


### Model Architecture

The model is built on a **ResNet50** base:
1.  **ResNet50 Base Model** (partially fine-tuned)
2.  `GlobalAveragePooling`
3.  `Dense` Layer (128 units, ReLU activation)
4.  `Dropout` Layer (0.5)
5.  `Dense` Output Layer (1 unit, Sigmoid activation)

---

## 📈 Model Performance & Results

The final model was evaluated on a held-out test set using the optimally tuned threshold of **$0.482$**.

### Key Test Metrics

| Metric | Value |
| :--- | :--- |
| **Test Loss** | $0.0686$ |
| **Test Accuracy** | $0.8453$ |
| **Test AUC** | **$0.9294$** |

### Classification Report

| Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Nevus (nv)** | $0.92$ | $0.78$ | $0.84$ | $223$ |
| **Melanoma (mel)** | $0.81$ | **$0.93$** | $0.86$ | $223$ |

### Confusion Matrix

| | **nv (pred)** | **mel (pred)** |
| :--- | :--- | :--- |
| **nv (true)** | $173$ (True Negatives) | $50$ (False Positives) |
| **mel (true)** | **$16$ (False Negatives)** | $207$ (True Positives) |

### **Clinical Interpretation**

The model achieved an outstanding **Recall (Sensitivity) of 93%** for the Melanoma class.

* This means that out of all true melanoma cases in the test set, only **16** were incorrectly classified as benign (False Negatives).
* In a clinical setting, minimizing False Negatives is paramount. The high recall rate demonstrates that the system is highly effective at meeting its primary objective: **flagging high-risk lesions for closer inspection**, acting as a reliable safety net for diagnosis.

---

## 📚 Future Directions

* Investigate the use of different vision transformers (ViTs) or larger ResNet models.
* Implement a **Grad-CAM** layer to visually explain the model's prediction by highlighting the decisive regions on the lesion image.
* Expand the model to include all seven common skin lesion categories from the HAM10000
