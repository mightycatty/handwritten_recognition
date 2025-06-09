# Handwritten Sequence Recognition Based on DenseNet + CTC

Handwritten sequence recognition uses a DenseNet backbone combined with a CTC (Connectionist Temporal Classification) loss. 
Two separate models are provided for recognizing **letters** and **digits**.

---

## 📁 Key Components

- **`config/`**  
  Contains configuration files for model structure and training parameters.

- **`data_utils_pack/`**  
  Data preparation and preprocessing utilities, including data synthesis and packaging for training.

- **`train_digit/`, `train_letter/`**  
  Scripts for training digit and letter recognition models.

- **`inference/`, `inference_letter/`**  
  Demo scripts for model inference and visualizing prediction results.

---

## 🚀 How to Run

1. **Configure Training**  
   Edit `config/TrainingConfig` to set training parameters (e.g., batch size, learning rate, model path, etc.).

2. **Prepare Dataset**  
   Download the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) and place it in the path expected by `generator_digit/`.

3. **Start Training**  
   Run either:
   ```bash
   python train_digit/train.py
