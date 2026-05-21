# Pakistan Crop Recommendation Model

A machine learning project that recommends the most suitable crop to grow based on soil nutrients and climate conditions. Built using a dataset of 10 Pakistani crops across 1,000 samples.

---

## What This Project Does

Farmers often struggle to decide which crop will perform best given their land and local climate. This model takes basic soil test results and weather data as input and outputs the most suitable crop to grow, along with a confidence score.

---

## Dataset

The dataset contains 1,000 samples covering 10 major Pakistani crops: Kapas, Ganna, Gandum, Masoor, Aloo, Makki, Chawal, Sarson, Chana, and Moong. Each sample has 7 input features.

| Feature | Description |
|---|---|
| N | Nitrogen content in soil (kg/ha) |
| P | Phosphorus content in soil (kg/ha) |
| K | Potassium content in soil (kg/ha) |
| temperature | Average temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH level |
| rainfall | Annual rainfall (mm) |

The dataset is perfectly balanced with 100 samples per crop and contains no missing values.

---

## Models Compared

Five models were trained and evaluated using 5-fold cross-validation before selecting the final one.

| Model | CV Accuracy |
|---|---|
| Random Forest | see notebook |
| Gradient Boosting | see notebook |
| Support Vector Machine | see notebook |
| K-Nearest Neighbors | see notebook |
| Neural Network | see notebook |

---

## Neural Network Architecture

The final model is a fully connected neural network with Batch Normalization and Dropout layers to reduce overfitting. It uses Early Stopping and ReduceLROnPlateau callbacks during training so it stops automatically when it is no longer improving.

Input (7 features) → Dense 128 → BatchNorm → Dropout 0.3 → Dense 64 → BatchNorm → Dropout 0.2 → Dense 32 → Output (10 classes)

---

## How to Use

```python
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    inp = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    inp_scaled = scaler.transform(inp)
    probs = model.predict(inp_scaled, verbose=0)[0]
    idx = np.argmax(probs)
    return le.classes_[idx], round(probs[idx] * 100, 2)

crop, confidence = recommend_crop(101, 58, 18, 25.7, 81.4, 6.65, 78.6)
print(f"Recommended crop: {crop} ({confidence}% confidence)")
```

---

## Output Files

Running the notebook produces these files ready for deployment.

| File | Description |
|---|---|
| crop_model.tflite | Quantized model for mobile and edge devices |
| label_map.json | Maps class index to crop name |
| scaler_params.json | Mean and std values to normalize new inputs |
| confusion_matrix.png | Per-class prediction breakdown |
| training_curves.png | Accuracy and loss over epochs |

---

## Run It

Open the notebook directly on Kaggle without any setup.

Kaggle Notebook: https://www.kaggle.com/code/abdulqadirds/crop-recommendation-ipynb

---

## Requirements

```
tensorflow
scikit-learn
pandas
numpy
seaborn
matplotlib
```

Install with: `pip install tensorflow scikit-learn pandas numpy seaborn matplotlib`
