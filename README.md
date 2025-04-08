# Comparative Analysis of Machine Learning and Deep Learning Techniques for Rainfall Prediction ([Link](https://drive.google.com/file/d/1DkIMeXenqMF87nxPRC7wWX6qzVpodgqG/view?usp=sharing))

## 📜 Project Description
This project presents a comprehensive comparative analysis of **Machine Learning (ML)** and **Deep Learning (DL)** techniques for rainfall prediction, focusing on **Mumbai, India**. The study evaluates classical ML models such as Logistic Regression, Decision Tree, Random Forest, and Support Vector Machines (SVM), alongside a Deep Learning-based neural network model (Multilayer Perceptron - MLP).

The findings demonstrate that the **Random Forest algorithm** achieved the highest accuracy (AUC: 0.9279), while the neural network model exhibited competitive performance. This project aims to improve rainfall forecasting systems for applications like agriculture, urban planning, and disaster preparedness.

---

## 🚀 Features
- Implementation of various ML algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Deep Learning model using Multilayer Perceptron (MLP).
- Comprehensive evaluation using metrics like Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- Data preprocessing steps including feature engineering and correlation analysis.
- Visualizations: Correlation matrix, ROC curves, confusion matrices.

---

## 📊 Results Summary
| Algorithm            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 91%      | 90%       | 73%    | 77%      | 0.9041  |
| SVM                  | 89%      | 89%       | 76%    | 77%      | 0.8926  |
| Decision Tree        | 88%      | 88%       | 76%    | 76%      | 0.8733  |
| Random Forest        | **93%**  | **93%**   | **78%**| **83%**  | **0.9279** |
| Neural Network (MLP) | 90%      | 90%       | 79%    | 80%      | 0.9085  |

### Key Takeaways:
- **Random Forest** performed the best overall.
- The **Neural Network model** was competitive but did not outperform Random Forest.
- Traditional ML models like Logistic Regression also achieved strong baseline results.

---

## 📚 Dataset

### Source:
The dataset contains historical weather observations from Mumbai, including:
- Temperature
- Humidity
- Pressure
- Wind Speed
- Precipitation

### Preprocessing Steps:
1. Handling missing values using mean/mode imputation.
2. Feature engineering:
   - Extracted date/time components.
   - Created new features like daily average temperature, humidity, etc.
3. Defined target variables:
   - `RainToday` (binary variable based on current weather conditions).
   - `RainTomorrow` (lagged target variable for prediction).

For more details on the dataset and preprocessing steps, refer to the `reports/` folder.

---

## 📈 Evaluation Metrics

We evaluated model performance using the following metrics:
1. **Accuracy**: Overall correctness of predictions.
2. **Precision**: Correctly predicted rain days out of all predicted rain days.
3. **Recall (Sensitivity)**: Correctly predicted rain days out of actual rain days.
4. **F1-Score**: Harmonic mean of Precision and Recall.
5. **ROC-AUC**: Area under the ROC curve.

For more details on these metrics, refer to Section `3.3 Evaluation Metrics` in the report.

---

## 🌟 Future Work

Potential improvements include:
1. Using advanced DL architectures like LSTM or CNN-LSTM for capturing temporal dependencies.
2. Extending predictions to rainfall intensity categories instead of binary classification.
3. Incorporating spatial data for geographically weighted predictions.

---

## 🔗 References

Key references include:
1. [Scientific Reports: Predicting Rainfall Using ML/DL](https://www.nature.com/articles/s41598-024-77687-x)
2. [Hybrid CNN-LSTM Models for Weather Prediction](https://iwaponline.com/ws/article/22/5/4902/88212)

