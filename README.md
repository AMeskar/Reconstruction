# **Neutrino Energy Reconstruction Using Machine Learning**

This repository contains all code and documentation related to reconstructing **primary neutrino energy** from Monte Carlo (MC) simulated data using various **machine learning (ML)** models and **deep learning architectures**. The goal is to predict neutrino energy from truth-level physical features and compare different ML approaches.  

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Dataset and Data Processing](#dataset-and-data-processing)  
   1. [Data Sources](#data-sources)  
   2. [Preprocessing Steps](#preprocessing-steps)  
3. [Machine Learning Models Implemented](#machine-learning-models-implemented)  
   1. [Tree-Based Models (Ensemble Learning)](#tree-based-models-ensemble-learning)  
   2. [Distance-Based Model](#distance-based-model)  
   3. [Deep Learning Models](#deep-learning-models)  
4. [Deep Dive into the Mathematical Foundations](#deep-dive-into-the-mathematical-foundations)  
   1. [Universal Foundation of ML Algorithms](#universal-foundation-of-ml-algorithms)  
   2. [Key Equations in Regression Modeling](#key-equations-in-regression-modeling)  
   3. [Training Procedures](#training-procedures)  
5. [Model Evaluation Metrics](#model-evaluation-metrics)  
6. [Key Scripts and Files](#key-scripts-and-files)  
7. [Methodology Summary](#methodology-summary)  
8. [Results & Insights](#results--insights)  
9. [Learning Curve Analysis](#learning-curve-analysis)  
   1. [Code Explanation](#code-explanation)  
   2. [Mathematical Details](#mathematical-details)  
   3. [Interpreting the Learning Curves](#interpreting-the-learning-curves)  
10. [Deep Learning Section](#deep-learning-section)  
11. [Future Improvements](#future-improvements)   
12. [Contact Information](#Contact-Information)

---

## **1. Project Overview**

This project aims to **reconstruct the primary neutrino energy** from Monte Carlo (MC) simulations. We utilize multiple **machine learning** methods—such as **Random Forest**, **Gradient Boosting**, and **k-Nearest Neighbors**—as well as basic **deep learning** architectures like **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** to explore which approach provides the best energy prediction accuracy.

---

## **2. Dataset and Data Processing**

### **2.1 Data Sources**

- **Truth Data (`All_fields_in_one_file.csv`)**  
  Contains Monte Carlo **simulation truth values**, including spatial coordinates, directional components, and various event-level parameters.

- **Reconstructed Data (`Primary_Neutrino_Energy.csv`)**  
  Contains the reconstructed (simulated) **primary neutrino energy** values used as the regression target.

### **2.2 Preprocessing Steps**

1. **Loading**: Use `pandas` to load both truth data and reconstructed data.  
2. **Filtering**: Select only **numerical features** relevant to neutrino energy reconstruction.  
3. **Normalization**: Apply `StandardScaler()` to standardize features, ensuring no single feature dominates.  
4. **Train-Test Split**: Allocate **80%** of the data for training and **20%** for testing using `train_test_split()`.

---

## **3. Machine Learning Models Implemented**

### **3.1 Tree-Based Models (Ensemble Learning)**

| Model                                                          | Description                                                                                          |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Random Forest** (`RandomForestRegressor`)                   | An ensemble of decision trees where each tree votes, reducing variance and improving accuracy.       |
| **Gradient Boosting** (`GradientBoostingRegressor`)           | Iteratively boosts weak learners (decision trees) to minimize prediction error.                      |
| **Histogram-Based Gradient Boosting** (`HistGradientBoostingRegressor`) | A faster variant of gradient boosting using optimized histogram binning for continuous features.     |

### **3.2 Distance-Based Model**

| Model                                          | Description                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| **k-Nearest Neighbors** (`KNeighborsRegressor`) | Predicts based on the average of the k-nearest samples in the feature space. |

### **3.3 Deep Learning Models**

| Model                                   | Description                                                                                                            |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Artificial Neural Network (ANN)**     | A feedforward neural network with multiple dense (fully connected) layers.                                            |
| **Convolutional Neural Network (CNN)**  | Uses 1D convolutions to extract features/patterns from the input variables (treated like sequences or signals).       |

---

## **4. Deep Dive into the Mathematical Foundations**

### **4.1 Universal Foundation of ML Algorithms**

Despite their differences in structure and application, nearly all supervised machine learning models share the same high-level goal:

$$\[
\min_\theta \; \mathcal{L}(y, \hat{y})
\]$$

where:  
- $$\(\theta\)$$ represents the parameters of the model (e.g., weights in a neural network, split points in a decision tree).  
- $$\(y\)$$ is the true label (in our case, the true neutrino energy).  
- $$\(\hat{y}\)$$ is the predicted label (the model’s estimate of the neutrino energy).  
- $$\(\mathcal{L}\)$$ is a **loss function** that measures the discrepancy between the true label $$\(y\)$$ and the prediction $$\(\hat{y}\)$$.

### **4.2 Key Equations in Regression Modeling**

1. **Linear Regression** (Fundamental Example)

   $$\[
   \hat{y} = \beta_0 + \sum_{i=1}^n \beta_i x_i
   \]$$
   - $$\(\beta_i\)$$ are coefficients learned during training.  
   - $$\(x_i\)$$ are input features.

   **Cost Function (MSE)**:
   $$\[
   \text{MSE} = \frac{1}{N} \sum_{j=1}^N (y_j - \hat{y}_j)^2
   \]$$

2. **Decision Trees**  
   Splits features recursively to reduce a **criterion**, typically MSE (for regression):
   
<div style="background-color:black; padding:10px;">
<p style="color:white;">
$$
\text{Split Criterion} = \sum_{j \in \text{Left}} (y_j - \bar{y}_\text{Left})^2 + \sum_{j \in \text{Right}} (y_j - \bar{y}_\text{Right})^2
$$
</p>
</div>

 <div style="background-color:black; padding:10px;">
<p style="color:white;">
where $\bar{y}_\text{Left}$ and $\bar{y}_\text{right}$ are the mean values of the left and right child nodes, respectively.
</p>
</div> 

4. **Ensemble Methods** (e.g., Gradient Boosting)  
   **Gradient Boosting** fits new weak learners to the **residual errors**:
   $$\[
   \hat{y}^{(m)} = \hat{y}^{(m-1)} + \nu \cdot h_m(x),
   \]$$
   where $$\(h_m(x)\)$$ is the new weak learner fitted at iteration $$\(m\)$$, and $$\(\nu\)$$ is the learning rate.

5. **k-Nearest Neighbors** (KNN)  
   $$\[
   \hat{y} = \frac{1}{k} \sum_{i \in \text{kNN}} y_i
   \]$$
   where $$\(\text{kNN}\)$$ is the set of $$\(k\)$$ nearest neighbors.

### **4.3 Training Procedures**

- **Gradient Descent**:
  $$\[
  \theta \leftarrow \theta - \eta \, \frac{\partial \mathcal{L}}{\partial \theta}
  \]$$
  where $$\(\eta\)$$ is the learning rate.

- **Backpropagation** (for Neural Networks):  
  It involves computing partial derivatives of the network’s weights to the loss $$\(\mathcal{L}\)$$ and updating them iteratively via gradient descent.

---

## **5. Model Evaluation Metrics**

The regression performance of each model is evaluated using:

<table>
  <tr>
    <th>Metric</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><strong>Mean Absolute Error (MAE)</strong></td>
    <td><div style="background-color:black; padding:10px; color:white;">$\text{MAE} = \frac{1}{N}\sum_{j=1}^N \bigl|y_j - \hat{y}_j\bigr|$</div></td>
  </tr>
  <tr>
    <td><strong>Mean Squared Error (MSE)</strong></td>
    <td><div style="background-color:black; padding:10px; color:white;">$\text{MSE} = \frac{1}{N}\sum_{j=1}^N (y_j - \hat{y}_j)^2$</div></td>
  </tr>
  <tr>
    <td><strong>R² Score</strong> (Coefficient of Determination)</td>
    <td>Measures how well future samples are likely to be predicted by the model; the ideal score is 1.0.</td>
  </tr>
  <tr>
    <td><strong>Pearson Correlation Coefficient</strong></td>
    <td>Evaluates the linear relationship between truth and predicted energy values.</td>
  </tr>
</table>
---

## **6. Key Scripts and Files**

| File Name                  | Description                                                                    |
|----------------------------|--------------------------------------------------------------------------------|
| `data_processing.py`       | Loads, filters, and processes Monte Carlo simulation data.                     |
| `feature_selection.py`     | Extracts relevant physical parameters for model training.                      |
| `ml_models.py`             | Trains tree-based and distance-based regression models (RF, GB, kNN).          |
| `deep_learning.py`         | Defines and trains ANN and CNN architectures.                                  |
| `evaluation.py`            | Computes error metrics (MAE, MSE, R², Pearson) and generates visualizations.   |
| `correlation_analysis.py`  | Computes feature correlation with reconstructed energy.                        |
| `visualization.py`         | Generates histograms, scatter plots, and bar charts for model comparison.      |
| **`learning_curve.py`**    | Demonstrates how to compute and plot learning curves using multiple regressors. |

---

## **7. Methodology Summary**

1. **Data Preprocessing**  
   - Load `df_truth` and `df_reco`, remove non-numeric columns, and standardize the features.

2. **Model Training**  
   - Using selected features, train multiple models (`RandomForestRegressor`, `GradientBoostingRegressor`, `kNeighborsRegressor`, `ANN`, `CNN`).  
   - Perform hyperparameter tuning where possible to improve performance.

3. **Model Evaluation**  
   - Compute **MAE, MSE, R²**, and **Pearson Correlation**.  
   - Analyze **feature importance** and correlation matrices.

4. **Visualization & Interpretation**  
   - **Histograms** comparing predicted vs. actual energy distributions.  
   - **Bar charts** comparing performance metrics across models.  
   - **Scatter plots** illustrating residuals or correlations with truth.

---

## **8. Results & Insights**

1. **Tree-Based Models** (Random Forest, Gradient Boosting):  
   - Provided robust performance with relatively small errors.  
   - Typically less sensitive to outliers than simpler models.

2. **k-Nearest Neighbors**:  
   - Performs well with sufficient training samples, but can suffer from high-dimensional data sparsity.  
   - Computationally expensive during inference for large datasets.

3. **Deep Learning Models** (ANN, CNN):  
   - Showed potential if expanded to more sophisticated architectures (e.g., deeper networks).  
   - Requires careful hyperparameter tuning (learning rate, batch size, etc.).

4. **Feature Engineering**:  
   - Standardization and physics-informed features substantially improved performance.

Overall, **ensemble-based methods** (Random Forest, Gradient Boosting) emerged as strong contenders, but **deep learning approaches** could outperform them with more extensive tuning and data.

---

## **Learning Curve Analysis**

Learning curves illustrate how a model’s performance evolves as the training set size increases. They help diagnose whether a model is **overfitting** or **underfitting**, and whether **collecting more data** or **tuning model complexity** is likely to yield improvement. Below is a detailed explanation of how we compute learning curves in this project, along with the mathematical foundations and an emphasis on the **fully automated** nature of our implementation.

---

### **1. Automated Workflow**

One of the key strengths of our learning curve implementation is that it is **fully automated**. Once you specify which model you want to train (e.g., by passing a command-line argument for the model’s ID), the script:

- **Loads** the dataset from an HDF5 file.  
- **Selects** a range of training subset sizes (e.g., 10%, 20%, …, 100%).  
- **Randomly samples** each training subset multiple times (with different random seeds).  
- **Trains** the chosen model on each subset.  
- **Evaluates** the performance on both the training subset and a fixed validation set.  
- **Computes** multiple metrics (R², MSE, MAE, etc.).  
- **Generates** plots showing the mean and standard deviation of these metrics vs. the training set size.  

This end-to-end procedure requires **zero manual intervention** once launched. It ensures consistency, reproducibility, and the ability to **easily switch** between different models or training configurations simply by changing a parameter.

---

### **2. Mathematical Foundations**

## 2.1 General Learning Curve Concept

A learning curve tracks a performance metric $$\( M \)$$ (e.g., MSE, MAE, R²) as a function of the training set size $$\( m \)$$. Formally, we denote:

<p align="center">
  $$ \mathcal{M}_{\text{train}}(m), \quad \mathcal{M}_{\text{valid}}(m) $$
</p>

where:

$$ \mathcal{M}_{\text{train}}(m) $$ is the metric computed on a training subset of size$$ \( m \)$$.  
$$ \mathcal{M}_{\text{valid}}(m) $$ is the metric computed on the validation set after training on the same $$\( m \)$$ samples.  
Because we randomly sample subsets multiple times (say $$\( S \)$$ times) to reduce statistical fluctuations, we get repeated measurements:

<p align="center">
  $$ \mathcal{M}_{\text{train}, s}(m), \quad \mathcal{M}_{\text{valid}, s}(m) \quad \text{for} \; s = 1, 2, \ldots, S $$
</p>

We then compute the average metric:

<p align="center">
  $$ \overline{\mathcal{M}}_{\text{train}}(m) = \frac{1}{S} \sum_{s=1}^S \mathcal{M}_{\text{train}, s}(m), \quad \overline{\mathcal{M}}_{\text{valid}}(m) = \frac{1}{S} \sum_{s=1}^S \mathcal{M}_{\text{valid}, s}(m) $$
</p>

And the standard deviation:

<p align="center">
  $$ \sigma_{\text{train}}(m) = \sqrt{\frac{1}{S} \sum_{s=1}^S \Bigl(\mathcal{M}_{\text{train}, s}(m) - \overline{\mathcal{M}}_{\text{train}}(m)\Bigr)^2} $$
</p>

<p align="center">
  $$ \sigma_{\text{valid}}(m) = \sqrt{\frac{1}{S} \sum_{s=1}^S \Bigl(\mathcal{M}_{\text{valid}, s}(m) - \overline{\mathcal{M}}_{\text{valid}}(m)\Bigr)^2} $$
</p>

We plot $$ \overline{\mathcal{M}}_{\text{train}}(m) $$ and $$ \overline{\mathcal{M}}_{\text{valid}}(m) $$ against $$\( m \)$$, often along with the $$\( \pm \sigma \)$$ “shaded region” to show variability.

---

### **3. Metrics Used**

Within our implementation, we compute several **regression metrics**:

1. **R² Score** $$(\(r^2\))$$:  
   $$\[
   r^2 = 1 - \frac{\sum (y_{\text{true}} - y_{\text{pred}})^2}{\sum (y_{\text{true}} - \bar{y}_{\text{true}})^2}.
   \]$$
   Measures the proportion of variance explained by the model.

2. **Mean Squared Error (MSE)**:  
   $$\[
   \text{MSE} = \frac{1}{N} \sum_{i=1}^N \bigl(y_{\text{true},i} - y_{\text{pred},i}\bigr)^2.
   \]$$
   Penalizes large errors more heavily.

3. **Mean Absolute Error (MAE)**:  
   $$\[
   \text{MAE} = \frac{1}{N} \sum_{i=1}^N \bigl|y_{\text{true},i} - y_{\text{pred},i}\bigr|.
   \]$$
   A robust measure of how far off predictions are on average.

4. **Explained Variance**, **Max Error**, **Median Absolute Error**, etc. are also computed for completeness.

---

### **4. Implementation Steps (High-Level)**

1. **Data Loading & Preprocessing**  
   - Open the HDF5 file and load the chosen **features** along with the **target** $$(\(E\))$$.  
   - Apply a **log-transform** $$(\(\log_{10}(E)\))$$ to stabilize training if desired.  
   - Use a **StandardScaler** so that all features have zero mean and unit variance.  
   - Split the data into **training** and **validation** sets (e.g., 80%/20%).

2. **Training Subset Definition**  
   - Define an array of training set sizes (e.g., from 10% to 100% in equal increments).

3. **Model Training & Evaluation**  
   - For each size $$\(m\)$$:
     - **Randomly sample** $$\(m\)$$ data points from the training set multiple times (different seeds).  
     - **Fit** the model on each sample.  
     - **Predict** on the sampled training subset and on the **validation** set.  
     - **Compute** metrics for both training and validation predictions.

4. **Results Aggregation**  
   - Take the **mean and standard deviation** of each metric across repeated samples.  
   - Store results in dictionaries or data structures that facilitate **plotting**.

5. **Plotting**  
   - Plot **training** curves and **validation** curves against the training subset size.  
   - Use a **shaded region** to represent $$\(\pm1\)$$ standard deviation around the mean metric.

---

### **5. Interpreting the Learning Curves**

1. **Underfitting (High Bias)**  
   - Both training and validation metrics converge to a suboptimal level.  
   - Increasing training data does not significantly improve results.

2. **Overfitting (High Variance)**  
   - Training metric is much better than validation metric.  
   - Validation metric gradually improves as $$\(m\)$$ increases, indicating more data helps.

3. **Good Fit**  
   - Training and validation curves are close to each other, indicating generalization is good.

---

### **6. Why This Automation Matters**

- **Reproducibility**: By automating every step (subsampling, training, evaluating, plotting), we ensure consistent, repeatable experiments.  
- **Convenience**: Switching between models (e.g., Random Forest vs. XGBoost) is as simple as changing a single parameter.  
- **Efficiency**: Batch-running experiments over several models or hyperparameter settings are streamlined.  
- **Scalability**: The workflow can easily be extended to larger datasets or additional metrics without manual intervention.

By fully automating the learning curve generation process, we provide a robust and flexible platform for diagnosing our neutrino energy reconstruction models, revealing how they respond to various training set sizes and where improvements might be most beneficial.

---

## **10. Deep Learning Section**

For a **comprehensive deep dive** into our deep learning architectures, including **layer-by-layer explanations** and **additional experiments**, please refer to [https://github.com/AMeskar/DeepLearning](https://github.com/AMeskar/DeepLearning).  

In that repository, you will find:
- Detailed explanations of **ANN** and **CNN** model architectures.  
- Advanced topics like **regularization**, **dropout**, and **batch normalization**.  
- Examples of **hyperparameter tuning** and **best practices** for training deep neural networks.

---

## **11. Future Improvements**

1. **Weighted Training**  
   - High-energy events could be assigned larger weights to emphasize their importance during model training.

2. **Hyperparameter Optimization**  
   - Systematic grid search, random search, or Bayesian optimization for all model types.

3. **Feature Engineering**  
   - Incorporate domain knowledge from neutrino physics to craft additional input variables or transformations.

4. **Advanced Deep Learning Architectures**  
   - Explore **Transformers**, **LSTM** (if sequential data is relevant), or other specialized architectures.

5. **Uncertainty Quantification**  
   - Investigate methods like **Monte Carlo Dropout**, **Bayesian Neural Networks**, or **quantile regression** to estimate the uncertainty of energy predictions.

---

## **12. Contact Information**

- Email: Meskaramine2@gmail.com  
- LinkedIn: [linkedin.com/in/amine-meskar](https://www.linkedin.com/in/amine-meskar)  
