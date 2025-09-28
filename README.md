# Task 4: Classification with Logistic Regression

##  Objective
Build a binary classifier using **Logistic Regression** to understand the basics of classification, evaluation metrics, and decision thresholds.

---

##  Tools & Libraries
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  

---

## Dataset
For this task, we use the **Breast Cancer Wisconsin Dataset**, which is available directly from `scikit-learn`.

- Features: Medical measurements of tumors  
- Target: `0 = malignant`, `1 = benign`

---

## Steps
1. **Load dataset** using scikit-learn.  
2. **Train/test split** (e.g., 80/20).  
3. **Standardize features** with `StandardScaler`.  
4. **Fit Logistic Regression model** from scikit-learn.  
5. **Evaluate model** using:
   - Confusion Matrix
   - Precision
   - Recall
   - F1-score
   - ROC-AUC Curve  
6. **Tune decision threshold** to see effect on metrics.  
7. **Explain Sigmoid Function** (used to map outputs into probabilities).  

---

##  Results
- Logistic Regression achieved good accuracy on the dataset.  
- Metrics like **precision, recall, and ROC-AUC** were used to evaluate the model.  
- Adjusting the decision threshold showed how to balance between **false positives and false negatives**.  
