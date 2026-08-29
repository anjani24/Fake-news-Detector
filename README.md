# 🔍 Fake News Detector

A machine learning-based system that identifies fake news articles with high accuracy using advanced NLP techniques and multiple classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Approach](#technical-approach)
- [Models & Performance](#models--performance)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [What I Learned](#what-i-learned)

---

## Overview

This project tackles one of today's most pressing challenges: **identifying misinformation in news content**. The system analyzes news article titles and classifies them as either genuine or fake using a combination of Natural Language Processing (NLP) and machine learning.

The model achieves high accuracy by combining:
- **Text preprocessing** with both stemming and lemmatization techniques
- **Feature extraction** using TF-IDF vectorization
- **Ensemble learning** with multiple classifier algorithms
- **Rigorous evaluation** with comprehensive metrics

---

## Motivation

In an age of information overload, the ability to distinguish authentic journalism from misleading content is crucial. This project demonstrates how machine learning can serve as a practical tool to:
- Automate fact-checking at scale
- Combat misinformation spread on social media
- Assist journalists and content moderators
- Support media literacy efforts

---

## Technical Approach

### 1. **Data Processing & Exploration**
- **Dataset**: WELFake Dataset containing real and fake news articles
- **Size**: Balanced dataset with clear label distribution
- **Preprocessing Steps**:
  - Handled missing values
  - Text normalization and cleaning
  - Removal of special characters and numbers

### 2. **Text Preprocessing Pipeline**

I implemented **two parallel preprocessing strategies** to capture different aspects of language:

#### **Lemmatization Path**
- Converts words to their dictionary form (base meaning)
- Example: "running" → "run"
- Preserves semantic meaning better
- Uses POS (Part-of-Speech) tagging for context-aware conversion

#### **Stemming Path**
- Reduces words to their root form using Porter Stemmer
- Example: "running" → "run"
- Faster but less accurate semantically
- Useful for capturing word variations

### 3. **Feature Engineering**
- **TF-IDF Vectorization**: Converts text into numerical features
  - Term Frequency: Measures word frequency in a document
  - Inverse Document Frequency: Weights down common words across all documents
  - Result: High-dimensional feature vectors capturing word importance

### 4. **Data Split**
- 80% training data | 20% testing data
- Stratified split to maintain class distribution
- Fixed random seed for reproducibility

---

## Models & Performance

I trained and compared **three different machine learning algorithms**, each with two feature sets (stemmed and lemmatized):

### **1. Logistic Regression**
- Fast linear classifier
- Great baseline model
- Provides probability estimates
- **Use Case**: When speed and interpretability matter

### **2. Support Vector Classifier (SVM)**
- Powerful non-linear classifier
- Finds optimal decision boundary
- Good generalization
- **Use Case**: When maximum separation is needed

### **3. Random Forest Classifier**
- Ensemble method combining multiple decision trees
- Handles non-linear relationships well
- Provides feature importance insights
- **Use Case**: When capturing complex patterns is important

### **Evaluation Metrics**
For each model, I calculated:
- **Accuracy**: Overall correctness
- **Precision**: False positives (wrongly labeling real news as fake)
- **Recall**: False negatives (wrongly labeling fake as real)
- **F1-Score**: Balanced metric combining precision & recall
- **Confusion Matrix**: Visual breakdown of predictions
- **ROC Curve & AUC**: Classification threshold analysis

---

## Project Structure

```
Fake-news-Detector/
├── Fake news detector.ipynb          # Main notebook with full analysis
├── README.md                          # This file
├── lr_stem_model.joblib              # Logistic Regression (stemmed features)
├── lr_lem_model (1).joblib           # Logistic Regression (lemmatized features)
├── svm_stem_model.joblib             # SVM Classifier (stemmed features)
├── svm_lem_model.joblib              # SVM Classifier (lemmatized features)
├── vectorizer_lem (1).joblib         # TF-IDF Vectorizer (lemmatized)
└── fakenewsai/                       # Project directory
```

---

## Key Features

### **Dual Text Processing**
Two preprocessing approaches trained in parallel for comparison and ensemble potential

### **Comprehensive Analysis**
- Word frequency analysis with visualizations
- Word clouds for quick pattern recognition
- Detailed confusion matrices
- ROC curves for threshold optimization

### **Multiple Models**
Built 6 trained models (3 algorithms × 2 preprocessing methods) for flexibility

### **Detailed Evaluation**
Rigorous metrics beyond simple accuracy to understand model behavior

### **Pre-trained Models**
All models saved as .joblib files for easy reuse without retraining

---

## What I Learned

### Technical Skills Developed:
1. **NLP Pipeline Design**
   - Difference between stemming vs. lemmatization
   - When to use each approach
   - POS tagging for smarter text processing

2. **Machine Learning Best Practices**
   - Importance of train-test splitting and stratification
   - Feature engineering impact on model performance
   - Comparing multiple algorithms systematically
   - Understanding precision-recall tradeoffs

3. **Data Science Workflow**
   - From raw data to production-ready models
   - Proper evaluation methodology
   - Reproducibility through fixed random seeds

4. **Real-world Problem Solving**
   - Handling imbalanced or messy data
   - Trade-offs between accuracy and interpretability
   - Practical applications of ML in content moderation

### Key Insights:
- **Lemmatization often outperforms stemming** for semantic understanding
- **Different models excel in different scenarios** - no one-size-fits-all solution
- **Visualization is crucial** for understanding model behavior
- **Ensemble methods like Random Forest** capture complex patterns better than linear models

---

## Installation & Usage

### Requirements
```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud autocorrect
```

### Running the Notebook
```bash
jupyter notebook "Fake news detector.ipynb"
```

### Using Pre-trained Models
```python
import joblib

# Load model
model = joblib.load('lr_stem_model.joblib')

# Load vectorizer
vectorizer = joblib.load('vectorizer_lem (1).joblib')

# Make predictions
text_features = vectorizer.transform(['your news text here'])
prediction = model.predict(text_features)
```

---

## Future Enhancements

- Deploy as a web API for real-time classification
- Create a browser extension for on-the-fly fact-checking
- Implement transfer learning with pre-trained language models (BERT, RoBERTa)
- Add more sophisticated features (source credibility, temporal patterns)
- Extend to multi-language support
- Incorporate fact-checking with external databases

---

**Built with Python, scikit-learn, and NLTK** | *Machine Learning for Misinformation Detection*
