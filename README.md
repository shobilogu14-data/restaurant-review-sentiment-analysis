# Restaurant Review Analysis Using Machine Learning

### Aspect-Based Sentiment Analysis | Natural Language Processing | Deep Learning

**MSc Data Science Dissertation — University of East London**

## Project Overview

Customer reviews provide valuable information about restaurant performance, but manually analysing large volumes of feedback can be time-consuming.

This project explores how Natural Language Processing (NLP) and Machine Learning can automatically identify the aspects customers discuss and classify the sentiment expressed towards those aspects.

The objective is to transform unstructured customer feedback into meaningful insights that could support restaurant service improvements and business decision-making.

## Business Problem

A restaurant may receive hundreds or thousands of reviews discussing food quality, service, pricing and ambience.

A traditional overall sentiment score may not reveal which specific aspects are responsible for positive or negative customer experiences.

This project investigates whether aspect-based sentiment analysis can provide more detailed insights.

## Project Objectives

- Identify the restaurant-related aspects mentioned in customer reviews.
- Classify sentiment associated with individual aspects.
- Compare traditional machine-learning approaches with transformer-based models.
- Evaluate predictive performance using appropriate classification metrics.
- Explore model interpretability using explainable AI techniques.
- Demonstrate how the findings could support customer experience analysis.

## Technologies and Tools

**Programming:** Python

**Data Processing:** Pandas, NumPy

**Machine Learning:** Scikit-learn, Logistic Regression, Naive Bayes, Support Vector Machines, Random Forest and XGBoost

**Natural Language Processing:** TF-IDF, RoBERTa, Hugging Face Transformers

**Model Explainability:** SHAP, LIME

**Visualisation:** Matplotlib, Seaborn

## Methodology

The project follows a two-stage analytical approach.

**Stage 1 — Aspect Identification**

Customer reviews are preprocessed and transformed into numerical representations using TF-IDF. A multi-label classification approach is used to identify relevant restaurant aspects.

**Stage 2 — Sentiment Classification**

Sentiment classification models are evaluated to determine whether customer feedback expresses positive, negative or neutral sentiment towards the identified aspects.

Traditional machine-learning models are compared with a transformer-based RoBERTa approach.

## Model Evaluation

The analysis considers classification metrics including:

- Precision, recall and F1-score
- Confusion matrices
- ROC-AUC and PR-AUC
- Hamming loss and Jaccard score for multi-label classification
- Cohen's Kappa

Detailed model comparisons and verified experimental results will be added alongside the project code and visualisations.

## Business Applications

The proposed approach could help restaurant businesses:

- Identify recurring customer complaints.
- Understand satisfaction across different service aspects.
- Prioritise operational improvements.
- Monitor customer feedback trends.
- Support evidence-based customer experience decisions.

## Project Status

Academic research completed.

GitHub documentation, implementation files and reproducibility instructions are being organised for publication.

## Author

**Shobana Loganathan**

MSc Data Science (Distinction), University of East London

[LinkedIn](https://www.linkedin.com/in/shobana-l-a0032a19a) | [Portfolio](https://shobilogu14-data.github.io/)
