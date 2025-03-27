#  Bug Report Classification Tool

> Coursework submission for Intelligent Software Engineering (ISE)  
> Project Type: Tool Development (Bug Report Classification)

---

# Project Description

This project builds an automatic bug report classifier for open-source projects (e.g., PyTorch, TensorFlow, Keras). It uses various machine learning and deep learning models to classify bug reports into relevant categories (e.g., bug vs non-bug).

---



# How to Run

1. Install dependencies
Make sure you have Python 3.8+ and the following packages:
pip install -r requirements.txt
You can also refer to the provided System Requirements & Dependencies.pdf.

2. Run the tool
python bugreport.py
This will process all datasets and generate results (metrics, plots, saved models) in the model_results/ folder.

---

# Output Summary
For each project (e.g. pytorch), the script generates:

*_final_comparison.csv: Model performance summary (accuracy, precision, recall, F1, AUC)

*_confusion_matrix.png: Confusion matrix for each model

*_history.png: Training curves (for deep learning models)

.h5 files: Saved deep learning models

---

# Replication
Steps for replication are detailed in replication.pdf.
The project supports full re-execution using bugreport.py, requiring only the datasets/ folder.




