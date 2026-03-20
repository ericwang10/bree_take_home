# Bree: AI-Powered Loan Application Processor

This repository contains the solution for the Bree Machine Learning Intern take-home assignment. It includes a Jupyter Notebook with EDA, model training, and evaluation, the generated dataset, and this README file.

## How to Run and Reproduce Results

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Mac/Linux
    # venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter and run the notebook:**
    ```bash
    jupyter notebook "Bree_ML_Take_Home.ipynb" 
    ```
    Running all cells will reproduce all analyses, plots, and metric calculations. The `loan_applications.csv` file is included in the repository.

## My Approach and Key Decisions

My approach was to build a simple, highly interpretable model that outperforms the baseline not just on accuracy, but on real-world business value, factoring in Bree's specific cost structure.

**1. EDA & Feature Engineering:**
*   Filtered out 164 `ongoing` applications to train on a dataset of 1,836 resolved loans.
*   Identified that `missing_docs` and `overstated_income` were strong predictors of default.
*   Engineered these two signals into binary features to allow a linear model (Logistic Regression) to capture their non-linear impact.

**2. Model Selection: Logistic Regression**
*   **Why Logistic Regression?** I chose LR because this problem isn't overly complex and relies more on clarity and explainability to the business. In lending, trust and transparency are key. LR provides clear, auditable coefficients that are intuitive to understand.
*   **Key Steps:**
    *   Applied `class_weight='balanced'` to handle the ~30% class imbalance.
    *   Used `StandardScaler` on all features. This is crucial for LR, as it ensures the model's coefficients accurately reflect feature importance, rather than being skewed by the magnitude of features (e.g., income in thousands vs. a binary flag).

**3. Evaluation Strategy: Optimizing for Business Value**
*   **Asymmetric Costs:** I recognized that for Bree, a False Negative (missing a default) is far more expensive (loss of principal) than a False Positive (denying a good applicant, loss of a ~$5 express fee). This means **Recall** is our most important metric.
*   **Systematic Thresholding:** Instead of using the default 50% probability cutoff, I used an ROC curve to find the `optimal_threshold (0.41)`. This threshold was chosen to match the baseline's False Positive Rate (~47%), allowing for a direct "apples-to-apples" comparison of how many more defaults we could catch at the same risk tolerance.
*   **Outcome:** The model is a strict upgrade, catching 6 more defaults and denying 1 fewer good applicant on the 368-application test set. This translates to a net value of **+$8,005** for the business.

**4. Fairness Analysis**
*   **Finding:** The baseline system was biased, unfairly penalizing `self_employed` applicants despite them having the same ground-truth default rate as `employed` applicants.
*   **Correction:** The ML model organically corrected this bias by learning from the actual data, closing the approval rate gap.
*   **Recommendation:** To mitigate future regulatory risk while preserving predictive power, I recommend engineering a new binary feature: `has_income_source`, which groups `employed` and `self_employed` together.

## What I'd Do With More Time

*   **Benchmark Against Other Interpretable Models:** I would benchmark the Logistic Regression model against other classical ML models such as **Random Forest**, **XGBoost**, and potentially even **K-Nearest Neighbors**. For the tree-based models, I would use SHAP values to maintain the high level of interpretability required for this business problem.
*   **Systematic Hyperparameter Tuning:** I would use `GridSearchCV` or `RandomizedSearchCV` to systematically tune the hyperparameters of the Logistic Regression model (e.g., the regularization strength `C`) and any other benchmarked models to ensure we are extracting maximum performance.
*   **Deeper Feature Engineering:** I would explore creating more advanced features, such as interaction terms (e.g., does the ratio of `loan_amount` to `monthly_deposits` create a stronger signal?) to see if we can provide the model with even more predictive power.

## A Note on AI-Assisted Development

The take-home assignment encourages the use of AI tools. In that spirit, I used AI assistance for tasks such as generating boilerplate code, refining plots, adding comments, and drafting documentation like this README. However, the core ML strategy—the direction of the EDA, feature engineering, model selection, and final analysis—was driven entirely by my own judgment. All code has been personally reviewed, tested, and verified for correctness.
