# Model performance review

## Source and scope

This report transcribes model evaluation outputs from the uploaded `Projects_Risk_Prediction.ipynb`. The source notebook uses 4,000 records, an 80/20 stratified train-test split and `random_state=42`.

## Held-out test set

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7325 | 0.75 | 0.73 | 0.74 | 0.73 |
| Random Forest | 0.5025 | 0.56 | 0.47 | 0.48 | 0.48 |
| XGBoost | 0.5725 | 0.60 | 0.56 | 0.57 | 0.57 |

## XGBoost diagnostics

- Training accuracy: 1.0000
- Test accuracy: 0.5725
- Reported five-fold cross-validation mean: 0.57125

## Interpretation

Logistic Regression has the highest recorded test accuracy and macro F1 among the three models. XGBoost reaches perfect training accuracy but substantially lower test accuracy, indicating overfitting.

## Validation caveat

The notebook imputes missing values and one-hot encodes the full dataset before splitting into training and test subsets. The XGBoost cross-validation score is then computed on this already-preprocessed dataset. Consequently, the test and cross-validation estimates may be optimistic or otherwise affected by information leakage.

The next evaluation should use a scikit-learn `Pipeline` and `ColumnTransformer`, with imputation and encoding fitted independently inside each training split / cross-validation fold. Use stratified cross-validation and retain a final untouched test set. Recalculate all model metrics after this change.

## Reporting restrictions

The notebook does not provide a verified ROC-AUC result. The raw dataset and its original target-class mapping were not supplied with the notebook, so this report does not assign names to encoded classes or verify class-level business interpretations.
