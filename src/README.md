# Source code

The supplied analysis currently lives in a single Jupyter notebook. As the project is refactored, move reusable logic into Python modules, for example:

- `data_preprocessing.py`: train-fitted imputation and encoding.
- `train_model.py`: pipeline creation, model training and cross-validation.
- `evaluate_model.py`: held-out metrics, confusion matrices and model comparison.
- `predict.py`: load the complete fitted pipeline and produce predictions using the original input schema.

Do not duplicate notebook code mechanically. First define a single preprocessing-and-model pipeline and test it. Persist the fitted pipeline and any target-label mapping together so inference uses exactly the same transformations as training.
