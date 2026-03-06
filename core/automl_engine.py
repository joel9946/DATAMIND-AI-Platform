"""
╔══════════════════════════════════════════════════════════════════╗
║  automl_engine.py  —  DataMind Platform                         ║
║  THE ROBOT TRAINER                                               ║
║                                                                  ║
║  AutoML = Automated Machine Learning.                           ║
║  This file trains NINE different AI models on your data,        ║
║  races them against each other, and tells you which one wins.   ║
║                                                                  ║
║  Think of it like a bake-off: we give each contestant the same  ║
║  ingredients (your data) and the same amount of time, then      ║
║  taste-test all the cakes (check accuracy) and crown a winner.  ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import pandas as pd     # DataFrames — the spreadsheet-like data structure
import numpy as np      # Fast maths on arrays
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")   # Silence annoying deprecation warnings

# ── sklearn: The Machine Learning Toolkit ────────────────────────────────────
# sklearn (scikit-learn) is the most popular ML library in Python.
# Think of it as a giant toolbox where every tool is an AI algorithm.

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
# cross_val_score  : run k-fold cross-validation (train/test multiple times for honest estimate)
# train_test_split : split data into training portion and testing portion
# StratifiedKFold  : k-fold that preserves class balance in each fold (for classification)
# KFold            : k-fold without stratification (for regression)

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# StandardScaler  : rescale numbers so mean=0, std=1 (prevents large numbers dominating)
# LabelEncoder    : convert text labels to numbers ("cat"→0, "dog"→1)
# OneHotEncoder   : convert categories to binary columns ("red"→[1,0,0], "blue"→[0,1,0])

from sklearn.impute import SimpleImputer
# SimpleImputer : fill in missing (NaN) values
# strategy="median" : replace missing with the median value of that column

from sklearn.pipeline import Pipeline
# Pipeline : chain preprocessing steps + model into ONE object.
# Key benefit: prevents "data leakage" — the test data is NEVER used in preprocessing.

from sklearn.compose import ColumnTransformer
# ColumnTransformer : apply different processing to different column types
# e.g. scale numeric columns AND encode text columns in one step

# ── Classifier Models (for predicting categories) ────────────────────────────
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
# LogisticRegression : despite the name, this is a CLASSIFIER
#                      Uses a sigmoid curve to output a probability (0 to 1)
# Ridge, Lasso, ElasticNet : linear regression variants with different regularisation

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
# RandomForest : builds MANY decision trees and averages their votes (ensemble)
#               Like asking 100 people and taking the majority vote
# GradientBoosting : builds trees SEQUENTIALLY, each fixing the previous one's errors
#                    Like a student correcting mistakes after each test
# ExtraTrees : like RandomForest but uses random split thresholds (faster, sometimes better)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# DecisionTree : a flowchart of yes/no questions leading to a prediction
# "Is Age > 18? Yes → look at income. No → look at parent data..."

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# KNN (K-Nearest Neighbors) : to classify a new point, look at its K nearest neighbours
# and take a majority vote. "You are like your neighbours."

from sklearn.svm import SVC, SVR
# SVM (Support Vector Machine) : finds the best boundary line between classes
# SVC = SVM Classifier, SVR = SVM Regressor
# kernel="rbf" = uses a radial basis function to handle non-linear boundaries

from sklearn.naive_bayes import GaussianNB
# Naive Bayes : uses probability theory (Bayes' theorem) to classify
# Very fast, works well on text data. "Naive" because it assumes all features are independent.

from sklearn.linear_model import LinearRegression
# LinearRegression : fits a straight line through data
# Predicts y = m1*x1 + m2*x2 + ... + b (classic maths equation)

# ── Metrics : How we score each model's performance ──────────────────────────
from sklearn.metrics import (
    accuracy_score,       # % of predictions that were correct
    f1_score,             # Balanced score of precision and recall (better for imbalanced data)
    roc_auc_score,        # Area Under the Curve — how well model separates classes
    precision_score,      # Of all predicted positives, how many were actually positive?
    recall_score,         # Of all actual positives, how many did we catch?
    mean_squared_error,   # Average squared error for regression (sensitive to big errors)
    mean_absolute_error,  # Average absolute error for regression
    r2_score,             # R² score: 1.0 = perfect, 0 = no better than guessing mean
    classification_report # Detailed text report of precision/recall per class
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse the same visual theme as EDA engine (consistency across the platform)
COLORS = {
    "primary":   "#00e5ff",
    "secondary": "#7c4dff",
    "accent":    "#ff6d00",
    "success":   "#00e676",
    "danger":    "#ff1744",
    "palette": ["#00e5ff", "#7c4dff", "#ff6d00", "#00e676", "#ff1744",
                "#40c4ff", "#b388ff", "#ffab40", "#69f0ae", "#ff8a80"],
}
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(2,4,9,0)",
    plot_bgcolor="rgba(8,15,26,0.5)",
    font=dict(family="Space Mono, monospace", color="#7ec8e3", size=11),
    margin=dict(l=40, r=20, t=50, b=40),
    # NOTE: gridcolor must NOT go here (invalid top-level Plotly Layout property).
    # Use update_xaxes(gridcolor=...) on the figure instead.
)
GRID_COLOR = "rgba(26,58,92,0.4)"  # Reusable grid line colour


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS: AutoMLEngine
# ══════════════════════════════════════════════════════════════════════════════

class AutoMLEngine:
    """
    End-to-end AutoML engine. Given a DataFrame and a target column, it:
    1. Detects problem type (classification or regression)
    2. Preprocesses features (fill gaps, encode text, scale numbers)
    3. Trains and cross-validates multiple model families
    4. Ranks models by performance
    5. Extracts feature importance
    6. Generates evaluation plots
    """

    # ── Class-Level Model Dictionaries ───────────────────────────────────────
    # These are defined on the CLASS (not on instances).
    # Every AutoMLEngine object shares these same definitions.
    # Like a menu written on the wall — every customer at every table sees it.

    # ── FAST mode model dictionaries (fewer trees = much faster) ─────────
    # Used when fast_mode=True. Accuracy drops slightly but speed improves 5-10x.
    CLASSIFIERS_FAST = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=30, random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=30, random_state=42, n_jobs=-1),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }
    # SVM is dropped in fast mode — it's very slow on large datasets.

    REGRESSORS_FAST = {
        "Linear Regression":   LinearRegression(),
        "Ridge":               Ridge(alpha=1.0),
        "Random Forest":       RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=30, random_state=42),
        "Extra Trees":         ExtraTreesRegressor(n_estimators=30, random_state=42, n_jobs=-1),
        "Decision Tree":       DecisionTreeRegressor(max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    # ── Full mode (default) ───────────────────────────────────────────────
    CLASSIFIERS = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
    }

    REGRESSORS = {
        "Linear Regression":   LinearRegression(),
        "Ridge":               Ridge(alpha=1.0),
        "Lasso":               Lasso(alpha=0.1),
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Extra Trees":         ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":       DecisionTreeRegressor(max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "SVR":                 SVR(kernel="rbf"),
    }

    # ── __init__ : Constructor ────────────────────────────────────────────────

    def __init__(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, fast_mode: bool = False):
        # fast_mode=True : use fewer trees → 5-10x faster, slightly lower accuracy
        # Recommended for datasets with >2000 rows or when you want quick results.
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.fast_mode = fast_mode          # Store the speed setting

        self.problem_type = self._detect_problem_type()
        # Automatically figure out if this is classification or regression.
        # Store the result so every other method can use it.

        self.label_encoder = None        # Will be created if target column has text labels
        self.preprocessor = None         # Will hold the ColumnTransformer after building
        self.results: List[Dict] = []    # Will hold one result dict per model after training
        self.best_model = None           # Will hold the winning pipeline after training
        self.best_model_name = None      # Will hold the name of the winning model
        self.feature_names: List[str] = []  # Will hold feature names after one-hot encoding
        self.X_test = None               # Held-out test features (for evaluation)
        self.y_test = None               # Held-out test labels (for evaluation)


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: Problem Type Detection
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_problem_type(self) -> str:
        """
        Heuristic to decide: classification or regression?

        Classification: predict a CATEGORY (survived/died, spam/not-spam)
        Regression:     predict a NUMBER (house price, temperature)

        The _ prefix on the method name means "private" — only called internally.
        """
        target = self.df[self.target_col].dropna()
        # Get the target column, drop missing values for analysis.

        if target.dtype in ["object", "category"]:
            return "classification"
            # Text labels (like "male"/"female" or "cat"/"dog") = definitely classification

        if target.nunique() <= 20:
            return "classification"
            # If there are 20 or fewer UNIQUE values, treat as categories.
            # e.g. 0 and 1 (binary), or 1 through 5 (star ratings)
            # This heuristic is used by H2O AutoML and Google AutoML too.

        return "regression"
        # Many unique numeric values → it's a continuous number → regression


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: Preprocessing Pipeline
    #  The pipeline ensures clean, consistent data preparation.
    # ══════════════════════════════════════════════════════════════════════════

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Builds a ColumnTransformer that:
        - Finds all numeric columns → fills gaps → scales them
        - Finds all text columns → fills gaps → one-hot encodes them

        Why a Pipeline inside a ColumnTransformer?
        Using Pipeline prevents DATA LEAKAGE: the scaler/encoder is fit ONLY
        on training data. When we later transform test data, we use the
        parameters learned from training data — never from test data.
        Like measuring a customer's old shoe size to make their new shoe,
        not measuring from the new shoe itself (that would be circular!).
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # All numeric columns in the input features X (not including the target)

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        # All text/category columns

        transformers = []
        # Start with an empty list of (name, transformer, columns) tuples.

        if numeric_cols:
            transformers.append((
                "num",   # Name for this transformer group (any string you choose)
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    # Step 1: fill missing values with the MEDIAN of that column.
                    # Median is better than mean for skewed data — outliers don't affect it.
                    # e.g. If Age values are [20, 25, 30, 999], median=27.5, mean=268.5

                    ("scaler", StandardScaler()),
                    # Step 2: rescale so mean=0, std=1 (z-score normalisation)
                    # Why? Because Age (0–80) would overpower a column of (0–1).
                    # After scaling, all columns are in the same "units".
                ]),
                numeric_cols   # Apply this pipeline to these columns
            ))

        if categorical_cols:
            transformers.append((
                "cat",   # Name for this transformer group
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    # Step 1: fill missing text values with the literal string "missing"
                    # We can't use median for text. Using a constant is the safe default.

                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    # Step 2: one-hot encode text.
                    # "Sex" column: "male" → [1, 0], "female" → [0, 1]
                    # Creates one binary column per unique category value.
                    # handle_unknown="ignore" → if test data has a new category not in training,
                    #                           just use all zeros instead of crashing.
                    # sparse_output=False → return a dense array, not a sparse matrix.
                ]),
                categorical_cols
            ))

        return ColumnTransformer(transformers=transformers, remainder="drop")
        # remainder="drop" → any columns we didn't explicitly handle get DROPPED.
        # This prevents accidental inclusion of irrelevant ID columns, etc.

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits the data into train/test sets and encodes text targets.
        Returns: (X_train, X_test, y_train, y_test)
        """
        df_clean = self.df.dropna(subset=[self.target_col])
        # Remove rows where the TARGET is missing (can't train on unknowns).

        X = df_clean.drop(columns=[self.target_col])
        # X = all columns EXCEPT the target. The "inputs" to the model.

        y = df_clean[self.target_col]
        # y = just the target column. The "answers" we want to predict.

        if self.problem_type == "classification" and y.dtype in ["object", "category"]:
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
            # fit_transform() learns the mapping: {"survived"→1, "died"→0}
            # then applies it. Now y contains numbers instead of strings.
            # pd.Series(..., index=y.index) preserves the original row index.

        return train_test_split(X, y, test_size=self.test_size, random_state=42)
        # Shuffles and splits data:
        # 80% → X_train, y_train  (we learn from these)
        # 20% → X_test, y_test    (we evaluate on these — NEVER used in training)
        # random_state=42 → same random shuffle every run (reproducibility)
        # Returns a tuple of 4 items: X_train, X_test, y_train, y_test


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: The Main Training Loop
    #  This is where the magic happens — all 9 models trained and evaluated.
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, progress_callback=None) -> List[Dict]:
        """
        The main AutoML race:
        1. Prepare data
        2. For each model: build pipeline → cross-validate → test → record metrics
        3. Sort by best metric
        4. Return ranked leaderboard

        progress_callback: an optional function(value, message) called after each model.
        Used to update the Streamlit progress bar in real time.
        This is the "callback pattern" — we pass a function as an argument.
        """
        X_train, X_test, y_train, y_test = self._prepare_data()
        # Unpack the 4 values returned by _prepare_data() into 4 separate variables.

        self.X_test = X_test    # Store test data for later use (e.g. in plots)
        self.y_test = y_test

        self.preprocessor = self._build_preprocessor(X_train)
        # Build the preprocessor using ONLY training data.
        # CRITICAL: must be built from X_train only, never X_test.

        models = (
            (self.CLASSIFIERS_FAST if self.fast_mode else self.CLASSIFIERS)
            if self.problem_type == "classification"
            else (self.REGRESSORS_FAST if self.fast_mode else self.REGRESSORS)
        )
        # Pick the right model dictionary: fast (fewer trees) or full (more trees).
        # Fast mode trains in seconds; full mode is more accurate but slower.

        cv = (StratifiedKFold(5, shuffle=True, random_state=42)
              if self.problem_type == "classification"
              else KFold(5, shuffle=True, random_state=42))
        # Pick the right cross-validator:
        # StratifiedKFold: for classification — each fold has the same class balance
        # KFold: for regression — simple random fold splits
        # 5 = use 5 folds (train 5 times, each time on a different 80/20 split)

        scoring = "f1_weighted" if self.problem_type == "classification" else "r2"
        # The metric we use to compare models:
        # f1_weighted = for classification (handles class imbalance well)
        # r2          = for regression (how much variance does the model explain?)

        results = []   # List to collect each model's results

        for i, (name, model) in enumerate(models.items()):
            # enumerate(models.items()) gives: (0, ("Logistic Regression", <model>)), etc.
            # We unpack: i = index, name = model name, model = the actual sklearn object

            if progress_callback:
                progress_callback(i / len(models), f"Training {name}...")
                # Call the progress callback if one was provided.
                # i / len(models) = fraction complete (0.0 to 1.0)
                # This updates the Streamlit progress bar after each model.

            try:
                pipe = Pipeline([
                    ("preprocessor", self.preprocessor),
                    # Step 1: apply the ColumnTransformer (impute, scale, encode)

                    ("model", model)
                    # Step 2: the actual ML model
                ])
                # Now "pipe" IS the complete ML system.
                # pipe.fit() → preprocesses then trains.
                # pipe.predict() → preprocesses then predicts.
                # The preprocessing parameters are LOCKED after fit() — no leakage.

                cv_scores = cross_val_score(
                    pipe,                  # The full pipeline to cross-validate
                    X_train, y_train,      # Training data only — NEVER use X_test here
                    cv=cv,                 # Our 5-fold cross-validator
                    scoring=scoring,       # The metric to compute at each fold
                    n_jobs=-1              # Use ALL available CPU cores in parallel (-1 = all)
                )
                # cross_val_score runs the pipeline 5 times on different data splits.
                # Each time: fit on 4 folds, evaluate on the 5th fold.
                # Returns array of 5 scores, e.g. [0.82, 0.81, 0.83, 0.80, 0.82]
                # This gives an HONEST estimate of real-world performance.

                pipe.fit(X_train, y_train)
                # Now fit on the FULL training set (not just 4/5 of it).
                # This is the "final" trained model for this candidate.

                y_pred = pipe.predict(X_test)
                # Make predictions on the held-out test data.
                # These rows were NEVER seen during training — true unseen evaluation.

                metrics = self._compute_metrics(y_test, y_pred, pipe, X_test)
                # Calculate accuracy, F1, RMSE, etc. depending on problem type.

                metrics["model_name"] = name           # Store which model this is
                metrics["cv_mean"] = round(cv_scores.mean(), 4)   # Average CV score
                metrics["cv_std"]  = round(cv_scores.std(),  4)   # How consistent the CV was
                metrics["pipeline"] = pipe             # Store the trained pipeline itself
                results.append(metrics)                # Add this model's results to the list

            except Exception as e:
                pass
                # Some models fail (e.g. SVM on very large datasets runs out of memory).
                # We silently skip failures — the race continues without them.

        sort_key = "f1_weighted" if self.problem_type == "classification" else "r2"
        results.sort(key=lambda x: x.get(sort_key, -999), reverse=True)
        # Sort the results list by the primary metric, best first.
        # lambda x: x.get(sort_key, -999) → a tiny function that extracts the score.
        # .get(sort_key, -999) → if the key doesn't exist, use -999 (worst possible).
        # reverse=True → highest score first (descending order).

        self.results = results   # Store on self so other methods can access it

        if results:
            self.best_model = results[0]["pipeline"]
            # results[0] is the best model (sorted above).
            # ["pipeline"] is the trained Pipeline object.

            self.best_model_name = results[0]["model_name"]

            try:
                self.feature_names = (
                    self.best_model.named_steps["preprocessor"]
                    .get_feature_names_out().tolist()
                )
                # After one-hot encoding, "Sex" becomes "cat__Sex_male", "cat__Sex_female".
                # get_feature_names_out() gives us the full list of all feature names
                # after all transformations — we need these to label the importance chart.
            except Exception:
                self.feature_names = []

        if progress_callback:
            progress_callback(1.0, "Training complete!")
            # Signal 100% done to the progress bar.

        return results


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: Metrics Computation
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_metrics(self, y_true, y_pred, pipe, X_test) -> Dict:
        """Computes the right metrics for the problem type."""

        if self.problem_type == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                # accuracy = correct predictions / total predictions
                # e.g. 82 correct out of 100 = 0.82

                "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                # F1 = 2 * (precision * recall) / (precision + recall)
                # "weighted" = weighted by how many of each class there are
                # zero_division=0 → if a class has no predictions, return 0 instead of crashing

                "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                # Precision = "of all the times I predicted 'positive', how often was I right?"
                # High precision → few false alarms

                "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                # Recall = "of all actual positives, how many did I catch?"
                # High recall → few misses
            }

            if len(np.unique(y_true)) == 2:
                # ROC-AUC only makes sense for BINARY (2-class) problems.
                try:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    # predict_proba returns probabilities: [[0.8, 0.2], [0.3, 0.7], ...]
                    # [:, 1] selects the SECOND column = probability of the POSITIVE class

                    metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
                    # ROC-AUC = area under the ROC curve.
                    # 1.0 = perfect. 0.5 = random guessing. 0.0 = perfectly wrong.
                except Exception:
                    pass
                    # SVM might fail here if probability=True was not set. Skip it.

        else:
            # Regression metrics
            metrics = {
                "r2": round(r2_score(y_true, y_pred), 4),
                # R² (R-squared): 1.0 = model explains all variation.
                # 0.0 = model is no better than just predicting the mean.
                # Negative = model is WORSE than predicting the mean.

                "mae": round(mean_absolute_error(y_true, y_pred), 4),
                # MAE = average absolute error = average of |actual - predicted|
                # Easy to interpret: "my predictions are off by X units on average"

                "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
                # RMSE = Root Mean Squared Error = sqrt(average of (actual - predicted)²)
                # Penalises large errors more than MAE does (squaring amplifies big gaps)

                "mape": round(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100, 2),
                # MAPE = Mean Absolute Percentage Error
                # (y_true - y_pred) / y_true = relative error (as fraction)
                # * 100 = as a percentage
                # 1e-8 added to denominator to avoid division by zero
            }
        return metrics


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: Feature Importance
    #  How much does each input column contribute to the predictions?
    # ══════════════════════════════════════════════════════════════════════════

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.DataFrame]:
        """
        Extracts feature importance from the best model.
        Tree models: use built-in feature_importances_ attribute.
        Linear models: use magnitude of coefficients (bigger coef = more important).
        """
        if not self.best_model or not self.feature_names:
            return None
            # No best model yet (training hasn't been run), or no feature names available.

        model_step = self.best_model.named_steps["model"]
        # Access the SECOND step of the pipeline: the actual trained model object.
        # named_steps["model"] → the step we named "model" when building the Pipeline.

        importance_vals = None
        try:
            if hasattr(model_step, "feature_importances_"):
                importance_vals = model_step.feature_importances_
                # hasattr() checks if the object has this attribute (like checking if a dog has fur).
                # Tree-based models (Random Forest, Gradient Boosting, etc.) have feature_importances_.
                # e.g. array([0.04, 0.31, 0.02, ...]) — fraction of importance per feature.

            elif hasattr(model_step, "coef_"):
                coef = model_step.coef_
                importance_vals = np.abs(coef.ravel() if coef.ndim > 1 else coef)
                # Linear models (Logistic Regression, Ridge, Lasso) have coef_ (coefficients).
                # np.abs() = absolute value (we care about magnitude, not direction)
                # coef.ndim > 1 → for multi-class, coef is a 2D matrix. .ravel() flattens to 1D.
        except Exception:
            return None

        if importance_vals is None:
            return None   # Model doesn't support importance extraction (e.g. KNN, SVM without coef)

        n = min(len(self.feature_names), len(importance_vals))
        # Take the minimum length in case sizes don't match (safety check).

        fi = pd.DataFrame({
            "feature": self.feature_names[:n],       # Feature names
            "importance": importance_vals[:n],        # Importance scores
        }).sort_values("importance", ascending=False).head(top_n)
        # sort_values("importance", ascending=False) → most important first
        # .head(top_n) → keep only the top 20

        fi["importance_normalized"] = fi["importance"] / fi["importance"].sum()
        # Divide each score by the total so all values add up to 1.0 (100%).
        # Now importance is shown as a fraction, making comparison easy.

        return fi


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 6: Visualisation Functions
    # ══════════════════════════════════════════════════════════════════════════

    def plot_leaderboard(self) -> go.Figure:
        """Horizontal bar chart ranking all models by their primary metric."""
        if not self.results:
            return None

        primary = "f1_weighted" if self.problem_type == "classification" else "r2"
        names  = [r["model_name"] for r in self.results]   # List of model names
        scores = [r.get(primary, 0) for r in self.results] # List of scores (0 if missing)

        colors = [COLORS["primary"] if n == self.best_model_name else COLORS["secondary"]
                  for n in names]
        # Ternary list comprehension: cyan for the winner, purple for all others.
        # Highlights the winner visually.

        fig = go.Figure(go.Bar(
            x=scores,         # Scores on x-axis (horizontal bars)
            y=names,          # Model names on y-axis
            orientation="h",  # Horizontal orientation
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores],   # Label each bar with its score
            textposition="outside",               # Label goes outside the bar end
        ))
        fig.update_layout(
            title=f"<b>Model Leaderboard — {primary.upper()}</b>",
            xaxis_title=primary,
            height=350,
            **LAYOUT_BASE
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        return fig

    def plot_feature_importance(self) -> Optional[go.Figure]:
        """Horizontal bar chart of the top feature importances from the best model."""
        fi = self.get_feature_importance()
        if fi is None or fi.empty:
            return None

        fig = go.Figure(go.Bar(
            x=fi["importance_normalized"].tolist(),   # Normalised importance on x-axis
            y=fi["feature"].tolist(),                 # Feature names on y-axis
            orientation="h",
            marker=dict(
                color=fi["importance_normalized"].tolist(),
                colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
                # Color gradient: low importance → purple, high importance → cyan
            ),
            text=[f"{v:.3f}" for v in fi["importance_normalized"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"<b>Feature Importance — {self.best_model_name}</b>",
            xaxis_title="Normalized Importance",
            height=400,
            **LAYOUT_BASE
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        return fig

    def plot_metrics_radar(self) -> Optional[go.Figure]:
        """
        Radar (spider web) chart comparing the top 5 models across all metrics.
        Each axis of the web is one metric. Each model is one coloured polygon.
        A bigger polygon = better model.
        """
        if not self.results:
            return None

        if self.problem_type == "classification":
            metric_keys = ["accuracy", "f1_weighted", "precision", "recall", "cv_mean"]
        else:
            metric_keys = ["r2", "cv_mean"]
            # For regression we only have 2 metrics that are on 0-1 scale.
            # RMSE and MAE are in the original units and can't go on the same radar.

        top_n = min(5, len(self.results))
        # Show at most 5 models on the radar (too many makes it unreadable).

        fig = go.Figure()

        for i, result in enumerate(self.results[:top_n]):
            values = [result.get(m, 0) for m in metric_keys]
            # Get each metric value for this model. Default to 0 if not found.

            values_closed = values + [values[0]]
            # Close the polygon by repeating the first value at the end.
            # Without this, the web shape would have a gap between the last and first spokes.

            metrics_closed = metric_keys + [metric_keys[0]]
            # Similarly close the metric names list.

            fig.add_trace(go.Scatterpolar(
                r=values_closed,         # Radial distances (the metric values)
                theta=metrics_closed,    # The axis labels (metric names)
                fill="toself",           # Fill the area inside the polygon
                name=result["model_name"],
                line_color=COLORS["palette"][i],       # Border colour
                fillcolor=COLORS["palette"][i],         # Fill colour
                opacity=0.3 if i > 0 else 0.5,
                # Best model (i=0) is 50% opaque. Others are 30% (more transparent).
            ))

        fig.update_layout(
            title="<b>Model Comparison Radar</b>",
            polar=dict(
                bgcolor="rgba(8,15,26,0.5)",   # Dark background inside the radar
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(26,58,92,0.5)"),
                # radialaxis: the circular grid lines. range=[0,1] = 0 to 100%.
                angularaxis=dict(gridcolor="rgba(26,58,92,0.5)"),
                # angularaxis: the spoke lines going outward from the centre.
            ),
            height=400,
            **LAYOUT_BASE
        )
        return fig

    def get_leaderboard_df(self) -> pd.DataFrame:
        """Returns a clean summary table of all model results for display in the UI."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            row = {
                "Model":   r["model_name"],
                "CV Mean": r["cv_mean"],
                "CV Std":  f"±{r['cv_std']}",
                # f"±{r['cv_std']}" → e.g. "±0.0234"
                # ± (plus-or-minus) shows the uncertainty in the CV score.
                # A small std means the model performs consistently across folds.
            }
            if self.problem_type == "classification":
                row.update({
                    "Accuracy":      r.get("accuracy",    "-"),
                    "F1 (weighted)": r.get("f1_weighted", "-"),
                    "Precision":     r.get("precision",   "-"),
                    "Recall":        r.get("recall",      "-"),
                })
                # .update() adds/overwrites keys in the dictionary.
                # .get("key", "-") returns "-" (dash) if that key doesn't exist.
                if "roc_auc" in r:
                    row["ROC-AUC"] = r["roc_auc"]
                    # Only add ROC-AUC column if it was calculated (binary classification only).
            else:
                row.update({
                    "R²":       r.get("r2",   "-"),
                    "MAE":      r.get("mae",  "-"),
                    "RMSE":     r.get("rmse", "-"),
                    "MAPE (%)": r.get("mape", "-"),
                })
            rows.append(row)

        return pd.DataFrame(rows)
        # Convert list of row-dicts into a neat table.
