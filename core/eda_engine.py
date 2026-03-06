"""
╔══════════════════════════════════════════════════════════════════╗
║  eda_engine.py  —  DataMind Platform                            ║
║  THE DATA DETECTIVE                                              ║
║                                                                  ║
║  EDA = Exploratory Data Analysis.                               ║
║  This file is like a detective that investigates a new dataset. ║
║  It checks everything: how big is it? Are there missing pieces? ║
║  Are there weird outlier values? Do any columns relate to each  ║
║  other? Then it draws charts and writes a summary report.       ║
║                                                                  ║
║  A data scientist does this manually at the start of every      ║
║  project. DataMind does it automatically in seconds.            ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────

import pandas as pd
# pandas is the #1 data tool in Python. A "DataFrame" (df) is like an Excel
# spreadsheet inside Python. pd.read_csv() loads a CSV file into a DataFrame.

import numpy as np
# numpy gives us fast maths on arrays of numbers.
# np.number means "any numeric type" (int, float, etc.)
# np.sqrt() = square root. np.mean() = average. Very commonly used.

import plotly.express as px
# plotly.express is a high-level charting library.
# px.scatter() makes a scatter plot in one line.
# "express" means quick and easy — good for standard charts.

import plotly.graph_objects as go
# plotly.graph_objects is the lower-level, more customisable charting library.
# go.Figure(), go.Histogram(), go.Heatmap() — we build charts piece by piece.
# More control, more code.

from plotly.subplots import make_subplots
# make_subplots creates a grid of charts inside one figure.
# Like a newspaper page with multiple graphs arranged in rows and columns.

from typing import Dict, Any, List, Tuple, Optional
# Type hints (see ollama_client.py for explanation).

import warnings
warnings.filterwarnings("ignore")
# Some libraries produce annoying yellow warning messages in the terminal.
# This line silences them. Like putting on earplugs for background noise.


# ── VISUAL THEME CONSTANTS ────────────────────────────────────────────────────
# These set the colours and fonts for ALL charts in DataMind.
# Defined once here so every chart looks consistent — like a brand style guide.

PLOT_TEMPLATE = "plotly_dark"
# Use Plotly's built-in dark theme as a starting point.

COLORS = {
    "primary":   "#00e5ff",   # Electric cyan — the main accent colour
    "secondary": "#7c4dff",   # Deep purple — second accent
    "accent":    "#ff6d00",   # Warm orange — highlights
    "success":   "#00e676",   # Green — good results
    "danger":    "#ff1744",   # Red — warnings / bad results
    "palette": [              # A list of 10 colours used for multiple series on one chart
        "#00e5ff", "#7c4dff", "#ff6d00", "#00e676", "#ff1744",
        "#40c4ff", "#b388ff", "#ffab40", "#69f0ae", "#ff8a80"
    ],
}
# Hex colour codes: #RRGGBB where RR=red, GG=green, BB=blue in hexadecimal.
# #00e5ff = 0 red, 229 green, 255 blue → electric cyan.

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(2,4,9,0)",       # Background of the WHOLE figure: nearly transparent
    plot_bgcolor="rgba(8,15,26,0.5)",    # Background of the chart AREA: semi-transparent dark
    font=dict(family="Space Mono, monospace", color="#7ec8e3", size=11),
    # font: use Space Mono (a monospace coding font), cyan-ish colour, size 11
    margin=dict(l=40, r=20, t=50, b=40),
    # Margins: left=40px, right=20px, top=50px, bottom=40px — breathing room
    # NOTE: gridcolor is NOT a valid top-level Layout property in Plotly.
    # It must be set inside xaxis/yaxis dicts. We apply it via update_xaxes/update_yaxes calls.
)

# Grid line colour used in update_xaxes() and update_yaxes() calls below each chart.
# Kept as a constant so it's easy to change the grid colour in one place.
GRID_COLOR = "rgba(26,58,92,0.4)"  # Dark blue, 40% transparent
# dict() creates a dictionary. We spread (**) this into every chart's layout
# so all charts share the same dark-theme style.


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS: EDAEngine
# ══════════════════════════════════════════════════════════════════════════════

class EDAEngine:
    """
    Runs a full automated EDA suite on any Pandas DataFrame.

    The workflow mirrors what a senior data scientist would do manually:
    1. Profile the data (shape, types, missing values)
    2. Analyze distributions (histograms, box plots)
    3. Compute correlations (heatmap, scatter matrix)
    4. Detect outliers (IQR method)
    5. Analyze the target variable (if provided)
    6. Generate a text summary for the LLM to narrate
    """

    # ── __init__ ──────────────────────────────────────────────────────────────

    def __init__(self, df: pd.DataFrame):
        # df: pd.DataFrame — this means "df must be a pandas DataFrame"
        # type hint = the label on the box saying what's inside.

        self.df = df.copy()
        # IMPORTANT: .copy() makes a new copy of the data.
        # If we don't copy, changing self.df would change the original too.
        # Like photocopying a document before writing notes on it.

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # select_dtypes(include=[np.number]) → filter to only numeric columns
        # .columns → get just the column names
        # .tolist() → convert from Index object to a plain Python list
        # Result: ["Age", "Fare", "Pclass"] — only number columns

        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # "object" dtype = text / string columns (like "male", "female", "London")
        # "category" dtype = pandas-optimised categories
        # Result: ["Name", "Sex", "Cabin"] — only text columns

        self.datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        # datetime = date+time columns (like "2023-01-15 09:30:00")
        # Useful for time series analysis.

        self.n_rows, self.n_cols = df.shape
        # df.shape returns a tuple: (number_of_rows, number_of_columns)
        # e.g. (891, 12) for the Titanic dataset
        # We "unpack" it into two separate variables: n_rows=891, n_cols=12


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: Data Profiling
    #  Like taking a full medical checkup for the dataset.
    # ══════════════════════════════════════════════════════════════════════════

    def get_profile(self) -> Dict[str, Any]:
        """Returns a comprehensive data profile dictionary."""

        missing = self.df.isnull().sum()
        # isnull() → creates a True/False table: True where data is missing
        # .sum()   → counts True values per column (True=1, False=0)
        # Result: a Series like {"Age": 177, "Cabin": 687, "Embarked": 2}

        missing_pct = (missing / self.n_rows * 100).round(2)
        # Divide missing count by total rows, multiply by 100 = percentage missing
        # .round(2) = round to 2 decimal places
        # e.g. 177/891*100 = 19.87% of Age values are missing

        profile = {
            "shape": {"rows": self.n_rows, "columns": self.n_cols},
            # How big is the dataset?

            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1e6, 2),
            # memory_usage(deep=True) → how many bytes does each column use in RAM?
            # .sum() → total bytes
            # / 1e6 → convert bytes to megabytes (1e6 = 1,000,000)
            # round(..., 2) → keep 2 decimal places

            "duplicate_rows": int(self.df.duplicated().sum()),
            # duplicated() → marks rows that are exact copies of an earlier row as True
            # .sum() → count how many duplicate rows exist
            # int() → convert numpy integer to regular Python int (for JSON safety)

            "numeric_columns":    len(self.numeric_cols),    # How many number columns
            "categorical_columns": len(self.categorical_cols), # How many text columns
            "datetime_columns":   len(self.datetime_cols),   # How many date columns

            "missing": {
                col: {"count": int(missing[col]), "percent": float(missing_pct[col])}
                for col in self.df.columns if missing[col] > 0
            },
            # Dictionary comprehension: loop over all columns,
            # include only columns that have at least 1 missing value.
            # For each, store {"count": 177, "percent": 19.87}

            "dtypes": self.df.dtypes.astype(str).to_dict(),
            # .dtypes → data type of each column (int64, float64, object, etc.)
            # .astype(str) → convert dtype objects to strings (for JSON serialisation)
            # .to_dict() → convert Series to a plain dict

            "unique_counts": self.df.nunique().to_dict(),
            # nunique() → count unique values per column
            # e.g. "Sex" has 2 unique values ("male", "female")
        }
        return profile

    def get_descriptive_stats(self) -> pd.DataFrame:
        """Extended descriptive stats including skewness and kurtosis."""

        if not self.numeric_cols:
            return pd.DataFrame()
            # If there are no numeric columns, return an empty table.
            # Safety check — always handle edge cases.

        stats = self.df[self.numeric_cols].describe().T
        # .describe() gives: count, mean, std, min, 25%, 50%, 75%, max
        # It's like a sports stats card for each column.
        # .T = "transpose" — flip rows and columns so each row is a column name.

        stats["skewness"] = self.df[self.numeric_cols].skew().round(3)
        # Skewness: how lopsided is the distribution?
        # 0 = perfectly symmetric. Positive = tail on the right. Negative = tail on left.
        # e.g. Income data is usually right-skewed (a few very rich people pull the tail right).

        stats["kurtosis"] = self.df[self.numeric_cols].kurtosis().round(3)
        # Kurtosis: how "peaked" or "flat" is the distribution?
        # High kurtosis = very pointy bell curve with fat tails (more extreme values).
        # Normal distribution has kurtosis = 0.

        stats["missing_%"] = (self.df[self.numeric_cols].isnull().mean() * 100).round(2)
        # isnull().mean() = fraction of missing values per column (0.0 to 1.0)
        # * 100 = convert to percentage
        # So if 20% of Age values are missing, missing_% = 20.0

        return stats.round(3)
        # Round everything to 3 decimal places for clean display.

    def get_summary_for_llm(self) -> str:
        """
        Creates a rich text summary to send to the AI for narrative generation.
        The AI reads this and writes a human-friendly story about the data.
        """
        profile = self.get_profile()
        # Call our profiling function to get all the stats.

        lines = [
            f"Dataset: {self.n_rows:,} rows × {self.n_cols} columns",
            # :, adds comma separators to large numbers: 1000000 → 1,000,000

            f"Memory: {profile['memory_mb']} MB",
            f"Duplicates: {profile['duplicate_rows']:,}",
            f"Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols[:10])}",
            # ', '.join([...]) → join a list into a string with ", " between each item
            # [:10] → take only the first 10 items (don't flood the prompt)

            f"Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols[:10])}",
        ]

        if profile["missing"]:
            # Only add the missing section if there ARE missing values.
            lines.append("\nMissing Values:")
            for col, info in list(profile["missing"].items())[:10]:
                # .items() → loop over key-value pairs: ("Age", {"count": 177, "percent": 19.87})
                # [:10] → only report the first 10 columns with missing data
                lines.append(f"  - {col}: {info['count']:,} ({info['percent']}%)")

        if self.numeric_cols:
            try:
                stats = self.get_descriptive_stats()
                lines.append("\nKey Statistics:")
                for col in self.numeric_cols[:8]:
                    # Only report stats for first 8 numeric columns (keep it concise)
                    if col in stats.index:
                        row = stats.loc[col]
                        # stats.loc[col] → get the row for this column by label
                        lines.append(
                            f"  {col}: mean={row['mean']:.2f}, std={row['std']:.2f}, "
                            f"min={row['min']:.2f}, max={row['max']:.2f}, "
                            f"skew={row.get('skewness', 'N/A')}"
                        )
                        # :.2f = format as float with 2 decimal places
            except Exception:
                pass
                # If anything goes wrong building stats, silently skip it.
                # We don't want a stats error to crash the whole summary.

        return "\n".join(lines)
        # "\n".join(lines) → join all lines with newline characters between them
        # Turns a list of strings into one big multi-line string.


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: Chart-Making Functions
    #  Each function creates one Plotly figure (an interactive chart).
    # ══════════════════════════════════════════════════════════════════════════

    def plot_distributions(self, max_cols: int = 9) -> Optional[go.Figure]:
        """
        Creates a grid of histograms — one per numeric column.
        A histogram shows how values are spread out:
        tall bars = common values, short bars = rare values.
        """
        cols = self.numeric_cols[:max_cols]
        # Take only the first max_cols (9) numeric columns.
        # Avoids creating a huge grid for datasets with 50+ columns.

        if not cols:
            return None
            # No numeric columns → nothing to plot → return None (nothing).

        n = len(cols)
        ncols = min(3, n)
        # Number of chart columns in the grid. At most 3 per row.
        # min(3, n) = 3 if we have 3+ charts, otherwise use n.

        nrows = (n + ncols - 1) // ncols
        # Integer division to calculate how many rows we need.
        # e.g. 7 charts in 3 columns: (7 + 3 - 1) // 3 = 9 // 3 = 3 rows
        # This is the ceiling division formula.

        fig = make_subplots(
            rows=nrows, cols=ncols,
            # Create a grid of nrows × ncols empty chart slots.

            subplot_titles=[f"<b>{c}</b>" for c in cols],
            # Title above each subplot. f"<b>{c}</b>" = bold column name.
            # This is a list comprehension: make a title for each column name.

            vertical_spacing=0.12,   # 12% gap between rows
            horizontal_spacing=0.08  # 8% gap between columns
        )

        for i, col in enumerate(cols):
            # enumerate() gives us both the index (i) and the value (col).
            # i=0, col="Age"   then  i=1, col="Fare"   etc.

            row, col_idx = divmod(i, ncols)
            # divmod(i, ncols) returns (quotient, remainder).
            # e.g. divmod(5, 3) = (1, 2) → row 1, column 2 (0-indexed)
            # This maps a flat index to a (row, column) grid position.

            data = self.df[col].dropna()
            # self.df[col] → select the column as a Series
            # .dropna() → remove NaN (missing) values before plotting

            fig.add_trace(
                go.Histogram(
                    x=data,                # The actual data values
                    name=col,              # Label for the legend
                    marker_color=COLORS["palette"][i % len(COLORS["palette"])],
                    # i % len(palette) → cycle through colours.
                    # If i=12 and palette has 10 colours: 12 % 10 = 2 → use colour[2].
                    # This ensures we never run out of colours.
                    opacity=0.8,           # 80% solid (20% see-through)
                    showlegend=False,      # Don't show legend (each chart has its own title)
                    nbinsx=30,             # Divide data into 30 bins (bars)
                ),
                row=row + 1, col=col_idx + 1
                # +1 because Plotly uses 1-based indexing (rows start at 1, not 0)
            )

        fig.update_layout(
            title="<b>Feature Distributions</b>",
            height=250 * nrows,   # Chart height grows with number of rows
            **LAYOUT_BASE         # ** "unpacks" the dict — spreads all keys as arguments
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)  # X-axis grid lines: dark blue
        fig.update_yaxes(gridcolor=GRID_COLOR)  # Y-axis grid lines: dark blue
        return fig

    def plot_correlation_heatmap(self) -> Optional[go.Figure]:
        """
        Pearson correlation heatmap.
        Shows how strongly every pair of numeric columns is related.
        Range: -1 (perfectly opposite) to 0 (unrelated) to +1 (perfectly together).
        Colour: red = strong negative, dark = near zero, cyan = strong positive.
        """
        if len(self.numeric_cols) < 2:
            return None
            # Need at least 2 columns to compare. With only 1, there's nothing to correlate.

        corr = self.df[self.numeric_cols].corr()
        # .corr() computes the Pearson correlation coefficient between every pair of columns.
        # Returns a square DataFrame: rows AND columns are all the same column names.
        # e.g. corr["Age"]["Fare"] = 0.12 (weak positive correlation)

        fig = go.Figure(go.Heatmap(
            z=corr.values,                        # The 2D array of correlation numbers
            x=corr.columns.tolist(),              # Column names on the x-axis
            y=corr.index.tolist(),                # Column names on the y-axis
            colorscale=[
                [0.0, "#ff1744"],   # -1 correlation → red colour
                [0.5, "#0d1b2e"],   #  0 correlation → very dark blue (neutral)
                [1.0, "#00e5ff"],   # +1 correlation → cyan colour
            ],
            # colorscale: a list of [position, colour] pairs.
            # position 0 = minimum value (-1), position 1 = maximum value (+1).
            # Values in between get interpolated (blended) automatically.

            zmin=-1, zmax=1,
            # Force the colour scale to go from -1 to +1,
            # even if no value reaches those extremes.

            text=corr.round(2).values,     # Show the correlation number inside each cell
            texttemplate="%{text}",        # How to format the text label
            textfont={"size": 10, "color": "#e8f4fd"},   # Text style inside cells
            hoverongaps=False,             # Don't show tooltip for empty cells
        ))
        fig.update_layout(
            title="<b>Correlation Matrix</b>",
            height=500,        # 500 pixels tall
            **LAYOUT_BASE
        )
        return fig

    def plot_missing_values(self) -> Optional[go.Figure]:
        """Bar chart showing what percentage of each column is missing (empty)."""

        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        # isnull().sum() → count of missing values per column
        # / len(self.df) → divide by total rows = fraction
        # * 100 → convert to percentage
        # .sort_values(ascending=False) → sort from most missing to least

        missing_pct = missing_pct[missing_pct > 0]
        # Keep only columns that have at least SOME missing values.
        # Boolean indexing: [condition] keeps only rows where condition is True.

        if missing_pct.empty:
            return None
            # All data is complete! No chart needed → return None.

        colors = [COLORS["danger"] if v > 20 else COLORS["accent"] if v > 5 else COLORS["primary"]
                  for v in missing_pct.values]
        # Colour each bar based on how bad the missing data is:
        # > 20% missing → red (danger)     e.g. Cabin column
        # > 5% missing  → orange (accent)  e.g. Age column
        # ≤ 5% missing  → cyan (primary)   e.g. Embarked column
        # This is called a "ternary expression" (inline if-else).

        fig = go.Figure(go.Bar(
            x=missing_pct.index.tolist(),   # Column names on x-axis
            y=missing_pct.values,           # Percentage values on y-axis
            marker_color=colors,            # Individual bar colours from list above
            text=[f"{v:.1f}%" for v in missing_pct.values],   # Label on top of each bar
            textposition="outside",         # Put the label above the bar (not inside)
        ))
        fig.update_layout(
            title="<b>Missing Value Analysis</b>",
            xaxis_title="Column",
            yaxis_title="Missing %",
            height=350,
            **LAYOUT_BASE
        )
        return fig

    def plot_categorical_counts(self, max_cols: int = 6, max_categories: int = 15) -> Optional[go.Figure]:
        """
        Horizontal bar charts showing the count of each category in text columns.
        e.g. For the "Sex" column: Male=577, Female=314
        """
        cols = self.categorical_cols[:max_cols]  # First 6 text columns
        if not cols:
            return None

        n = len(cols)
        ncols = min(2, n)   # At most 2 charts side by side
        nrows = (n + ncols - 1) // ncols   # How many rows needed

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=[f"<b>{c}</b>" for c in cols],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        for i, col in enumerate(cols):
            row, col_idx = divmod(i, ncols)

            vc = self.df[col].value_counts().head(max_categories)
            # .value_counts() → counts how many times each unique value appears
            # Result: Series with value as index, count as value. Sorted by count.
            # .head(max_categories) → keep only the top 15 most common categories

            fig.add_trace(
                go.Bar(
                    y=vc.index.astype(str).tolist(),   # Categories on y-axis (horizontal bars)
                    x=vc.values,                        # Counts on x-axis
                    orientation="h",                    # "h" = horizontal bars
                    marker_color=COLORS["palette"][i % len(COLORS["palette"])],
                    showlegend=False,
                    name=col,
                ),
                row=row + 1, col=col_idx + 1
            )

        fig.update_layout(
            title="<b>Categorical Feature Counts</b>",
            height=300 * nrows,
            **LAYOUT_BASE
        )
        return fig

    def plot_outlier_analysis(self) -> Optional[go.Figure]:
        """
        Box plots for all numeric columns.
        A box plot shows:
        - The box: the "middle 50%" of the data (Q1 to Q3)
        - The line in the box: the median (middle value)
        - The whiskers: reasonable range (1.5 × IQR)
        - The dots beyond whiskers: OUTLIERS (extreme unusual values)
        """
        cols = self.numeric_cols[:12]  # First 12 numeric columns
        if not cols:
            return None

        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(go.Box(
                y=self.df[col].dropna().values,    # All values for this column
                name=col,                           # Column name as the label
                marker_color=COLORS["palette"][i % len(COLORS["palette"])],
                boxmean=True,         # Show the MEAN as a dashed line inside the box
                jitter=0.3,           # Add slight horizontal randomness to dots
                                      # (prevents all dots stacking on top of each other)
                pointpos=-1.8,        # Position the individual dots to the left of the box
            ))

        fig.update_layout(
            title="<b>Outlier Analysis (Box Plots)</b>",
            height=450,
            showlegend=False,  # No legend needed — each box is labelled on x-axis
            **LAYOUT_BASE
        )
        return fig

    def plot_target_analysis(self, target_col: str) -> Optional[go.Figure]:
        """
        Deep analysis of the TARGET column — the thing you're trying to predict.
        - If it's categories (like "survived"/"died"): pie chart + bar chart
        - If it's numbers (like "house price"): histogram
        """
        if target_col not in self.df.columns:
            return None
            # Safety check: does this column actually exist?

        target = self.df[target_col].dropna()

        if target.dtype in ["object", "category"] or target.nunique() <= 20:
            # This condition detects a CLASSIFICATION target:
            # - Text values (like "survived", "died") OR
            # - Few unique numeric values (like 0 or 1 for binary classification)

            vc = target.value_counts()
            # Count how many of each class exists.
            # e.g. {"0": 549, "1": 342} for Titanic survived column

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]]
                # specs tells Plotly the TYPE of each subplot.
                # Normally all subplots are "xy" (axes). Here we need one "pie".
            )

            fig.add_trace(go.Pie(
                labels=vc.index.astype(str).tolist(),  # Category names
                values=vc.values.tolist(),             # How many of each
                marker_colors=COLORS["palette"],
                textinfo="label+percent",  # Show "label: 38.4%" on each slice
                hole=0.4,                  # 0.4 = donut hole in centre (40% hollow)
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=vc.index.astype(str).tolist(),  # Categories on x-axis
                y=vc.values.tolist(),             # Counts on y-axis
                marker_color=COLORS["primary"],
            ), row=1, col=2)

        else:
            # REGRESSION target: just a histogram of the continuous values
            fig = go.Figure(go.Histogram(
                x=target,
                marker_color=COLORS["primary"],
                opacity=0.8,
                nbinsx=50,   # 50 bars
            ))

        fig.update_layout(
            title=f"<b>Target Variable: {target_col}</b>",
            height=400,
            **LAYOUT_BASE
        )
        return fig


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: Outlier Detection (IQR Method)
    #  IQR = InterQuartile Range = the "middle 50%" of the data.
    #  Values too far outside this range are called outliers.
    # ══════════════════════════════════════════════════════════════════════════

    def detect_outliers_iqr(self) -> pd.DataFrame:
        """
        IQR-based outlier detection for every numeric column.
        The IQR method is robust — extreme outliers don't distort it,
        unlike z-score which uses the mean (which outliers CAN distort).
        """
        results = []
        # Start with an empty list. We'll append one row per column.

        for col in self.numeric_cols:
            data = self.df[col].dropna()
            # Remove missing values — can't compute stats on NaN.

            Q1 = data.quantile(0.25)
            # Q1 = "first quartile" = the value below which 25% of data falls.
            # If you sorted all students by height, Q1 is the height of the 25th percentile student.

            Q3 = data.quantile(0.75)
            # Q3 = "third quartile" = the value below which 75% of data falls.

            IQR = Q3 - Q1
            # IQR = the span of the middle 50% of the data.
            # Tall IQR = data is spread out. Small IQR = data is clustered.

            lower = Q1 - 1.5 * IQR
            # Anything BELOW this is an outlier on the low end.
            # The 1.5 multiplier is the standard Tukey rule (from statistician John Tukey).

            upper = Q3 + 1.5 * IQR
            # Anything ABOVE this is an outlier on the high end.

            outliers = data[(data < lower) | (data > upper)]
            # Boolean filtering: keep only values that fall OUTSIDE the bounds.
            # | means OR: flag the row if it's below lower OR above upper.

            results.append({
                "column": col,
                "outlier_count": len(outliers),           # How many outliers found
                "outlier_%": round(len(outliers) / len(data) * 100, 2),  # As a percentage
                "lower_bound": round(lower, 3),           # The lower threshold
                "upper_bound": round(upper, 3),           # The upper threshold
            })

        return pd.DataFrame(results)
        # pd.DataFrame(list_of_dicts) → creates a table from a list of row-dictionaries.
        # Each dict becomes one row. Keys become column names.


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: Custom Scatter Plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_scatter(self, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
        """
        Interactive scatter plot: each dot is one row in the data.
        x_col   = what goes on the horizontal axis
        y_col   = what goes on the vertical axis
        color_col = optional: which column to use for dot colours
        """
        kwargs = dict(x=self.df[x_col], y=self.df[y_col])
        # Start building the arguments dictionary.
        # dict() creates a dict from keyword=value pairs.

        if color_col and color_col in self.df.columns:
            kwargs["color"] = self.df[color_col]
            # If a colour column was provided and it exists, add it to kwargs.
            # This makes dots different colours based on that column's value.

        fig = px.scatter(**kwargs, template="plotly_dark", opacity=0.7)
        # **kwargs "unpacks" the dictionary as keyword arguments.
        # Like saying px.scatter(x=..., y=..., color=...) but built dynamically.

        fig.update_layout(
            title=f"<b>{x_col} vs {y_col}</b>",
            xaxis_title=x_col,
            yaxis_title=y_col,
            **LAYOUT_BASE
        )
        return fig


    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: Time Series Plot
    # ══════════════════════════════════════════════════════════════════════════

    def plot_time_series(self, date_col: str, value_col: str) -> Optional[go.Figure]:
        """
        Line chart for time-based data with a 7-period rolling average overlay.
        Rolling average smooths out noise so you can see the underlying trend.
        Like drawing a smooth curve through jagged price data.
        """
        if date_col not in self.df.columns or value_col not in self.df.columns:
            return None
            # Safety check: both columns must exist.

        df_sorted = self.df[[date_col, value_col]].dropna().sort_values(date_col)
        # [[date_col, value_col]] → select only those 2 columns
        # .dropna() → remove rows with any missing values in either column
        # .sort_values(date_col) → sort by date (oldest first)

        rolling = df_sorted[value_col].rolling(window=7, min_periods=1).mean()
        # .rolling(window=7) → a sliding window of 7 rows at a time
        # .mean() → average of those 7 rows
        # min_periods=1 → at the start (where we don't have 7 rows yet), use what we have
        # Result: a smoothed version of the value column

        fig = go.Figure()
        # Create an empty figure — we'll add traces (data series) one by one.

        fig.add_trace(go.Scatter(
            x=df_sorted[date_col], y=df_sorted[value_col],
            mode="lines",          # Draw a line connecting the dots (not dots alone)
            name="Actual",         # Label for the legend
            line=dict(color=COLORS["primary"], width=1),  # Thin cyan line
            opacity=0.6            # Semi-transparent so rolling average is visible on top
        ))

        fig.add_trace(go.Scatter(
            x=df_sorted[date_col], y=rolling,
            mode="lines",
            name="7-period MA",    # MA = Moving Average
            line=dict(color=COLORS["accent"], width=2.5),  # Thicker orange line on top
        ))

        fig.update_layout(
            title=f"<b>{value_col} over Time</b>",
            height=400,
            **LAYOUT_BASE
        )
        return fig
