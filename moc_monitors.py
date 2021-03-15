import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias

import logging

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    A class to compute differences (by some metric) between two DataFrames:
    a baseline and a sample.

    Methods
    -------
    calculate_drift:
        Calculates drift between baseline and sample datasets according to
        a pre-defined metric or a user-defined metric.

    plot_numerical:
        Plots distribution of numerical features of baseline and sample datasets.

    plot_categorical:
        Creates a proportion histogram between the 2 datasets for categorical
        columns.

    Args
    ----
    df_baseline: <pandas.DataFrame>
        Pandas DataFrame of the baseline dataset. 

    df_sample: <pandas.DataFrame>
        Pandas DataFrame of the sample dataset.

    categorical_columns: <list of str>
        A list of categorical columns in the dataset. If not provided, categorical 
        columns will be inferred from column types.

    numerical_columns: <list of str>
        A list of numerical columns in the dataset. If not provided, numerical 
        columns will be inferred from column types.

    score_column: <str>
        Column containing predicted values (as computed by underlying model).

    label_column: <str>
        Column containing actual values (ground truths).

    # label_type: <str>
    #     'categorical' or 'numerical' to reflect classification or regression.
    """

    def __init__(
        self,
        df_baseline,
        df_sample,
        categorical_columns=None,
        numerical_columns=None,
        score_column=None,
        label_column=None,
        # label_type=None,
    ):
        assert isinstance(
            df_baseline, pd.DataFrame
        ), "df_baseline should be of type <pandas.DataFrame>."

        assert isinstance(
            df_sample, pd.DataFrame
        ), "df_baseline should be of type <pandas.DataFrame>."

        assert all(
            df_baseline.columns == df_sample.columns
        ), "df_baseline and df_sample should have the same column names."

        assert all(
            df_baseline.dtypes == df_sample.dtypes
        ), "df_baseline and df_sample should have the same column types."

        assert isinstance(
            categorical_columns, (list, type(None))
        ), "categorical_columns should be of type <list>."

        assert isinstance(
            numerical_columns, (list, type(None))
        ), "numerical_columns should be of type <list>."

        assert isinstance(
            score_column, (str, type(None))
        ), "score_column should be of type <str>."

        assert isinstance(
            label_column, (str, type(None))
        ), "label_column should be of type <str>."

        if score_column:
            assert (
                score_column in df_baseline.columns
            ), "score_column does not exist in df_baseline."

        if label_column:
            assert (
                label_column in df_baseline.columns
            ), "label_column does not exist in df_baseline."

        #if label_type:
        #    assert isinstance(label_type, str), "label_type should be of type <str>"
        #    assert label_type in (
        #        "categorical",
        #        "numerical",
        #    ), "label_type should be either 'categroical' or 'numerical'."

        df_baseline_ = copy.deepcopy(df_baseline)
        df_sample_ = copy.deepcopy(df_sample)

        # infer categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = [
                c
                for c in df_baseline_.columns
                if (
                    (df_baseline_.dtypes[c] == "object")
                    and (c != score_column)
                    and (c != label_column)
                )
            ]
            logger.info("Identified categorical column(s): ", categorical_columns)

        # cast categorical values as strings
        df_baseline_[categorical_columns] = df_baseline_[categorical_columns].astype(
            str
        )
        df_sample_[categorical_columns] = df_sample_[categorical_columns].astype(str)

        # infer numerical columns if not specified
        if numerical_columns is None:
            num_types = ["float64", "float32", "int32", "int64", "uint8"]
            numerical_columns = [
                c
                for c in df_baseline_.columns
                if (
                    (df_baseline_.dtypes[c] in num_types)
                    and (c != score_column)
                    and (c != label_column)
                )
            ]
            logger.info("Identified numerical column(s): ", numerical_columns)

        # cast numerical values as floats
        df_baseline_[numerical_columns] = df_baseline_[numerical_columns].astype(float)
        df_sample_[numerical_columns] = df_sample_[numerical_columns].astype(float)

        # Set attributes
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.score_column = score_column
        self.label_column = label_column

        # self.label_type = label_type

        self.df_baseline = df_baseline_
        self.df_sample = df_sample_


    def calculate_drift(
        self, 
        pre_defined_metric=None,
        user_defined_metric=None
        ):

        """
        Calculates drift between baseline and sample datasets according to
        a pre-defined metric (jensen-shannon distance or KS) or a user-defined metric.

        param: pre_defined_metric: "jensen-shannon" or "ks".
        param: user_defined_metric: function defined by user to compute drift.

        return: drift measures as computed by some metric function.
        """

        if pre_defined_metric and user_defined_metric:
            print("One of pre_defined_metric or user_defined_metric must be None.")

        elif pre_defined_metric:
            # Remove capitalization
            pre_defined_metric = pre_defined_metric.lower()

            assert pre_defined_metric in (
                "jensen-shannon",
                "ks",
            ), "pre_defined_metric should be either 'jensen-shannon' or 'ks'."

            if pre_defined_metric == "jensen-shannon":
                return js_metric(
                    df_1=self.df_baseline,
                    df_2=self.df_sample,
                    numerical_columns=self.numerical_columns,
                    categorical_columns=self.categorical_columns
                )

            elif pre_defined_metric == "ks":
                return ks_metric(
                    df_1=self.df_baseline, 
                    df_2=self.df_sample, 
                    numerical_columns=self.numerical_columns
                )
        
        # No pre_defined_metric specified - check if use_defined_metric is provided
        elif user_defined_metric:
            return user_defined_metric
        
        # Raise error
        else:
            print("A metric (user_defined or pre_defined) must be provided.")


    def plot_numerical(self, plot_numerical_columns=None, alpha=0.5):
        """
        Plots distribution of numerical features of baseline and sample datasets

        Args
        ----
        plot_numerical_columns: <list of str>
            List of numerical columns to plot, uses all if not specified

        alpha: <float>
            Transparency of the scatter plot

        Returns
        ----
        Resulting plot
        """
        assert isinstance(
            plot_numerical_columns, (list, type(None))
        ), "plot_numerical_columns should be of type list"

        if plot_numerical_columns is None:
            plot_numerical_columns = self.numerical_columns

        df_baseline = self.df_baseline[plot_numerical_columns].copy()
        df_sample = self.df_sample[plot_numerical_columns].copy()

        df_baseline["source"] = "baseline"
        df_sample["source"] = "sample"

        plot_df = pd.concat([df_baseline, df_sample])

        logger.info("Plotting the following numerical column(s):", plot_numerical_columns)

        num_numerical_features = len(plot_numerical_columns)
        column_wrap = 4
        ncols = min(num_numerical_features, column_wrap)
        nrows = 1 + (num_numerical_features - 1) // column_wrap

        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            sharex=False,
            sharey=False,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False,
        )

        indices = []
        for i in range(nrows):
            for j in range(ncols):
                indices.append((i, j))

        for i in range(len(indices) - num_numerical_features):
            axes.flat[-1 - i].set_visible(False)  # to remove plot

        cnt = 0
        for feature in plot_numerical_columns:

            sns.kdeplot(
                ax=axes[indices[cnt]],
                data=plot_df,
                x=feature,
                hue="source",
                fill=True,
                alpha=alpha,
                common_norm=False,
            )
            cnt += 1

        # plt.legend()
        plt.close(fig)

        return fig


    def plot_categorical(self, plot_categorical_columns=None):
        """Plot histograms to compare categorical columns

        Args
        ----
        plot_categorical_columns: <list of str>
            List of categorical columns to plot, uses all if no specified

        Returns
        ----
        Resulting plot

        """
        assert isinstance(
            plot_categorical_columns, (list, type(None))
        ), "plot_categorical_columns should be of type list"

        # Count distinct values in each column
        col_nunique = self.df_baseline.nunique()

        # won't plot categoricals with more than 20 values
        if plot_categorical_columns is None:
            plot_categorical_columns = [
                col
                for col in col_nunique.index
                if ((col_nunique[col] <= 20) & (col in self.categorical_columns))
            ]

        logger.info(
            "Plotting the following categorical column(s):", plot_categorical_columns
        )

        fig, ax = plt.subplots(
            len(plot_categorical_columns),
            2,
            figsize=(10, 5 * len(plot_categorical_columns)),
        )

        for i, col in enumerate(plot_categorical_columns):

            if len(plot_categorical_columns) == 1:
                _ax0 = ax[0]
                _ax1 = ax[1]
            elif len(plot_categorical_columns) > 1:
                _ax0 = ax[i, 0]
                _ax1 = ax[i, 1]

            # Get all values and counts from baseline and sample dfs
            df_baseline_values = (
                self.df_baseline[col]
                .value_counts(normalize=True, dropna=False)
                .index.values
            )
            df_sample_values = (
                self.df_sample[col]
                .value_counts(normalize=True, dropna=False)
                .index.values
            )

            # Get all unique values in the union of both lists above
            all_values = np.union1d(df_baseline_values, df_sample_values)

            # recount values in each df to include missing values in each - impute by zero
            df_baseline_values = (
                self.df_baseline[col].value_counts()[all_values].fillna(0)
            )
            df_sample_values = self.df_sample[col].value_counts()[all_values].fillna(0)

            # generate side-by-side barplots
            (
                df_baseline_values.rename("Proportion")
                .sort_index()
                .reset_index()
                .pipe((sns.barplot, "data"), x="index", y="Proportion", ax=_ax0)
            )
            _ax0.set_title(col + ", baseline")
            _ax0.set(xlabel=col)
            (
                df_sample_values.rename("Proportion")
                .sort_index()
                .reset_index()
                .pipe((sns.barplot, "data"), x="index", y="Proportion", ax=_ax1)
            )
            _ax1.set(xlabel=col)
            _ax1.set_title(col + ", sample")

        plt.close(fig)

        return fig


class ModelEvaluator:
    """
    A class to evaluate the performance of a ML model on baseline and sample datasets.

    Methods
    -------
    compare_performance:
        Compares model performance on baseline and sample datasets.

    Args
    ----
    df_baseline: <pandas.DataFrame>
        Pandas DataFrame of the baseline dataset. 

    df_sample: <pandas.DataFrame>
        Pandas DataFrame of the sample dataset.

    score_column: <str>
        Column containing predicted values (as computed by underlying model).

    label_column: <str>
        Column containing actual values (ground truths).

    label_type: <str>
        'categorical' or 'numerical' to reflect classification or regression.
    """

    def __init__(
        self,
        df_baseline,
        df_sample,
        score_column,
        label_column,
        label_type=None):
        
        assert isinstance(
            df_baseline, pd.DataFrame
        ), "df_baseline should be of type <pandas.DataFrame>."

        assert isinstance(
            df_sample, pd.DataFrame
        ), "df_baseline should be of type <pandas.DataFrame>."

        assert all(
            df_baseline.columns == df_sample.columns
        ), "df_baseline and df_sample should have the same column names."

        assert all(
            df_baseline.dtypes == df_sample.dtypes
        ), "df_baseline and df_sample should have the same column types."

        assert isinstance(
            score_column, str
        ), "score_column should be of type <str>."

        assert isinstance(
            label_column, str
        ), "label_column should be of type <str>."

        assert (
            score_column in df_baseline.columns
        ), "score_column does not exist in df_baseline."

        assert (
            label_column in df_baseline.columns
        ), "label_column does not exist in df_baseline."

        if label_type:
            assert isinstance(label_type, str), "label_type should be of type <str>"
            assert label_type in (
                "categorical",
                "numerical",
            ), "label_type should be either \
                'categroical' (classification) or 'numerical' (regression)."
        
        self.df_baseline = df_baseline
        self.df_sample = df_sample
        self.score_column = score_column
        self.label_column = label_column
        self.label_type = label_type


    def _rmse(targets, predictions):
        return np.sqrt(np.mean((predictions - targets) ** 2))


    def compare_performance(self):
        """
        A method to compare model performance on baseline and sample datasets.
        Will call _eval_classifier or _eval_regressor depending on label_type.

        param: score_column <str>: column containing predicted values.
        param: label_column <str>: column containing actual values.

        return: a DataFrame of ML metrics computed on baseline and sample datasets.
        """

        if self.label_type == "categorical":
            self._eval_classifier()

        elif self.label_type == "numerical":
            self._eval_regressor()

        return self.performance_comparison


    def _eval_regressor(self):
        """
        A funtion to compute RMSE, MAE, and R2 score on baseline and sample datasets.

        return: a Pandas DataFrame of the results indexed by data source.
        """

        y_pred_baseline = self.df_baseline[self.score_column]
        y_pred_sample = self.df_sample[self.score_column]

        y_label_baseline = self.df_baseline[self.label_column]
        y_label_sample = self.df_sample[self.label_column]

        rmse_baseline = self._rmse(y_label_baseline, y_pred_baseline)
        mae_baseline = mean_absolute_error(y_label_baseline, y_pred_baseline)
        r2_baseline = r2_score(y_label_baseline, y_pred_baseline)

        rmse_sample = self._rmse(y_label_sample, y_pred_sample)
        mae_sample = mean_absolute_error(y_label_sample, y_pred_sample)
        r2_sample = r2_score(y_label_sample, y_pred_sample)

        metrics_df = pd.DataFrame(
            {
                "RMSE": [rmse_baseline, rmse_sample],
                "MAE": [mae_baseline, mae_sample],
                "R2": [r2_baseline, r2_sample],
            },
            index=["baseline", "sample"],
        )

        self.performance_comparison = metrics_df


    def _eval_classifier(self):
        """
        A function to compute accuracy, precision, recall, F1 score, and AUC on
        baseline and sample datasets.

        return: a Pandas DataFrame of the results indexed by data source.
        """

        y_pred_baseline = self.df_baseline[self.score_column]
        y_pred_sample = self.df_sample[self.score_column]

        y_label_baseline = self.df_baseline[self.label_column]
        y_label_sample = self.df_sample[self.label_column]

        precision_baseline = precision_score(y_label_baseline, y_pred_baseline)
        recall_baseline = recall_score(y_label_baseline, y_pred_baseline)
        acc_baseline = accuracy_score(y_label_baseline, y_pred_baseline)
        f1_baseline = f1_score(y_label_baseline, y_pred_baseline)
        try:
            auc_baseline = roc_auc_score(y_label_baseline, y_pred_baseline)
        except ValueError:
            auc_baseline = "NA"

        precision_sample = precision_score(y_label_sample, y_pred_sample)
        recall_sample = recall_score(y_label_sample, y_pred_sample)
        acc_sample = accuracy_score(y_label_sample, y_pred_sample)
        f1_sample = f1_score(y_label_sample, y_pred_sample)
        try:
            auc_sample = roc_auc_score(y_label_sample, y_pred_sample)
        except ValueError:
            auc_sample = "NA"

        metrics_df = pd.DataFrame(
            {
                "Accuracy": [acc_baseline, acc_sample],
                "Precision": [precision_baseline, precision_sample],
                "Recall": [recall_baseline, recall_sample],
                "F1": [f1_baseline, f1_sample],
                "AUC": [auc_baseline, auc_sample],
            },
            index=["baseline", "sample"],
        )

        self.performance_comparison = metrics_df


class BiasMonitor:
    
    def __init__(
        self,
        df=None,
        score_column=None,
        label_column=None,
        protected_class=None,
        reference_group=None,
    ):

        self.df = df
        self.score_column = score_column
        self.label_column = label_column
        self.protected_class = protected_class
        self.reference_group = reference_group

    def compute_group_metrics(
        self, 
        pre_defined_metric=None, 
        user_defined_metric=None
        ):

        if pre_defined_metric:
            assert pre_defined_metric in (
                "aequitas_group"
            ), "pre_defined_metric should be one of ['aequitas_group']"

            if pre_defined_metric == "aequitas_group":
                return aequitas_group(
                    self.df,
                    self.score_column,
                    self.label_column,
                    self.protected_class
                )
        elif user_defined_metric:
            return user_defined_metric
        
        # Raise error
        else:
            print("A metric (user_defined or pre_defined) must be provided ")

    def compute_bias_metrics(
        self, 
        pre_defined_metric=None, 
        user_defined_metric=None
        ):

        if pre_defined_metric:
            assert pre_defined_metric in (
                "aequitas_bias"
            ), "pre_defined_metric should be one of ['aequitas_bias']"

            if pre_defined_metric == "aequitas_bias":
                return aequitas_bias(
                    self.df,
                    self.score_column,
                    self.label_column,
                    self.protected_class,
                    self.reference_group
                )
        elif user_defined_metric:
            return user_defined_metric
        
        # Raise error
        else:
            print("A metric (user_defined or pre_defined) must be provided ")





def ks_metric(df_1, df_2, numerical_columns):
    ks_tests = [
        ks_2samp(
            data1=df_1.loc[:, feat],
            data2=df_2.loc[:, feat],
        )
        for feat in numerical_columns
    ]
    pvalues = [x[1] for x in ks_tests]
    list_of_pval = [f"{feat}_p-value" for feat in numerical_columns]

    ks_pvalues = dict(zip(list_of_pval, pvalues))

    return ks_pvalues


def js_metric(df_1, df_2, numerical_columns, categorical_columns):
    """
    A function to compute the jensen-shannon distances between columns of
    similar DataFrames.

    For categorical columns, the probability of each category will be
    computed separately for `df_baseline` and `df_sample`, and the Jensen
    Shannon distance between the 2 probability arrays will be computed. 
    
    For numerical columns, the values will first be fitted into a gaussian KDE
    separately for `df_baseline` and `df_sample`, and a probability array
    will be sampled from them and compared with the Jensen Shannon distance.

    param: df_1: baseline DataFrame
    param: df_2: sample DataFrame
    param: numerical_columns: list of numerical columns
    param: categorical_columns: list of categorical columns

    return: sorted list of tuples containing the column names and Jensen-Shannon 
    distances.
    """

    res = {}
    STEPS = 100

    for col in categorical_columns:
        # to ensure similar order, concat before computing probability
        col_baseline = df_1[col].to_frame()
        col_sample = df_2[col].to_frame()
        col_baseline["source"] = "baseline"
        col_sample["source"] = "sample"

        col_ = pd.concat([col_baseline, col_sample], ignore_index=True)

        # aggregate and convert to probability array
        arr = (
            col_.groupby([col, "source"])
            .size()
            .to_frame()
            .reset_index()
            .pivot(index=col, columns="source")
            .droplevel(0, axis=1)
        )
        arr_ = arr.div(arr.sum(axis=0), axis=1)
        arr_.fillna(0, inplace=True)

        # calculate js distance
        js_distance = jensenshannon(
            arr_["baseline"].to_numpy(), arr_["sample"].to_numpy()
        )

        res.update({col: js_distance})

    for col in numerical_columns:
        # fit gaussian_kde
        col_baseline = df_1[col]
        col_sample = df_2[col]
        kde_baseline = gaussian_kde(col_baseline)
        kde_sample = gaussian_kde(col_sample)

        # get range of values
        min_ = min(col_baseline.min(), col_sample.min())
        max_ = max(col_baseline.max(), col_sample.max())
        range_ = np.linspace(start=min_, stop=max_, num=STEPS)

        # sample range from KDE
        arr_baseline_ = kde_baseline(range_)
        arr_sample_ = kde_sample(range_)

        arr_baseline = arr_baseline_ / np.sum(arr_baseline_)
        arr_sample = arr_sample_ / np.sum(arr_sample_)

        # calculate js distance
        js_distance = jensenshannon(arr_baseline, arr_sample)

        res.update({col: js_distance})

    list_output = sorted(res.items(), key=lambda x: x[1], reverse=True)
    dict_output = dict(list_output)

    return dict_output


def aequitas_group(df, score_column, label_column, protected_class):
    # To measure Bias towards protected_class, filter DataFrame
    # to score, label (ground truth), and protected class
    data_scored = df[
        [
            score_column,
            label_column,
            protected_class,
        ]
    ]

    # Aequitas expects ground truth under 'label_value'
    data_scored = data_scored.rename(columns={label_column: "label_value"})

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Group Metrics
    g = Group()
    xtab, _ = g.get_crosstabs(data_scored_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = g.list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ["attribute_name", "attribute_value"] + absolute_metrics
    ].round(2)

    # For example:
    """
        attribute_name  attribute_value     tpr     tnr  ... precision
    0   gender          female              0.60    0.88 ... 0.75
    1   gender          male                0.49    0.90 ... 0.64
    """

    return absolute_metrics_df


def aequitas_bias(df, score_column, label_column, protected_class, reference_group):
    
    # To measure Bias towards protected_class, filter DataFrame
    # to score, label (ground truth), and protected class
    data_scored = df[
        [
            score_column,
            label_column,
            protected_class,
        ]
    ]

    data_scored = data_scored.rename(columns={label_column: "label_value"})

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Bias Metrics
    b = Bias()
    g = Group()
    xtab, _ = g.get_crosstabs(data_scored_processed)

    # Disparities calculated in relation <protected_class> for class groups
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={protected_class: reference_group},
        alpha=0.05,
        mask_significance=True,
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = b.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ["attribute_name", "attribute_value"] + calculated_disparities
    ]

    # For example:
    """
        attribute_name	attribute_value    ppr_disparity   precision_disparity
    0   gender          female             0.714286        1.41791
    1   gender          male               1.000000        1.000000
    """

    return disparity_metrics_df