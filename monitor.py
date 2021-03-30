from typing import List

import pandas as pd
from pyspark.sql import SparkSession

from moc_monitors import DriftDetector
from moc_schema_infer import set_detector_parameters


# modelop.init
def begin():
    print("Begin function...")
    global SPARK
    SPARK = SparkSession.builder.appName("DriftTest").getOrCreate()

    # Read schema
    global SCHEMA
    SCHEMA = pd.read_json("df_sample_scored_input_schema.avsc", orient="records")
    # set schema index to be "name"
    SCHEMA.set_index("name", inplace=True)


# modelop.metrics
def metrics(external_inputs: List, external_outputs: List, external_model_assets: List):
    print("Metrics function...")

    comparator_path, baseline_path, output_path = parse_assets(
        external_inputs, external_outputs, external_model_assets
    )

    # Load the assets as Pandas dataframes
    baseline_df = SPARK.read.option("header", "true").csv(baseline_path).toPandas()
    comparator_df = SPARK.read.option("header", "true").csv(comparator_path).toPandas()

    print("Detecting drift...")
    print("Baseline df:")
    print(baseline_df)
    print("Comparator df:")
    print(comparator_df)

    # Run the monitoring to detect drift
    detector_parameters = set_detector_parameters(SCHEMA)

    drift_detector = DriftDetector(
        df_baseline=baseline_df,
        df_sample=comparator_df,
        categorical_columns=detector_parameters["categorical_columns"],
        numerical_columns=detector_parameters["numerical_columns"],
        score_column=detector_parameters["score_column"][0],
        label_column=detector_parameters["label_column"][0],
    )

    output = drift_detector.calculate_drift(
        pre_defined_metric="jensen-shannon", user_defined_metric=None
    )

    print("Monitor output:")
    print(output)

    # Spark doesn't like numpy float values, so convert the output to
    # pandas before casting to a Spark Dataframe
    #
    # But, pandas needs the dict to have list values, so first we need
    # to wrap each value in a list
    print("Converting drift output to Spark...")
    output_as_lists = {k: [v] for k, v in output.items()}
    output_pandas_df = pd.DataFrame.from_dict(output_as_lists)

    # Cast to Spark dataframe
    output_df = SPARK.createDataFrame(output_pandas_df)
    print("Spark metrics output:")
    output_df.show()

    print("Writing output to", output_path)
    # Use coalesce() so that the output is a single file for easy reading
    output_df.coalesce(1).write.mode("overwrite").option("header", "true").format(
        "json"
    ).save(output_path)

    SPARK.stop()


def parse_assets(
    external_inputs: List, external_outputs: List, external_model_assets: List
):
    """
    Returns a tuple of (comparator_path, baseline_path, output_path) for paths to HDFS
    assets
    """
    print("Parsing assets...")

    ### Input assets
    # Grab input assets from arguments
    comparator_assets = [
        a for a in external_inputs if a["assetRole"] == "COMPARATOR_DATA"
    ]
    if len(comparator_assets) != 1:
        raise ValueError(
            "There must be one item in input assets with COMPARATOR_DATA role, found {}".format(
                len(comparator_assets)
            )
        )
    comparator_asset = comparator_assets[0]

    baseline_assets = [a for a in external_inputs if a["assetRole"] == "TRAINING_DATA"]
    if len(baseline_assets) != 1:
        raise ValueError(
            "There must be one item in input assets with TRAINING_DATA role, found {}".format(
                len(baseline_assets)
            )
        )
    baseline_asset = baseline_assets[0]

    # If either asset is a JSON, error out because Spark doesn't like JSON
    if ("fileFormat" in comparator_asset) and (
        comparator_asset["fileFormat"] == "JSON"
    ):
        raise ValueError("Comparator data file format is set as JSON but must be CSV")
    if ("fileFormat" in baseline_asset) and (baseline_asset["fileFormat"] == "JSON"):
        raise ValueError("Baseline data file format is set as JSON but must be CSV")

    # Pull the HDFS file paths for the assets
    comparator_path = comparator_asset["fileUrl"]
    baseline_path = baseline_asset["fileUrl"]

    ### Output asset
    # Grab the output asset
    if len(external_outputs) != 1:
        raise ValueError("There must be one output asset, found 0")
    output_path = external_outputs[0]["fileUrl"]

    return (comparator_path, baseline_path, output_path)
