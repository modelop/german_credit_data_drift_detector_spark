# German Credit Spark Data Drift Monitor Example

This repo is an example Spark data drift monitor model that is conformed for use with ModelOp Center and the ModelOp Spark Runtime Service.

This model is intended to be an associated model for the [mehri-odg/german_credit_python](https://github.com/merhi-odg/german_credit_python) base model. The base model is not a Spark model, but this associated model is all that is run for data drift monitoring, so it can still be run in Spark.

## Assets

Below are the assets that are used to run this example:

| Asset Type | Repo File | HDFS Path | Description |
| --- | --- | --- | --- |
| Model Schema | `df_sample_scored_input_schema.avsc` | n/a (copied via Spark runtime service) | Input schema for the model. Copied from the base model input schema. |
| Input Asset (Baseline/Training Data) | `df_baseline_scored.csv` | `/hadoop/demo/german_credit_model/df_baseline_scored.csv` | An attachment on the base model snapshot. Input file for the model `metrics()` function. The HDFS path can vary based on the `external_inputs` param of the `metrics()` function  |
| Input Asset (Comparator Data) | `df_sample_scored.csv` | `/hadoop/demo/german_credit_model/df_sample_scored.csv` | An associated asset. Input file for the model `metrics()` function. The HDFS path can vary based on the `external_inputs` param of the `metrics()` function  |
| Output Asset | n/a | `/hadoop/demo/german_credit_model/drift_analysis.csv` | Output from the model `metrics()` function that the MLC consumes to display on the ModelOp Center UI. The HDFS path can vary based on the `OUTPUT_FILE` `fileUrl` that is passed to the MLC when fired by API. |

## Testing the Model

1. Verify that your ModelOp Center instance has at least one Spark runtime service connected to an existing Spark cluster
2. Verify that the Spark runtime service has the tag `test`
3. Import this model into ModelOp Center via GitHub link
4. Create a snapshot of this model. Select "Spark" for the runtime type and (optionally) select a Spark runtime service
5. Import the base model [mehri-odg/german_credit_python](https://github.com/merhi-odg/german_credit_python)
6. Add the training data asset as a URL HDFS attachment to the base model.
    - URL: `hdfs:///hadoop/demo/german_credit_model/df_baseline_scored.csv`
7. Create a snapshot of the base model
   - (Optional) Select the ModelOp Runtime as the runtime type for the base model
   - Add the snapshot of the associated model (this repo) as an associated model
   - Set the associated model type to "Data Drift Model"
   - Add the comparator data as an associated asset HDFS URL
       - URL: `hdfs:///hadoop/demo/german_credit_model/df_sample_scored.csv`
   - (Optional) Provide a DMN file for the MLC to parse with the test results
8. Record the UUID of the base model snapshot
9. Create a drift detection job by firing the CronTriggeredDataDriftTest MLC using the URL and JSON payload below:
    - URL: http://mlc.mocaasin.modelop.center/rest/signal
    - JSON payload (**note that `MODEL_ID.value` must be updated**):
      ```json
      {
          "name": "com.modelop.mlc.definitions.Signals_MODEL_DATA_DRIFT_TEST",
          "variables": {
              "MODEL_ID": {
                  "value": "5a31b3c6-f033-4948-b433-bae6656baa26"
              },
              "OUTPUT_FILE": {
                  "value": {
                      "assetType": "EXTERNAL_FILE",
                      "assetRole": "UNKNOWN",
                      "fileUrl": "hdfs:///hadoop/demo/german_credit_model/drift_analysis.csv",
                      "filename": "drift_analysis.csv",
                      "fileFormat": "CSV",
                      "repositoryInfo": {
                          "repositoryType": "HDFS_REPOSITORY",
                          "host": "",
                          "port": 0
                      }
                  }
              }
          }
      }
      ```
10. Wait for the created job to enter a `COMPLETE` state
11. Navigate to the base model snapshot's test results and verify that the test results look similar:
    ```json
    {
      "number_existing_credits": 0.1661573547955825,
      "number_people_liable": 0.15643137378956762,
      "score": 0.1374666427619285,
      "label_value": 0.12909200180788885,
      "present_residence_since": 0.09589797018577487,
      "installment_rate": 0.0923431071984353,
      "purpose": 0.08901857345091589,
      "credit_amount": 0.06579786015199766,
      "age_years": 0.062297942889786954,
      "present_employment_since": 0.06092273471356515,
      "duration_months": 0.05571648405502104,
      "savings_account": 0.04709760284899042,
      "credit_history": 0.03574135542427669,
      "property": 0.03477985804578699,
      "telephone": 0.026242637204975147,
      "job": 0.024366893631309054,
      "foreign_worker": 0.01806487817678789,
      "checking_status": 0.01600519274897279,
      "installment_plans": 0.015781433298469323,
      "housing": 0.010258094196074643,
      "debtors_guarantors": 0.004689237456452252
    }
    ```

### Manual Tests

The `job_submit` folder includes files that can send a job to MOC directly. you can update the global environment variables in `job_submit/submit.py` to point to your instance and run different types of jobs:

```python
BASE_URL = "http://my.modelop.center.instance"
MODEL_FILENAME = "monitor.py"
JOB_FILENAME = "job_submitter/german_credit_data_drift_job.json"
```

Then run `submit.py` to submit the job:

```
python3 job_submitter/submit.py
```
