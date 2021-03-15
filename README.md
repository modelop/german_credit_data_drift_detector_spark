# Spark Drift Monitor Job

TODO:
- add rest of files from german_credit_data_drift_detector repo
- explanation of repo
- layout of files
- how to run this example (section is below)

## MLC Discussion

- Output **folder** instead of file?
- Need test results to show up for main model

## MLC Trigger

```json
{
  "name": "com.modelop.mlc.definitions.Signals_MODEL_DATA_DRIFT_TEST",
  "variables": {
    "MODEL_ID": {
      "value": "f0ff95d4-fc87-4098-ab84-d5bf873a5449"
    }
  }
}
```

http://internal-a2297ab0-wttest-gatewaying-d216-1674145052.us-east-2.elb.amazonaws.com/mlc-service/rest/signal

## Putting Files in Hadoop

```shell
k cp df_baseline_scored.csv $(getpod cluster):/home/cloudera/.
k cp df_sample_scored.csv $(getpod cluster):/home/cloudera/.
hadoop fs -mkdir -p /hadoop/demo/german_credit_model/
hadoop fs -put *.csv /hadoop/demo/german_credit_model
```

## Set up to run this

- import model
- add input attachments (roles)
- add output **folder** (with tag?)
- upload files per attachment details
- run with mlc? cron?
- download: `hadoop fs -getmerge /hadoop/demo/german_credit_model/drift_detector_output.csv/* drift_detector_output.csv` && `k cp`

