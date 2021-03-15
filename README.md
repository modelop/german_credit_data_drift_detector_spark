# Spark Drift Monitor Job

TODO: upload to github https://github.com/modelop/german_credit_drift_detector_spark
add sami as admin and core as readers

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

