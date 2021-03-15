import json
import os
import urllib

import requests


base_url = "http://internal-a2297ab0-wttest-gatewaying-d216-1674145052.us-east-2.elb.amazonaws.com/"
model_filename = "model.py"
job_filename = "full_drift_job.json"
model_filename = "monitor.py"

# URLs
job_endpoint = urllib.parse.urljoin(base_url, "model-manage/api/jobs")

# Read files
with open(job_filename, "r") as f:
    job_json = f.read()
payload = json.loads(job_json)

if model_filename:
    # Update JSON payload
    for asset in payload["model"]["storedModel"]["modelAssets"]:
        if "primaryModelSource" in asset:
            if asset["primaryModelSource"]:
                model_source_object = asset
                break
    else:
        raise ValueError("No primary model source found in job object")

    with open(model_filename, "r") as f:
            model_source = f.read()
    model_source_object["name"] = model_filename
    model_source_object["sourceCode"] = model_source

# Create job
response = requests.post(job_endpoint, json=payload)
if not response.ok:
    raise Exception(
        "Job create error: ({}) {}".format(response.status_code, response.text)
    )

returned_job = json.loads(response.text)
job_id = returned_job["id"]
print("Job id: {}".format(job_id))

job_url = urllib.parse.urljoin(base_url, "#/jobs/{}".format(job_id))
os.system("open {}".format(job_url))