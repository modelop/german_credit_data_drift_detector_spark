import json
import os
import urllib

import requests


BASE_URL = "http://mocaasin.modelop.center"
MODEL_FILENAME = "monitor.py"
JOB_FILENAME = "job_submitter/german_credit_data_drift_job.json"

###

req_verify = False
if "mocaasin" in BASE_URL:
    from oauth2 import OAuth2

    oauth_url = BASE_URL + "/gocli/token"
    req_auth = OAuth2(oauth_url, "john", "john-secret")
else:
    req_auth = None

# Set url
job_endpoint = urllib.parse.urljoin(BASE_URL, "model-manage/api/jobs")

# Read files
with open(JOB_FILENAME, "r") as f:
    job_json = f.read()
payload = json.loads(job_json)

if MODEL_FILENAME:
    # Update JSON payload
    for asset in payload["model"]["storedModel"]["modelAssets"]:
        if "primaryModelSource" in asset:
            if asset["primaryModelSource"]:
                model_source_object = asset
                break
    else:
        raise ValueError("No primary model source found in job object")

    with open(MODEL_FILENAME, "r") as f:
        model_source = f.read()
    model_source_object["name"] = MODEL_FILENAME
    model_source_object["sourceCode"] = model_source

# Create job
response = requests.post(job_endpoint, json=payload, auth=req_auth, verify=req_verify)
if not response.ok:
    raise Exception(
        "Job create error: ({}) {}".format(response.status_code, response.text)
    )

returned_job = json.loads(response.text)
job_id = returned_job["id"]
print("Job id: {}".format(job_id))

job_url = urllib.parse.urljoin(BASE_URL, "#/jobs/{}".format(job_id))
os.system("open {}".format(job_url))
