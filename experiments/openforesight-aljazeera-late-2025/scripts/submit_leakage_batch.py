"""Submit leakage batch to OpenAI."""
import sys
from openai import OpenAI
client = OpenAI()

# Upload file
f = client.files.create(
    file=open("/data/eval/aljz/leakage_batch_dayminus1.jsonl","rb"),
    purpose="batch",
)
print(f"Uploaded file: {f.id}")

# Submit batch
b = client.batches.create(
    input_file_id=f.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description":"dayminus1 leakage check"},
)
print(f"Batch ID: {b.id}")
print(f"Status: {b.status}")
with open("/data/eval/aljz/leakage_batch_id.txt","w") as fh:
    fh.write(b.id)
