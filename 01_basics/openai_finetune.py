#
# data preparation
#

import json

training_data = [{"prompt": "빨강이", "completion": "Data Scientist"},
{"prompt": "주황이", "completion": "Developer"},
{"prompt": "노랑이", "completion": "Developer"},
{"prompt": "초록이", "completion": "Developer"},
{"prompt": "파랑이", "completion": "Developer"},
{"prompt": "검둥이", "completion": "Data Scientist"},
{"prompt": "보랑이", "completion": "Developer"},
{"prompt": "남둥이", "completion": "Developer"},
{"prompt": "하늘이", "completion": "Data Scientist"},
{"prompt": "개나리", "completion": "Data Scientist"}]

file_name = "training_data.jsonl"
with open(file_name, "w", encoding="utf-8") as output_file:
    for entry in training_data:
        json.dump(entry, output_file)
        output_file.write("\n")


#
# finetune
#

from openai import OpenAI
# client generation
client = OpenAI(api_key="sk-")

# file upload
upload_response = client.files.create(
    file=open(file_name, "rb"),
    purpose='fine-tune',
)

file_id = upload_response.id

# model generation for finetune
fine_tune_response = client.fine_tuning.jobs.create(
      training_file=file_id,
      model="davinci-002"
)

# fine-tuning job 리스트 10개 나열
# client.fine_tuning.jobs.list(limit=10)

# fine-tuning 상태 확인
# client.fine_tuning.jobs.retrieve(fine_tune_response.id) 

# fine-tuning job에서 10개의 이벤트 나열
# fine_tune_events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tune_response.id)

#
# inference
#

completion = client.completions.create(
    model="ft:davinci-002:personal::9B2EojTR",
    prompt="주황이는?")
print(completion.choices[0].text)