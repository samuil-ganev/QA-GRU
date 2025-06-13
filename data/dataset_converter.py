import json

input_file = "train-v2.0.json"
output_file = "raw_dataset.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as out_f:
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa.get("is_impossible", False):
                    question = qa["question"].strip()

                    if qa["answers"]:
                        answer = qa["answers"][0]["text"].strip()
                        out_f.write(f"{question}\t{answer}\n")

print(f"Извлечени въпроси и отговори са записани в: {output_file}")
