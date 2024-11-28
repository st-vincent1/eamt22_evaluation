import json

json_data = []
files = {}
split = "detector_sample"
for suffix in ["cxt", "en", "pl", "gold"]:#"mark", "context", "context_excluding_formality", "formality_text"]:
    with open(f"tester/{split}.{suffix}", "r") as file:
        files[suffix] = file.read().splitlines()

for i in range(len(files["cxt"])):
    json_data.append(
        {
            "english": files["en"][i],
            "polish": files["pl"][i],
            "detected_context_csv": files["cxt"][i],
            "manually_labelled_context_csv": files["gold"][i],
            # "marking": files["mark"][i],
            # "context_text": files["context"][i],
            # "context_text_excluding_formality": files["context_excluding_formality"][i],
            # "formality_text": files["formality_text"][i],
        }
    )

with open(f"data/detector_testing_sample.json", "w") as file:
    json.dump(json_data, file, indent=4, ensure_ascii=False)