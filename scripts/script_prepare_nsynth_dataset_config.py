import os
import json
from pprint import pprint


NSYNTH_DATASET_PATH = "/home/sarah/Projects/master_atiam/pam/nsynth-test"
DATA_DIR = "/home/sarah/Projects/master_atiam/pam/deep-eurorack-control/data"


if __name__ == "__main__":
    instruments = ["bass", "string"]

    with open(os.path.join(NSYNTH_DATASET_PATH, "examples.json"), "r") as f:
        data = json.load(f)

    data_strings = {k: v for k, v in data.items() if k.startswith("string")}
    print(f"nb samples : {len(data_strings.keys())}")
    pprint(data_strings["string_acoustic_056-047-075"])

    with open(os.path.join(DATA_DIR, "nsynth_string.json"), "w", encoding='utf-8') as f:
        json.dump(data_strings, f, indent=4)
