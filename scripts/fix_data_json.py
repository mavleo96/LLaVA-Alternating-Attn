"""Add missing <image> tokens to the first human turn in LLaVA-style JSON.

Given a JSON file containing a list of records with a `conversations` field, this script:
  1) Loads the JSON file from `--json_path`.
  2) For each record, checks the first conversation turn (`conversations[0]["value"]`).
  3) If the string does not already contain `<image>`, it prepends `<image>\\n`.
The modified JSON overwrites the input file in place and the script reports how many
records were changed.
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    fix_count = 0
    for record in data:
        if "<image>" not in record["conversations"][0]["value"]:
            record["conversations"][0]["value"] = "<image>\n" + record["conversations"][0]["value"]
            fix_count += 1

    print(f"Fixed {fix_count} records")

    with open(args.json_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()