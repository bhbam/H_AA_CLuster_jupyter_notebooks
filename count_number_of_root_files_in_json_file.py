import argparse
import json


def count_files(file_path):
  with open(file_path, "r") as f:
    root_files = json.load(f)
  print(f"Total files in '{file_path}': {len(root_files)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Count ROOT file paths in a JSON file."
  )
  parser.add_argument(
      "--input_file", "-i", type=str, help="Path to the input .json file"
  )

  args = parser.parse_args()
  count_files(args.input_file)
