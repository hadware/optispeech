import argparse
import csv
from pathlib import Path

from optispeech.utils import get_script_logger

log = get_script_logger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "original_dataset_dir",
        type=Path,
        help="Original dataset dir",
    )
    parser.add_argument(
        "preprocessed_dataset_dir",
        type=Path,
        help="Preprocessed dataset dir",
    )
    args = parser.parse_args()
    dataset_folder : Path = args.original_dataset_dir
    preprocessed_folder : Path = args.preprocessed_dataset_dir
    splits = [dataset_folder / split_name for split_name in ["train", "val"]]
    all_data_files_ids = {p.stem for p in (preprocessed_folder / "data").glob("*.npz")}

    for split in splits:

        with (split / "metadata.csv").open() as split_file:
            csv_reader = csv.reader(split_file, delimiter="|")
            split_file_ids = {r[0] for r in csv_reader if r}
        kept_ids = split_file_ids & all_data_files_ids

        split_index_filepath = preprocessed_folder / f"{split.stem}.txt"
        with split_index_filepath.open("w") as split_index_file:
            for kept_id in kept_ids:
                kept_id = preprocessed_folder.absolute() / "data" / f"{kept_id}"
                split_index_file.write(f"{kept_id}\n")

    log.info("Process done!")


if __name__ == "__main__":
    main()
