"""Train the label-first Rank1 variant reported in Table 7.

At inference, use the resulting adapter with ``rank0`` so generation
stops after the relevance label.
"""

import argparse

from .common import add_training_arguments, load_training_dataset, train_lora_adapter


def convert_to_inverse_format(row):
    reasoning, label = row["output"].split("</think>", maxsplit=1)
    completion = f"{label.strip()}\n{reasoning.strip()}\n</think>"
    return {
        "prompt": [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Rank1 with label-first inverse training"
    )
    add_training_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = load_training_dataset(args.sft_data_path, convert_to_inverse_format)
    train_lora_adapter(args, train_data, "rank1_inverse")


if __name__ == "__main__":
    main()
