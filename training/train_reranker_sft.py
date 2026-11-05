"""Train Rank0 or the matched Rank1 implementation."""

import argparse

from .common import add_training_arguments, load_training_dataset, train_lora_adapter


def convert_to_conversation(row, remove_reasoning):
    assistant_response = row["output"]
    if remove_reasoning:
        try:
            assistant_response = assistant_response.split("</think>", maxsplit=1)[1].strip()
        except IndexError as error:
            raise ValueError("Training output is missing a </think> tag") from error

    return {
        "prompt": [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ],
        "completion": [{"role": "assistant", "content": assistant_response}],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Rank0 or the matched Rank1 implementation"
    )
    add_training_arguments(parser)
    parser.add_argument(
        "--remove_reasoning",
        action="store_true",
        help="Train Rank0 by removing reasoning from each target",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_data = load_training_dataset(
        args.sft_data_path,
        lambda row: convert_to_conversation(row, args.remove_reasoning),
    )
    variant = "rank0" if args.remove_reasoning else "rank1_ours"
    train_lora_adapter(args, train_data, variant)


if __name__ == "__main__":
    main()
