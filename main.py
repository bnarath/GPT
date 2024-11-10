import argparse
import logging
from nano_gpt.run import Nano_GPT


def _parse_args():
    arg_parser = argparse.ArgumentParser(conflict_handler="resolve")
    arg_parser.add_argument(
        "--option",
        dest="option",
        required=False,
        choices=["nano_gpt"],
        default="nano_gpt",
        help="Choice of GPT",
    )
    arg_parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default=1e-3,
        help="Learning rate of optimizer",
    )
    arg_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        required=False,
        type=int,
        default=32,
        help="No. of batches in each epoch",
    )
    arg_parser.add_argument(
        "--block_size",
        dest="block_size",
        required=False,
        type=int,
        default=8,
        help="Chuck or sequence size - time dimension",
    )
    arg_parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        required=False,
        type=int,
        default=3000,
        help="No. of epochs",
    )
    arg_parser.add_argument(
        "--max_new_tokens",
        dest="max_new_tokens",
        required=False,
        type=int,
        default=400,
        help="No of new tokens to generate",
    )
    arg_parser.add_argument(
        "--eval_interval",
        dest="eval_interval",
        required=False,
        type=int,
        default=300,
        help="Epoch interval for evaluation",
    )
    arg_parser.add_argument(
        "--eval_iters",
        dest="eval_iters",
        required=False,
        type=int,
        default=300,
        help="No of batches to consider for train and test evaluation to get mean loss value (to avoid noisy loss)",
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = _parse_args()
    logging.info(f"Arguments: {args}")
    if args.option == "nano_gpt":
        nano_gpt = Nano_GPT(
            lr=args.lr,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_epochs=args.num_epochs,
            max_new_tokens=args.max_new_tokens,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
        )
        nano_gpt.run()
