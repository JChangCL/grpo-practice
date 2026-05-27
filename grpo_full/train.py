from __future__ import annotations

import argparse

from grpo_full.config import load_config
from grpo_full.grpo import GRPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
