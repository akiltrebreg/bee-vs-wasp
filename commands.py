import fire

from bee_vs_wasp.infer import InferConfig
from bee_vs_wasp.infer import infer as infer_fn
from bee_vs_wasp.train import train as train_fn


class BeeCLI:
    """Command-line interface for bee-vs-wasp project."""

    @staticmethod
    def train():
        """Start training process."""
        train_fn()

    @staticmethod
    def infer(model_path: str, dataset_root: str = "./data"):
        """Run inference on test dataset."""
        cfg = InferConfig(model_path=model_path, dataset_root=dataset_root)
        infer_fn(cfg)


def main():
    """Main entry point."""
    fire.Fire(BeeCLI)


if __name__ == "__main__":
    main()
