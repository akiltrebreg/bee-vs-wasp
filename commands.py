import subprocess
import sys

import fire


class BeeCLI:
    """Command-line interface for bee-vs-wasp project."""

    @staticmethod
    def train(*overrides):
        """Start training process with optional Hydra config overrides.

        Args:
            *overrides: Hydra config overrides (e.g., num_epochs=1, model=simple_cnn)

        Examples:
            python commands.py train num_epochs=1 data.batch_size=8
            python commands.py train model=simple_cnn num_epochs=1
        """
        cmd = ["python", "-m", "bee_vs_wasp.train", *overrides]
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)


def main():
    """Main entry point."""
    fire.Fire(BeeCLI)


if __name__ == "__main__":
    main()
