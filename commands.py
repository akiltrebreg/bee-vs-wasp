import subprocess

import fire


def train():
    """
    Fire wrapper to run the Hydra-decorated train function
    """
    subprocess.run(["python", "-m", "bee_vs_wasp.train"], check=True)


if __name__ == "__main__":
    fire.Fire({"train": train})
