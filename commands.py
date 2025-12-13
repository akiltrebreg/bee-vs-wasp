import fire

from bee_vs_wasp.infer import infer  # noqa: F401
from bee_vs_wasp.train import train  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
