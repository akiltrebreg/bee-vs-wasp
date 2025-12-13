import fire

from .infer import infer  # noqa: F401
from .train import train  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
