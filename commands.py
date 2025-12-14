import fire

from bee_vs_wasp.train import train


def main() -> None:
    fire.Fire({"train": train})


if __name__ == "__main__":
    main()
