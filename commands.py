import fire

from bee_vs_wasp.train import train as hydra_train


class CLI:
    def train(self) -> None:
        hydra_train()


def main() -> None:
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
