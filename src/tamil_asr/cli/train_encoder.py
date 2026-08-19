from tamil_asr.cli.staged_asr import run_stage_cli
from tamil_asr.training.staged_asr.recipe import ENCODER_RECIPE


def main() -> None:
    run_stage_cli(ENCODER_RECIPE)


if __name__ == "__main__":
    main()
