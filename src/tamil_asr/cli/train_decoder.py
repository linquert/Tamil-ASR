from tamil_asr.cli.staged_asr import run_stage_cli
from tamil_asr.training.staged_asr.recipe import DECODER_RECIPE


def main() -> None:
    run_stage_cli(DECODER_RECIPE)


if __name__ == "__main__":
    main()
