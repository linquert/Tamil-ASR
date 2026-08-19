from tamil_asr.cli.staged_asr import run_stage_cli
from tamil_asr.training.staged_asr.recipe import JOINT_RECIPE


def main() -> None:
    run_stage_cli(JOINT_RECIPE)


if __name__ == "__main__":
    main()
