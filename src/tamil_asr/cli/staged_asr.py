from __future__ import annotations

import argparse
from pathlib import Path

from tamil_asr.training.staged_asr.executor import execute
from tamil_asr.training.staged_asr.recipe import StageRecipe
from tamil_asr.training.staged_asr.run import DataPaths, ModelPaths, RuntimeOptions, StageRun
from tamil_asr.training.staged_asr.preflight import preflight_stage


def _parser(recipe: StageRecipe) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Staged ASR {recipe.stage.order + 1}: {recipe.stage.phase_name.replace('_', ' ')}"
    )
    paths = parser.add_argument_group("required paths")
    paths.add_argument("--model", required=True, help="Qwen3-ASR base model directory")
    paths.add_argument("--qwen-repo", required=True, help="Local Qwen3-ASR Python checkout")
    paths.add_argument("--train-manifest", required=True, help="Audited training manifest")
    paths.add_argument("--validation-manifest", required=True, help="Disjoint audited validation manifest")
    paths.add_argument("--output-dir", required=True)

    lineage = parser.add_mutually_exclusive_group()
    if recipe.requires_previous_stage is not None:
        lineage.add_argument(
            "--previous-stage",
            default="",
            help=f"Completed {recipe.requires_previous_stage.value} checkpoint",
        )
    lineage.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default="",
        help="Resume this stage; omit the path to use the latest checkpoint in --output-dir",
    )

    runtime = parser.add_argument_group("runtime limits")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--sample-rate", type=int, default=16_000)
    runtime.add_argument("--max-batch-audio-seconds", type=float, default=20.0)
    runtime.add_argument("--max-batch-size", type=int, default=8)
    runtime.add_argument("--gradient-accumulation", type=int, default=8)
    runtime.add_argument("--num-workers", type=int, default=2)
    runtime.add_argument("--prefetch-factor", type=int, default=2)
    runtime.add_argument("--log-every", type=int, default=10)
    runtime.add_argument("--eval-every", type=int, default=200)
    runtime.add_argument("--save-every", type=int, default=200)
    runtime.add_argument("--eval-examples", type=int, default=32)
    runtime.add_argument("--max-new-tokens", type=int, default=192)
    runtime.add_argument("--max-steps", type=int, default=None, help="Explicit smoke/debug cap only")
    runtime.add_argument(
        "--prompt",
        default="Transcribe the Tamil speech exactly. Output only the transcription.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight",
        action="store_true",
        help="Run one real forward/backward contract check without an optimizer step",
    )
    action.add_argument(
        "--allow-training",
        action="store_true",
        help="Required guard: this command performs optimizer updates",
    )
    return parser


def _run_from_args(args: argparse.Namespace) -> StageRun:
    previous_value = getattr(args, "previous_stage", "")
    previous = Path(previous_value) if previous_value else None
    resume: Path | str | None = args.resume
    if resume and resume != "latest":
        resume = Path(resume)
    return StageRun(
        model=ModelPaths(base_model=Path(args.model), qwen_repository=Path(args.qwen_repo)),
        data=DataPaths(
            train_manifest=Path(args.train_manifest),
            validation_manifest=Path(args.validation_manifest),
            prompt=args.prompt,
        ),
        runtime=RuntimeOptions(
            output_dir=Path(args.output_dir),
            seed=args.seed,
            sample_rate=args.sample_rate,
            max_batch_audio_seconds=args.max_batch_audio_seconds,
            max_batch_size=args.max_batch_size,
            gradient_accumulation=args.gradient_accumulation,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            log_every=args.log_every,
            evaluate_every=args.eval_every,
            save_every=args.save_every,
            evaluation_examples=args.eval_examples,
            max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps,
        ),
        previous_stage=previous,
        resume=resume,
    )


def run_stage_cli(recipe: StageRecipe) -> None:
    args = _parser(recipe).parse_args()
    stage_run = _run_from_args(args)
    if args.preflight:
        preflight_stage(recipe, stage_run)
    else:
        execute(recipe=recipe, run=stage_run, allow_training=args.allow_training)
