#!/usr/bin/env python3
"""Optuna search wrapper for classify_gnn.py.

This script tunes a small architectural search space by launching the existing
graph classifier as a subprocess for each trial, then reading the saved
final_results.json artifact for the objective value.

Example:
    python optuna_classify_gnn.py \
      --n-trials 20 \
      --models gcn gin sage gat \
      --poolings mean max attention \
      --hidden-dims 64 128 256 512 \
      --min-layers 2 \
      --max-layers 6 \
      --device 0 \
      -- --data-dir /home/ogharbi/datasets/SHREC25/ssl_graphs_1k \
         --feature-preset essential \
         --epochs 200 \
         --patience 20 \
         --batch-size 64 \
         --num-workers 4 \
         --lr 0.001 \
         --weight-decay 0.0001
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import optuna
from optuna.samplers import TPESampler


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT / 'optuna_classify_gnn.db'}"
DEFAULT_CLASSIFY_SCRIPT = PROJECT_ROOT / 'classify_gnn.py'
SEARCH_CONTROLLED_FLAGS = {
    '--device',
    '--hidden-dim',
    '--model',
    '--num-heads',
    '--num-layers',
    '--output-dir',
    '--pooling',
    '--run-name',
}
FEATURE_CONTROLLED_FLAGS = {
    '--drop-features',
    '--feature-preset',
    '--keep-features',
    '--skip-xyz',
}
DEFAULT_FEATURE_INDICES = list(range(8))
DEFAULT_FEATURE_GROUPS = {
    'coords': (5, 6, 7),
    'normals': (2, 3, 4),
    'potential': (0,),
    'normal_potential': (1,),
    'mean_curvature': (10,),
}
DEFAULT_FEATURE_GROUP_NAMES = [
    'coords',
    'normals',
    'potential',
    'normal_potential',
    'mean_curvature',
]


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description='Optuna architecture search for SHREC25 classify_gnn.py'
    )
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials to run')
    parser.add_argument('--study-name', type=str, default='classify_gnn_architecture_search', help='Optuna study name')
    parser.add_argument('--storage', type=str, default=DEFAULT_STORAGE, help='Optuna storage URL')
    parser.add_argument('--direction', type=str, default='maximize', choices=['maximize', 'minimize'], help='Optimization direction')
    parser.add_argument('--sampler-seed', type=int, default=42, help='TPE sampler seed')
    parser.add_argument('--device', type=int, default=0, help='CUDA device passed to classify_gnn.py')
    parser.add_argument('--python-executable', type=str, default=sys.executable, help='Python executable used for trial subprocesses')
    parser.add_argument('--classify-script', type=str, default=str(DEFAULT_CLASSIFY_SCRIPT), help='Path to classify_gnn.py')
    parser.add_argument('--trial-output-dir', type=str, default='runs_optuna', help='Output directory for per-trial classify_gnn runs')
    parser.add_argument('--run-prefix', type=str, default='optuna_gnn', help='Prefix for generated run names')
    parser.add_argument('--max-trial-seconds', type=int, default=None, help='Optional timeout per training trial')
    parser.add_argument('--metric-source', type=str, default='best_val_metrics', choices=['best_val_metrics', 'best_train_metrics', 'final_test_metrics', 'auto'], help='Which result block to optimize')
    parser.add_argument('--metric-name', type=str, default='f1_macro', choices=['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'loss'], help='Metric inside the selected result block')
    parser.add_argument('--models', nargs='+', default=['gcn', 'gin', 'sage', 'resgcn', 'gat'], help='Candidate backbone models')
    parser.add_argument('--poolings', nargs='+', default=['mean', 'max', 'sum', 'attention'], help='Candidate graph pooling methods')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 128, 256, 512], help='Candidate hidden dimensions')
    parser.add_argument('--min-layers', type=int, default=2, help='Minimum number of GNN layers')
    parser.add_argument('--max-layers', type=int, default=6, help='Maximum number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Fixed number of GAT heads for all trials')
    parser.add_argument('--search-feature-subsets', action='store_true', help='Search non-empty combinations of semantic feature groups by default, or raw feature indices when --feature-indices is provided')
    parser.add_argument('--feature-groups', type=str, nargs='+', default=None, help='Feature groups eligible for subset search. Available groups: coords, normals, potential, normal_potential, mean_curvature. Defaults to all five unless --feature-indices is provided')
    parser.add_argument('--feature-indices', type=int, nargs='+', default=None, help='Raw feature indices eligible for subset search. When provided, overrides --feature-groups and searches individual channels instead of groups')
    parser.add_argument('--min-feature-count', type=int, default=1, help='Minimum number of kept groups or raw features when --search-feature-subsets is enabled')
    parser.add_argument('--max-feature-count', type=int, default=None, help='Maximum number of kept groups or raw features when --search-feature-subsets is enabled (default: all listed groups or indices)')
    parser.add_argument('--feature-subset-arg-mode', type=str, default='auto', choices=['auto', 'keep', 'drop'], help='How feature subsets are emitted to classify_gnn.py: keep list, drop list, or whichever is shorter')

    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == '--':
        extra_args = extra_args[1:]
    return args, extra_args


def validate_args(args: argparse.Namespace, extra_args: Sequence[str]) -> None:
    if args.n_trials <= 0:
        raise ValueError(f'--n-trials must be > 0 (got {args.n_trials})')
    if args.min_layers < 2:
        raise ValueError(f'--min-layers must be >= 2 (got {args.min_layers})')
    if args.max_layers < args.min_layers:
        raise ValueError(
            f'--max-layers ({args.max_layers}) must be >= --min-layers ({args.min_layers})'
        )
    if args.num_heads <= 0:
        raise ValueError(f'--num-heads must be > 0 (got {args.num_heads})')
    if not args.models:
        raise ValueError('At least one model must be provided via --models')
    if not args.poolings:
        raise ValueError('At least one pooling method must be provided via --poolings')
    if not args.hidden_dims:
        raise ValueError('At least one hidden dimension must be provided via --hidden-dims')
    if any(hidden_dim <= 0 for hidden_dim in args.hidden_dims):
        raise ValueError('--hidden-dims values must all be > 0')

    if args.min_feature_count <= 0:
        raise ValueError(f'--min-feature-count must be > 0 (got {args.min_feature_count})')

    conflicting = [token for token in extra_args if token in SEARCH_CONTROLLED_FLAGS]
    if conflicting:
        joined = ', '.join(sorted(set(conflicting)))
        raise ValueError(
            f'Forwarded classify_gnn.py args cannot include search-controlled flags: {joined}'
        )

    if args.search_feature_subsets:
        feature_conflicts = [token for token in extra_args if token in FEATURE_CONTROLLED_FLAGS]
        if feature_conflicts:
            joined = ', '.join(sorted(set(feature_conflicts)))
            raise ValueError(
                'Forwarded classify_gnn.py args cannot include feature-selection flags when '
                f'--search-feature-subsets is enabled: {joined}'
            )

        if args.feature_indices is not None and args.feature_groups is not None:
            raise ValueError('Use either --feature-indices or --feature-groups, not both')

        if args.feature_indices is not None:
            search_unit_count = len(normalize_feature_indices(args.feature_indices))
        else:
            search_unit_count = len(normalize_feature_group_names(args.feature_groups))

        if args.max_feature_count is None:
            args.max_feature_count = search_unit_count
        if args.max_feature_count <= 0:
            raise ValueError(f'--max-feature-count must be > 0 (got {args.max_feature_count})')
        if args.min_feature_count > args.max_feature_count:
            raise ValueError(
                f'--min-feature-count ({args.min_feature_count}) must be <= --max-feature-count ({args.max_feature_count})'
            )
        if args.max_feature_count > search_unit_count:
            raise ValueError(
                f'--max-feature-count ({args.max_feature_count}) exceeds available search units ({search_unit_count})'
            )

    classify_script = Path(args.classify_script)
    if not classify_script.exists():
        raise FileNotFoundError(f'classify script not found: {classify_script}')


def normalize_feature_indices(values: Sequence[int] | None) -> List[int]:
    raw_values = DEFAULT_FEATURE_INDICES if values is None else list(values)
    if not raw_values:
        raise ValueError('Feature index list cannot be empty')

    normalized = []
    seen = set()
    for value in raw_values:
        if value < 0:
            raise ValueError(f'Feature indices must be >= 0 (got {value})')
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def normalize_feature_group_names(values: Sequence[str] | None) -> List[str]:
    raw_values = DEFAULT_FEATURE_GROUP_NAMES if values is None else list(values)
    if not raw_values:
        raise ValueError('Feature group list cannot be empty')

    normalized = []
    seen = set()
    for value in raw_values:
        if value not in DEFAULT_FEATURE_GROUPS:
            available = ', '.join(DEFAULT_FEATURE_GROUP_NAMES)
            raise ValueError(f'Unknown feature group {value!r}. Available groups: {available}')
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def enumerate_feature_subsets(
    feature_indices: Sequence[int],
    min_feature_count: int,
    max_feature_count: int,
) -> List[Tuple[int, ...]]:
    subsets: List[Tuple[int, ...]] = []
    for subset_size in range(min_feature_count, max_feature_count + 1):
        subsets.extend(tuple(combo) for combo in combinations(feature_indices, subset_size))
    if not subsets:
        raise ValueError('Feature subset search produced no candidate subsets')
    return subsets


def expand_feature_groups(group_names: Sequence[str]) -> List[int]:
    expanded = set()
    for group_name in group_names:
        expanded.update(DEFAULT_FEATURE_GROUPS[group_name])
    return sorted(expanded)


def format_feature_subset_token(keep_indices: Sequence[int]) -> str:
    return ','.join(str(idx) for idx in keep_indices)


def build_feature_trial_args(
    all_feature_indices: Sequence[int],
    keep_indices: Sequence[int],
    arg_mode: str,
) -> Tuple[List[str], str, List[int], List[int]]:
    keep_list = list(keep_indices)
    keep_set = set(keep_list)
    drop_list = [idx for idx in all_feature_indices if idx not in keep_set]

    resolved_mode = arg_mode
    if not drop_list:
        resolved_mode = 'keep'
    elif arg_mode == 'auto':
        resolved_mode = 'drop' if len(drop_list) < len(keep_list) else 'keep'

    if resolved_mode == 'drop':
        return ['--drop-features', *[str(idx) for idx in drop_list]], 'drop', keep_list, drop_list
    return ['--keep-features', *[str(idx) for idx in keep_list]], 'keep', keep_list, drop_list


def sanitize_token(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)


def build_feature_suffix(mode: str, keep_indices: Sequence[int], drop_indices: Sequence[int]) -> str:
    values = keep_indices if mode == 'keep' else drop_indices
    prefix = 'k' if mode == 'keep' else 'd'
    return f"f{prefix}{'-'.join(str(idx) for idx in values)}"


def build_group_feature_suffix(group_names: Sequence[str]) -> str:
    return f"fg{'-'.join(sanitize_token(name) for name in group_names)}"


def build_feature_search_choices(args: argparse.Namespace):
    if not args.search_feature_subsets:
        return [], [], 'none'

    if args.feature_indices is not None:
        search_feature_indices = normalize_feature_indices(args.feature_indices)
        feature_subset_choices = [
            {
                'token': format_feature_subset_token(subset),
                'keep_indices': list(subset),
                'keep_groups': [],
            }
            for subset in enumerate_feature_subsets(
                search_feature_indices,
                args.min_feature_count,
                args.max_feature_count,
            )
        ]
        return feature_subset_choices, search_feature_indices, 'indices'

    search_feature_groups = normalize_feature_group_names(args.feature_groups)
    all_group_feature_indices = expand_feature_groups(search_feature_groups)
    feature_subset_choices = []
    for subset_size in range(args.min_feature_count, args.max_feature_count + 1):
        for group_subset in combinations(search_feature_groups, subset_size):
            feature_subset_choices.append(
                {
                    'token': format_feature_subset_token(group_subset),
                    'keep_indices': expand_feature_groups(group_subset),
                    'keep_groups': list(group_subset),
                }
            )
    if not feature_subset_choices:
        raise ValueError('Feature subset search produced no candidate group subsets')
    return feature_subset_choices, all_group_feature_indices, 'groups'


def build_run_name(
    prefix: str,
    trial_number: int,
    model: str,
    pooling: str,
    hidden_dim: int,
    num_layers: int,
    feature_suffix: str | None = None,
) -> str:
    run_name = (
        f'{sanitize_token(prefix)}_t{trial_number:04d}_'
        f'{sanitize_token(model)}_'
        f'{sanitize_token(pooling)}_'
        f'h{hidden_dim}_l{num_layers}'
    )
    if feature_suffix:
        run_name = f'{run_name}_{sanitize_token(feature_suffix)}'
    return run_name


def choose_metric_block(results: dict, requested_source: str) -> Tuple[str, dict]:
    if requested_source != 'auto':
        if requested_source not in results:
            available = ', '.join(sorted(results.keys()))
            raise KeyError(
                f'Requested metric source {requested_source!r} not found in final_results.json. '
                f'Available keys: {available}'
            )
        return requested_source, results[requested_source]

    for source in ('best_val_metrics', 'best_train_metrics', 'final_test_metrics'):
        if source in results:
            return source, results[source]

    available = ', '.join(sorted(results.keys()))
    raise KeyError(f'Could not resolve metric source automatically. Available keys: {available}')


def read_metric(results_path: Path, metric_source: str, metric_name: str) -> Tuple[float, str, dict]:
    with results_path.open() as f:
        results = json.load(f)

    resolved_source, block = choose_metric_block(results, metric_source)
    if metric_name not in block:
        available = ', '.join(sorted(block.keys()))
        raise KeyError(
            f'Metric {metric_name!r} not found in {resolved_source}. Available metrics: {available}'
        )
    return float(block[metric_name]), resolved_source, results


def build_trial_command(
    args: argparse.Namespace,
    forwarded_args: Sequence[str],
    run_name: str,
    model: str,
    pooling: str,
    hidden_dim: int,
    num_layers: int,
    feature_args: Sequence[str],
) -> List[str]:
    return [
        args.python_executable,
        str(Path(args.classify_script).resolve()),
        '--device', str(args.device),
        '--model', model,
        '--pooling', pooling,
        '--hidden-dim', str(hidden_dim),
        '--num-layers', str(num_layers),
        '--num-heads', str(args.num_heads),
        '--run-name', run_name,
        '--output-dir', args.trial_output_dir,
        *feature_args,
        *forwarded_args,
    ]


def append_command_log(log_path: Path, command: Sequence[str]) -> None:
    with log_path.open('a') as log_file:
        log_file.write(f'$ {shlex.join(command)}\n\n')


def build_best_command(study: optuna.Study) -> str:
    trial = study.best_trial
    cmd = trial.user_attrs.get('command')
    if not cmd:
        return ''
    return cmd


def objective_factory(args: argparse.Namespace, forwarded_args: Sequence[str]):
    output_root = Path(args.trial_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_subset_choices, search_feature_indices, feature_search_mode = build_feature_search_choices(args)
    feature_subset_map = {
        choice['token']: choice
        for choice in feature_subset_choices
    }

    def objective(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical('model', args.models)
        pooling = trial.suggest_categorical('pooling', args.poolings)
        hidden_dim = trial.suggest_categorical('hidden_dim', args.hidden_dims)
        num_layers = trial.suggest_int('num_layers', args.min_layers, args.max_layers)
        feature_args: List[str] = []
        feature_mode = 'keep'
        keep_indices: List[int] = []
        drop_indices: List[int] = []

        if feature_subset_choices:
            feature_subset_token = trial.suggest_categorical(
                'feature_subset',
                list(feature_subset_map.keys()),
            )
            selected_feature_choice = feature_subset_map[feature_subset_token]
            feature_args, feature_mode, keep_indices, drop_indices = build_feature_trial_args(
                search_feature_indices,
                selected_feature_choice['keep_indices'],
                args.feature_subset_arg_mode,
            )
            trial.set_user_attr('feature_keep_indices', keep_indices)
            trial.set_user_attr('feature_drop_indices', drop_indices)
            trial.set_user_attr('feature_arg_mode', feature_mode)
            trial.set_user_attr('feature_count', len(keep_indices))
            if selected_feature_choice['keep_groups']:
                trial.set_user_attr('feature_keep_groups', selected_feature_choice['keep_groups'])
                trial.set_user_attr('feature_group_count', len(selected_feature_choice['keep_groups']))

        feature_suffix = None
        if feature_subset_choices:
            if feature_search_mode == 'groups':
                feature_suffix = build_group_feature_suffix(selected_feature_choice['keep_groups'])
            else:
                feature_suffix = build_feature_suffix(feature_mode, keep_indices, drop_indices)

        if model == 'gat' and hidden_dim % args.num_heads != 0:
            trial.set_user_attr(
                'skip_reason',
                f'gat requires hidden_dim divisible by num_heads ({hidden_dim} % {args.num_heads} != 0)',
            )
            raise optuna.exceptions.TrialPruned('invalid GAT hidden_dim/num_heads combination')

        run_name = build_run_name(
            args.run_prefix,
            trial.number,
            model,
            pooling,
            hidden_dim,
            num_layers,
            feature_suffix=feature_suffix,
        )
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / 'trial.log'
        command = build_trial_command(
            args,
            forwarded_args,
            run_name,
            model,
            pooling,
            hidden_dim,
            num_layers,
            feature_args,
        )

        append_command_log(log_path, command)
        trial.set_user_attr('run_name', run_name)
        trial.set_user_attr('run_dir', str(run_dir.resolve()))
        trial.set_user_attr('log_path', str(log_path.resolve()))
        trial.set_user_attr('command', shlex.join(command))

        with log_path.open('a') as log_file:
            try:
                subprocess.run(
                    command,
                    cwd=str(PROJECT_ROOT),
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.max_trial_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                trial.set_user_attr('timeout_seconds', args.max_trial_seconds)
                raise optuna.exceptions.TrialPruned(
                    f'trial exceeded timeout of {args.max_trial_seconds}s'
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f'classify_gnn.py failed for trial {trial.number}; see {log_path}'
                ) from exc

        results_path = run_dir / 'final_results.json'
        if not results_path.exists():
            raise FileNotFoundError(f'Expected results file not found: {results_path}')

        score, resolved_source, results = read_metric(results_path, args.metric_source, args.metric_name)
        trial.set_user_attr('metric_source', resolved_source)
        trial.set_user_attr('objective_score', score)

        if 'best_val_metrics' in results:
            trial.set_user_attr('best_val_f1_macro', results['best_val_metrics'].get('f1_macro'))
        if 'final_test_metrics' in results:
            trial.set_user_attr('final_test_f1_macro', results['final_test_metrics'].get('f1_macro'))

        return score

    return objective


def print_study_summary(study: optuna.Study, args: argparse.Namespace) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print('\n' + '=' * 70)
    print('OPTUNA CLASSIFY_GNN SEARCH COMPLETE')
    print('=' * 70)
    print(f'Study:         {study.study_name}')
    print(f'Storage:       {args.storage}')
    print(f'Direction:     {args.direction}')
    print(f'Metric:        {args.metric_source}.{args.metric_name}')
    print(f'Trials total:  {len(study.trials)}')
    print(f'Completed:     {len(completed)}')
    print(f'Pruned:        {len(pruned)}')
    print(f'Failed:        {len(failed)}')
    if args.search_feature_subsets:
        if args.feature_indices is not None:
            feature_indices = normalize_feature_indices(args.feature_indices)
            print(f'Feature search: raw indices {feature_indices} (keep {args.min_feature_count}..{args.max_feature_count})')
        else:
            feature_groups = normalize_feature_group_names(args.feature_groups)
            print(f'Feature search: groups {feature_groups} (keep {args.min_feature_count}..{args.max_feature_count})')

    if not completed:
        print('\nNo completed trials.')
        return

    best = study.best_trial
    print('\nBest trial:')
    print(f'  Number:      {best.number}')
    print(f'  Value:       {best.value:.6f}')
    print(f'  Model:       {best.params["model"]}')
    print(f'  Pooling:     {best.params["pooling"]}')
    print(f'  Hidden dim:  {best.params["hidden_dim"]}')
    print(f'  Num layers:  {best.params["num_layers"]}')
    if 'feature_subset' in best.params:
        if best.user_attrs.get('feature_keep_groups'):
            print(f'  Groups:      {best.user_attrs.get("feature_keep_groups", [])}')
        print(f'  Features:    keep={best.user_attrs.get("feature_keep_indices", [])}')
        print(f'  Feature arg: {best.user_attrs.get("feature_arg_mode", "all")}')
    print(f'  Run dir:     {best.user_attrs.get("run_dir", "")}' )

    best_command = build_best_command(study)
    if best_command:
        print('\nBest command:')
        print(best_command)


def main() -> None:
    args, forwarded_args = parse_args()
    validate_args(args, forwarded_args)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        sampler=TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )

    objective = objective_factory(args, forwarded_args)
    study.optimize(objective, n_trials=args.n_trials, catch=(RuntimeError, FileNotFoundError, KeyError))
    print_study_summary(study, args)


if __name__ == '__main__':
    main()