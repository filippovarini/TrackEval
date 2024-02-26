import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402


def run_mot_challenge(**kwargs):
    """
    Runs the MOT Challenge evaluation with configurable settings via keyword arguments.

    :param kwargs: Keyword arguments to override default evaluation, dataset, and metrics configurations.
    :return: A tuple containing the evaluation results and messages.
    """
    # Setup default configurations
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    # Update config with kwargs
    for setting, value in kwargs.items():
        if setting in config:
            if isinstance(config[setting], bool) and isinstance(value, str):
                config[setting] = value == 'True'
            elif isinstance(config[setting], int):
                config[setting] = int(value)
            elif isinstance(config[setting], list) or config[setting] is None:
                if not isinstance(value, list):
                    value = [value]
                config[setting] = value
            else:
                config[setting] = value

    # Separate configs based on their category
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Initialize evaluator, dataset, and metrics
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if not metrics_list:
        raise Exception('No metrics selected for evaluation')

    # Run evaluation
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    results = output_res['MotChallenge2DBox']['MPNTrack']
    cleaned_results = {}
    for key in results:
        cleaned_results[key] = results[key]['pedestrian']['CLEAR']

    return cleaned_results
