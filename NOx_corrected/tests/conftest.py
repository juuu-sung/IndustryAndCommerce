"""Keep synthetic checks runnable without publishing operational asset files."""
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'data/raw/manifest.json', 'data/retrieved/20261001/download_requests.json',
    'data/extended_20261001/manifest.json', 'data/unit_fuel_20261001/manifest.json',
    'data/operations_20261002/manifest.json',
    'artifacts/concentration/metadata.json', 'artifacts/legacy/metadata.json',
    'artifacts/concentration_20261001/metadata.json',
    'artifacts/concentration_unit_fuel_20261001/metadata.json',
    'reports/temporal_validation_20261001/COMPLETE',
    'reports/operations_validation_20261002/models/C/metadata.json',
    'reports/readiness_20261002/protocol.json',
]
ASSET_FIXTURES = {'predictor', 'candidate', 'development', 'operations', 'store', 'seed_db'}
ASSET_TESTS = {
    'test_source_joins_retain_rows_and_calendar_split',
    'test_official_overlap_and_hashes',
    'test_training_scope_and_frozen_overwrite_protection',
    'test_new_saved_predictions_match_inference',
    'test_new_history_bootstrap_does_not_backdate_retrieved_data',
    'test_unit_boolean_and_bad_reason_rejected',
    'test_complete_override_retains_targets_and_other_plants',
    'test_actual_fuel_history_uses_previous_calendar_month',
    'test_saved_actual_experiment_protocol_scores_and_api_policy',
    'test_trial_is_independent_of_outer_targets_and_reloads',
    'test_changed_protocol_is_rejected_before_fitting',
}


def pytest_addoption(parser):
    parser.addoption('--require-local-assets', action='store_true',
                     help='Fail collection if full local data/model snapshots are unavailable')


def pytest_collection_modifyitems(config, items):
    missing = [name for name in REQUIRED if not (PROJECT / name).is_file()]
    if missing and config.getoption('--require-local-assets'):
        raise pytest.UsageError('Restore local assets before full verification: ' + ', '.join(missing))
    for item in items:
        needs_assets = bool(ASSET_FIXTURES.intersection(item.fixturenames)) or item.originalname in ASSET_TESTS
        if needs_assets:
            item.add_marker(pytest.mark.requires_assets)
            if missing:
                item.add_marker(pytest.mark.skip(reason='Local data/model assets are not in the public repository'))
