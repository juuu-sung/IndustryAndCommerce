"""Read calendar observations from the same frozen bytes verified by the registry."""
from dataclasses import dataclass

from .registry import (
    CaseRegistry, RegistrationError, QUALITY_STATES, _csv_rows, _csv_number,
    _digest, _json, _month, _number, _text_list,
)


@dataclass(frozen=True)
class Observation:
    month: str
    value: float | None
    quality_status: str
    quality_reasons: tuple[str, ...] = ()
    row_sha256: str | None = None
    missing_reason: str | None = None

    def __post_init__(self):
        _month(self.month)
        if self.value is not None and _number(self.value, 'observation') < 0:
            raise RegistrationError('Concentration observations cannot be negative')
        if self.quality_status not in QUALITY_STATES:
            raise RegistrationError('Unknown observation quality status')
        if not isinstance(self.quality_reasons, tuple):
            raise RegistrationError('Observation quality reasons must be an immutable tuple')
        _text_list(list(self.quality_reasons), 'observation quality reasons')
        if (self.value is None) != (self.quality_status == 'unavailable'):
            raise RegistrationError('Observation availability and quality status disagree')

    def as_dict(self):
        return {
            'month': self.month, 'value': self.value, 'unit': 'ppm',
            'quality_status': self.quality_status,
            'quality_reasons': list(self.quality_reasons),
            'observation_row_sha256': self.row_sha256,
            'missing_reason': self.missing_reason,
        }


def previous_months(month):
    """The exact preceding three calendar months, oldest first."""
    _month(month)
    serial = int(month[:4]) * 12 + int(month[5:]) - 1
    months = []
    for lag in (3, 2, 1):
        year, offset = divmod(serial - lag, 12)
        candidate = f'{year:04d}-{offset + 1:02d}'
        _month(candidate)
        months.append(candidate)
    return tuple(months)


class ArchiveRepository:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else CaseRegistry()

    def load(self, case_key):
        manifest, blobs = self.registry.read_verified(case_key)
        case = manifest['case']
        rows = _csv_rows(blobs['history.csv'], 'history.csv',
                         {'plant', 'unit', 'month', 'target', 'quality_status', 'quality_reasons'})
        def observation(month):
            row = rows.get((case['plant'], case['unit'], month))
            if row is None:
                return Observation(month, None, 'unavailable', missing_reason='missing_calendar_month')
            value = None if row['target'] == '' else _csv_number(row['target'], 'history.target')
            return Observation(
                month, value, row['quality_status'],
                tuple(_text_list(_json(row['quality_reasons'], 'quality_reasons'), 'quality_reasons')),
                _digest(row), 'missing_target' if value is None else None,
            )
        current = observation(case['month'])
        prior = tuple(observation(month) for month in previous_months(case['month']))
        if current.row_sha256 != manifest['snapshot']['observation_row_sha256']:
            raise RegistrationError('Current observation differs from the registered row')
        return manifest, current, prior
