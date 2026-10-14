"""Strict request contracts and public errors for local report APIs."""
from dataclasses import dataclass
import re

from .registry import METHODS, RegistrationError, _case_key, _hash

REPORT_VERSION = 'monthly_evaluation_report_v1'


class ReportError(ValueError):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code, self.status = code, status


def invalid_report():
    return ReportError('report_invalid', 'Report content or version could not be verified')


def report_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'r_[0-9a-f]{64}', value):
        raise ReportError('invalid_report_id', 'Report ID must use the issued content identifier', 400)
    return value


def selected_methods(value, available=METHODS):
    if (not isinstance(value, (list, tuple)) or not value
            or len(value) > len(METHODS) or any(not isinstance(x, str) for x in value)
            or len(value) != len(set(value)) or any(x not in available for x in value)):
        raise ReportError('invalid_methods', 'Methods must be unique available saved prediction methods', 400)
    return tuple(method for method in METHODS if method in value)


@dataclass(frozen=True)
class ReportRequest:
    case_key: str
    analysis_mode: str = 'evaluation_replay'
    methods: tuple[str, ...] | None = None
    expected_case_version_id: str | None = None

    def __post_init__(self):
        try:
            _case_key(self.case_key)
            if self.expected_case_version_id is not None:
                _hash(self.expected_case_version_id, 'expected_case_version_id')
        except RegistrationError as error:
            raise ReportError('invalid_input', 'Invalid case key or expected case version', 400) from error
        if self.analysis_mode != 'evaluation_replay':
            raise ReportError('unsupported_analysis_mode', 'Only saved evaluation replay is supported', 400)
        if self.methods is not None:
            object.__setattr__(self, 'methods', selected_methods(self.methods))

    @classmethod
    def parse(cls, data):
        fields = {'case_key', 'analysis_mode', 'methods', 'expected_case_version_id'}
        if not isinstance(data, dict) or 'case_key' not in data or set(data) - fields:
            raise ReportError('invalid_input', 'Provide a case key and only supported report fields', 400)
        if 'methods' in data and not isinstance(data['methods'], list):
            raise ReportError('invalid_methods', 'Methods must be a JSON array', 400)
        if 'expected_case_version_id' in data and not isinstance(data['expected_case_version_id'], str):
            raise ReportError('invalid_input', 'Expected case version must be a SHA-256 string', 400)
        return cls(**data)
