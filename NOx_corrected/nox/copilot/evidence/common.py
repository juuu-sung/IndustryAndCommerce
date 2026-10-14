from pathlib import Path
import re

from ...schema import ROOT
from ..contracts import ReportError
from ..registry import _digest, _sha

DIRECTORY = ROOT / 'data/runtime/copilot/evidence_reports'
VERSION = 'source_bound_monthly_report_v1'


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'er_[0-9a-f]{64}', value):
        raise ReportError('invalid_evidence_report_id', 'Use an issued evidence report ID', 400)
    return value


def implementation_hash():
    directory = Path(__file__).parent
    return _digest({p.name: _sha(p.read_bytes()) for p in sorted(directory.glob('*.py'))})


def semantic(report):
    import copy
    value = copy.deepcopy(report)
    value.pop('report_id', None)
    value.pop('generated_at', None)
    value['base_report'].pop('report_generated_at', None)
    value['base_report']['calculation'].pop('generated_at', None)
    return value
