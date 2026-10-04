"""Independently re-fetch flagged monthly exports without modifying first receipts."""
from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd

from .operating_patterns import HOURS, digest, fetch, read_hourly, write_json
from .schema import ROOT


def main():
    root = ROOT / 'data/hourly_operations_20261002'
    raw = root / 'sources'
    output = root / 'refetch'
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(root / 'monthly_reconciliation.csv', parse_dates=['month'])
    rejected = audit.loc[audit.operation_quality_status.eq('review_required')]
    def check(month):
        name = 'hourly_yd_' + month.strftime('%Y%m') + '.csv'
        original = raw / name
        receipt = json.loads(original.with_name(name + '.provenance.json').read_text())
        fetched = fetch(name, receipt['url'], receipt['parameters'], output)
        a, b = read_hourly(original, month.strftime('%Y-%m')), read_hourly(fetched, month.strftime('%Y-%m'))
        pd.testing.assert_frame_equal(a[['unit', 'date', *HOURS, '총량(KW)']], b[['unit', 'date', *HOURS, '총량(KW)']])
        return {'month': month.strftime('%Y-%m'), 'original_sha256': digest(original),
                'refetch_sha256': digest(fetched), 'byte_identical': digest(original) == digest(fetched),
                'numeric_values_identical': True, 'corrected': False}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, sorted(rejected.month.unique())))
    write_json(root / 'refetch_verification.json', {'monthly_files': len(results), 'results': results,
        'interpretation': 'Repeated provider export, not independent instrumentation confirmation; flagged features remain quarantined'})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
