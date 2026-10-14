"""Create and export a verified report locally without an HTTP server or LLM."""
import argparse
import json
from pathlib import Path

from .contracts import ReportError, ReportRequest
from .registry import CaseRegistry, DEFAULT_REGISTRY
from .repositories import DEFAULT_REPORTS, ReportRepository, ReportService


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-key', required=True)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--registry-dir', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    try:
        service = ReportService(CaseRegistry(args.registry_dir), ReportRepository(args.report_dir))
        request = ReportRequest(args.case_key, methods=args.methods)
        report, created = service.create(request)
        html = service.export_html(report['report_id'])
        print(json.dumps({'status': 'draft', 'created': created,
                          'report_id': report['report_id'], 'context_hash': report['context_hash'],
                          'json': str(service.repository.path_for(report['report_id'])),
                          'html': str(html)}, ensure_ascii=False, allow_nan=False))
    except (ReportError, OSError, ValueError) as error:
        code = error.code if isinstance(error, ReportError) else 'reports_unavailable'
        # Unexpected local errors may include server paths; keep generic run logs.
        message = str(error) if isinstance(error, ReportError) else 'Report storage or configuration is unavailable'
        print(json.dumps({'status': 'blocked', 'error': {'code': code, 'message': message}}, ensure_ascii=False))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
