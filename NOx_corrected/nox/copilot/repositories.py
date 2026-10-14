"""Immutable local report storage and source-aware service coordination."""
from pathlib import Path
import os
import tempfile

from ..schema import ROOT
from .calculations import calculate_case
from .contracts import ReportError, ReportRequest, report_id
from .registry import CaseRegistry, RegistrationError, _case_key, _json, _write_new, summary
from .report_builder import ReportBuilder, semantic_report
from .validators import validate_report
from .report_sources import load_source_quality

DEFAULT_REPORTS = ROOT / 'data/runtime/copilot/reports'


class ReportRepository:
    def __init__(self, directory=DEFAULT_REPORTS):
        self.directory = Path(directory).resolve()

    def path_for(self, identifier):
        return self.directory / (report_id(identifier) + '.json')

    def read(self, identifier):
        path = self.path_for(identifier)
        try:
            if path.is_symlink():
                raise ReportError('report_invalid', 'Report content or version could not be verified')
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise ReportError('report_not_found', 'Report was not found', 404) from error
        except OSError as error:
            raise ReportError('reports_unavailable', 'Report storage is unavailable', 503) from error
        try:
            report = _json(data, 'report')
            validate_report(report)
            if report['report_id'] != identifier:
                raise ReportError('report_invalid', 'Report content or version could not be verified')
        except RegistrationError as error:
            raise ReportError('report_invalid', 'Report content or version could not be verified') from error
        return report

    def save(self, report):
        validate_report(report)
        path = self.path_for(report['report_id'])
        if path.exists() or path.is_symlink():
            existing = self.read(report['report_id'])
            if semantic_report(existing) != semantic_report(report):
                raise ReportError('report_conflict', 'Existing report content differs from this request')
            return existing, False
        try:
            _write_new(path, report)
        except FileExistsError:
            existing = self.read(report['report_id'])
            if semantic_report(existing) != semantic_report(report):
                raise ReportError('report_conflict', 'Existing report content differs from this request')
            return existing, False
        except OSError as error:
            raise ReportError('reports_unavailable', 'Report storage is unavailable', 503) from error
        return report, True


def source_error(error):
    # Public messages are fixed; never expose original errors, file paths or row data.
    unavailable = ('unavailable', 'missing or linked outside bundle', 'became unavailable')
    if isinstance(error, OSError) or any(value in str(error).lower() for value in unavailable):
        return ReportError('source_unavailable', 'Registered source assets are unavailable', 503)
    return ReportError('source_conflict', 'Registered sources changed or could not be verified', 409)


class ReportService:
    def __init__(self, registry=None, repository=None):
        self.registry = registry if registry is not None else CaseRegistry()
        self.repository = repository if repository is not None else ReportRepository()
        if self.repository.directory.is_relative_to(self.registry.directory):
            raise ValueError('Report storage must be outside the case registry')

    def calculation(self, case_key):
        try:
            path = self.registry.path_for(case_key)
            if not path.exists():
                raise ReportError('case_not_found', 'Registered case was not found', 404)
            return calculate_case(case_key, self.registry)
        except (RegistrationError, OSError) as error:
            raise source_error(error) from error

    def cases(self):
        items = []
        try:
            if not self.registry.directory.exists():
                return items
            for path in sorted(self.registry.directory.glob('*.json')):
                try:
                    _case_key(path.stem)
                except RegistrationError:
                    continue
                try:
                    manifest = self.registry.verify(path.stem)
                    items.append({**summary(manifest), 'status': 'available',
                                  'target': manifest['target'],
                                  'available_methods': list(manifest['snapshot']['saved_predictions'])})
                except (RegistrationError, OSError) as error:
                    public = source_error(error)
                    items.append({'case_key': path.stem, 'status': 'blocked',
                                  'error': {'code': public.code, 'message': str(public)}})
        except OSError as error:
            raise ReportError('registry_unavailable', 'Case registry is unavailable', 503) from error
        return items

    def create(self, request):
        if not isinstance(request, ReportRequest):
            request = ReportRequest.parse(request)
        calculation = self.calculation(request.case_key)
        version = calculation['provenance']['case_version_id']
        if request.expected_case_version_id is not None and request.expected_case_version_id != version:
            raise ReportError('case_version_conflict', 'Case version differs from the requested version')
        try:
            quality = load_source_quality(calculation, self.registry)
        except (RegistrationError, OSError) as error:
            raise source_error(error) from error
        report = ReportBuilder().build(calculation, request.methods, quality)
        # Recheck the source at the save boundary, including changes during building.
        try:
            manifest = self.registry.verify(request.case_key)
        except (RegistrationError, OSError) as error:
            raise source_error(error) from error
        # The output location is configured by the server, never by the HTTP client.
        source = (self.registry.directory / manifest['artifact_path']).resolve()
        if self.repository.directory.is_relative_to(source):
            raise ReportError('reports_unavailable', 'Report output configuration is unavailable', 503)
        if manifest['case_version_id'] != version:
            raise ReportError('case_version_conflict', 'Case version changed during report generation')
        return self.repository.save(report)

    def get(self, identifier):
        report = self.repository.read(identifier)
        try:
            calculation = self.calculation(report['case_key'])
        except ReportError as error:
            if error.code == 'case_not_found':
                raise ReportError('source_unavailable', 'Registered source assets are unavailable', 503) from error
            raise
        validate_report(report, expected_calculation=calculation)
        try:
            quality = load_source_quality(calculation, self.registry)
        except (RegistrationError, OSError) as error:
            raise source_error(error) from error
        if quality != report['source_quality']:
            raise ReportError('source_conflict', 'Report coverage differs from registered sources')
        return report

    def export_html(self, identifier):
        """Export the same validated view without overwriting existing HTML."""
        from .renderers import render_html
        report = self.get(identifier)
        content = render_html(report).encode('utf-8')
        path = self.repository.directory / (report_id(identifier) + '.html')
        def existing():
            if path.is_symlink() or path.read_bytes() != content:
                raise ReportError('report_invalid', 'Saved HTML differs from the validated report')
            return path
        try:
            if path.exists() or path.is_symlink():
                return existing()
            descriptor, temporary = tempfile.mkstemp(prefix='report_', suffix='.tmp', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    return existing()
            finally:
                Path(temporary).unlink(missing_ok=True)
        except OSError as error:
            raise ReportError('reports_unavailable', 'Report export storage is unavailable', 503) from error
        return path
