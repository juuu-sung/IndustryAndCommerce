"""Freeze original documents and page/section text with explicit source scope."""
from dataclasses import dataclass, asdict
from pathlib import Path
import re

from .common import CORPUS, RagError, asset_bytes, dependency, identifier, now, read_json, write_new
from ..registry import _canonical, _digest, _sha


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    title: str
    source_url: str
    publisher: str
    version_label: str
    language: str
    country: str
    scope: str
    format: str
    published_date: str | None = None
    access: str = 'public'
    use_conditions: str = 'Local research reference; redistribution conditions require confirmation'

    def __post_init__(self):
        identifier(self.document_id)
        if any(not isinstance(value, str) or not value.strip() for value in
               (self.title, self.source_url, self.publisher, self.version_label, self.use_conditions)):
            raise RagError('invalid_document', 'Source metadata must contain nonempty text', 400)
        if (self.language not in ('ko', 'en') or self.scope not in ('project_method', 'general_technical', 'korean_tms_general')
                or self.country not in ('KR', 'US') or self.format not in ('md', 'pdf', 'html')
                or self.access not in ('public', 'restricted')):
            raise RagError('invalid_document', 'Unsupported document scope or format', 400)
        if self.published_date is not None:
            from datetime import date
            try:
                date.fromisoformat(self.published_date)
            except (TypeError, ValueError) as error:
                raise RagError('invalid_document', 'Publication date must be known or null', 400) from error


def markdown_units(text):
    lines = text.splitlines(keepends=True)
    starts, headings = [0], ['문서 시작']
    stack = []
    fence = None
    for i, line in enumerate(lines):
        delimiter = re.match(r'^\s*(`{3,}|~{3,})', line)
        if delimiter:
            marker = delimiter[1][0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        match = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
        if not match:
            continue
        level, title = len(match[1]), match[2]
        stack = stack[:level - 1] + [title]
        if i == 0:
            headings[0] = ' / '.join(stack)
        else:
            starts.append(i)
            headings.append(' / '.join(stack))
    units = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        value = ''.join(lines[start:end])
        if len(value.strip()) < 12:
            continue
        units.append({'unit_key': f'section_{start + 1}', 'text': value,
                      'locator': {'kind': 'markdown_lines', 'section': headings[pos],
                                  'line_start': start + 1, 'line_end': end}})
    return units


def extract(spec, data):
    if spec.format == 'md':
        return markdown_units(data.decode('utf-8-sig'))
    if spec.format == 'pdf':
        import io
        reader = dependency('pypdf').PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise RagError('unsupported_pdf', 'Encrypted PDFs require separate review', 400)
        units = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text(extraction_mode='layout') or ''
            if len(text.strip()) < 20:
                continue  # Scanned/empty pages are never guessed into searchable text.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            labels = [line for line in lines[:5] + lines[-5:] if re.fullmatch(r'\d+-\d+', line)]
            units.append({'unit_key': f'page_{i + 1}', 'text': text,
                          'locator': {'kind': 'pdf_page', 'page_index': i,
                                      'pdf_page_number': i + 1,
                                      'printed_page_label': labels[0] if labels else None}})
        if not units:
            raise RagError('unsupported_pdf', 'PDF has no reviewed text layer; OCR is not enabled', 400)
        return units
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, 'html.parser')
    for item in soup(['script', 'style', 'nav']):
        item.decompose()
    if spec.document_id == 'epa_sncr_fact':
        if 'EPA-452/F-03-031' not in soup.get_text():
            raise RagError('unsupported_html', 'NEPIS document identity differs', 400)
        return [{'unit_key': f'block_{i + 1}', 'text': node.get_text(),
                 'locator': {'kind': 'html_text_block', 'block_index': i,
                             'block_number': i + 1, 'section': 'NEPIS public text'}}
                for i, node in enumerate(soup.find_all('pre')) if len(node.get_text().strip()) >= 20]
    text = soup.get_text('\n', strip=True)
    start = text.find('굴뚝원격감시체계(CleanSYS)란?')
    stop = text.find('이 페이지에서 제공하는 정보', start)
    if start < 0 or stop < start:
        raise RagError('unsupported_html', 'Official content boundaries could not be verified', 400)
    value = text[start:stop]
    return [{'unit_key': 'cleansys_overview', 'text': value,
             'locator': {'kind': 'html_extracted_lines', 'section': '굴뚝원격감시체계 운영',
                         'line_start': 1, 'line_end': len(value.splitlines())}}]


class Corpus:
    def __init__(self, directory=CORPUS):
        self.directory = Path(directory).resolve()

    def register(self, sources):
        documents, units, hashes = [], [], set()
        for spec, path in sources:
            data = asset_bytes(path)
            source_sha = _sha(data)
            if source_sha in hashes or spec.document_id in {d['document_id'] for d in documents}:
                raise RagError('duplicate_document', 'Duplicate document identity or source content', 400)
            hashes.add(source_sha)
            parsed = extract(spec, data)
            raw_name = spec.document_id + '.' + spec.format
            version = _digest({'metadata': asdict(spec), 'source_sha256': source_sha})
            document = {**asdict(spec), 'document_version_id': version, 'source_sha256': source_sha,
                        'source_size_bytes': len(data), 'raw_file': raw_name,
                        'extraction_status': 'text_layer_parsed', 'table_structure_verified': False,
                        'plant_specific_evidence': False, 'measurement_validity_certified': False}
            documents.append(document)
            for item in parsed:
                unit = {**item, 'unit_id': spec.document_id + ':' + item['unit_key'],
                        'document_id': spec.document_id, 'document_version_id': version}
                unit['unit_sha256'] = _digest(unit)
                units.append(unit)
            raw = self.directory / 'raw' / raw_name
            raw.parent.mkdir(parents=True, exist_ok=True)
            if raw.exists() and asset_bytes(raw) != data:
                raise RagError('rag_version_conflict', 'Registered source differs; use a new corpus directory')
            if not raw.exists():
                with raw.open('xb') as output:
                    output.write(data)
        content = {'schema_version': 1, 'parser_version': 'page_section_v2',
                   'documents': documents, 'units_sha256': _digest(units)}
        content['corpus_version_id'] = _digest(content)
        manifest_path = self.directory / 'manifest.json'
        if manifest_path.exists():
            old = self.verify()[0]
            if old['corpus_version_id'] != content['corpus_version_id']:
                raise RagError('rag_version_conflict', 'Existing document corpus differs')
            return old
        manifest = {**content, 'registered_at': now()}
        write_new(self.directory / 'units.json', units)
        write_new(manifest_path, manifest)
        self.verify()
        return manifest

    def verify(self):
        try:
            return self._verify()
        except RagError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
            raise RagError('rag_asset_invalid', 'Document corpus structure is invalid', 503) from error

    def _verify(self):
        manifest = read_json(self.directory / 'manifest.json')
        units = read_json(self.directory / 'units.json')
        if (not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int
                or manifest['schema_version'] != 1 or not isinstance(units, list)
                or _digest(units) != manifest.get('units_sha256')):
            raise RagError('rag_asset_invalid', 'Document manifest or extracted text differs')
        core = {k: v for k, v in manifest.items() if k not in ('registered_at', 'corpus_version_id')}
        if manifest['corpus_version_id'] != _digest(core):
            raise RagError('rag_asset_invalid', 'Document corpus version differs')
        documents = {doc['document_id']: doc for doc in manifest['documents']}
        if len(documents) != len(manifest['documents']):
            raise RagError('rag_asset_invalid', 'Duplicate document metadata')
        for doc in documents.values():
            identifier(doc['document_id'])
            if doc['raw_file'] != doc['document_id'] + '.' + doc['format']:
                raise RagError('rag_asset_invalid', 'Document file descriptor differs')
            data = asset_bytes(self.directory / 'raw' / doc['raw_file'])
            if _sha(data) != doc['source_sha256'] or len(data) != doc['source_size_bytes']:
                raise RagError('rag_asset_invalid', 'Registered document changed')
            meta = {key: doc[key] for key in DocumentSpec.__dataclass_fields__}
            DocumentSpec(**meta)
            if doc['document_version_id'] != _digest({'metadata': meta, 'source_sha256': doc['source_sha256']}):
                raise RagError('rag_asset_invalid', 'Document version mismatch')
        unit_ids = set()
        for unit in units:
            if (unit['unit_id'] in unit_ids or unit['document_id'] not in documents
                    or unit['document_version_id'] != documents[unit['document_id']]['document_version_id']
                    or unit['unit_sha256'] != _digest({k: v for k, v in unit.items() if k != 'unit_sha256'})):
                raise RagError('rag_asset_invalid', 'Extracted unit provenance differs')
            unit_ids.add(unit['unit_id'])
        return manifest, units
