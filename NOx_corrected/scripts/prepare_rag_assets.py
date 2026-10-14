"""Fetch pinned public embedding assets and official NOx reference documents.

Downloaded weights/documents stay under the private runtime directory. This does
not install packages into the existing Python environment or train a model.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'data/runtime/copilot/rag'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
SOURCES = {
    'epa_scr_fact': 'https://www.epa.gov/sites/default/files/2020-08/documents/fscr.pdf',
    'epa_scr_manual': 'https://www.epa.gov/sites/default/files/2017-12/documents/scrcostmanualchapter7thedition_2016revisions2017.pdf',
    'epa_sncr_manual': 'https://www.epa.gov/sites/default/files/2017-12/documents/sncrcostmanualchapter7thedition20162017revisions.pdf',
    'keco_cleansys': 'https://www.keco.or.kr/group05/lay1/S300T734C736/contents.do',
}


def main():
    incoming = RUNTIME / 'incoming'
    incoming.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for identifier, url in SOURCES.items():
        path = incoming / (identifier + ('.html' if identifier == 'keco_cleansys' else '.pdf'))
        if not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'NOx-Copilot research source collection'})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read(10_000_001)
                if len(data) > 10_000_000:
                    raise ValueError('Public document exceeds the configured source limit')
            if path.suffix == '.pdf' and not data.startswith(b'%PDF-'):
                raise ValueError('Official PDF URL did not return a PDF')
            path.write_bytes(data)
        receipts[identifier] = {'source_url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'size_bytes': path.stat().st_size}
        print(json.dumps({'document': identifier, 'bytes': path.stat().st_size}), flush=True)
    from huggingface_hub import snapshot_download
    model_dir = RUNTIME / 'models' / REVISION
    snapshot_download('intfloat/multilingual-e5-small', revision=REVISION,
                      local_dir=str(model_dir), allow_patterns=[
                          'config.json', 'model.safetensors', 'tokenizer.json',
                          'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model',
                      ])
    model_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in model_dir.iterdir() if path.is_file()}
    receipt = {'model': 'intfloat/multilingual-e5-small', 'revision': REVISION,
               'model_files': model_files, 'documents': receipts}
    receipt_path = RUNTIME / 'asset_receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError('Existing asset receipt differs; do not silently rebaseline')
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    # The SCR fact-sheet PDF has copy restrictions and is excluded from indexing.
    # NEPIS exposes the separate SNCR fact sheet as official public text.
    text_url = 'https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100RQ6Y.TXT'
    text_path = incoming / 'epa_sncr_fact.html'
    if not text_path.exists():
        with urllib.request.urlopen(text_url, timeout=40) as response:
            text_path.write_bytes(response.read(1_000_000))
    print(json.dumps({'status': 'downloaded', 'model_revision': REVISION, 'documents': len(receipts)}), flush=True)


if __name__ == '__main__':
    main()
