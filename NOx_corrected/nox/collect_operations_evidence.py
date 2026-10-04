"""Read public disclosure/measurement pages; preserve receipts, including failures."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json

from bs4 import BeautifulSoup

from .operating_patterns import digest, fetch, write_json
from .schema import ROOT

SOURCES = {
    'scr_disclosure.html': 'https://www.koenergy.kr/kosep/fr/bo/board/main.do?menuCd=FN02011006',
    'daily_emission_page.html': 'https://www.koenergy.kr/kosep/gv/nf/dt/nfdt16/main.do',
    'total_mass_guidance.html': 'https://www.keco.or.kr/web/lay1/S1T166C1020/contents.do',
    'cleansys_home.html': 'https://cleansys.or.kr/',
    'daily_nox_metadata.html': 'https://www.data.go.kr/data/15131510/fileData.do',
    'cleansys_metadata.html': 'https://www.data.go.kr/data/15136837/fileData.do',
    'scr_portal_search.html': 'https://www.data.go.kr/tcs/dss/selectDataSetList.do?keyword=%ED%83%88%EC%A7%88%EC%84%A4%EB%B9%84%20%EC%95%94%EB%AA%A8%EB%8B%88%EC%95%84',
}


def main():
    output = ROOT / 'data/hourly_operations_20261002/public_evidence'
    output.mkdir(parents=True, exist_ok=True)
    def get(item):
        name, url = item
        try:
            path = fetch(name, url, {}, output, method='GET')
            soup = BeautifulSoup(path.read_bytes(), 'html.parser')
            for node in soup(['script', 'style']):
                node.decompose()
            text_path = path.with_suffix('.txt')
            text_path.write_text(soup.get_text('\n', strip=True), encoding='utf-8')
            return {'file': name, 'url': url, 'status': 'retrieved', 'sha256': digest(path)}
        except Exception as error:
            return {'file': name, 'url': url, 'status': 'retrieval_failed', 'error': str(error)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(get, SOURCES.items()))
    write_json(output / 'retrieval_status.json', {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'sources': results, 'actual_scr_timeseries_not_inferred_from_procurement': True})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
