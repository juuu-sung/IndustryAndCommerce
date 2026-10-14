"""Initial real source catalog; no fabricated plant manual or status evidence."""
from ...schema import ROOT
from .common import RUNTIME
from .ingest import DocumentSpec


def default_sources():
    docs = ROOT.parent / 'docs'
    project = [
        ('project_data', '데이터·타깃 정의와 품질 관리', 'DATA_QUALITY.md'),
        ('project_evaluation', '모델 평가·비교 범위', 'EVALUATION.md'),
        ('project_target_validation', '기관 타깃 검증과 재학습 절차', 'TARGET_VALIDATION.md'),
        ('project_api', 'Flask·Gemini 연동과 검증 범위', 'API_GEMINI.md'),
        ('project_calculations', '달력 이력·저장 예측 오차 계산', 'COPILOT_CALCULATIONS.md'),
        ('project_reports', '보고서 JSON·HTML·API', 'COPILOT_REPORTS.md'),
    ]
    sources = [(DocumentSpec(identifier, title,
                            'https://github.com/juuu-sung/IndustryAndCommerce/blob/main/docs/' + name,
                            'IndustryAndCommerce 프로젝트', 'local snapshot 2026-10-14 (content SHA-256)',
                            'ko', 'KR', 'project_method', 'md',
                            use_conditions='Project method snapshot; local changes may not yet be published'), docs / name)
               for identifier, title, name in project]
    incoming = RUNTIME / 'incoming'
    for identifier, title, name, url, version, date in (
        ('epa_scr_manual', 'Control Cost Manual Chapter 2: Selective Catalytic Reduction', 'epa_scr_manual.pdf',
         'https://www.epa.gov/sites/default/files/2017-12/documents/scrcostmanualchapter7thedition_2016revisions2017.pdf', 'revised 2019-06-12', '2019-06-12'),
        ('epa_sncr_manual', 'Control Cost Manual Chapter 1: Selective Noncatalytic Reduction', 'epa_sncr_manual.pdf',
         'https://www.epa.gov/sites/default/files/2017-12/documents/sncrcostmanualchapter7thedition20162017revisions.pdf', 'revised 2019-04-25', '2019-04-25'),
    ):
        sources.append((DocumentSpec(identifier, title, url, 'US EPA', version, 'en', 'US',
                                     'general_technical', 'pdf', date,
                                     use_conditions='US EPA public technical reference; cited third-party material retains its terms'), incoming / name))
    sources.append((DocumentSpec('epa_sncr_fact', 'Selective Non-Catalytic Reduction (SNCR) Fact Sheet',
                                 'https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100RQ6Y.TXT',
                                 'US EPA', 'EPA-452/F-03-031 public NEPIS text', 'en', 'US',
                                 'general_technical', 'html', use_conditions='Official public text; OCR and numeric tables require source review'),
                    incoming / 'epa_sncr_fact.html'))
    sources.append((DocumentSpec('keco_cleansys', '한국환경공단 굴뚝원격감시체계 운영',
                                 'https://www.keco.or.kr/group05/lay1/S300T734C736/contents.do',
                                 '한국환경공단', 'page updated 2024-02-20; captured 2026-10-14',
                                 'ko', 'KR', 'korean_tms_general', 'html'), incoming / 'keco_cleansys.html'))
    return sources
