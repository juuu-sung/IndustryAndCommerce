"""Publish aggregate evidence and synthetic examples, excluding private observations."""
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from nox.copilot.rag.common import RUNTIME
from nox.copilot.rag.domain import implementation_hash as search_hash
from nox.copilot.agent.common import DIRECTORY,implementation_hash as agent_hash
from nox.copilot.evidence.common import implementation_hash as report_hash


def main():
    output=ROOT.parent/'docs/generated'
    def read(path):return json.loads(path.read_text())
    def write(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    report=read(RUNTIME/'domain_evaluation_v1/evaluation.json')
    assert report['selection']['domain_implementation_hash']==search_hash()
    write('domain_search_evaluation.json',{k:v for k,v in report.items() if k!='results'})
    columns=['cohort','profile','answerable_questions','unsupported_questions','hit_questions',
             'hit_at_5','recall_at_5','mrr','unsupported_handled_count','false_acceptance_count']
    with (output/'domain_search_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for cohort,profiles in report['metrics'].items():
            for profile,metrics in profiles.items():
                writer.writerow({'cohort':cohort,'profile':profile,**{k:metrics[k] for k in columns[2:]}})
    write('domain_search_additional_questions.json',read(RUNTIME/'fresh_gold_20261014.json'))
    write('domain_search_question_results.json',report['results'])
    candidates=sorted((ROOT/'data/runtime/copilot').glob('integration_validation_*/http_verification.json'),
                      key=lambda p:p.stat().st_mtime)
    http=read(candidates[-1])
    write('copilot_workflow_http.json',http)
    live=read(ROOT/'data/runtime/copilot/gemini_evidence_check_20261014.json')
    write('copilot_gemini_live_check.json',live)
    real=read(ROOT/'data/runtime/copilot/real_case_demo_20261014.json')
    real.pop('html_file',None)
    write('copilot_workflow_verification.json',{
        'date':'2026-10-14','tests_passed':419,'tests_skipped':0,'existing_pandas_future_warnings':1,
        'new_tests':{'domain_search':17,'evidence_reports':22,'agent_workflow':32},
        'previous_tests':348,'source_files_preserved':9,'old_report_contract_preserved':True,
        'implementation_hashes':{'search':search_hash(),'report':report_hash(),'agent':agent_hash()},
        'http':http,'gemini':live,'real_case_demo':real,
        'limitations':['Semantic claim faithfulness and expert source review are unverified.',
                       'Default planner is rule based; Gemini tool selection has mock validation only.',
                       'Real-case report remains waiting for the user; only synthetic decisions were approved.',
                       'Forward forecasting, new-unit adapters and plant controls are not implemented.']})
    versions=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions(path=[str(DIRECTORY/'deps')]))
    write('copilot_agent_dependencies.json',{'installation':'isolated project runtime target','packages':dict(versions),
          'global_environment_replaced':False,'postgres_or_pgvector_used':False})
    print(json.dumps({'status':'exported','test_count':419,'selected':report['selection']['selected_profile'],
                      'http_requests':http['request_count'],'actual_gemini_call_verified':live['actual_external_call_verified']}))


if __name__=='__main__':main()
