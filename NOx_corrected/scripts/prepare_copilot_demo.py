"""Prepare a real registered-case draft and leave it waiting for the user's review."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from nox.copilot.agent.service import AgentService
from nox.copilot.evidence.render import render_html


def main():
    agent=AgentService(background=False)
    try:
        request={'request':'영동 1호기 2025년 10월 관측과 저장 예측을 비교해 근거 보고서 초안을 작성해 주세요.',
                 'case_key':'yeongdong_u1_2025_10_fold_c_v1','task_type':'case_report',
                 'planner':'rule','llm_enabled':True,'idempotency_key':'yeongdong-evidence-demo-20261014'}
        run,_=agent.create(request)
        if run['status']!='waiting_review':
            raise ValueError('Real-case demo did not reach review: '+str(run['error_code']))
        report,view=agent.report(run['run_id'])
        path=agent.evidence.repository.directory/(report['report_id']+'.html')
        if path.exists():
            assert path.read_text()==render_html(report,view)
        else:path.write_text(render_html(report,view))
        summary={'run_id':run['run_id'],'status':run['status'],'report_id':run['report_id'],
                 'report_revision':run['report_revision'],'context_hash':run['context_hash'],
                 'rule_tool_calls':len(run['tool_calls']),'facts':len(report['facts']),
                 'citations':len(report['citations']),'analysis_status':report['analysis']['status'],
                 'analysis_reason':report['analysis']['reason'],
                 'actual_external_call_verified':report['analysis']['actual_external_call_verified'],
                 'real_report_approved':False,'provider_measurement_validity':'pending',
                 'html_file':str(path)}
        output=ROOT/'data/runtime/copilot/real_case_demo_20261014.json'
        output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='html_file'},ensure_ascii=False,indent=2))
    finally:
        agent.close()
        encoder=agent.evidence.retrieval.base.encoder
        if encoder is not None:encoder.close()


if __name__=='__main__':main()
