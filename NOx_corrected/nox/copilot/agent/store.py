"""Transactional workflow state and append-only, source-bound execution receipts."""
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .common import DIRECTORY,AgentError,now,run_id
from ..registry import _digest,_json

STATUSES={'queued','running','waiting_input','waiting_review','succeeded','failed','cancelled'}


class WorkflowStore:
    def __init__(self,directory=DIRECTORY):
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'workflow.sqlite'
        if self.path.is_symlink():raise AgentError('agent_storage_invalid','Workflow storage is invalid',503)
        with self.connect() as db:
            db.executescript('''
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY,idempotency_key TEXT UNIQUE,request_hash TEXT NOT NULL,
 request_json TEXT NOT NULL,status TEXT NOT NULL,state_version INTEGER NOT NULL,
 report_id TEXT,report_revision INTEGER,context_hash TEXT,pins_json TEXT,
 pending_json TEXT,owner_pid INTEGER,error_code TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
 run_id TEXT NOT NULL,seq INTEGER NOT NULL,payload_json TEXT NOT NULL,
 previous_hash TEXT,event_hash TEXT NOT NULL,PRIMARY KEY(run_id,seq)
);
CREATE TABLE IF NOT EXISTS decisions (
 run_id TEXT NOT NULL,revision INTEGER NOT NULL,report_id TEXT NOT NULL,
 context_hash TEXT NOT NULL,decision TEXT NOT NULL,actor TEXT NOT NULL,
 decision_id TEXT NOT NULL,created_at TEXT NOT NULL,PRIMARY KEY(run_id,revision)
);
CREATE TABLE IF NOT EXISTS tool_results (
 run_id TEXT NOT NULL,tool_name TEXT NOT NULL,arguments_hash TEXT NOT NULL,
 pins_hash TEXT NOT NULL,result_json TEXT NOT NULL,result_hash TEXT NOT NULL,
 PRIMARY KEY(run_id,tool_name)
);
''')
        os.chmod(self.path,0o600)

    def connect(self):
        db=sqlite3.connect(self.path,timeout=20)
        db.row_factory=sqlite3.Row
        return db

    def _event(self,db,identifier,payload):
        row=db.execute('SELECT seq,event_hash FROM events WHERE run_id=? ORDER BY seq DESC LIMIT 1',(identifier,)).fetchone()
        seq=row['seq']+1 if row else 1
        previous=row['event_hash'] if row else None
        core={'run_id':identifier,'seq':seq,'payload':payload,'previous_hash':previous}
        db.execute('INSERT INTO events VALUES (?,?,?,?,?)',(identifier,seq,json.dumps(payload,ensure_ascii=False,allow_nan=False),previous,_digest(core)))

    def create(self,payload):
        digest=_digest(payload)
        key=payload.get('idempotency_key')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if key is not None:
                old=db.execute('SELECT * FROM runs WHERE idempotency_key=?',(key,)).fetchone()
                if old:
                    if old['request_hash']!=digest:
                        raise AgentError('idempotency_conflict','The same request key was used for different input')
                    return self._decode(old),False
            identifier='ag_'+uuid.uuid4().hex
            stamp=now()
            db.execute('INSERT INTO runs (run_id,idempotency_key,request_hash,request_json,status,state_version,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)',
                       (identifier,key,digest,json.dumps(payload,ensure_ascii=False,allow_nan=False),'queued',1,stamp,stamp))
            self._event(db,identifier,{'kind':'run_created','request_hash':digest,'at':stamp})
        return self.get(identifier),True

    def _decode(self,row):
        value=dict(row)
        for key in ('request_json','pins_json','pending_json'):
            raw=value.pop(key)
            value[key.replace('_json','')]=_json(raw.encode(),'workflow state') if raw is not None else None
        return value

    def get(self,identifier):
        run_id(identifier)
        with self.connect() as db:
            row=db.execute('SELECT * FROM runs WHERE run_id=?',(identifier,)).fetchone()
        if row is None:raise AgentError('run_not_found','Workflow run was not found',404)
        return self._decode(row)

    def transition(self,identifier,status,*,allowed=None,expected_version=None,event=None,**fields):
        if status not in STATUSES:raise ValueError('Unsupported workflow status')
        allowed_fields={'report_id','report_revision','context_hash','pins_json','pending_json','owner_pid','error_code'}
        if set(fields)-allowed_fields:raise ValueError('Unknown workflow fields')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM runs WHERE run_id=?',(run_id(identifier),)).fetchone()
            if row is None:raise AgentError('run_not_found','Workflow run was not found',404)
            if (allowed is not None and row['status'] not in allowed) or (expected_version is not None and row['state_version']!=expected_version):
                raise AgentError('run_state_conflict','Workflow state changed; reload the current run')
            data={**fields,'status':status,'state_version':row['state_version']+1,'updated_at':now()}
            db.execute('UPDATE runs SET '+','.join(name+'=?' for name in data)+' WHERE run_id=?',(*data.values(),identifier))
            self._event(db,identifier,event or {'kind':'state_changed','from':row['status'],'to':status,'version':data['state_version'],'at':data['updated_at']})
        return self.get(identifier)

    def event(self,identifier,payload):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._event(db,run_id(identifier),payload)

    def events(self,identifier):
        self.get(identifier)
        with self.connect() as db:
            rows=db.execute('SELECT * FROM events WHERE run_id=? ORDER BY seq',(identifier,)).fetchall()
        previous=None
        result=[]
        for expected_seq,row in enumerate(rows,1):
            payload=_json(row['payload_json'].encode(),'workflow event')
            core={'run_id':identifier,'seq':row['seq'],'payload':payload,'previous_hash':row['previous_hash']}
            if row['seq']!=expected_seq or row['previous_hash']!=previous or row['event_hash']!=_digest(core):
                raise AgentError('workflow_audit_invalid','Execution receipt integrity could not be verified')
            previous=row['event_hash']
            result.append({**core,'event_hash':previous})
        return result

    def cached_tool(self,identifier,name,args,pins):
        with self.connect() as db:
            row=db.execute('SELECT * FROM tool_results WHERE run_id=? AND tool_name=?',(run_id(identifier),name)).fetchone()
        if row is None:return None
        result=_json(row['result_json'].encode(),'cached tool result')
        if row['arguments_hash']!=_digest(args) or row['pins_hash']!=_digest(pins) or row['result_hash']!=_digest(result):
            raise AgentError('tool_cache_conflict','Cached tool arguments, sources or result changed')
        return result

    def cache_tool(self,identifier,name,args,pins,result,elapsed_ms,planner,planner_live_verified=False):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT * FROM tool_results WHERE run_id=? AND tool_name=?',(identifier,name)).fetchone()
            if old:
                if old['arguments_hash']!=_digest(args) or old['pins_hash']!=_digest(pins) or old['result_hash']!=_digest(result):
                    raise AgentError('tool_cache_conflict','A different result exists for this tool and source version')
                return
            db.execute('INSERT INTO tool_results VALUES (?,?,?,?,?,?)',
                       (identifier,name,_digest(args),_digest(pins),json.dumps(result,ensure_ascii=False,allow_nan=False),_digest(result)))
            self._event(db,identifier,{'kind':'tool_completed','tool':name,'args':args,
                                      'result_hash':_digest(result),'elapsed_ms':elapsed_ms,'planner':planner,
                                      'planner_live_verified':planner_live_verified})

    def decide(self,identifier,data,actor):
        required={'decision','expected_state_version','expected_report_id','expected_revision','expected_context_hash'}
        if (not isinstance(data,dict) or set(data)!=required or data['decision'] not in ('approve','reject')
                or type(data['expected_state_version']) is not int or type(data['expected_revision']) is not int):
            raise AgentError('invalid_review_decision','Provide a decision bound to the current run and report version',400)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM runs WHERE run_id=?',(run_id(identifier),)).fetchone()
            if row is None:raise AgentError('run_not_found','Workflow run was not found',404)
            old=db.execute('SELECT * FROM decisions WHERE run_id=? AND revision=?',(identifier,data['expected_revision'])).fetchone()
            if old:
                if all(old[k]==v for k,v in {'report_id':data['expected_report_id'],'context_hash':data['expected_context_hash'],
                                           'decision':data['decision'],'actor':actor}.items()):
                    return dict(old),False
                raise AgentError('review_conflict','A different decision already exists for this revision')
            if (row['status']!='waiting_review' or row['state_version']!=data['expected_state_version']
                    or row['report_id']!=data['expected_report_id'] or row['report_revision']!=data['expected_revision']
                    or row['context_hash']!=data['expected_context_hash']):
                raise AgentError('review_version_conflict','Review must use the current report ID, revision and content hash')
            record={'run_id':identifier,'revision':row['report_revision'],'report_id':row['report_id'],
                    'context_hash':row['context_hash'],'decision':data['decision'],'actor':actor,'created_at':now()}
            record['decision_id']=_digest(record)
            db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?)',
                       (identifier,record['revision'],record['report_id'],record['context_hash'],record['decision'],actor,record['decision_id'],record['created_at']))
            pending=json.dumps({'action':'review','decision_id':record['decision_id']})
            db.execute('UPDATE runs SET status=?,pending_json=?,state_version=state_version+1,updated_at=? WHERE run_id=?',('queued',pending,now(),identifier))
            self._event(db,identifier,{'kind':'user_decision','record':record})
        return record,True

    def decision(self,identifier,revision):
        with self.connect() as db:
            row=db.execute('SELECT * FROM decisions WHERE run_id=? AND revision=?',(run_id(identifier),revision)).fetchone()
        return dict(row) if row else None

    def unfinished(self):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM runs WHERE status IN ('queued','running')").fetchall()
        return [self._decode(row) for row in rows]
