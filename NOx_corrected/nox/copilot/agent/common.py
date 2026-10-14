from dataclasses import dataclass
from datetime import datetime,timezone
import importlib
import os
from pathlib import Path
import re
import secrets
import sys

from ...schema import ROOT
from ..registry import _case_key,_digest,_sha,RegistrationError

DIRECTORY=ROOT/'data/runtime/copilot/agent'


class AgentError(ValueError):
    def __init__(self,code,message,status=409):
        super().__init__(message)
        self.code,self.status=code,status


def now():
    return datetime.now(timezone.utc).isoformat()


def run_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'ag_[0-9a-f]{32}',value):
        raise AgentError('invalid_run_id','Use an issued workflow run ID',400)
    return value


def implementation_hash():
    directory=Path(__file__).parent
    return _digest({p.name:_sha(p.read_bytes()) for p in sorted(directory.glob('*.py'))})


def dependency(name):
    local=DIRECTORY/'deps'
    if local.is_dir() and str(local) not in sys.path:
        sys.path.insert(0,str(local))
    try:
        return importlib.import_module(name)
    except (ImportError,OSError) as error:
        raise AgentError('agent_dependency_unavailable','Optional Agent dependencies are unavailable',503) from error


@dataclass(frozen=True)
class RunRequest:
    request:str
    case_key:str|None=None
    task_type:str='auto'
    planner:str='rule'
    llm_enabled:bool=False
    idempotency_key:str|None=None

    def __post_init__(self):
        if not isinstance(self.request,str) or not self.request.strip() or len(self.request)>2000:
            raise AgentError('invalid_request','Request must contain 1-2000 characters',400)
        if self.task_type not in ('auto','case_report','document_search') or self.planner not in ('rule','gemini') or type(self.llm_enabled) is not bool:
            raise AgentError('invalid_request','Unsupported task or planner configuration',400)
        if self.case_key is not None:
            try:_case_key(self.case_key)
            except RegistrationError as error:raise AgentError('invalid_case_key','Invalid registered case key',400) from error
        if self.idempotency_key is not None and (not isinstance(self.idempotency_key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',self.idempotency_key)):
            raise AgentError('invalid_idempotency_key','Use a bounded idempotency key',400)

    @classmethod
    def parse(cls,value):
        allowed={'request','case_key','task_type','planner','llm_enabled','idempotency_key'}
        if not isinstance(value,dict) or 'request' not in value or set(value)-allowed:
            raise AgentError('invalid_request','Provide a request and only supported workflow fields',400)
        return cls(**value)


class Reviewer:
    def __init__(self,token=None,*,directory=DIRECTORY,identity='local_reviewer'):
        self.identity=identity
        configured=token if token is not None else os.getenv('COPILOT_REVIEW_TOKEN')
        if configured is not None:
            if not isinstance(configured,str) or len(configured)<16:
                raise AgentError('invalid_review_configuration','Review token must contain at least 16 characters',503)
            self.token=configured
            return
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'review_token'
        if path.is_symlink():
            raise AgentError('invalid_review_configuration','Review token storage is invalid',503)
        try:
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as output:output.write(secrets.token_urlsafe(32))
        except FileExistsError:pass
        self.token=path.read_text().strip()
        if len(self.token)<16:
            raise AgentError('invalid_review_configuration','Review token storage is invalid',503)

    def authenticate(self,value):
        if not isinstance(value,str) or not secrets.compare_digest(value,self.token):
            raise AgentError('review_authentication_required','A valid local review credential is required',401)
        return self.identity
