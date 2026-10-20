"""Evidence-bearing facts and deterministic conflict resolution shared by STM/LTM."""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass,asdict,fields
from datetime import datetime,timezone
from app.config.settings import settings

def timestamp(value):
    if isinstance(value,(float,int)): return float(value)
    try: return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() if datetime.fromisoformat(value).tzinfo is None else datetime.fromisoformat(value).timestamp()
    except (TypeError,ValueError): return 0.0

@dataclass
class MemoryFact:
    content: str
    category: str
    created_at: str
    source_session: str = ''
    fact_id: str = ''
    attribute: str = ''
    value: str = ''
    scope: str = 'user'
    product_scope: str = ''
    status: str = 'active'
    source_type: str = 'legacy_unverified'
    source_message_id: str = ''
    source_quote: str = ''
    source_position: int = 0
    confidence: float = .3
    importance: float = .4
    stability: float = .4
    expires_at: float | None = None
    last_accessed: float = 0
    observed_at: float = 0
    supersedes: str = ''
    requires_refresh: bool = False

    def __post_init__(self):
        if not self.observed_at: self.observed_at=timestamp(self.created_at)
        if not self.attribute: self.attribute='legacy:'+hashlib.sha256(self.content.casefold().encode()).hexdigest()[:20]
        if not self.fact_id:
            identity=[self.source_session,self.source_message_id,self.attribute,self.value,self.content]
            self.fact_id=hashlib.sha256(json.dumps(identity,ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def load(cls,item):
        return cls(**{k:v for k,v in item.items() if k in {f.name for f in fields(cls)}})

    def active(self,now=None):
        now=time.time() if now is None else now
        return self.status=='active' and (self.expires_at is None or self.expires_at>now)

    def trusted(self):
        return self.source_type in {'user_explicit','business_tool'} and bool(self.source_message_id) and bool(self.source_session)


def order(fact):
    # Source observation time wins, never completion time of a slow extractor.
    return (fact.observed_at,fact.source_position,fact.source_message_id,fact.fact_id)

def retention(fact,now):
    age=max(0,now-max(fact.last_accessed,fact.observed_at))/86400
    return 3*fact.importance+2*fact.stability+fact.confidence+math.exp(-age/30)

def merge(existing,incoming,max_facts,now=None):
    now=time.time() if now is None else now
    byid={f.fact_id:MemoryFact.load(asdict(f)) for f in existing}
    for f in incoming:
        if f.fact_id not in byid: byid[f.fact_id]=MemoryFact.load(asdict(f))
        else: byid[f.fact_id].last_accessed=max(byid[f.fact_id].last_accessed,f.last_accessed)
    groups={}
    for f in byid.values():
        if f.status == 'retracted': continue
        groups.setdefault((f.scope,f.attribute),[]).append(f)
    for group in groups.values():
        # Do not let inferred/unverified content replace explicit evidence.
        verified=[f for f in group if f.trusted()]
        winner=max(verified or group,key=order)
        for f in group:
            if f is not winner: f.status='superseded'
            if f is not winner: winner.supersedes=f.fact_id
        if winner.expires_at is not None and winner.expires_at<=now: winner.status='expired'
    active=sorted((f for f in byid.values() if f.active(now)),key=lambda f:(retention(f,now),order(f)),reverse=True)
    for f in active[max_facts:]: f.status='evicted'
    inactive=sorted((f for f in byid.values() if not f.active(now)),key=order,reverse=True)
    # Retain one head per attribute as a tombstone: dropping it can resurrect stale writes.
    heads={max(group,key=order).fact_id for group in groups.values()}
    retained={f.fact_id:f for f in inactive[:settings.memory_archive_limit]}
    retained.update({f.fact_id:f for f in inactive if f.fact_id in heads})
    return active[:max_facts]+list(retained.values())

PRODUCT_SCOPES={'shoes':('鞋子','鞋'),'clothes':('衣服','上衣','外套'),'trousers':('裤子','长裤')}

def product_scope(text):
    found={name for name,words in PRODUCT_SCOPES.items() if any(w in text for w in words)}
    return next(iter(found)) if len(found)==1 else None


def select(facts,query='',category=None,top_k=None,scope=None,now=None,semantic_scores=None):
    from app.agent.rag.lexical import tokenize
    now=time.time() if now is None else now
    stop={'我','的','是','了','吗','你','请','一下','有','什么'}
    terms=set(tokenize(query))-stop
    requested_scope=product_scope(query)
    local_attributes={f.attribute for f in facts if f.scope==scope and f.scope!='user'
                      and f.active(now) and f.trusted() and not f.requires_refresh}
    ranked=[]
    for f in facts:
        if not f.active(now) or not f.trusted() or f.requires_refresh: continue
        if category and f.category!=category: continue
        if f.scope!='user' and f.scope!=scope: continue
        if f.product_scope and requested_scope and f.product_scope!=requested_scope: continue
        if f.scope=='user' and f.attribute in local_attributes: continue
        words=set(tokenize(f.content+' '+f.category+' '+f.attribute+' '+f.value))
        overlap=len(terms & words)
        semantic=(semantic_scores or {}).get(f.fact_id,0)
        if terms and not overlap and semantic<settings.memory_semantic_min_score: continue
        ranked.append((overlap*10+semantic*10+retention(f,now),order(f),f))
    selected=[f for _,_,f in sorted(ranked,key=lambda row:(row[0],row[1]),reverse=True)[:top_k or settings.memory_top_k]]
    for f in selected: f.last_accessed=now
    return selected

def extract(messages,session,now=None):
    """Precision-first explicit facts. Unsupported language remains in conversation evidence.

    No model inference is accepted as a user fact. Common preference updates and
    negation share one attribute, instead of deduplicating the literal sentence.
    """
    from app.agent.context import dumps,tool_fields
    now=time.time() if now is None else now
    result=[]
    for index,msg in enumerate(messages):
        role=msg.get('role')
        text=msg.get('content') or ''
        if not isinstance(text,str): continue
        # Stamp once; subsequent consolidation must not refresh the observation time.
        import uuid
        msg.setdefault('_observed_at',now)
        msg.setdefault('_message_id',uuid.uuid4().hex)
        observed=timestamp(msg.get('_observed_at')) or now
        mid=hashlib.sha256(dumps([session,msg.get('_message_id') or index,role,text]).encode()).hexdigest()
        def add(quote,attribute,value,position=0,category='preference',dynamic=False,context=None,domain=''):
            context=context if context is not None else text
            temporary=any(t in context for t in ('今天','这次','本次','暂时','最近'))
            ttl=settings.memory_temporary_ttl_days if temporary else settings.memory_preference_ttl_days
            expires=observed if dynamic else observed+ttl*86400
            if '今天' in context and not dynamic:
                from datetime import timedelta
                from zoneinfo import ZoneInfo
                local=datetime.fromtimestamp(observed,ZoneInfo('Asia/Shanghai'))
                expires=(local+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
            result.append(MemoryFact(quote,category,datetime.fromtimestamp(observed,timezone.utc).isoformat(),session,
                attribute=attribute,value=str(value),scope='session:'+session if any(t in context for t in ('这次','本次')) else 'user',product_scope=domain,
                source_type='business_tool' if role=='tool' else 'user_explicit',source_message_id=mid,
                source_quote=quote,source_position=position,confidence=1,importance=.8 if category=='identity' else .6,
                stability=.2 if temporary or dynamic else .8,expires_at=expires,observed_at=observed,requires_refresh=dynamic))
        if role=='user':
            # Questions, quotations and third-person reports are not user assertions.
            for segment in re.split(r'(?<=[。！？!?\n；;])',text):
                if segment.endswith(('？','?')): continue
                sentence=segment.rstrip('。！？!?\n；;')
                if any(x in sentence for x in ('如果','假如','不是不','不再不','他说','她说','用户说','客户说','你','并非','不是喜欢','不一定','可能','也许','不确定','才怪','呵呵','下周','本周','这周','不只','不仅','吗','是否','“','"')): continue
                for m in re.finditer(r'(?:我)?(?:最近|现在|以前|今天|这次|本次|暂时)?(不再喜欢|不太喜欢|不怎么喜欢|不喜欢|不偏好|讨厌|喜欢|偏好)([^，,。；;！？!?\n]{1,30}?)(?=我?(?:现在|最近|以前|今天|但|不过)|[，,]|$)',sentence):
                    # Do not infer that another person's preferences belong to the user.
                    before=sentence[:m.start()]
                    if any(x in before for x in ('他','她','朋友','妈妈','爸爸','客户')): continue
                    target=m.group(2).strip()
                    target=re.split(r'但|不过|现在|最近|以前|今天',target)[0].strip()
                    if not target or target in {'了','它','这个','那个','一些'}: continue
                    clause_start=max(sentence.rfind('，',0,m.start()),sentence.rfind(',',0,m.start()))+1
                    clause=sentence[clause_start:m.end()]
                    if ('以前' in m.group(0) or '曾经' in clause) and '现在' not in m.group(0): continue
                    domain=product_scope(clause) or ''
                    if sum(any(w in clause for w in words) for words in PRODUCT_SCOPES.values())>1: continue
                    if domain:
                        target=re.sub(r'(?:的)?(?:鞋子|鞋|衣服|上衣|外套|裤子|长裤)$','',target).strip()
                        if not target: continue
                    quote=m.group(0)[:len(m.group(0))-len(m.group(2))+len(target)]
                    add(clause if domain else quote,'preference:'+(domain+':' if domain else '')+target,'avoid' if m.group(1) in ('不再喜欢','不太喜欢','不怎么喜欢','不喜欢','不偏好','讨厌') else 'like',text.find(sentence)+m.start(),context=clause,domain=domain)
                for m in re.finditer(r'我(?:现在)?是(注册|白银|黄金|钻石)会员',sentence):
                    add(m.group(0),'identity:membership',m.group(1),text.find(sentence)+m.start(),'identity',True)
                for m in re.finditer(r'我的(?:尺码|鞋码)是([A-Za-z0-9.]+)',sentence):
                    add(m.group(0),'identity:size',m.group(1),text.find(sentence)+m.start(),'identity')
        elif role=='tool':
            for key,value in tool_fields(text).items():
                if key.split('.')[-1] in {'status','refund_status','amount','total_amount','refund_amount','execution_status'}:
                    add(dumps({key:value}),'business:'+key,str(value),category='issue',dynamic=True)
    return result
