from normalizer.config import NormalizeConfig
from normalizer.normalizer import Normalizer

def test_org_merge_suffix():
    n = Normalizer(NormalizeConfig())
    mentions = {"doc_id":"DOC-1","entities":[
        {"id":"E1","text":"Acme, S.A. de C.V.","label":"ORG","conf":0.9},
        {"id":"E2","text":"ACME SA de CV","label":"ORG","conf":0.9}
    ]}
    out = n.run(mentions)
    assert len(out.entities)==1
    e = out.entities[0]
    assert e.attrs["org_core"]=="acme" and "S.A. de C.V." in (e.attrs.get("org_suffix") or [])

def test_person_key_and_names():
    n = Normalizer(NormalizeConfig())
    mentions = {"doc_id":"D","entities":[
        {"id":"E1","text":"Ana María López","label":"PERSON","conf":0.95}
    ]}
    out = n.run(mentions)
    e = out.entities[0]
    assert e.attrs["given_name"].startswith("Ana")
    assert e.attrs["family_name"]=="López"
    assert e.attrs["person_key"]
