from normalizer.config import NormalizeConfig
from normalizer.normalizer import Normalizer

def test_money_mxn_and_usd():
    n = Normalizer(NormalizeConfig(default_currency="MXN"))
    mentions = {"doc_id":"DOC-1","entities":[
        {"id":"E1","text":"$ 2,300.50","label":"MONEY","conf":0.9},
        {"id":"E2","text":"US$ 1,250.00","label":"MONEY","conf":0.9},
    ]}
    out = n.run(mentions)
    vals = {e.attrs["normalized_value"]: e.attrs["currency"] for e in out.entities}
    assert 2300.50 in vals and vals[2300.50]=="MXN"
    assert 1250.00 in vals and vals[1250.00]=="USD"
