import json
from normalizer.config import NormalizeConfig
from normalizer.normalizer import Normalizer

def test_date_precision_day_es_format():
    cfg = NormalizeConfig(date_locale="es")
    n = Normalizer(cfg)
    mentions = {"doc_id":"DOC-1","entities":[
        {"id":"E1","text":"12/01/2023","label":"DATE","conf":0.9}
    ]}
    out = n.run(mentions)
    e = out.entities[0]
    assert e.type=="DATE"
    assert e.attrs["value_iso"]=="2023-01-12"
    assert e.attrs["precision"]=="day"

def test_date_precision_year_only():
    n = Normalizer(NormalizeConfig())
    mentions = {"doc_id":"D","entities":[{"id":"E","text":"1999","label":"DATE","conf":0.9}]}
    out = n.run(mentions)
    e = out.entities[0]
    assert e.attrs["value_iso"]=="1999"
    assert e.attrs["precision"]=="year"
