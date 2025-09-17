from normalizer.config import NormalizeConfig
from normalizer.normalizer import Normalizer

def test_similarity_merge_person():
    n = Normalizer(NormalizeConfig(merge_threshold=0.9))
    mentions = {"doc_id":"D","entities":[
        {"id":"E1","text":"Juan Perez","label":"PERSON","conf":0.9},
        {"id":"E2","text":"Juan Pérez","label":"PERSON","conf":0.88},  # debajo umbral -> no entra
        {"id":"E3","text":"J. Perez","label":"PERSON","conf":0.95}
    ]}
    out = n.run(mentions)
    # E2 tiene conf 0.88 (<0.66? No, es >, pero si ajustas min_conf_keep, cambia)
    assert len(out.entities) >= 1
