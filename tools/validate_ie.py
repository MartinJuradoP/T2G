# -*- coding: utf-8 -*-
"""
Herramienta simple para validar la coherencia entre Mentions y Triples.
- Reporta % de relaciones de Mentions que están soportadas por Triples (misma oracion y etiqueta canónica).
- Reporta discrepancias comunes.
"""
import os, glob, json
from collections import defaultdict

def load_can_rel(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    m = defaultdict(list)
    for r in d.get("relations", []):
        can = r.get("canonical_label") or r.get("label")
        sid = r.get("sentence_idx")
        m[(d.get("doc_id"), sid, can)].append(r)
    return m

def main(mentions_dir, triples_dir):
    m_files = sorted(glob.glob(os.path.join(mentions_dir, "*_mentions.json")))
    t_files = sorted(glob.glob(os.path.join(triples_dir, "*_triples.json")))

    t_index = {}
    for tp in t_files:
        d = json.load(open(tp, "r", encoding="utf-8"))
        for tr in d.get("triples", []):
            can = tr.get("relation")
            sid = tr.get("meta", {}).get("sentence_idx")
            if can is None or sid is None:
                continue
            key = (d.get("doc_id"), int(sid), can)
            t_index.setdefault(key, 0)
            t_index[key] += 1

    total = 0
    supported = 0
    for mp in m_files:
        dm = json.load(open(mp, "r", encoding="utf-8"))
        doc = dm.get("doc_id")
        for r in dm.get("relations", []):
            can = r.get("canonical_label") or r.get("label")
            sid = r.get("sentence_idx")
            total += 1
            if (doc, sid, can) in t_index:
                supported += 1

    rate = (supported / total) * 100 if total else 0.0
    print(f"[VALIDATE] Mentions relations supported by Triples: {supported}/{total} = {rate:.2f}%")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mentions", required=True)
    ap.add_argument("--triples", required=True)
    args = ap.parse_args()
    main(args.mentions, args.triples)
