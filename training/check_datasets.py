import json, collections, time
from datasets import load_dataset
out = {}
def rec(name, f):
    t=time.time()
    try:
        r = f(); r["secs"]=round(time.time()-t,1); out[name]=r; print("OK", name, json.dumps(r)[:900])
    except Exception as e:
        out[name]={"error":repr(e)[:300]}; print("FAIL", name, repr(e)[:300])

def ge():
    d = load_dataset("google-research-datasets/go_emotions","simplified")
    names = d["train"].features["labels"].feature.names
    return {"splits":{k:len(v) for k,v in d.items()}, "labels":names, "cols":d["train"].column_names}
rec("go_emotions", ge)

def dd():
    d = load_dataset("li2017dailydialog/daily_dialog", revision="refs/convert/parquet")
    f = d["train"].features
    c = collections.Counter(e for x in d["train"]["emotion"] for e in x)
    return {"splits":{k:len(v) for k,v in d.items()}, "cols":d["train"].column_names, "emotion":str(f["emotion"])[:300], "turn_emotion_counts":dict(c)}
rec("daily_dialog_parquet_branch", dd)

def bdd():
    d = load_dataset("pixelsandpointers/better_daily_dialog")
    return {"splits":{k:len(v) for k,v in d.items()}, "features":str(d["train"].features)[:400]}
rec("better_daily_dialog", bdd)

def es():
    d = load_dataset("thu-coai/esconv")
    x = d["train"][0]
    return {"splits":{k:len(v) for k,v in d.items()}, "cols":d["train"].column_names, "sample":str(x)[:600]}
rec("esconv", es)

def ei():
    r={}
    for e in ["anger","fear","joy","sadness"]:
        d = load_dataset(f"Holmeister/ei_reg_{e}")
        r[e]={k:len(v) for k,v in d.items()}
    return r
rec("ei_reg", ei)

def eb():
    d = load_dataset("csv", data_files="https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv")
    t = d["train"]
    return {"rows":len(t), "cols":t.column_names, "split_counts":dict(collections.Counter(t["split"])), "V_range":[min(t["V"]),max(t["V"])], "A_range":[min(t["A"]),max(t["A"])]}
rec("emobank_github_csv", eb)

def ed():
    d = load_dataset("facebook/empathetic_dialogues", revision="refs/convert/parquet")
    return {"splits":{k:len(v) for k,v in d.items()}, "cols":d["train"].column_names, "n_emotions":len(set(d["train"]["context"]))}
rec("empathetic_dialogues_parquet_branch", ed)

def tei():
    d = load_dataset("stepp1/tweet_emotion_intensity")
    return {"splits":{k:len(v) for k,v in d.items()}, "classes":dict(collections.Counter(d["train"]["class"])), "intens":dict(collections.Counter(d["train"]["sentiment_intensity"]))}
rec("tweet_emotion_intensity", tei)

def dai():
    d = load_dataset("dair-ai/emotion","split")
    return {"splits":{k:len(v) for k,v in d.items()}, "labels":d["train"].features["label"].names}
rec("dair_emotion", dai)
json.dump(out, open("dataset_check.json","w"), indent=1)
