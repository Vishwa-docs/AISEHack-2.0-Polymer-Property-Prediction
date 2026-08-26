"""C155: corrected rich-bank plus dummy-capped charge residual for Ei."""
from pathlib import Path
import json, runpy, time
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdPartialCharges
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog("rdApp.*")
ROOT=Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2"); OUT=ROOT/"experiments/LOCAL_DIAGNOSTIC_ONLY"; OUT.mkdir(parents=True,exist_ok=True); started=time.time()
env=runpy.run_path(str(ROOT/"tools/round2_c148_corrected_tree_ionic.py"))
F=env["F"]; canon=F["canon_list"]; X0=env["X_ei"]; ei_rows=env["ei_rows"]; base=env["ei_base"]; both=env["both_ei"]; L=env["L"]; ti=env["ti"]; test=env["test"]; obs=env["OBS"]; path2=env["path2"]

def charge_features(smiles):
    out=[]
    for text in smiles:
        row=np.full(10,np.nan)
        try:
            m=Chem.MolFromSmiles(text); rw=Chem.RWMol(Chem.Mol(m))
            for a in rw.GetAtoms():
                if a.GetAtomicNum()==0: a.SetAtomicNum(6); a.SetFormalCharge(0); a.SetNoImplicit(False)
            cm=rw.GetMol(); Chem.SanitizeMol(cm); rdPartialCharges.ComputeGasteigerCharges(cm); q=np.asarray([float(a.GetProp("_GasteigerCharge")) for a in cm.GetAtoms()])
            if np.isfinite(q).all():
                d=np.asarray(Chem.GetDistanceMatrix(cm,useBO=False),float); aa=list(cm.GetAtoms()); het=np.asarray([a.GetAtomicNum() not in (0,1,6) for a in aa],bool); w=np.abs(q[:,None]*q[None,:]); wh=w*(het[:,None]|het[None,:]); row=[q.mean(),q.std(),q.min(),q.max(),np.ptp(q),np.abs(q).mean(),np.abs(q).sum()/max(cm.GetNumHeavyAtoms(),1),np.abs(q[het]).mean() if het.any() else 0.,float((w*d).sum()/max(w.sum(),1e-12)),float((wh*d).sum()/max(wh.sum(),1e-12))]
        except Exception: pass
        out.append(row)
    return np.asarray(out,float)

C=charge_features(canon); X=np.hstack([X0,C]).astype(np.float64); y=L[ei_rows,ti["ei"]]; groups=np.asarray(F["scaffolds"],dtype=object)[ei_rows]; pos={r:i for i,r in enumerate(ei_rows)}; oof=base.copy(); folds=[]
for fold,(tr,va) in enumerate(GroupKFold(5).split(ei_rows,y,groups),1):
    m=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=30.0)); m.fit(X[ei_rows[tr]],y[tr]-base[tr]); p=base[va]+.20*m.predict(X[ei_rows[va]])
    for k,r in enumerate(ei_rows[va]):
        if not both[r]: oof[pos[r]]=p[k]
    folds.append({"fold":fold,"fit_rows":int(len(tr)),"validation_rows":int(len(va))})
metrics={"base_r2":float(r2_score(y,base)),"candidate_r2":float(r2_score(y,oof))}; metrics["delta_r2"]=metrics["candidate_r2"]-metrics["base_r2"]; print("C155 Ei OOF",json.dumps(metrics,sort_keys=True),flush=True)
m=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=30.0)); m.fit(X[ei_rows],y-base); test_fi=test.fi.to_numpy(); test_pred=m.predict(np.hstack([X0[test_fi],C[test_fi]]).astype(np.float64))
candidate=pd.read_csv(path2)
for i,row in test.iterrows():
    fi=int(row.fi)
    if row.target_type=="ei" and not (obs[fi,ti["egc"]] and obs[fi,ti["eea"]]): candidate.loc[i,"target"]+=.20*test_pred[i]
name="R2-C155-rich-charge-ei-residual-LOCAL_DIAGNOSTIC_ONLY"; path=OUT/f"{name}.csv"; candidate.to_csv(path,index=False); report={"experiment":name,"official_only_fitting":True,"local_eval_read":False,"pretrained_weights":False,"mechanism":"corrected C148 rich target-masked bank plus 10 dummy-capped Gasteiger charge features, GroupKFold(5), Ridge residual weight 0.20, missing-partner Ei only","metrics":metrics,"folds":folds,"candidate_path":str(path),"elapsed_seconds":time.time()-started}; (OUT/f"{name}-oof.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print("C155 candidate",path,len(candidate),flush=True)
