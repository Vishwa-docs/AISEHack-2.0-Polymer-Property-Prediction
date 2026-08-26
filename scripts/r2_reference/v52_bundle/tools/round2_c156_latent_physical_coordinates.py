"""C156: fold-local multi-output physical-coordinate reconstruction."""
from pathlib import Path
import json, runpy, time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2"); OUT=ROOT/"experiments/LOCAL_DIAGNOSTIC_ONLY"; OUT.mkdir(parents=True,exist_ok=True); started=time.time()
e=runpy.run_path(str(ROOT/"tools/round2_c148_corrected_tree_ionic.py"))
F=e["F"]; ti=e["ti"]; L=e["L"]; OBS=e["OBS"]; test=e["test"]; X=e["TREEX"].astype(np.float64); CUR=e["CUR"]; path2=e["path2"]
groups=np.asarray(F["scaffolds"],dtype=object); base=CUR.copy(); both_ei=OBS[:,ti["ei"]]&OBS[:,ti["egc"]]&OBS[:,ti["eea"]]; base[both_ei,ti["ei"]]=.5*CUR[both_ei,ti["ei"]]+.5*(L[both_ei,ti["egc"]]+L[both_ei,ti["eea"]])
def fit_model(x,y): return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=50.0)).fit(x,y)
def r2(y,p): return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))

elec=np.where(OBS[:,ti["ei"]]&OBS[:,ti["eea"]])[0]; chi=(L[elec,ti["ei"]]+L[elec,ti["eea"]])/2; gap=L[elec,ti["ei"]]-L[elec,ti["eea"]]
opt=np.where(OBS[:,ti["eps"]]&OBS[:,ti["nc"]])[0]; ionic=L[opt,ti["eps"]]-L[opt,ti["nc"]]**2; optical=L[opt,ti["nc"]]**2
if np.nanmin(ionic)<=0: raise RuntimeError("non-positive ionic coordinate")
ei_rows=np.where(OBS[:,ti["ei"]])[0]; eps_rows=np.where(OBS[:,ti["eps"]])[0]; nc_rows=np.where(OBS[:,ti["nc"]])[0]; ei_pos={r:i for i,r in enumerate(ei_rows)}; eps_pos={r:i for i,r in enumerate(eps_rows)}; nc_pos={r:i for i,r in enumerate(nc_rows)}
ei_base=base[ei_rows,ti["ei"]].copy(); eps_base=base[eps_rows,ti["eps"]].copy(); nc_base=base[nc_rows,ti["nc"]].copy(); ei_c=ei_base.copy(); eps_c=eps_base.copy(); nc_c=nc_base.copy(); folds=[]

# Electronic coordinates use folds formed only from paired (Ei, Eea) rows.
for fold,(tr,va) in enumerate(GroupKFold(5).split(elec,groups=groups[elec]),1):
    er=elec[tr]; m_e=fit_model(X[er],np.column_stack([chi[tr],gap[tr]])); ep=m_e.predict(X[ei_rows])
    # Only validation Ei rows use the latent reconstruction. Both observed
    # partners retain the fixed identity carrier above.
    val_e=set(elec[va].tolist())
    for r in ei_rows:
        if r not in val_e or both_ei[r]: continue
        if OBS[r,ti["eea"]]: raw=2*ep[ei_pos[r],0]-L[r,ti["eea"]]
        else: raw=ep[ei_pos[r],0]+.5*ep[ei_pos[r],1]
        ei_c[ei_pos[r]]=.5*ei_base[ei_pos[r]]+.5*raw
    folds.append({"kind":"electronic","fold":fold,"fit_rows":int(len(er)),"validation_rows":int(len(elec[va]))})

# Optical coordinates use an independent fold partition formed only from
# paired (EPS, Nc) rows. This prevents the electronic fold indices from being
# accidentally applied to a different target population.
for fold,(tr,va) in enumerate(GroupKFold(5).split(opt,groups=groups[opt]),1):
    op_tr=opt[tr]; op_va=opt[va]
    m_o=fit_model(X[op_tr],np.column_stack([np.log(np.clip(ionic[tr],1e-6,None)),optical[tr]])); pred_o=m_o.predict(X[op_va])
    for r,pr in zip(op_va,pred_o):
        ionp=np.exp(np.clip(pr[0],-8,4)); n2=max(pr[1],.05**2)
        if OBS[r,ti["eps"]]: nc_c[nc_pos[r]]=.5*nc_base[nc_pos[r]]+.5*np.sqrt(max(L[r,ti["eps"]]-ionp,.05**2))
        if OBS[r,ti["nc"]]: eps_c[eps_pos[r]]=.5*eps_base[eps_pos[r]]+.5*(L[r,ti["nc"]]**2+ionp)
    folds.append({"kind":"optical","fold":fold,"fit_rows":int(len(op_tr)),"validation_rows":int(len(op_va))})
metrics={"ei":{"base_r2":r2(L[ei_rows,ti["ei"]],ei_base),"candidate_r2":r2(L[ei_rows,ti["ei"]],ei_c)},"eps":{"base_r2":r2(L[eps_rows,ti["eps"]],eps_base),"candidate_r2":r2(L[eps_rows,ti["eps"]],eps_c)},"nc":{"base_r2":r2(L[nc_rows,ti["nc"]],nc_base),"candidate_r2":r2(L[nc_rows,ti["nc"]],nc_c)}}
for v in metrics.values(): v["delta_r2"]=v["candidate_r2"]-v["base_r2"]
print("C156 OOF",json.dumps(metrics,sort_keys=True),flush=True)

me=fit_model(X[elec],np.column_stack([chi,gap])); pred_e=me.predict(X); mo=fit_model(X[opt],np.column_stack([np.log(np.clip(ionic,1e-6,None)),optical])); pred_o=mo.predict(X); candidate=pd.read_csv(path2)
for i,row in test.iterrows():
    fi=int(row.fi)
    if row.target_type=="ei" and not (OBS[fi,ti["egc"]] and OBS[fi,ti["eea"]]):
        raw=2*pred_e[fi,0]-L[fi,ti["eea"]] if OBS[fi,ti["eea"]] else pred_e[fi,0]+.5*pred_e[fi,1]; candidate.loc[i,"target"]=.5*candidate.loc[i,"target"]+.5*raw
    elif row.target_type=="eps" and OBS[fi,ti["nc"]]: candidate.loc[i,"target"]=.5*candidate.loc[i,"target"]+.5*(L[fi,ti["nc"]]**2+np.exp(np.clip(pred_o[fi,0],-8,4)))
    elif row.target_type=="nc" and OBS[fi,ti["eps"]]: candidate.loc[i,"target"]=.5*candidate.loc[i,"target"]+.5*np.sqrt(max(L[fi,ti["eps"]]-np.exp(np.clip(pred_o[fi,0],-8,4)),.05**2))
name="R2-C156-latent-physical-coordinates-LOCAL_DIAGNOSTIC_ONLY"; path=OUT/f"{name}.csv"; candidate.to_csv(path,index=False); report={"experiment":name,"official_only_fitting":True,"local_eval_read":False,"pretrained_weights":False,"mechanism":"structure-only multi-output Ridge; electronic chi/gap and optical log-ionic/Nc2 coordinates; GroupKFold(5), target-excluded paired fits, fixed 0.5 reconstruction blend","electronic_rows":int(len(elec)),"optical_rows":int(len(opt)),"folds":folds,"metrics":metrics,"candidate_path":str(path),"elapsed_seconds":time.time()-started}; (OUT/f"{name}-oof.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print("C156 candidate",path,len(candidate),flush=True)
