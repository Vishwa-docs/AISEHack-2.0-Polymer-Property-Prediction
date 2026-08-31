"""C154: transductive graph-Laplacian residual smoothing for Ei/Nc/EPS."""
from pathlib import Path
import json, pickle, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

ROOT=Path("/home/vishwa/Desktop/AISEHack-2.0/Polymer Prediction Challenge Round 2")
BASE=ROOT/"ppp-round-2"; SCR=Path("/tmp/claude-1001/-home-vishwa-Desktop-AISEHack-2-0/b00b2b9e-6110-479c-9f4b-9e173328b74e/scratchpad"); OUT=ROOT/"experiments/LOCAL_DIAGNOSTIC_ONLY"; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260804; T=["tg","egc","egb","ei","eea","nc","eps"]; ti={t:i for i,t in enumerate(T)}; ACTIVE=["ei","nc","eps"]; K=32; SHRINK=.20; ITER=20; started=time.time()
def r2(y,p): return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
def clean(x):
    x=np.asarray(x,np.float64); x[~np.isfinite(x)]=np.nan; return np.clip(x,-1e10,1e10)

F=pickle.loads((SCR/"features.pkl").read_bytes()); P=pickle.loads((SCR/"physics.pkl").read_bytes()); G=pickle.loads((SCR/"pgfp.pkl").read_bytes()); idx,cmap,blocks=F["idx"],F["canon_map"],F["blocks"]; ns=len(F["canon_list"])
train=pd.read_csv(BASE/"train.csv"); test=pd.read_csv(BASE/"test.csv"); archive=pd.read_csv(BASE/"archive/train.csv")
for frame in (train,test,archive): frame["canon"]=frame["smiles"].map(cmap); frame["fi"]=frame["canon"].map(idx).astype(int)
L=np.full((ns,len(T)),np.nan)
for j,target in enumerate(T):
    for frame in (archive,train):
        vals=frame.loc[frame.target_type.eq(target)].groupby("canon")["target"].mean()
        for canon,value in vals.items(): L[idx[canon],j]=value
OBS=np.isfinite(L)

PG=G["M"]; PGN=PG/np.maximum(1.0,PG.sum(1,keepdims=True)); dense=np.hstack([clean(blocks["desc"]),clean(blocks["extra"]),clean(blocks["oligo"]),clean(blocks["ipc"]),clean(P["M"]),clean(PGN),clean(G["morph"])])
keep=np.array([np.nanstd(dense[:,j])>1e-12 and np.isfinite(dense[:,j]).mean()>.3 for j in range(dense.shape[1])]); dense=dense[:,keep]; med=np.nanmedian(dense,0); bad=np.where(~np.isfinite(dense)); dense[bad]=np.take(med,bad[1])
dense_qt=QuantileTransformer(n_quantiles=1000,output_distribution="normal",random_state=SEED).fit_transform(dense)
sparse=np.hstack([np.log1p(blocks["morgan"]),blocks["maccs"],np.log1p(blocks["ap"]),np.log1p(blocks["tt"]),np.log1p(blocks["rk"]),np.log1p(PG)]).astype(np.float32); sparse=sparse[:,(sparse!=0).sum(0)>=4]
embed=np.hstack([PCA(64,random_state=SEED).fit_transform(dense_qt),TruncatedSVD(64,random_state=SEED).fit_transform(sparse)]).astype(np.float64); embed/=(embed.std(0)+1e-9)
dist,nbr=NearestNeighbors(n_neighbors=K+1,metric="euclidean",n_jobs=10).fit(embed).kneighbors(embed); dist,nbr=dist[:,1:],nbr[:,1:]; weights=1/(dist+.05)**2; weights/=weights.sum(1,keepdims=True)

base=np.load(SCR/"out_clean_corrected/PFINAL.npy"); both=OBS[:,ti["ei"]]&OBS[:,ti["egc"]]&OBS[:,ti["eea"]]; base_oof=base.copy(); base_oof[both,ti["ei"]]=.5*base[both,ti["ei"]]+.5*(L[both,ti["egc"]]+L[both,ti["eea"]]); groups=np.asarray(F["scaffolds"],dtype=object)
results={}
for target in ACTIVE:
    j=ti[target]; rows=np.where(OBS[:,j])[0]; y=L[rows,j]; parent=base_oof[rows,j]; cand=parent.copy(); fold_rows=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(rows,y,groups[rows]),1):
        anchors=rows[tr]; anchor_resid=y[tr]-base_oof[anchors,j]; anchor_mask=np.zeros(ns,bool); anchor_mask[anchors]=True; state=np.zeros(ns); state[anchors]=anchor_resid
        for _ in range(ITER):
            propagated=(weights*state[nbr]).sum(1); state[~anchor_mask]=propagated[~anchor_mask]; state[anchors]=anchor_resid
        cand[[np.where(rows==r)[0][0] for r in rows[va]]]=parent[[np.where(rows==r)[0][0] for r in rows[va]]]+SHRINK*state[rows[va]]; fold_rows.append({"fold":fold,"anchor_rows":int(len(anchors)),"validation_rows":int(len(va))})
    metrics={"base_r2":r2(y,parent),"candidate_r2":r2(y,cand)}; metrics["delta_r2"]=metrics["candidate_r2"]-metrics["base_r2"]; results[target]={"metrics":metrics,"folds":fold_rows}; print("C154",target,json.dumps(results[target],sort_keys=True),flush=True)

base_path=OUT/"R2-C148-et-eps-only-LOCAL_DIAGNOSTIC_ONLY.csv"; candidate=pd.read_csv(base_path); test_fi=test.fi.to_numpy()
for target in ACTIVE:
    j=ti[target]; rows=np.where(OBS[:,j])[0]; anchor_resid=L[rows,j]-base_oof[rows,j]; anchor_mask=np.zeros(ns,bool); anchor_mask[rows]=True; state=np.zeros(ns); state[rows]=anchor_resid
    for _ in range(ITER):
        propagated=(weights*state[nbr]).sum(1); state[~anchor_mask]=propagated[~anchor_mask]; state[rows]=anchor_resid
    for i in np.flatnonzero(test.target_type.eq(target).to_numpy()): candidate.loc[i,"target"]+=SHRINK*state[int(test_fi[i])]
name="R2-C154-graph-laplacian-residual-LOCAL_DIAGNOSTIC_ONLY"; path=OUT/f"{name}.csv"; candidate.to_csv(path,index=False)
report={"experiment":name,"official_only_fitting":True,"local_eval_read":False,"pretrained_weights":False,"mechanism":"official train+test unlabeled kNN graph, K=32 inverse-distance weights, 20-step harmonic residual propagation, fixed shrinkage 0.20, target-excluded GroupKFold(5), active Ei/Nc/EPS","graph_nodes":int(ns),"k":K,"iterations":ITER,"shrinkage":SHRINK,"targets":results,"candidate_path":str(path),"elapsed_seconds":time.time()-started}
(OUT/f"{name}-oof.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print("C154 candidate",path,len(candidate),flush=True)
