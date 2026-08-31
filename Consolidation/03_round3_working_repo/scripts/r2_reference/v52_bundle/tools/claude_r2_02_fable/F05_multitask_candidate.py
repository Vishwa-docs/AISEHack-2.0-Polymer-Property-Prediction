"""F05 matched concat-selector multitask control."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01

def main():
    archive=os.environ.get('FABLE_INCLUDE_ARCHIVE','1')=='1'; branch='with_archive' if archive else 'without_archive'
    data=fc.load_data(include_archive=archive); root=Path(fc.ROUND2_DIR)
    base=pd.read_csv(root/('submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv' if archive else 'experiments/CLEAN_OFFICIAL_ONLY/R2-F03-CLEAN-20260805-1924/candidate.csv')).set_index('id')['target'].copy()
    allc=list(dict.fromkeys(data.train['can'].tolist()+data.test['can'].tolist())); ix={c:i for i,c in enumerate(allc)}
    raw=np.hstack([f01.descriptor_block(allc),f01.morgan_count_block(allc)])
    sc=StandardScaler().fit(np.nan_to_num(raw)); pca=PCA(n_components=min(96,raw.shape[0]-1,raw.shape[1]),random_state=fc.SEED).fit(sc.transform(np.nan_to_num(raw)))
    feat=pca.transform(sc.transform(np.nan_to_num(raw))); ti={t:i for i,t in enumerate(fc.TARGETS)}
    rows=[]; ys=[]
    means=np.zeros(7); stds=np.ones(7)
    for t in fc.TARGETS:
        y=data.train.loc[data.train['target_type'].eq(t),'target'].to_numpy(float); means[ti[t]]=y.mean(); stds[ti[t]]=max(y.std(),1e-6)
    for _,r in data.train.iterrows():
        z=np.zeros(7); z[ti[r['target_type']]]=1
        rows.append(np.r_[feat[ix[r['can']]],z]); ys.append((float(r['target'])-means[ti[r['target_type']]])/stds[ti[r['target_type']]])
    model=MLPRegressor(hidden_layer_sizes=(128,64),activation='relu',alpha=1e-3,learning_rate_init=1e-3,max_iter=250,early_stopping=True,validation_fraction=.1,n_iter_no_change=20,random_state=fc.SEED,verbose=False)
    model.fit(np.asarray(rows),np.asarray(ys))
    for t in fc.TARGETS:
        te=data.test[data.test['target_type'].eq(t)]; z=np.zeros((len(te),7)); z[:,ti[t]]=1
        pred=model.predict(np.hstack([feat[[ix[c] for c in te['can']]],z]))*stds[ti[t]]+means[ti[t]]
        base.loc[te['id'].astype(int)]=pred
    out=pd.DataFrame({'id':data.test['id'].astype(int),'target':base.loc[data.test['id'].astype(int)].to_numpy(float)})
    path=root/'final_submission'/branch/f'R2-F05-MULTITASK-{branch}-candidate.csv'; out.to_csv(path,index=False); print({'path':str(path),'rows':len(out),'iterations':model.n_iter_})
if __name__=='__main__': main()
