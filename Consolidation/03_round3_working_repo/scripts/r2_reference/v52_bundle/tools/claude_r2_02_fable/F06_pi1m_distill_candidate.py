"""F06 bounded PI1M pseudo-label student experiment."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fable_common as fc
import F01_ei_eea_egb_chain_engine as f01

def main():
    archive=os.environ.get('FABLE_INCLUDE_ARCHIVE','1')=='1'; branch='with_archive' if archive else 'without_archive'
    seed=int(os.environ.get('FABLE_SEED_OVERRIDE', str(fc.SEED)))
    force_label_archive=os.environ.get('FABLE_FORCE_LABEL_ARCHIVE')
    include_label_archive=archive if force_label_archive is None else force_label_archive=='1'
    data=fc.load_data(include_archive=include_label_archive); root=Path(fc.ROUND2_DIR)
    base=pd.read_csv(root/('submissions/R2-BEST-KNOWN-CLEAN-COMPOSITE-20260805.csv' if archive else 'experiments/CLEAN_OFFICIAL_ONLY/R2-F03-CLEAN-20260805-1924/candidate.csv')).set_index('id')['target'].copy()
    pi=pd.read_csv(root/'ppp-round-2/PI1M.csv',nrows=50001)['SMILES'].astype(str).iloc[1:50001].tolist()
    for target in ('egb','ei','eea','nc','eps'):
        tr=data.train[data.train['target_type'].eq(target)].reset_index(drop=True); te=data.test[data.test['target_type'].eq(target)].reset_index(drop=True)
        cans=tr['can'].tolist(); tcans=te['can'].tolist(); pcans=[fc.canon_nostereo(s) for s in pi]
        valid=[i for i,c in enumerate(pcans) if c is not None]
        pcans=[pcans[i] for i in valid]
        xtr=f01.descriptor_block(cans); xte=f01.descriptor_block(tcans); xpi=f01.descriptor_block(pcans)
        sc=StandardScaler().fit(np.nan_to_num(xtr)); a=sc.transform(np.nan_to_num(xtr)); b=sc.transform(np.nan_to_num(xte)); q=sc.transform(np.nan_to_num(xpi))
        teacher=ExtraTreesRegressor(300,min_samples_leaf=2,random_state=seed,n_jobs=-1).fit(a,tr['target'].to_numpy(float)); yp=teacher.predict(q)
        student=MLPRegressor(hidden_layer_sizes=(96,48),alpha=1e-3,learning_rate_init=1e-3,max_iter=180,early_stopping=True,random_state=seed,n_iter_no_change=15)
        xr=np.vstack([q,a,a,a,a,a]); yr=np.concatenate([yp]+[tr['target'].to_numpy(float)]*5)
        student.fit(xr,yr); pred=student.predict(b); base.loc[te['id'].astype(int)]=pred
    out=pd.DataFrame({'id':data.test['id'].astype(int),'target':base.loc[data.test['id'].astype(int)].to_numpy(float)})
    output_override = os.environ.get('FABLE_OUTPUT_CSV')
    if output_override:
        path = Path(output_override).expanduser().resolve()
    else:
        path = root/'final_submission'/branch/f'R2-F06-PI1M-{branch}-candidate.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing F06 output: {path}")
    out.to_csv(path,index=False); print({'path':str(path),'rows':len(out),'pi1m_rows':len(pcans)})
if __name__=='__main__': main()
