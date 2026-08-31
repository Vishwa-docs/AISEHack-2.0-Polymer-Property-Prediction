#!/usr/bin/env python3
"""patch_v57_arms.py - inject per-target arms into a copy of the V57 standalone.
Usage: patch_v57_arms.py <src.py> <dst.py> <arm1[,arm2,...]>
Arms are appended right before the final-validity check or replace the char/spread blocks.
All alphas are fitted on the C282 OOF residuals (train-only). No answer panels involved."""
import sys

ANCHOR_FINAL = "    if not np.isfinite(final).all():"
CHAR_START = "    char_delta = np.zeros(len(test_df))"
CHAR_END = "    # ---- final: splice char targets (tg/egc/egb/nc/eps) and spread targets (ei/eea) ----"
SPREAD_START = "        if t in ('ei', 'eea'):"
SPREAD_END = "        else:\n            final[mte] = base_target[mte] + char_delta[mte]"

CHAR_TUNED = '''    char_delta = np.zeros(len(test_df))
    for t in TARGETS_ORDER:
        tm = tr_tt == t
        idx = np.where(tm)[0]
        y = resid[idx].copy()
        pred = np.zeros(len(test_smiles))
        oof_pred = np.zeros(len(idx))
        kf = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        use_tfidf = (t == 'nc' or t == 'eps')
        use_huber = (t == 'tg' and HUBER_TG)
        for tr_f, va_f in kf.split(idx):
            if use_tfidf:
                vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False, sublinear_tf=True)
                ridge_alpha = 30.0
            else:
                vec = CountVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
                ridge_alpha = 40.0
            if use_huber:
                model = make_pipeline_model(vec, HuberRegressor(alpha=ridge_alpha, max_iter=2000, epsilon=1.35))
            else:
                model = make_pipeline_model(vec, RidgeModel(alpha=ridge_alpha, solver='lsqr', max_iter=5000, tol=1e-4))
            model.fit([tr_smiles[i] for i in idx[tr_f]], y[tr_f])
            pred += model.predict(test_smiles) / 5
            oof_pred[va_f] = model.predict([tr_smiles[i] for i in idx[va_f]])
        mte = test_df['target_type'].to_numpy(object) == t
        a_t = float(np.clip(np.sum(y * oof_pred) / max(np.sum(oof_pred ** 2), 1e-12), 0.0, 1.0))
        char_delta[mte] = a_t * pred[mte]
'''

SPREAD_TUNED = '''        if t in ('ei', 'eea'):
            tr_vals = train_df.loc[train_df['target_type'] == t, 'target'].to_numpy(float)
            lo = float(np.quantile(tr_vals, 0.001)) - float(np.std(tr_vals, ddof=1)) * 0.25
            hi = float(np.quantile(tr_vals, 0.999)) + float(np.std(tr_vals, ddof=1)) * 0.25
            med = float(np.median(tr_vals))
            _om = oof[oof['target_type'] == t]
            _y2 = _om['target'].to_numpy(float)
            _p2 = _om['prediction'].to_numpy(float)
            _s = float(np.clip(np.sum((_y2 - med) * (_p2 - med)) / max(np.sum((_p2 - med) ** 2), 1e-12), 0.5, 2.0))
            spread = med + _s * (base_target[mte] - med)
            final[mte] = np.clip(spread, lo, hi)
'''

ARM_KRIGING = '''    # ---- P5A-101 arm: tg Tanimoto kNN residual kriging (alpha on C282 OOF) ----
    _t = 'tg'
    _om = oof[oof['target_type'] == _t].reset_index(drop=True)
    _res = _om['target'].to_numpy(float) - _om['prediction'].to_numpy(float)
    _fp_tr = morgan_fp_list(list(_om['canonical']))
    _sim = np.array([DataStructs.BulkTanimotoSimilarity(_fp, _fp_tr) for _fp in _fp_tr], dtype=np.float64)
    np.fill_diagonal(_sim, 0.0)
    _w = _sim / np.maximum(_sim.sum(axis=1, keepdims=True), 1e-9)
    _d_oof = _w @ _res
    _alpha = float(np.clip(np.sum(_res * _d_oof) / max(np.sum(_d_oof ** 2), 1e-12), 0.0, 1.0))
    _fp_te = morgan_fp_list([canonicalize(s) for s in test_df['smiles'].tolist()])
    _sim_te = np.array([DataStructs.BulkTanimotoSimilarity(_fp, _fp_tr) for _fp in _fp_te], dtype=np.float64)
    _wte = _sim_te / np.maximum(_sim_te.sum(axis=1, keepdims=True), 1e-9)
    _d_te = _wte @ _res
    _mte = test_tt == _t
    final[_mte] += _alpha * _d_te[_mte]
    print('P5A-101 tg-kriging alpha=%.4f' % _alpha, flush=True)
'''

ARM_CALIB = '''    # ---- P5A-102 arm: per-target linear calibration of base on C282 OOF ----
    _cal_delta = np.zeros(len(test_df))
    for _t in TARGETS_ORDER:
        _om = oof[oof['target_type'] == _t]
        _y = _om['target'].to_numpy(float)
        _p = _om['prediction'].to_numpy(float)
        if len(_y) < 20:
            continue
        _a, _b = np.polyfit(_p, _y, 1)
        _d_oof = (_a * _p + _b) - _p
        _alpha = float(np.clip(np.sum((_y - _p) * _d_oof) / max(np.sum(_d_oof ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        _cal_delta[_mte] = _alpha * (_a * base_target[_mte] + _b - base_target[_mte])
    final += _cal_delta
    print('P5A-102 calib applied', flush=True)
'''

PARTNERS_HEAD = '''    # ---- partner machinery (Ridge on char n-grams, KFold OOF, exact train-partner overlay) ----
    _all_can = list(oof['canonical'])
    _test_can = [canonicalize(s) for s in test_df['smiles'].tolist()]
    _tt_arr = oof['target_type'].to_numpy(object)
    _y_arr = oof['target'].to_numpy(float)
    _pred_oof = {_pt: np.full(len(oof), np.nan) for _pt in ('egc', 'eea', 'nc')}
    _pred_test = {_pt: np.zeros(len(test_df)) for _pt in ('egc', 'eea', 'nc')}
    _lab_map = {}
    for _i in range(len(oof)):
        _lab_map.setdefault(oof['canonical'].values[_i], {})[oof['target_type'].values[_i]] = oof['target'].values[_i]
    for _pt in ('egc', 'eea', 'nc'):
        _idx = np.where(_tt_arr == _pt)[0]
        _yv = _y_arr[_idx]
        _kf2 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        for _tr2, _va2 in _kf2.split(_idx):
            _vec2 = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=32768, lowercase=False)
            _mod2 = make_pipeline_model(_vec2, RidgeModel(alpha=30.0, solver='lsqr', max_iter=3000, tol=1e-4))
            _mod2.fit([_all_can[i] for i in _idx[_tr2]], _yv[_tr2])
            _pred_oof[_pt][_idx[_va2]] = _mod2.predict([_all_can[i] for i in _idx[_va2]])
            _pred_test[_pt] += _mod2.predict(_test_can) / 5
        for _m in range(len(test_df)):
            _d = _lab_map.get(_test_can[_m])
            if _d and _pt in _d:
                _pred_test[_pt][_m] = _d[_pt]
'''

ARM_EI = PARTNERS_HEAD + '''    # ---- P5A-103 arm: ei via egc+eea partner identity (imputed: exact labels where available) ----
    _t = 'ei'
    _idx = np.where(_tt_arr == _t)[0]
    _y = _y_arr[_idx]
    _p = oof['prediction'].to_numpy(float)[_idx]
    _c_oof = _pred_oof['egc'][_idx] + _pred_oof['eea'][_idx]
    _d_oof = _c_oof - _p
    _ok = np.isfinite(_d_oof)
    _alpha = float(np.clip(np.sum((_y[_ok] - _p[_ok]) * _d_oof[_ok]) / max(np.sum(_d_oof[_ok] ** 2), 1e-12), 0.0, 1.0)) if int(_ok.sum()) > 10 else 0.0
    _mte = test_tt == _t
    final[_mte] += _alpha * (_pred_test['egc'][_mte] + _pred_test['eea'][_mte] - base_target[_mte])
    print('P5A-103 ei-identity alpha=%.4f' % _alpha, flush=True)
'''

ARM_EPS = PARTNERS_HEAD + '''    # ---- P5A-104 arm: eps via nc^2 + ionic partner (imputed: exact labels where available) ----
    _eps_d = train_df[train_df['target_type'] == 'eps'][['smiles', 'target']]
    _nc_d = train_df[train_df['target_type'] == 'nc'][['smiles', 'target']]
    _jj = _eps_d.merge(_nc_d, on='smiles', suffixes=('_e', '_n'))
    _ionic_med = float(np.median(_jj['target_e'] - _jj['target_n'] ** 2)) if len(_jj) else 0.767
    _t = 'eps'
    _idx = np.where(_tt_arr == _t)[0]
    _y = _y_arr[_idx]
    _p = oof['prediction'].to_numpy(float)[_idx]
    _c_oof = _pred_oof['nc'][_idx] ** 2 + _ionic_med
    _d_oof = _c_oof - _p
    _ok = np.isfinite(_d_oof)
    _alpha = float(np.clip(np.sum((_y[_ok] - _p[_ok]) * _d_oof[_ok]) / max(np.sum(_d_oof[_ok] ** 2), 1e-12), 0.0, 1.0)) if int(_ok.sum()) > 10 else 0.0
    _mte = test_tt == _t
    final[_mte] += _alpha * (_pred_test['nc'][_mte] ** 2 + _ionic_med - base_target[_mte])
    print('P5A-104 eps-ionic alpha=%.4f' % _alpha, flush=True)
'''

ARM_EGB = PARTNERS_HEAD + '''    # ---- P5A-105 arm: egb via egc covariate ----
    _t = 'egb'
    _idx = np.where(_tt_arr == _t)[0]
    _y = _y_arr[_idx]
    _p = oof['prediction'].to_numpy(float)[_idx]
    _egc_oof = _pred_oof['egc'][_idx]
    _ok = np.isfinite(_egc_oof)
    _a, _b = np.polyfit(_egc_oof[_ok], _y[_ok], 1)
    _c_oof = _a * _egc_oof + _b
    _d_oof = _c_oof - _p
    _alpha = float(np.clip(np.sum((_y[_ok] - _p[_ok]) * _d_oof[_ok]) / max(np.sum(_d_oof[_ok] ** 2), 1e-12), 0.0, 1.0)) if int(_ok.sum()) > 10 else 0.0
    _mte = test_tt == _t
    final[_mte] += _alpha * (_a * _pred_test['egc'][_mte] + _b - base_target[_mte])
    print('P5A-105 egb-egc alpha=%.4f' % _alpha, flush=True)
'''

ARM_SHRINK = '''    # ---- P5A-109 arm: per-target shrink toward train median (alpha on OOF) ----
    _shrink = np.zeros(len(test_df))
    for _t in TARGETS_ORDER:
        _om = oof[oof['target_type'] == _t]
        _y = _om['target'].to_numpy(float)
        _p = _om['prediction'].to_numpy(float)
        _med = float(np.median(train_df.loc[train_df['target_type'] == _t, 'target']))
        _d_oof = _med - _p
        _alpha = float(np.clip(np.sum((_y - _p) * _d_oof) / max(np.sum(_d_oof ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        _shrink[_mte] = _alpha * (_med - base_target[_mte])
    final += _shrink
    print('P5A-109 median-shrink applied', flush=True)
'''

ARM_SMILER3 = '''    # ---- P5A-107 arm: tg residual model on smile_r3-fitted char-SVD features ----
    _sr3 = pd.read_csv(os.path.join(data_dir, 'smile_r3.csv'), nrows=1000000)
    _smp = _sr3['smiles'].sample(n=400000, random_state=2026).tolist()
    _hv = HashingVectorizer(analyzer='char', ngram_range=(2, 4), n_features=2 ** 18, alternate_sign=False)
    _H = _hv.fit_transform(_smp)
    _svd = TruncatedSVD(n_components=48, random_state=2026)
    _svd.fit(_H)
    _t = 'tg'
    _idx = np.where(tr_tt == _t)[0]
    _y = resid[_idx].copy()
    _Xtr_s = _svd.transform(_hv.transform([tr_smiles[i] for i in _idx]))
    _Xte_s = _svd.transform(_hv.transform(test_smiles))
    _kf3 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
    _pred3 = np.zeros(len(test_smiles))
    _oof3 = np.zeros(len(_idx))
    for _tr3, _va3 in _kf3.split(_idx):
        _m3 = RidgeModel(alpha=40.0, solver='lsqr', max_iter=5000, tol=1e-4)
        _m3.fit(_Xtr_s[_tr3], _y[_tr3])
        _pred3 += _m3.predict(_Xte_s) / 5
        _oof3[_va3] = _m3.predict(_Xtr_s[_va3])
    _a3 = float(np.clip(np.sum(_y * _oof3) / max(np.sum(_oof3 ** 2), 1e-12), 0.0, 1.0))
    _mte = test_tt == _t
    final[_mte] += _a3 * _pred3[_mte]
    print('P5A-107 tg smile_r3 arm alpha=%.4f' % _a3, flush=True)
'''

ARM_MAE = '''    # ---- P5A-112 arm: tg MAE-optimal residual model (absolute_error HGB on char n-grams) ----
    from sklearn.metrics import mean_absolute_error as _mae_metric
    _t = 'tg'
    _idx = np.where(tr_tt == _t)[0]
    _y = resid[_idx].copy()
    _kf4 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
    _pred4 = np.zeros(len(test_smiles))
    _oof4 = np.zeros(len(_idx))
    for _tr4, _va4 in _kf4.split(_idx):
        _vec4 = CountVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
        _m4 = make_pipeline_model(_vec4, HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, max_depth=6, random_state=2026))
        _m4.fit([tr_smiles[i] for i in _idx[_tr4]], _y[_tr4])
        _pred4 += _m4.predict(test_smiles) / 5
        _oof4[_va4] = _m4.predict([tr_smiles[i] for i in _idx[_va4]])
    _best_a, _best_mae = 0.0, float(_mae_metric(_y, np.zeros_like(_y)))
    for _a in np.linspace(0.0, 1.0, 21):
        _m = float(_mae_metric(_y, _a * _oof4))
        if _m < _best_mae:
            _best_mae, _best_a = _m, float(_a)
    _a_r2 = float(np.clip(np.sum(_y * _oof4) / max(np.sum(_oof4 ** 2), 1e-12), 0.0, 1.0))
    _mte = test_tt == _t
    final[_mte] += _best_a * _pred4[_mte]
    print('P5A-112 tg-MAE alpha_mae=%.4f (oof MAE %.3f -> %.3f; alpha_r2=%.4f)' % (_best_a, float(_mae_metric(_y, np.zeros_like(_y))), _best_mae, _a_r2), flush=True)
'''

ARM_WEAKAUG = '''    # ---- P5A-113 arm: weak-target residual models with train-only random-SMILES augmentation ----
    def _random_smiles(smiles_str, n=8, seed=2026):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return [smiles_str] * n
            out = []
            for _ in range(n):
                out.append(AllChem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True))
            return out
        except Exception:
            return [smiles_str] * n
    for _t in ('ei', 'eea', 'nc', 'eps'):
        _idx = np.where(tr_tt == _t)[0]
        _y = resid[_idx].copy()
        if len(_y) < 30:
            continue
        _sm_a, _grp_a, _y_a = [], [], []
        for _k, _i in enumerate(_idx):
            _vs = _random_smiles(tr_smiles[_i], n=8, seed=2026 + _k % 97)
            _sm_a.extend(_vs)
            _grp_a.extend([_k] * len(_vs))
            _y_a.extend([_y[_k]] * len(_vs))
        _grp_a = np.array(_grp_a)
        _y_a = np.array(_y_a, dtype=float)
        _polys = np.unique(_grp_a)
        _kf5 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _pred5 = np.zeros(len(test_smiles))
        _oof5 = np.zeros(len(_idx))
        for _tr5, _va5 in _kf5.split(_polys):
            _trg = np.isin(_grp_a, _polys[_tr5])
            _vag = np.isin(_grp_a, _polys[_va5])
            _vec5 = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
            _m5 = make_pipeline_model(_vec5, RidgeModel(alpha=30.0, solver='lsqr', max_iter=5000, tol=1e-4))
            _m5.fit([_sm_a[i] for i in np.where(_trg)[0]], _y_a[_trg])
            _pred5 += _m5.predict(test_smiles) / 5
            _va_orig = np.unique(_grp_a[_vag])
            _oof5[_va_orig] = _m5.predict([tr_smiles[_idx[k]] for k in _va_orig])
        _a5 = float(np.clip(np.sum(_y * _oof5) / max(np.sum(_oof5 ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        final[_mte] += _a5 * _pred5[_mte]
        _sse = float(np.sum((_y - _a5 * _oof5) ** 2))
        _sst = float(np.sum((_y - _y.mean()) ** 2))
        print('P5A-113 weak-aug %s alpha=%.4f (oof R2 %.4f)' % (_t, _a5, 1.0 - _sse / max(_sst, 1e-9)), flush=True)
'''


ARM_NCEPS = '''    # ---- P5A-117 arm: nc-eps exact-partner consistency (nc = sqrt(eps - ionic), eps = nc^2 + ionic) ----
    _eps_d = train_df[train_df['target_type'] == 'eps'][['smiles', 'target']]
    _nc_d = train_df[train_df['target_type'] == 'nc'][['smiles', 'target']]
    _jj = _eps_d.merge(_nc_d, on='smiles', suffixes=('_e', '_n'))
    _ionic_med = float(np.median(_jj['target_e'] - _jj['target_n'] ** 2)) if len(_jj) else 0.767
    _lab_ex = {}
    for _i in range(len(oof)):
        _lab_ex.setdefault(oof['canonical'].values[_i], {})[oof['target_type'].values[_i]] = oof['target'].values[_i]
    _test_can2 = [canonicalize(s) for s in test_df['smiles'].tolist()]
    _t = 'nc'
    _idx = np.where(tr_tt == _t)[0]
    _y = resid[_idx].copy()
    _cand_oof = np.full(len(_idx), np.nan)
    for _k, _i in enumerate(_idx):
        _d = _lab_ex.get(tr_smiles[_i])
        if _d and 'eps' in _d:
            _cand_oof[_k] = np.sqrt(max(float(_d['eps']) - _ionic_med, 0.05))
    _p_oof = oof['prediction'].to_numpy(float)[_idx]
    _d_oof = _cand_oof - _p_oof
    _ok = np.isfinite(_d_oof)
    _alpha = float(np.clip(np.sum((_y[_ok]) * _d_oof[_ok]) / max(np.sum(_d_oof[_ok] ** 2), 1e-12), 0.0, 1.0)) if int(_ok.sum()) > 10 else 0.0
    _mte = test_tt == _t
    _cand_te = np.full(len(test_df), np.nan)
    for _m in np.where(_mte)[0]:
        _d = _lab_ex.get(_test_can2[_m])
        if _d and 'eps' in _d:
            _cand_te[_m] = np.sqrt(max(float(_d['eps']) - _ionic_med, 0.05))
    _d_te = _cand_te - base_target
    _use = np.isfinite(_d_te) & _mte
    final[_use] += _alpha * _d_te[_use]
    print('P5A-117 nc-eps exact: nc alpha=%.4f (n=%d)' % (_alpha, int(_ok.sum())), flush=True)
'''

ARM_MAEWEAK = '''    # ---- P5A-118 arm: weak-target MAE-tuned residual models (char Ridge, alpha by OOF MAE) ----
    from sklearn.metrics import mean_absolute_error as _mae2
    for _t in ('ei', 'eea', 'nc', 'eps'):
        _idx = np.where(tr_tt == _t)[0]
        _y = resid[_idx].copy()
        if len(_y) < 30:
            continue
        _kf6 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _pred6 = np.zeros(len(test_smiles))
        _oof6 = np.zeros(len(_idx))
        for _tr6, _va6 in _kf6.split(_idx):
            _vec6 = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
            _m6 = make_pipeline_model(_vec6, RidgeModel(alpha=30.0, solver='lsqr', max_iter=5000, tol=1e-4))
            _m6.fit([tr_smiles[i] for i in _idx[_tr6]], _y[_tr6])
            _pred6 += _m6.predict(test_smiles) / 5
            _oof6[_va6] = _m6.predict([tr_smiles[i] for i in _idx[_va6]])
        _best6, _bm6 = 0.0, float(_mae2(_y, np.zeros_like(_y)))
        for _a in np.linspace(0.0, 1.0, 21):
            _mm = float(_mae2(_y, _a * _oof6))
            if _mm < _bm6:
                _bm6, _best6 = _mm, float(_a)
        _mte = test_tt == _t
        final[_mte] += _best6 * _pred6[_mte]
        print('P5A-118 mae-weak %s alpha=%.4f (oof MAE %.3f -> %.3f)' % (_t, _best6, float(_mae2(_y, np.zeros_like(_y))), _bm6), flush=True)
'''

ARM_TGGBM = '''    # ---- P5A-119 arm: tg GBM residual model (huber HGB on char n-grams) ----
    _t = 'tg'
    _idx = np.where(tr_tt == _t)[0]
    _y = resid[_idx].copy()
    _kf7 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
    _pred7 = np.zeros(len(test_smiles))
    _oof7 = np.zeros(len(_idx))
    for _tr7, _va7 in _kf7.split(_idx):
        _vec7 = CountVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
        _m7 = make_pipeline_model(_vec7, HistGradientBoostingRegressor(loss='huber', max_iter=400, max_depth=7, l2_regularization=1.0, random_state=2026))
        _m7.fit([tr_smiles[i] for i in _idx[_tr7]], _y[_tr7])
        _pred7 += _m7.predict(test_smiles) / 5
        _oof7[_va7] = _m7.predict([tr_smiles[i] for i in _idx[_va7]])
    _a7 = float(np.clip(np.sum(_y * _oof7) / max(np.sum(_oof7 ** 2), 1e-12), 0.0, 1.0))
    _mte = test_tt == _t
    final[_mte] += _a7 * _pred7[_mte]
    _sse7 = float(np.sum((_y - _a7 * _oof7) ** 2))
    _sst7 = float(np.sum((_y - _y.mean()) ** 2))
    print('P5A-119 tg-gbm alpha=%.4f (oof R2 %.4f)' % (_a7, 1.0 - _sse7 / max(_sst7, 1e-9)), flush=True)
'''

ARM_WEAKSTACK = PARTNERS_HEAD + '''    # ---- P5A-121 arm: weak-target stacker (Ridge on partner preds + base, OOF alpha) ----
    for _t in ('ei', 'eea', 'nc', 'eps'):
        _idx = np.where(_tt_arr == _t)[0]
        _y = _y_arr[_idx]
        _p = oof['prediction'].to_numpy(float)[_idx]
        _cols = [pt for pt in ('egc', 'eea', 'nc') if pt != _t]
        _F = np.column_stack([_pred_oof[pt][_idx] for pt in _cols] + [_p])
        _F = np.nan_to_num(_F, nan=0.0)
        _Fte = np.column_stack([_pred_test[pt] for pt in _cols] + [base_target])
        _Fte = np.nan_to_num(_Fte, nan=0.0)
        _kf8 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _oof8 = np.zeros(len(_idx))
        _pred8 = np.zeros(len(test_df))
        for _tr8, _va8 in _kf8.split(_idx):
            _m8 = RidgeModel(alpha=10.0)
            _m8.fit(_F[_tr8], _y[_tr8])
            _oof8[_va8] = _m8.predict(_F[_va8])
            _pred8 += _m8.predict(_Fte) / 5
        _d8 = _oof8 - _p
        _a8 = float(np.clip(np.sum((_y - _p) * _d8) / max(np.sum(_d8 ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        final[_mte] += _a8 * (_pred8[_mte] - base_target[_mte])
        print('P5A-121 weak-stack %s alpha=%.4f' % (_t, _a8), flush=True)
'''

ARM_WEAKKERNEL = '''    # ---- P5A-122 arm: weak-target Tanimoto KRR residual models ----
    for _t in ('ei', 'eea', 'nc', 'eps'):
        _idx = np.where(tr_tt == _t)[0]
        _y = resid[_idx].copy()
        if len(_y) < 30:
            continue
        _fpA = morgan_fp_list([tr_smiles[i] for i in _idx])
        _fpB = morgan_fp_list(test_smiles)
        _Ktr = np.array([DataStructs.BulkTanimotoSimilarity(f, _fpA) for f in _fpA], dtype=np.float64)
        _Kte = np.array([DataStructs.BulkTanimotoSimilarity(f, _fpA) for f in _fpB], dtype=np.float64)
        _kfK = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _oofK = np.zeros(len(_idx))
        _predK = np.zeros(len(test_smiles))
        for _trK, _vaK in _kfK.split(_idx):
            _mK = KernelRidge(kernel='precomputed', alpha=1.0)
            _mK.fit(_Ktr[np.ix_(_trK, _trK)], _y[_trK])
            _oofK[_vaK] = _mK.predict(_Ktr[np.ix_(_vaK, _trK)])
            _predK += _mK.predict(_Kte[:, _trK]) / 5
        _aK = float(np.clip(np.sum(_y * _oofK) / max(np.sum(_oofK ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        final[_mte] += _aK * _predK[_mte]
        print('P5A-122 weak-kernel %s alpha=%.4f' % (_t, _aK), flush=True)
'''

ARM_VALCAL = '''    # ---- P5A-125 arm: per-target value-aware calibration (Ridge on [base, base^2]) ----
    for _t in TARGETS_ORDER:
        _om = oof[oof['target_type'] == _t]
        _y = _om['target'].to_numpy(float)
        _p = _om['prediction'].to_numpy(float)
        if len(_y) < 30:
            continue
        _F = np.column_stack([_p, _p ** 2])
        _kfV = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _oofV = np.zeros(len(_y))
        for _trV, _vaV in _kfV.split(_p):
            _mV = RidgeModel(alpha=5.0)
            _mV.fit(_F[_trV], _y[_trV])
            _oofV[_vaV] = _mV.predict(_F[_vaV])
        _d = _oofV - _p
        _aV = float(np.clip(np.sum((_y - _p) * _d) / max(np.sum(_d ** 2), 1e-12), 0.0, 1.0))
        _mV2 = RidgeModel(alpha=5.0)
        _mV2.fit(_F, _y)
        _mte = test_tt == _t
        _Ft = np.column_stack([base_target[_mte], base_target[_mte] ** 2])
        _cand = _mV2.predict(_Ft)
        final[_mte] += _aV * (_cand - base_target[_mte])
        print('P5A-125 value-calib %s alpha=%.4f' % (_t, _aV), flush=True)
'''

ARM_TGAUGCHAR = '''    # ---- P5A-123 arm: tg char residual model with x8 random-SMILES augmentation ----
    def _random_smiles123(smiles_str, n=8, seed=2026):
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                return [smiles_str] * n
            return [AllChem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=True) for _ in range(n)]
        except Exception:
            return [smiles_str] * n
    _t = 'tg'
    _idx = np.where(tr_tt == _t)[0]
    _y = resid[_idx].copy()
    _sm_a = []
    _grp_a = []
    for _k, _i in enumerate(_idx):
        _vs = _random_smiles123(tr_smiles[_i], n=8, seed=2026 + _k % 97)
        _sm_a.extend(_vs)
        _grp_a.extend([_k] * len(_vs))
    _grp_a = np.array(_grp_a)
    _y_a = np.repeat(_y, 8)
    _polys = np.unique(_grp_a)
    _kf9 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
    _pred9 = np.zeros(len(test_smiles))
    _oof9 = np.zeros(len(_idx))
    for _tr9, _va9 in _kf9.split(_polys):
        _trg = np.isin(_grp_a, _polys[_tr9])
        _vag = np.isin(_grp_a, _polys[_va9])
        _vec9 = CountVectorizer(analyzer='char', ngram_range=(2, 7), max_features=65536, lowercase=False)
        _m9 = make_pipeline_model(_vec9, RidgeModel(alpha=40.0, solver='lsqr', max_iter=5000, tol=1e-4))
        _m9.fit([_sm_a[i] for i in np.where(_trg)[0]], _y_a[_trg])
        _pred9 += _m9.predict(test_smiles) / 5
        _va_orig = np.unique(_grp_a[_vag])
        _oof9[_va_orig] = _m9.predict([tr_smiles[_idx[k]] for k in _va_orig])
    _a9 = float(np.clip(np.sum(_y * _oof9) / max(np.sum(_oof9 ** 2), 1e-12), 0.0, 1.0))
    _mte = test_tt == _t
    final[_mte] += _a9 * _pred9[_mte]
    print('P5A-123 tg-aug-char alpha=%.4f' % _a9, flush=True)
'''

PARTNERS_HEAD3 = '''    # ---- partner machinery v3 (egc/eea/nc/ei, exact train-partner overlay) ----
    _all_can = list(oof['canonical'])
    _test_can = [canonicalize(s) for s in test_df['smiles'].tolist()]
    _tt_arr = oof['target_type'].to_numpy(object)
    _y_arr = oof['target'].to_numpy(float)
    _pred_oof = {_pt: np.full(len(oof), np.nan) for _pt in ('egc', 'eea', 'nc', 'ei')}
    _pred_test = {_pt: np.zeros(len(test_df)) for _pt in ('egc', 'eea', 'nc', 'ei')}
    _lab_map = {}
    for _i in range(len(oof)):
        _lab_map.setdefault(oof['canonical'].values[_i], {})[oof['target_type'].values[_i]] = oof['target'].values[_i]
    for _pt in ('egc', 'eea', 'nc', 'ei'):
        _idx = np.where(_tt_arr == _pt)[0]
        _yv = _y_arr[_idx]
        _kf2 = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        for _tr2, _va2 in _kf2.split(_idx):
            _vec2 = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), max_features=32768, lowercase=False)
            _mod2 = make_pipeline_model(_vec2, RidgeModel(alpha=30.0, solver='lsqr', max_iter=3000, tol=1e-4))
            _mod2.fit([_all_can[i] for i in _idx[_tr2]], _yv[_tr2])
            _pred_oof[_pt][_idx[_va2]] = _mod2.predict([_all_can[i] for i in _idx[_va2]])
            _pred_test[_pt] += _mod2.predict(_test_can) / 5
        for _m in range(len(test_df)):
            _d = _lab_map.get(_test_can[_m])
            if _d and _pt in _d:
                _pred_test[_pt][_m] = _d[_pt]
'''

ARM_EEA = PARTNERS_HEAD3 + '''    # ---- P5A-126 arm: eea via ei - egc identity (exact partners overlaid) ----
    _t = 'eea'
    _idx = np.where(_tt_arr == _t)[0]
    _y = _y_arr[_idx]
    _p = oof['prediction'].to_numpy(float)[_idx]
    _c_oof = _pred_oof['ei'][_idx] - _pred_oof['egc'][_idx]
    _d_oof = _c_oof - _p
    _ok = np.isfinite(_d_oof)
    _alpha = float(np.clip(np.sum((_y[_ok] - _p[_ok]) * _d_oof[_ok]) / max(np.sum(_d_oof[_ok] ** 2), 1e-12), 0.0, 1.0)) if int(_ok.sum()) > 10 else 0.0
    _mte = test_tt == _t
    final[_mte] += _alpha * (_pred_test['ei'][_mte] - _pred_test['egc'][_mte] - base_target[_mte])
    print('P5A-126 eea-identity alpha=%.4f' % _alpha, flush=True)
'''

ARM_IMPSTACK = PARTNERS_HEAD3 + '''    # ---- P5A-127 arm: full imputation stacker (all partners + base + base^2, Ridge) ----
    for _t in ('ei', 'eea', 'nc', 'eps'):
        _idx = np.where(_tt_arr == _t)[0]
        _y = _y_arr[_idx]
        _p = oof['prediction'].to_numpy(float)[_idx]
        _cols = [pt for pt in ('egc', 'eea', 'nc', 'ei') if pt != _t]
        _F = np.column_stack([_pred_oof[pt][_idx] for pt in _cols] + [_p, _p ** 2])
        _F = np.nan_to_num(_F, nan=0.0)
        _Fte = np.column_stack([_pred_test[pt] for pt in _cols] + [base_target, base_target ** 2])
        _Fte = np.nan_to_num(_Fte, nan=0.0)
        _kfI = KFoldModel(n_splits=5, shuffle=True, random_state=2026)
        _oofI = np.zeros(len(_idx))
        _predI = np.zeros(len(test_df))
        for _trI, _vaI in _kfI.split(_idx):
            _mI = RidgeModel(alpha=20.0)
            _mI.fit(_F[_trI], _y[_trI])
            _oofI[_vaI] = _mI.predict(_F[_vaI])
            _predI += _mI.predict(_Fte) / 5
        _dI = _oofI - _p
        _aI = float(np.clip(np.sum((_y - _p) * _dI) / max(np.sum(_dI ** 2), 1e-12), 0.0, 1.0))
        _mte = test_tt == _t
        final[_mte] += _aI * (_predI[_mte] - base_target[_mte])
        print('P5A-127 imp-stack %s alpha=%.4f' % (_t, _aI), flush=True)
'''

ARMS = {    "kriging": ("append", ARM_KRIGING),
    "calib": ("append", ARM_CALIB),
    "ei": ("append", ARM_EI),
    "eps": ("append", ARM_EPS),
    "egb": ("append", ARM_EGB),
    "shrink": ("append", ARM_SHRINK),
    "smiler3": ("append", ARM_SMILER3),
    "mae_tg": ("append", ARM_MAE),
    "weak_aug": ("append", ARM_WEAKAUG),
    "char_tune": ("char", "    HUBER_TG = False\n" + CHAR_TUNED),
    "char_huber": ("char", "    HUBER_TG = True\n" + CHAR_TUNED),
    "spread_tune": ("spread", SPREAD_TUNED),
    "nc_eps": ("append", ARM_NCEPS),
    "mae_weak": ("append", ARM_MAEWEAK),
    "tg_gbm": ("append", ARM_TGGBM),
    "weak_stack": ("append", ARM_WEAKSTACK),
    "weak_kernel": ("append", ARM_WEAKKERNEL),
    "tg_aug_char": ("append", ARM_TGAUGCHAR),
    "valcal": ("append", ARM_VALCAL),
    "eea_id": ("append", ARM_EEA),
    "imp_stack": ("append", ARM_IMPSTACK),
}

def main():
    src, dst, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    code = open(src).read()
    for name in spec.split(","):
        kind, block = ARMS[name.strip()]
        if kind == "append":
            assert code.count(ANCHOR_FINAL) == 1, "final anchor not unique"
            code = code.replace(ANCHOR_FINAL, block + "\n" + ANCHOR_FINAL)
        elif kind == "char":
            assert code.count(CHAR_START) == 1 and code.count(CHAR_END) == 1, "char block not unique"
            i = code.index(CHAR_START)
            j = code.index(CHAR_END)
            code = code[:i] + block + "\n" + code[j:]
        elif kind == "spread":
            assert code.count(SPREAD_START) == 1 and code.count(SPREAD_END) == 1, "spread block not unique"
            i = code.index(SPREAD_START)
            j = code.index(SPREAD_END)
            code = code[:i] + block + code[j + len(SPREAD_END):]
    open(dst, "w").write(code)
    print("patched %s with %s" % (dst, spec))

if __name__ == "__main__":
    main()
