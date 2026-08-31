"""R3-C000: No-archive baseline reproduction from V52, adapted for Round 3.
Standalone, single-run pipeline without archive, without oracle.
Reads ~/Desktop/r3_runtime/train.csv, test.csv (or local Dataset/).
Writes submission.csv. Validates 4940 rows.
Based on final_compound.py but archive-free.
"""
from pathlib import Path
import hashlib, json, math, os, platform, time, warnings, sys

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 20260804
TARGETS = ["tg", "egc", "egb", "ei", "eea", "nc", "eps"]
TARGET_INDEX = {name: i for i, name in enumerate(TARGETS)}
np.random.seed(SEED)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def locate_data():
    # Try R3 runtime on GPU, then local Dataset, then R2 ppp-round-2
    candidates = [
        Path.home() / "Desktop/r3_runtime",
        Path("/Users/daver/Desktop/AISEHack 2.0 Polymr Property Prediction Round 3/Dataset"),
        Path.cwd() / "Dataset",
        Path("/kaggle/input/ppp-round-3"),
        Path("/kaggle/input/aisehack-2-0"),
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "train.csv").is_file() and (c / "test.csv").is_file():
            return c.resolve()
    raise FileNotFoundError("train.csv/test.csv not found in candidates")

DATA_DIR = locate_data()
print("data_dir:", DATA_DIR)
print("python:", platform.python_version())

# Verify hashes if possible
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
# No archive in R3
assert list(train.columns) == ["smiles", "target", "target_type"]
assert list(test.columns) == ["id", "smiles", "target_type"]
assert set(train.target_type.str.lower()) == set(TARGETS)
assert set(test.target_type.str.lower()) == set(TARGETS)
assert test.id.is_unique and np.array_equal(test.id.to_numpy(), np.arange(1, len(test) + 1))

def canonicalize(value):
    molecule = Chem.MolFromSmiles(str(value).replace("[*]", "*"))
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

for frame in (train, test):
    frame["target_type"] = frame["target_type"].astype(str).str.lower()
    frame["canonical"] = frame["smiles"].map(canonicalize)

all_keys = sorted(set(train.canonical) | set(test.canonical))
key_index = {key: i for i, key in enumerate(all_keys)}
train["structure_index"] = train.canonical.map(key_index).astype(int)
test["structure_index"] = test.canonical.map(key_index).astype(int)

n_structures = len(all_keys)
labels = np.full((n_structures, len(TARGETS)), np.nan, dtype=np.float64)
for source in (train,):
    grouped = source.groupby(["structure_index", "target_type"], as_index=False)["target"].mean()
    for row in grouped.itertuples(index=False):
        labels[int(row.structure_index), TARGET_INDEX[row.target_type]] = float(row.target)
observed = np.isfinite(labels)

summary = pd.DataFrame({
    "target": TARGETS,
    "train_rows": [int((train.target_type == t).sum()) for t in TARGETS],
    "canonical_labelled_structures": [int(observed[:, j].sum()) for j in range(len(TARGETS))],
    "test_rows": [int((test.target_type == t).sum()) for t in TARGETS],
})
print(summary.to_string())
print(train.groupby("target_type")["target"].agg(["count", "mean", "std", "min", "max"]).reindex(TARGETS).to_string())
print("unique canonical structures:", n_structures)

co_label = pd.DataFrame(observed.astype(int), columns=TARGETS).T @ pd.DataFrame(observed.astype(int), columns=TARGETS)
print("co-labelled structure counts:")
print(co_label.to_string())

from collections import Counter
from scipy import sparse
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from rdkit import DataStructs
from rdkit.Chem import Crippen, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors

molecules = []
for key in all_keys:
    molecule = Chem.MolFromSmiles(key)
    if molecule is None:
        raise ValueError("canonical structure did not parse")
    molecules.append(molecule)

descriptor_items = [(name, fn) for name, fn in Descriptors._descList if name != "Ipc"]

def descriptor_block(mols):
    output = np.full((len(mols), len(descriptor_items)), np.nan, dtype=np.float64)
    for i, molecule in enumerate(mols):
        for j, (_, fn) in enumerate(descriptor_items):
            try:
                value = float(fn(molecule))
                output[i, j] = value if math.isfinite(value) else np.nan
            except Exception:
                pass
    return output

def conjugation_stats(molecule):
    eligible = {
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() != 0 and (
            atom.GetIsAromatic() or atom.GetHybridization() in (Chem.HybridizationType.SP, Chem.HybridizationType.SP2)
        )
    }
    graph = {node: [] for node in eligible}
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in eligible and b in eligible:
            graph[a].append(b)
            graph[b].append(a)
    def farthest(start):
        distances = {start: 0}
        queue = [start]
        for node in queue:
            for nxt in graph[node]:
                if nxt not in distances:
                    distances[nxt] = distances[node] + 1
                    queue.append(nxt)
        return max(distances, key=distances.get), distances
    best = 0
    unseen = set(graph)
    while unseen:
        start = next(iter(unseen))
        first, distances = farthest(start)
        component = set(distances)
        unseen -= component
        _, diameter = farthest(first)
        best = max(best, max(diameter.values(), default=0) + 1)
    return len(eligible), best

def topology_block(mols, smiles):
    rows = []
    for molecule, text in zip(mols, smiles, strict=True):
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        charge_values = []
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)
            for atom in atoms:
                raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "nan"
                value = float(raw)
                if math.isfinite(value):
                    charge_values.append(value)
        except Exception:
            charge_values = []
        charges = np.asarray(charge_values if charge_values else [0.0], dtype=float)
        stars = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
        star_neighbors = [
            nb.GetAtomicNum() for index in stars for nb in molecule.GetAtomWithIdx(index).GetNeighbors()
        ]
        try:
            backbone = list(Chem.GetShortestPath(molecule, stars[0], stars[1])) if len(stars) == 2 else []
        except Exception:
            backbone = []
        conjugated_atoms, longest_conjugated_path = conjugation_stats(molecule)
        weights = []
        for bond in bonds:
            order = bond.GetBondTypeAsDouble()
            weights.append(order * (1.5 if bond.GetIsAromatic() else 1.0))
        eig = np.linalg.eigvalsh(np.asarray(Chem.GetAdjacencyMatrix(molecule), dtype=float)) if molecule.GetNumAtoms() else np.zeros(1)
        rows.append([
            len(text), molecule.GetNumAtoms(), molecule.GetNumHeavyAtoms(), sum(a.GetAtomicNum() == 0 for a in atoms),
            molecule.GetRingInfo().NumRings(), sum(a.GetIsAromatic() for a in atoms),
            sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms), sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in atoms),
            rdMolDescriptors.CalcNumRotatableBonds(molecule), sum(b.GetBondTypeAsDouble() == 2 for b in bonds),
            sum(b.GetBondTypeAsDouble() == 3 for b in bonds), text.count("("), Chem.GetFormalCharge(molecule),
            Descriptors.MolWt(molecule), Crippen.MolLogP(molecule), Crippen.MolMR(molecule),
            rdMolDescriptors.CalcTPSA(molecule), rdMolDescriptors.CalcLabuteASA(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule), rdMolDescriptors.CalcNumHBA(molecule), rdMolDescriptors.CalcNumHBD(molecule),
            charges.min(), charges.max(), np.abs(charges).mean(), charges.std(), conjugated_atoms, longest_conjugated_path,
            len(backbone), max(0, molecule.GetNumAtoms() - len(backbone)), len(stars), sum(star_neighbors),
            sum(weights), sum(b.GetIsAromatic() for b in bonds),
            sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms) / max(1, molecule.GetNumHeavyAtoms()),
            molecule.GetNumHeavyAtoms() / max(1, len(text)), molecule.GetRingInfo().NumRings() / max(1, molecule.GetNumHeavyAtoms()),
            sum(a.GetAtomicNum() == 8 for a in atoms), sum(a.GetAtomicNum() == 7 for a in atoms), sum(a.GetAtomicNum() == 14 for a in atoms),
            sum(a.GetAtomicNum() == 16 for a in atoms), sum(a.GetAtomicNum() == 15 for a in atoms), sum(a.GetAtomicNum() == 9 for a in atoms),
            sum(a.GetAtomicNum() == 17 for a in atoms), sum(a.GetAtomicNum() == 35 for a in atoms), sum(a.GetAtomicNum() == 53 for a in atoms),
            eig[-1] - eig[0], eig[-1], eig[0],
        ])
    return np.asarray(rows, dtype=np.float64)

def atom_token(atom):
    return "X" if atom.GetAtomicNum() == 0 else f"{atom.GetSymbol()}{atom.GetDegree()}"

def polymer_genome_block(mols, max_features=384):
    counters = []
    vocabulary = Counter()
    for molecule in mols:
        atom_types = {atom.GetIdx(): atom_token(atom) for atom in molecule.GetAtoms()}
        counts = Counter()
        for atom in molecule.GetAtoms():
            counts["S|" + atom_types[atom.GetIdx()]] += 1
        for bond in molecule.GetBonds():
            pair = sorted((atom_types[bond.GetBeginAtomIdx()], atom_types[bond.GetEndAtomIdx()]))
            counts["P|" + "-".join(pair)] += 1
        for atom in molecule.GetAtoms():
            neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
            for left in range(len(neighbors)):
                for right in range(left + 1, len(neighbors)):
                    ends = sorted((atom_types[neighbors[left]], atom_types[neighbors[right]]))
                    counts["T|" + ends[0] + "-" + atom_types[atom.GetIdx()] + "-" + ends[1]] += 1
        counters.append(counts)
        vocabulary.update(counts)
    vocab = [key for key, _ in vocabulary.most_common(max_features)]
    position = {key: i for i, key in enumerate(vocab)}
    matrix = np.zeros((len(mols), len(vocab)), dtype=np.float32)
    for i, counts in enumerate(counters):
        for key, value in counts.items():
            if key in position:
                matrix[i, position[key]] = math.log1p(value)
    return matrix, vocab

DESC = descriptor_block(molecules)
TOPO = topology_block(molecules, all_keys)
PG, PG_NAMES = polymer_genome_block(molecules)
imputer = SimpleImputer(strategy="median", keep_empty_features=True)
X_HAND = imputer.fit_transform(np.hstack([DESC, TOPO, PG]))
X_HAND[~np.isfinite(X_HAND)] = 0.0

def bit_matrix(mols, radius, bits):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    result = np.zeros((len(mols), bits), dtype=np.float32)
    for i, molecule in enumerate(mols):
        vector = np.zeros(bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        result[i] = vector
    return result

MORGAN_R2 = bit_matrix(molecules, 2, 512)
MORGAN_R3 = bit_matrix(molecules, 3, 512)
count_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
count_rows, count_cols, count_values = [], [], []
for row, molecule in enumerate(molecules):
    for column, value in count_generator.GetCountFingerprint(molecule).GetNonzeroElements().items():
        count_rows.append(row); count_cols.append(int(column)); count_values.append(math.log1p(float(value)))
MORGAN_COUNTS = sparse.csr_matrix((count_values, (count_rows, count_cols)), shape=(len(molecules), 1024), dtype=np.float32)
vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=1024, alternate_sign=False, norm="l2", lowercase=False)
CHAR = vectorizer.transform(all_keys).astype(np.float32)
UNSUPERVISED = hstack([MORGAN_COUNTS, CHAR], format="csr")
svd = TruncatedSVD(n_components=64, random_state=SEED)
SVD = svd.fit_transform(UNSUPERVISED).astype(np.float32)
X_TREE = np.hstack([X_HAND.astype(np.float32), SVD, MORGAN_R2, MORGAN_R3]).astype(np.float32)
print("feature shapes:", {"handcrafted": X_HAND.shape, "pg_vocab": len(PG_NAMES), "svd": SVD.shape, "model": X_TREE.shape})

from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

def score_r2(y, prediction):
    return float(r2_score(np.asarray(y, float), np.asarray(prediction, float)))

def make_arms(n_rows, seed_offset):
    return ["ridge", "extra_trees", "lightgbm"] if HAVE_LGB else ["ridge", "extra_trees", "hist_gradient"]

def fit_arm(name, X, y, seed):
    if name == "ridge":
        scale = StandardScaler()
        z = scale.fit_transform(X)
        return (name, scale, Ridge(alpha=18.0).fit(z, y))
    if name == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=120, max_features=0.25, min_samples_leaf=2, n_jobs=2, random_state=seed)
        return (name, None, model.fit(X, y))
    if name == "hist_gradient":
        model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=17, min_samples_leaf=max(8, len(y) // 35), l2_regularization=1.0, random_state=seed)
        return (name, None, model.fit(X, y))
    model = lgb.LGBMRegressor(n_estimators=280, learning_rate=0.04, num_leaves=25, min_child_samples=max(8, len(y) // 35), subsample=0.85, subsample_freq=1, colsample_bytree=0.40, reg_lambda=1.0, random_state=seed, n_jobs=2, verbosity=-1)
    return (name, None, model.fit(X, y))

def predict_arm(fitted, X):
    name, scale, model = fitted
    return model.predict(scale.transform(X) if scale is not None else X)

def choose_weights(y, oof, names):
    matrix = np.column_stack([oof[name] for name in names])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    target = np.asarray(y) - np.mean(y)
    weights, _ = nnls(centered, target)
    if weights.sum() <= 1e-12:
        weights = np.ones(len(names), dtype=float)
    weights = weights / weights.sum()
    candidates = [(score_r2(y, matrix @ weights + np.mean(y - matrix @ weights)), weights)]
    for index, name in enumerate(names):
        one = np.zeros(len(names)); one[index] = 1.0
        candidates.append((score_r2(y, matrix @ one), one))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], float(candidates[0][0])

def fit_zoo(X, rows, y, prediction_rows, tag, X_prediction=None):
    rows = np.asarray(rows, dtype=int)
    prediction_rows = np.asarray(prediction_rows, dtype=int)
    X_prediction = X if X_prediction is None else X_prediction
    y = np.asarray(y, dtype=float)
    names = make_arms(len(rows), SEED)
    n_splits = 3 if len(rows) >= 1200 else 5
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED + len(tag))
    oof = {name: np.zeros(len(rows), dtype=float) for name in names}
    for fold, (tr, va) in enumerate(folds.split(rows)):
        for name in names:
            fitted = fit_arm(name, X[rows[tr]], y[tr], SEED + fold + 11)
            oof[name][va] = predict_arm(fitted, X[rows[va]])
    weights, selected_r2 = choose_weights(y, oof, names)
    full = {name: predict_arm(fit_arm(name, X[rows], y, SEED + 101), X_prediction[prediction_rows]) for name in names}
    blended = np.column_stack([full[name] for name in names]) @ weights
    report = {
        "tag": tag, "rows": int(len(rows)), "folds": int(n_splits),
        "arm_r2": {name: score_r2(y, oof[name]) for name in names},
        "weights": {name: float(weights[i]) for i, name in enumerate(names)},
        "selected_oof_r2": selected_r2,
    }
    print(tag, json.dumps(report, sort_keys=True))
    return oof, blended, report

pool = {target: np.flatnonzero(np.isfinite(labels[:, TARGET_INDEX[target]])) for target in TARGETS}
all_indices = np.arange(n_structures, dtype=int)
P1_SAFE = np.full((n_structures, len(TARGETS)), np.nan, dtype=float)
P1_FULL = np.full((n_structures, len(TARGETS)), np.nan, dtype=float)
OOF1 = {}
REPORTS = {"stage1": {}, "stage2": {}}
for target in TARGETS:
    j = TARGET_INDEX[target]
    rows = pool[target]
    oof, prediction, report = fit_zoo(X_TREE, rows, labels[rows, j], all_indices, "stage1_" + target)
    weights = np.array([report["weights"][name] for name in report["weights"]])
    names = list(report["weights"])
    stage1_oof = np.column_stack([oof[name] for name in names]) @ weights
    P1_FULL[:, j] = prediction
    P1_SAFE[:, j] = prediction
    P1_SAFE[rows, j] = stage1_oof
    OOF1[target] = stage1_oof
    REPORTS["stage1"][target] = report

def context_block(exclude_index, base_predictions):
    best = np.where(np.isfinite(labels), labels, base_predictions)
    availability = np.isfinite(labels).astype(np.float32)
    best = best.copy(); availability = availability.copy()
    best[:, exclude_index] = base_predictions[:, exclude_index]
    availability[:, exclude_index] = 0.0
    columns = []
    for j in range(len(TARGETS)):
        if j != exclude_index:
            columns.extend([best[:, j], availability[:, j]])
    tg, egc, egb, ei, eea, nc, eps = [best[:, i] for i in range(len(TARGETS))]
    context = [
        (ei + eea) / 2.0, ei - eea, eea + egc, ei - egc,
        egb - egc, egb, egc, nc ** 2, eps - nc ** 2,
        nc, eps, np.sqrt(np.maximum(eps - 0.65, 0.05)),
        np.maximum(nc ** 2 + 0.65, 0.0), availability.sum(axis=1), base_predictions[:, exclude_index],
    ]
    return np.column_stack(columns + context).astype(np.float32)

P_FINAL_FULL = P1_FULL.copy()
OOF_FINAL = {target: OOF1[target].copy() for target in TARGETS}
for target in TARGETS:
    j = TARGET_INDEX[target]
    rows = pool[target]
    context_safe = context_block(j, P1_SAFE)
    context_full = context_block(j, P1_FULL)
    augmented_safe = np.hstack([X_TREE, context_safe])
    augmented_full = np.hstack([X_TREE, context_full])
    oof, prediction, report = fit_zoo(augmented_safe, rows, labels[rows, j], all_indices, "stage2_" + target, X_prediction=augmented_full)
    names = list(report["weights"])
    weights = np.array([report["weights"][name] for name in names])
    stage2_oof = np.column_stack([oof[name] for name in names]) @ weights
    stage1_r2 = score_r2(labels[rows, j], OOF1[target])
    stage2_r2 = score_r2(labels[rows, j], stage2_oof)
    use_stage2 = stage2_r2 >= stage1_r2
    if use_stage2:
        P_FINAL_FULL[:, j] = prediction
        OOF_FINAL[target] = stage2_oof
    report["stage1_oof_r2"] = stage1_r2
    report["stage2_oof_r2"] = stage2_r2
    report["selected_stage"] = "stage2" if use_stage2 else "stage1"
    REPORTS["stage2"][target] = report
    print("selected", target, report["selected_stage"], "OOF", max(stage1_r2, stage2_r2))

oof_summary = pd.DataFrame({"target": TARGETS, "stage1": [score_r2(labels[pool[t], TARGET_INDEX[t]], OOF1[t]) for t in TARGETS], "selected": [score_r2(labels[pool[t], TARGET_INDEX[t]], OOF_FINAL[t]) for t in TARGETS]})
oof_summary.loc[len(oof_summary)] = ["mean", oof_summary.stage1.mean(), oof_summary.selected.mean()]
print(oof_summary.to_string())

pair_rows = np.flatnonzero(np.isfinite(labels[:, TARGET_INDEX["eps"]]) & np.isfinite(labels[:, TARGET_INDEX["nc"]]))
eps_y = labels[pair_rows, TARGET_INDEX["eps"]]
nc_y = labels[pair_rows, TARGET_INDEX["nc"]]
ionic_y = np.maximum(eps_y - nc_y ** 2, 0.05)
ionic_oof_raw, ionic_all, ionic_report = fit_zoo(X_TREE, pair_rows, np.log(ionic_y), all_indices, "ionic_log_coordinate")
ionic_names = list(ionic_report["weights"])
ionic_weights = np.array([ionic_report["weights"][name] for name in ionic_names])
ionic_oof = np.column_stack([ionic_oof_raw[name] for name in ionic_names]) @ ionic_weights
ionic_all = np.asarray(ionic_all, dtype=float)

parent_eps = OOF_FINAL["eps"].copy()
parent_nc = OOF_FINAL["nc"].copy()
pair_positions_eps = {int(row): i for i, row in enumerate(pool["eps"])}
pair_positions_nc = {int(row): i for i, row in enumerate(pool["nc"])}
eps_parent_pair = np.array([parent_eps[pair_positions_eps[int(row)]] for row in pair_rows])
nc_parent_pair = np.array([parent_nc[pair_positions_nc[int(row)]] for row in pair_rows])
eps_route = 0.5 * eps_parent_pair + 0.5 * (nc_y ** 2 + np.exp(np.clip(ionic_oof, -8, 4)))
nc_route = 0.5 * nc_parent_pair + 0.5 * np.sqrt(np.maximum(eps_y - np.exp(np.clip(ionic_oof, -8, 4)), 0.05 ** 2))
route_metrics = {
    "eps_parent": score_r2(eps_y, eps_parent_pair), "eps_route": score_r2(eps_y, eps_route),
    "nc_parent": score_r2(nc_y, nc_parent_pair), "nc_route": score_r2(nc_y, nc_route),
}
use_ionic_route = route_metrics["eps_route"] > route_metrics["eps_parent"] and route_metrics["nc_route"] >= route_metrics["nc_parent"] - 0.003
print("ionic route:", json.dumps(route_metrics, sort_keys=True), "selected:", use_ionic_route)

candidate_oof = {target: OOF_FINAL[target].copy() for target in TARGETS}
identity_routes = {}

def select_identity_route(target, mask, raw_values, description):
    rows = pool[target]
    positions = np.flatnonzero(mask)
    if len(positions) < 8:
        return
    parent = candidate_oof[target][positions]
    y = labels[rows[positions], TARGET_INDEX[target]]
    options = {
        "direct": np.asarray(raw_values, dtype=float),
        "half_parent": 0.5 * parent + 0.5 * np.asarray(raw_values, dtype=float),
    }
    scores = {name: score_r2(y, value) for name, value in options.items()}
    parent_score = score_r2(y, parent)
    chosen = max(scores, key=scores.get)
    if scores[chosen] > parent_score:
        candidate_oof[target][positions] = options[chosen]
        identity_routes[target] = {"description": description, "mode": chosen, "score": scores[chosen], "parent_score": parent_score}
    print("identity route", target, json.dumps({"parent": parent_score, **scores}, sort_keys=True), "selected:", identity_routes.get(target, {}).get("mode", "none"))

ei_rows = pool["ei"]
ei_mask = np.isfinite(labels[ei_rows, TARGET_INDEX["eea"]]) & np.isfinite(labels[ei_rows, TARGET_INDEX["egc"]])
select_identity_route("ei", ei_mask, labels[ei_rows[ei_mask], TARGET_INDEX["eea"]] + labels[ei_rows[ei_mask], TARGET_INDEX["egc"]], "Ei = Eea + Egc")
eea_rows = pool["eea"]
eea_mask = np.isfinite(labels[eea_rows, TARGET_INDEX["ei"]]) & np.isfinite(labels[eea_rows, TARGET_INDEX["egc"]])
select_identity_route("eea", eea_mask, labels[eea_rows[eea_mask], TARGET_INDEX["ei"]] - labels[eea_rows[eea_mask], TARGET_INDEX["egc"]], "Eea = Ei - Egc")

candidate_summary = pd.DataFrame({"target": TARGETS, "selected_oof_r2": [score_r2(labels[pool[t], TARGET_INDEX[t]], candidate_oof[t]) for t in TARGETS]})
candidate_summary.loc[len(candidate_summary)] = ["mean", candidate_summary.selected_oof_r2.mean()]
print(candidate_summary.to_string())

test_predictions = np.zeros((len(test), len(TARGETS)), dtype=float)
for i, row in enumerate(test.itertuples(index=False)):
    test_predictions[i, TARGET_INDEX[row.target_type]] = P_FINAL_FULL[int(row.structure_index), TARGET_INDEX[row.target_type]]

if use_ionic_route:
    pair_positions_eps = {int(row): i for i, row in enumerate(pool["eps"])}
    pair_positions_nc = {int(row): i for i, row in enumerate(pool["nc"])}
    for row, value in zip(pair_rows, eps_route, strict=True):
        candidate_oof["eps"][pair_positions_eps[int(row)]] = value
    for row, value in zip(pair_rows, nc_route, strict=True):
        candidate_oof["nc"][pair_positions_nc[int(row)]] = value
    for i, row in enumerate(test.itertuples(index=False)):
        fi = int(row.structure_index)
        if row.target_type == "eps" and np.isfinite(labels[fi, TARGET_INDEX["nc"]]):
            test_predictions[i, TARGET_INDEX["eps"]] = 0.5 * test_predictions[i, TARGET_INDEX["eps"]] + 0.5 * (labels[fi, TARGET_INDEX["nc"]] ** 2 + np.exp(np.clip(ionic_all[fi], -8, 4)))
        if row.target_type == "nc" and np.isfinite(labels[fi, TARGET_INDEX["eps"]]):
            test_predictions[i, TARGET_INDEX["nc"]] = 0.5 * test_predictions[i, TARGET_INDEX["nc"]] + 0.5 * np.sqrt(max(labels[fi, TARGET_INDEX["eps"]] - np.exp(np.clip(ionic_all[fi], -8, 4)), 0.05 ** 2))

for i, row in enumerate(test.itertuples(index=False)):
    fi = int(row.structure_index)
    if row.target_type == "ei" and "ei" in identity_routes and np.isfinite(labels[fi, TARGET_INDEX["eea"]]) and np.isfinite(labels[fi, TARGET_INDEX["egc"]]):
        raw = labels[fi, TARGET_INDEX["eea"]] + labels[fi, TARGET_INDEX["egc"]]
        test_predictions[i, TARGET_INDEX["ei"]] = raw if identity_routes["ei"]["mode"] == "direct" else 0.5 * test_predictions[i, TARGET_INDEX["ei"]] + 0.5 * raw
    if row.target_type == "eea" and "eea" in identity_routes and np.isfinite(labels[fi, TARGET_INDEX["ei"]]) and np.isfinite(labels[fi, TARGET_INDEX["egc"]]):
        raw = labels[fi, TARGET_INDEX["ei"]] - labels[fi, TARGET_INDEX["egc"]]
        test_predictions[i, TARGET_INDEX["eea"]] = raw if identity_routes["eea"]["mode"] == "direct" else 0.5 * test_predictions[i, TARGET_INDEX["eea"]] + 0.5 * raw

# Exact canonical overrides derived ONLY from official TRAIN rows (no archive in R3)
unique_values = {}
for (canonical, target), group in train.groupby(["canonical", "target_type"]):
    values = group.target.to_numpy(float)
    if np.unique(values).size == 1:
        unique_values[(canonical, target)] = float(values[0])
override_counts = Counter()
for i, row in enumerate(test.itertuples(index=False)):
    key = (row.canonical, row.target_type)
    if key in unique_values:
        test_predictions[i, TARGET_INDEX[row.target_type]] = unique_values[key]
        override_counts[row.target_type] += 1

# Joint dielectric consistency projection for test structures carrying both rows.
by_structure = {}
for i, row in enumerate(test.itertuples(index=False)):
    by_structure.setdefault(int(row.structure_index), {})[row.target_type] = i
for fi, positions in by_structure.items():
    if "eps" in positions and "nc" in positions:
        eps_pos, nc_pos = positions["eps"], positions["nc"]
        test_predictions[eps_pos, TARGET_INDEX["eps"]] = max(test_predictions[eps_pos, TARGET_INDEX["eps"]], test_predictions[nc_pos, TARGET_INDEX["nc"]] ** 2 + 0.02)

for target in TARGETS:
    j = TARGET_INDEX[target]
    lo, hi = np.nanpercentile(labels[:, j], [0.0, 100.0])
    span = max(hi - lo, 1e-6)
    mask = test.target_type.to_numpy() == target
    test_predictions[mask, j] = np.clip(test_predictions[mask, j], lo - 0.02 * span, hi + 0.02 * span)

OUTPUT_PATH = Path.cwd() / "submission.csv"
if "OUTPUT_PATH_OVERRIDE" in os.environ:
    OUTPUT_PATH = Path(os.environ["OUTPUT_PATH_OVERRIDE"])
submission_values = np.array([test_predictions[i, TARGET_INDEX[row.target_type]] for i, row in enumerate(test.itertuples(index=False))], dtype=float)
submission = pd.DataFrame({"id": test.id.astype(int), "target": submission_values})
assert len(submission) == len(test) and submission.id.is_unique and submission.id.equals(test.id)
assert np.isfinite(submission.target.to_numpy(float)).all()
submission.to_csv(OUTPUT_PATH, index=False)
print("official canonical overrides (train only):", dict(override_counts))
print("submission rows:", len(submission), "sha256:", sha256_file(OUTPUT_PATH))
print(submission.head(10).to_string())

final_rows = []
expected = {}
for target in TARGETS:
    j = TARGET_INDEX[target]
    rows = pool[target]
    y = labels[rows, j]
    p = candidate_oof[target]
    local = score_r2(y, p)
    expected[target] = float(local)
    final_rows.append({"target": target, "selected_oof_r2": local})
expected_table = pd.DataFrame(final_rows)
expected_table.loc[len(expected_table)] = {"target": "mean", "selected_oof_r2": expected_table.selected_oof_r2.mean()}
print(expected_table.to_string())
print("expected arithmetic mean OOF:", float(np.mean(list(expected.values()))))
print("submission path:", OUTPUT_PATH.resolve())
print("submission sha256:", sha256_file(OUTPUT_PATH))
