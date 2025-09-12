# scripts/make_rules_json.py
import os, json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.tree import DecisionTreeClassifier, _tree

MODEL_PATH = "models/rf_model.pkl"
TEST_PATH  = "data/processed/test_original.csv"
OUT_PATH   = "artifacts/surrogate_rules_deploy.json"

FEATURES = [
    "MSE ML","iHR V","MSE V","MSE AP","Weigth","Age",
    "Sex (M=1, F=2)","H-Y","Gait Speed","Duration (years)"
]

THRESH = 0.40  # soglia con cui binarizziamo le prob. RF per addestrare il surrogato


def find_target_column(df: pd.DataFrame) -> str:
    """Trova la colonna target binaria del test.
    Priorità: 'Triade', 'y', 'label', 'target', 'class'.
    In alternativa, cerca una colonna numerica binaria non inclusa in FEATURES."""
    candidates = ["Triade", "y", "label", "target", "class"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: cerca una binaria non-listata
    for c in df.columns:
        if c not in FEATURES:
            vals = df[c].dropna().unique()
            if len(vals) == 2:
                return c
    # se proprio non la trova…
    raise ValueError(
        "Impossibile identificare la colonna target nel test. "
        "Aggiungi esplicitamente 'Triade' (0/1) o aggiorna 'find_target_column'."
    )


def extract_paths_with_leaf_ids(clf: DecisionTreeClassifier, feature_names):
    """Estrae per ogni foglia:
       - la lista di condizioni testuali (path)
       - l'id della foglia (leaf_id) per poterla mappare a n e p_real.
    """
    tree = clf.tree_
    paths = []

    def recurse(node, conds):
        feat_idx = tree.feature[node]
        if feat_idx != _tree.TREE_UNDEFINED:
            f = feature_names[feat_idx]
            thr = tree.threshold[node]
            # SX: <= soglia
            recurse(tree.children_left[node],  conds + [f"{f} <= {thr:.3f}"])
            # DX: > soglia
            recurse(tree.children_right[node], conds + [f"{f} > {thr:.3f}"])
        else:
            # foglia
            leaf_id = node
            paths.append({"path": conds, "leaf_id": leaf_id})
    recurse(0, [])
    return paths


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # 1) Carica RF e test
    rf = load(MODEL_PATH)
    df = pd.read_csv(TEST_PATH)

    # X e y_true
    X = df[FEATURES].copy()
    target_col = find_target_column(df)
    y_true = df[target_col].astype(int).values

    # 2) y_hat per il surrogato (binarizzazione prob. RF)
    p = rf.predict_proba(X)[:, 1]
    y_hat = (p >= THRESH).astype(int)

    # Quick check: se per caso y_hat è mono-classe, abbassa/alza la soglia in modo automatico
    if len(np.unique(y_hat)) == 1:
        # prova una soglia alternativa attorno alla mediana
        alt = float(np.median(p))
        y_hat = (p >= alt).astype(int)
        if len(np.unique(y_hat)) == 1:
            # come ultima spiaggia, prova 0.5
            y_hat = (p >= 0.5).astype(int)

    # 3) Albero surrogato (depth=3 come nei notebook)
    dt = DecisionTreeClassifier(max_depth=3, random_state=7)
    dt.fit(X, y_hat)

    # 4) Mappa foglie -> n, p_real usando y_true
    #    Per fare questo, otteniamo l'id di foglia per ciascun campione
    #    Nota: DecisionTreeClassifier.apply(X) restituisce l'id *interno* del nodo foglia
    leaf_ids = dt.apply(X)  # shape (n_samples,)
    # costruiamo statistiche per foglia
    leaf_to_stats = {}
    unique_leaves = np.unique(leaf_ids)
    for lid in unique_leaves:
        mask = (leaf_ids == lid)
        n = int(mask.sum())
        if n > 0:
            p_real = float(np.mean(y_true[mask]))  # prevalenza reale classe 1
        else:
            p_real = 0.0
        leaf_to_stats[lid] = {"n": n, "p_real": p_real}

    # 5) Estrai i path testuali e attacca n e p_real usando leaf_id
    rules = []
    for r in extract_paths_with_leaf_ids(dt, FEATURES):
        leaf_id = r["leaf_id"]
        stats = leaf_to_stats.get(leaf_id, {"n": 0, "p_real": 0.0})
        rules.append({
            "path": r["path"],
            "n": stats["n"],
            "p_real": stats["p_real"]
        })

    # 6) Salva JSON
    with open(OUT_PATH, "w") as f:
        json.dump({"rules": rules}, f, indent=2)

    print(f"✅ Salvato: {OUT_PATH}  |  #regole: {len(rules)}")
    # feedback rapido su distribuzione p_real
    if rules:
        pvals = [r["p_real"] for r in rules]
        print(f"   p_real (min/med/max): {min(pvals):.2f} / {np.median(pvals):.2f} / {max(pvals):.2f}")


if __name__ == "__main__":
    main()