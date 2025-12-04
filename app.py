import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import io, warnings, re, math
warnings.filterwarnings('ignore')

# Handle numpy import with compatibility fixes
try:
    import numpy as np
except ImportError as e:
    st.error(f"NumPy import error: {e}")
    st.stop()

# Optional: torch for .pth checkpoints
try:
    import torch
except Exception:
    torch = None

# -------------------------------
# Page configuration & CSS
# -------------------------------
st.set_page_config(
    page_title="Electoral Voter Prediction Dashboard",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.25rem;
    }
    .voter-card, .prediction-card, .family-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.75rem 0;
    }
    .voter-card {
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        border: 2px solid #1f77b4;
    }
    .family-card {
        background-color: #f9f9f9;
        border-left: 3px solid #ff7f0e;
    }
    .code-badge {
        font-size: 0.85rem;
        background: #EFF2F6;
        padding: 2px 6px;
        border-radius: 6px;
        border: 1px solid #D7DFEA;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers
# -------------------------------
def get_any(row_or_dict, *names, default=None):
    """Fetch a value by trying multiple column/field names (case-insensitive)."""
    if isinstance(row_or_dict, pd.Series):
        data = row_or_dict.to_dict()
    else:
        data = dict(row_or_dict)
    # exact first
    for n in names:
        if n in data and pd.notna(data[n]):
            return data[n]
    # case-insensitive
    lower = {k.lower(): v for k, v in data.items()}
    for n in names:
        key = n.lower()
        if key in lower and pd.notna(lower[key]):
            return lower[key]
    return default

def to_float_safe(val, default=0.0):
    """Robust float parser: handles NaN, blanks, commas, %, currency, stray text."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.number)):
        try:
            v = float(val)
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "na", "n/a", "none", "null", "-"}:
        return default
    s = s.replace(",", "")  # remove thousands separators
    # handle percentages
    if s.endswith("%"):
        try:
            v = float(s[:-1]) / 100.0
            if math.isnan(v):
                return default
            return v
        except Exception:
            return default
    # extract first numeric token (handles currency & stray letters like 'C', '₹1,200.50')
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    try:
        return float(m.group()) if m else default
    except Exception:
        return default

def nz(x):
    """Count nonzero-ish elements in a numpy array."""
    try:
        return int(np.count_nonzero(np.asarray(x)))
    except Exception:
        return 0

# -------------------------------
# Model Predictor
# -------------------------------
class VoterPredictor:
    def __init__(self):
        self.model_state_dict = None
        self.feature_names = None
        self.party_names = None
        self.scaler = None
        self.vectorizer = None
        self.booth_id_to_idx = None
        self._raw_model_data = None

        # cached numpy arrays
        self._gamma0_array = None
        self._beta_P_array = None
        self._beta_T_array = None
        self._booth_effects_P_array = None
        self._booth_effects_T_array = None
        self._alpha0_value = None

        # canonical numeric feature order we scale in:
        self.num_cols = ['land_rate', 'construction_cost', 'population', 'male_female_ratio']

        # Accept numeric feature name aliases used by various training runs
        # (model's feature_names may contain any of these)
        self.numeric_alias_to_index = {
            'land_rate': 0, 'land_rate_per_sqm': 0,
            'construction_cost': 1, 'construction_cost_per_sqm': 1,
            'population': 2,
            'male_female_ratio': 3, 'MaleToFemaleRatio': 3
        }

    # ----- utils -----
    @staticmethod
    def _to_numpy(x):
        try:
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            if hasattr(x, "numpy"):
                return x.numpy()
            return np.array(x)
        except Exception:
            return np.array(x)

    @staticmethod
    def _convert_to_numpy_like(obj):
        """Convert lists, tuples, numeric-key dicts, and simple nested dict representations into numpy arrays.
        Returns None when conversion isn't sensible.
        """
        # Torch tensors / numpy handled by _to_numpy
        try:
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj
            # torch tensor
            if hasattr(obj, 'detach') or hasattr(obj, 'numpy'):
                return VoterPredictor._to_numpy(obj)
            # lists/tuples
            if isinstance(obj, (list, tuple)):
                try:
                    return np.asarray(obj)
                except Exception:
                    return None
            # dicts that are actually sequences keyed by numeric indices (strings or ints)
            if isinstance(obj, dict):
                # keys like '0','1',... or 0,1,... -> build ordered list
                keys = list(obj.keys())
                # try to parse numeric keys
                num_keys = []
                for k in keys:
                    try:
                        num_keys.append(int(k))
                    except Exception:
                        num_keys = None
                        break
                if num_keys is not None:
                    # sort by numeric key
                    ordered = [obj[str(i)] if str(i) in obj else obj.get(i) for i in sorted(num_keys)]
                    try:
                        return np.asarray([VoterPredictor._convert_to_numpy_like(v) if not isinstance(v, (int, float, np.number)) else v for v in ordered])
                    except Exception:
                        return np.asarray(ordered)
                # fallback: sometimes sklearn saved arrays as dicts under keys like 'mean_' where value itself is a dict
                # try to convert values if they are numeric-mapped dicts
                # if values are scalars, return array of values in original key order
                simple_vals = []
                all_scalar = True
                for v in obj.values():
                    if isinstance(v, (int, float, np.number)):
                        simple_vals.append(v)
                    else:
                        all_scalar = False
                        break
                if all_scalar and simple_vals:
                    return np.asarray(simple_vals)
            return None
        except Exception:
            return None

    def _extract_column(self, arr, col_idx):
        """
        Robustly extract a 1D column vector from arr at col_idx.
        Works with numpy arrays, scipy sparse matrices, and torch tensors.
        """
        a = arr
        if hasattr(a, "detach"):
            a = a.detach()
        if hasattr(a, "cpu"):
            a = a.cpu()

        try:
            col = a[:, col_idx]
        except Exception:
            col = np.asarray(a)[:, col_idx]

        if hasattr(col, "toarray"):   # sparse
            col = col.toarray().ravel()
        elif hasattr(col, "A1"):      # sparse shortcut
            col = col.A1
        else:
            col = np.asarray(col).ravel()
        return col

    # ----- model load -----
    def load_model(self, model_file):
        """Load Torch .pth (preferred) or pickle .pkl checkpoint into memory."""
        try:
            raw = model_file.read() if hasattr(model_file, "read") else open(model_file, "rb").read()
        except Exception as e:
            st.error(f"Could not read model file: {e}")
            return False

        model_data = None

        # Try torch first (typical .pth dict)
        if torch is not None:
            try:
                md = torch.load(io.BytesIO(raw), map_location="cpu")
                # Some users save whole objects; convert to dict sensibly
                if hasattr(md, 'state_dict'):
                    # e.g., a Lightning or nn.Module checkpoint
                    model_data = {'model_state_dict': md.state_dict()}
                elif isinstance(md, dict):
                    model_data = md
                else:
                    model_data = None
            except Exception:
                model_data = None

        # Fallback to pickle
        if model_data is None:
            try:
                # Reset the BytesIO position
                raw_io = io.BytesIO(raw)
                raw_io.seek(0)
                
                # Try loading with pickle - handle numpy compatibility
                try:
                    md = pickle.load(raw_io)
                except (ImportError, ModuleNotFoundError) as numpy_err:
                    if "numpy._core" in str(numpy_err):
                        # Handle numpy._core compatibility issue
                        st.warning("Detected numpy compatibility issue. Attempting to fix...")
                        raw_io.seek(0)
                        
                        # Try with custom unpickler that handles numpy._core
                        class NumpyCompatibleUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module == 'numpy._core._multiarray_umath':
                                    module = 'numpy.core._multiarray_umath'
                                elif module == 'numpy._core.multiarray':
                                    module = 'numpy.core.multiarray'
                                elif module.startswith('numpy._core'):
                                    module = module.replace('numpy._core', 'numpy.core')
                                return super().find_class(module, name)
                        
                        unpickler = NumpyCompatibleUnpickler(raw_io)
                        md = unpickler.load()
                    else:
                        raise numpy_err
                
                if hasattr(md, 'state_dict'):
                    model_data = {'model_state_dict': md.state_dict()}
                elif isinstance(md, dict):
                    model_data = md
                else:
                    model_data = None
            except Exception as e:
                st.error(f"Unsupported model format (.pth/.pkl expected). Details: {e}")
                return False

        if not isinstance(model_data, dict):
            st.error("Model file did not contain a dict-like checkpoint.")
            return False

        # keep raw model_data for fallback lookups
        self._raw_model_data = model_data

        # Accept alternate key names commonly used
        # Try a few fallbacks if exact keys aren't present.
        def first_present(d, keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        self.model_state_dict = first_present(
            model_data,
            ['model_state_dict', 'state_dict', 'weights', 'params'],
            {}
        )

        # extract metadata
        self.feature_names = first_present(
            model_data,
            ['feature_names', 'features', 'feature_list'],
            []
        )

        self.party_names = first_present(
            model_data,
            ['party_names', 'classes', 'class_names'],
            ['BJP', 'Congress', 'AAP', 'Others', 'NOTA']
        )

        self.scaler = first_present(model_data, ['scaler', 'standardizer', 'preprocessor'], None)
        self.vectorizer = first_present(model_data, ['vectorizer', 'dict_vectorizer', 'dv'], None)
        self.booth_id_to_idx = first_present(model_data, ['booth_id_to_idx', 'booth_map', 'booth_index'], {})

        # Sometimes the actual tensors are under weird keys inside model_state_dict
        # Normalize them into canonical names if we can infer them.
        if isinstance(self.model_state_dict, dict):
            msd = self.model_state_dict

            # direct keys
            cand_beta_P = first_present(msd, ['beta_P', 'betaP', 'party_beta', 'W_party', 'linear_P.weight'], None)
            cand_beta_T = first_present(msd, ['beta_T', 'betaT', 'turnout_beta', 'W_turnout', 'linear_T.weight'], None)
            cand_gamma0 = first_present(msd, ['gamma0', 'party_bias', 'b_party', 'linear_P.bias'], None)
            cand_alpha0 = first_present(msd, ['alpha0', 'turnout_bias', 'b_turnout', 'linear_T.bias'], None)
            cand_boothP = first_present(msd, ['booth_effects_P', 'boothP', 'booth_party'], None)
            cand_boothT = first_present(msd, ['booth_effects_T', 'boothT', 'booth_turnout'], None)

            # Compose a synthetic state dict with canonical names for downstream code
            norm = {}
            if cand_beta_P is not None: norm['beta_P'] = cand_beta_P
            if cand_beta_T is not None: norm['beta_T'] = cand_beta_T
            if cand_gamma0 is not None: norm['gamma0'] = cand_gamma0
            if cand_alpha0 is not None: norm['alpha0'] = cand_alpha0
            if cand_boothP is not None: norm['booth_effects_P'] = cand_boothP
            if cand_boothT is not None: norm['booth_effects_T'] = cand_boothT

            # Only override if we actually found something
            if norm:
                self.model_state_dict = norm

        ok = bool(self.model_state_dict) and len(self.feature_names) > 0
        if not ok:
            st.error("Missing model_state_dict or feature_names in checkpoint (or keys not recognized).")
            return False

        # If scaler/vectorizer are plain dicts (from pickle), wrap into simple runnable objects
        if isinstance(self.vectorizer, dict):
            vec = self.vectorizer
            class DummyVectorizer:
                def __init__(self, d):
                    self.feature_names_ = d.get('feature_names_', d.get('feature_names', [])) or []
                    self.vocabulary_ = d.get('vocabulary_', {})
                def transform(self, records):
                    n = len(records)
                    m = len(self.feature_names_)
                    out = np.zeros((n, m), dtype=float)
                    for i, rec in enumerate(records):
                        for j, fname in enumerate(self.feature_names_):
                            # fname like 'age=Age_18-25' -> key 'age' value 'Age_18-25'
                            if '=' in fname:
                                k, v = fname.split('=', 1)
                                # value in record might be plain token or include spaces
                                if k in rec and str(rec[k]) == v:
                                    out[i, j] = 1.0
                            else:
                                # fallback: check equality with any field
                                if any(str(v) == fname for v in rec.values()):
                                    out[i, j] = 1.0
                    return out
                def get_feature_names_out(self):
                    return list(self.feature_names_)
            self.vectorizer = DummyVectorizer(vec)

        if isinstance(self.scaler, dict):
            s = self.scaler
            # attempt to extract mean_ and scale_ robustly
            mean_arr = self._convert_to_numpy_like(s.get('mean_', None))
            scale_arr = self._convert_to_numpy_like(s.get('scale_', None))
            n_in = int(s.get('n_features_in_', 4)) if isinstance(s.get('n_features_in_'), (int, float, np.number)) else 4
            if mean_arr is None or len(mean_arr) != n_in:
                mean_arr = np.zeros(n_in, dtype=float)
            if scale_arr is None or len(scale_arr) != n_in:
                scale_arr = np.ones(n_in, dtype=float)
            class DummyScaler:
                def __init__(self, mean_, scale_):
                    self.mean_ = np.asarray(mean_, dtype=float)
                    self.scale_ = np.asarray(scale_, dtype=float)
                def transform(self, X):
                    X = np.asarray(X, dtype=float)
                    # only scale the first len(mean_) columns
                    m = min(X.shape[1], self.mean_.size)
                    X[:, :m] = (X[:, :m] - self.mean_[:m]) / (self.scale_[:m] + 1e-12)
                    return X
            self.scaler = DummyScaler(mean_arr, scale_arr)

        self._preprocess_model_weights()
        return True

    def _preprocess_model_weights(self):
        try:
            def _find_and_convert(key):
                # primary: self.model_state_dict, fallback: raw model_data
                val = None
                if isinstance(self.model_state_dict, dict) and key in self.model_state_dict:
                    val = self.model_state_dict.get(key)
                if (val is None or (isinstance(val, dict) and not val)) and self._raw_model_data is not None:
                    val = self._raw_model_data.get(key, val)
                if val is None:
                    return None
                # try direct numpy conversion helpers
                arr = self._convert_to_numpy_like(val)
                if arr is not None:
                    return np.asarray(arr)
                try:
                    return self._to_numpy(val)
                except Exception:
                    return None

            if _find_and_convert('gamma0') is not None:
                self._gamma0_array = np.asarray(_find_and_convert('gamma0')).reshape(-1)
            if _find_and_convert('beta_P') is not None:
                self._beta_P_array = np.asarray(_find_and_convert('beta_P'))  # [n_feat, n_party]
                # If it's [n_party, n_feat], transpose
                if self._beta_P_array.ndim == 2 and self._beta_P_array.shape[0] == len(self.party_names) and \
                   self._beta_P_array.shape[1] == len(self.feature_names):
                    self._beta_P_array = self._beta_P_array.T
            if _find_and_convert('beta_T') is not None:
                bT = np.asarray(_find_and_convert('beta_T'))
                if bT.ndim == 2:  # squeeze common cases
                    if bT.shape[0] == len(self.feature_names):
                        bT = bT.reshape(-1)
                    elif bT.shape[1] == len(self.feature_names):
                        bT = bT.reshape(-1)
                self._beta_T_array = bT.reshape(-1)
            if _find_and_convert('booth_effects_P') is not None:
                self._booth_effects_P_array = np.asarray(_find_and_convert('booth_effects_P'))  # [n_booth, n_party]
            if _find_and_convert('booth_effects_T') is not None:
                self._booth_effects_T_array = np.asarray(_find_and_convert('booth_effects_T')).reshape(-1)  # [n_booth]
            if _find_and_convert('alpha0') is not None:
                a0 = np.asarray(_find_and_convert('alpha0'))
                try:
                    self._alpha0_value = float(a0.reshape(())) if a0.size == 1 else float(np.ravel(a0)[0])
                except Exception:
                    self._alpha0_value = None
        except Exception as e:
            st.warning(f"Warning while preprocessing weights: {e}")

    # -------------------------------
    # Preprocessing using DictVectorizer + Scaler
    # -------------------------------
    def preprocess_voter_data_vectorized(self, voter_rows):
        """Return dense feature matrix aligned to model's feature_names."""
        # normalize to list[dict]
        if isinstance(voter_rows, dict):
            voter_rows = [voter_rows]
        elif isinstance(voter_rows, pd.Series):
            voter_rows = [voter_rows.to_dict()]
        elif isinstance(voter_rows, pd.DataFrame):
            voter_rows = voter_rows.to_dict('records')

        econ_to_income = {
            "LOW INCOME AREAS": "income_low",
            "LOWER MIDDLE CLASS": "income_low",
            "MIDDLE CLASS": "income_middle",
            "UPPER MIDDLE CLASS": "income_high",
            "PREMIUM AREAS": "income_high"
        }

        # Build categorical dicts per voter compatible with DictVectorizer
        cat_dicts = []
        X_num = np.zeros((len(voter_rows), len(self.num_cols)), dtype=float)

        for i, r in enumerate(voter_rows):
            # age group from numeric age
            age_val = get_any(r, 'age', default=0) or 0
            try:
                age_int = int(float(age_val))
            except Exception:
                age_int = 0
            if 18 <= age_int <= 25: age_group = "Age_18-25"
            elif 26 <= age_int <= 35: age_group = "Age_26-35"
            elif 36 <= age_int <= 45: age_group = "Age_36-45"
            elif 46 <= age_int <= 60: age_group = "Age_46-60"
            else: age_group = "Age_60+"

            # caste token -> trained tokens
            caste_raw = str(get_any(r, 'caste', default='SC')).upper()
            caste_map = {
                "SC": "Caste_Sc", "ST": "Caste_St", "OBC": "Caste_Obc",
                "BRAHMIN": "Caste_Brahmin", "KSHATRIYA": "Caste_Kshatriya",
                "VAISHYA": "Caste_Vaishya", "NO CASTE SYSTEM": "Caste_No_caste_system"
            }
            caste_tok = caste_map.get(caste_raw, f"Caste_{caste_raw.title()}")

            # religion token
            rel_raw = str(get_any(r, 'religion', default='HINDU')).upper()
            rel_map = {
                "HINDU": "Religion_Hindu", "MUSLIM": "Religion_Muslim",
                "SIKH": "Religion_Sikh", "CHRISTIAN": "Religion_Christian",
                "BUDDHIST": "Religion_Buddhist", "JAIN": "Religion_Jain"
            }
            rel_tok = rel_map.get(rel_raw, "Religion_Hindu")

            econ_raw = str(get_any(r, 'economic_category', default='MIDDLE CLASS')).upper()
            income_tok = econ_to_income.get(econ_raw, "income_middle")
            locality = str(get_any(r, 'Locality', 'locality', default=''))

            cat = {
                "age": age_group,
                "caste": caste_tok,
                "religion": rel_tok,
                "economic": econ_raw,
                "income": income_tok,
                "locality": locality,
                "econ_loc": f"{econ_raw} | {locality}"
            }
            cat_dicts.append(cat)

            # numeric columns in the scaler order (SAFE PARSING):
            X_num[i, 0] = to_float_safe(get_any(r, 'land_rate_per_sqm', 'land_rate', default=0.0), default=0.0)
            X_num[i, 1] = to_float_safe(get_any(r, 'construction_cost_per_sqm', 'construction_cost', default=0.0), default=0.0)
            X_num[i, 2] = to_float_safe(get_any(r, 'population', default=0.0), default=0.0)
            X_num[i, 3] = to_float_safe(get_any(r, 'MaleToFemaleRatio', 'male_female_ratio', default=1.0), default=1.0)

        # Encode categoricals
        if self.vectorizer is None:
            st.warning("Model checkpoint missing DictVectorizer; falling back to zero features for categoricals.")
            X_cat = None
            vec_feats = []
            vec_idx = {}
        else:
            X_cat = self.vectorizer.transform(cat_dicts)  # dense or sparse depending on saved vectorizer
            vec_feats = getattr(self.vectorizer, 'feature_names_', None)
            if vec_feats is None and hasattr(self.vectorizer, 'get_feature_names_out'):
                vec_feats = list(self.vectorizer.get_feature_names_out())
            if vec_feats is None:
                vec_feats = []
            vec_idx = {n: i for i, n in enumerate(vec_feats)}

            # Build mapping from checkpoint feature_names -> vectorizer column index.
            feature_to_vec_idx = {}
            # exact matches
            for i_f, fname in enumerate(self.feature_names):
                if fname in vec_idx:
                    feature_to_vec_idx[fname] = vec_idx[fname]

            # normalized matches (case-insensitive, collapse spaces)
            if len(feature_to_vec_idx) < len(self.feature_names):
                norm_vec = {str(n).lower().strip(): i for i, n in enumerate(vec_feats)}
                for fname in self.feature_names:
                    if fname in feature_to_vec_idx:
                        continue
                    nf = str(fname).lower().strip()
                    if nf in norm_vec:
                        feature_to_vec_idx[fname] = norm_vec[nf]
                    else:
                        # try collapsing multiple spaces and punctuation
                        nf2 = re.sub(r"\s+"," ", re.sub(r"[^0-9a-zA-Z=| ]+","", nf))
                        if nf2 in norm_vec:
                            feature_to_vec_idx[fname] = norm_vec[nf2]

            # If still missing and lengths match, assume index-alignment fallback
            used_index_alignment = False
            if len(feature_to_vec_idx) < len(self.feature_names) and len(vec_feats) == len(self.feature_names):
                used_index_alignment = True
                for j, fname in enumerate(self.feature_names):
                    if fname not in feature_to_vec_idx:
                        feature_to_vec_idx[fname] = j

            # final vec_idx mapping used for extraction: map feature name -> vec column index
            vec_idx = feature_to_vec_idx
            # persist for debug
            try:
                self._used_index_alignment = bool(used_index_alignment)
            except Exception:
                self._used_index_alignment = False

        # Scale numerics
        if hasattr(self.scaler, 'transform'):
            try:
                X_num = self.scaler.transform(X_num)
            except Exception:
                pass  # keep unscaled

        # Stitch into dense [n x len(feature_names)] aligned to checkpoint order
        X = np.zeros((len(voter_rows), len(self.feature_names)), dtype=np.float32)

        for j, fname in enumerate(self.feature_names):
            if fname in vec_idx and X_cat is not None:
                # robust extraction (works for dense or sparse)
                X[:, j] = self._extract_column(X_cat, vec_idx[fname])
            else:
                # numeric by alias (handles land_rate vs land_rate_per_sqm, etc.)
                if fname in self.numeric_alias_to_index:
                    k = self.numeric_alias_to_index[fname]
                    X[:, j] = X_num[:, k]
                else:
                    # unknown feature; remain zero
                    pass

        return X

    # -------------------------------
    # Vectorized Prediction
    # -------------------------------
    def predict_voters_vectorized(self, voter_rows):
        # convert to list[dict] for booth id extraction
        if isinstance(voter_rows, dict):
            voter_list = [voter_rows]
        elif isinstance(voter_rows, pd.Series):
            voter_list = [voter_rows.to_dict()]
        elif isinstance(voter_rows, pd.DataFrame):
            voter_list = voter_rows.to_dict('records')
        else:
            voter_list = list(voter_rows)

        X = self.preprocess_voter_data_vectorized(voter_list)
        n = len(voter_list)

        # booth indices with forced 2025 year
        booth_indices = []
        for r in voter_list:
            part_no = to_float_safe(get_any(r, 'partno', 'part_no', 'booth_no', default=1), default=1.0)
            part_no = int(part_no) if part_no == part_no else 1  # NaN check
            booth_id = f"{part_no}_2025"
            idx = self.booth_id_to_idx.get(booth_id, None)
            booth_indices.append(idx if idx is not None else -1)
        booth_indices = np.array(booth_indices)

        # Party logits
        if self._beta_P_array is not None and self._gamma0_array is not None:
            logits = X @ self._beta_P_array  # [n, n_party]
            logits = logits + self._gamma0_array[np.newaxis, :]

            if self._booth_effects_P_array is not None and self._booth_effects_P_array.size > 0:
                be = self._booth_effects_P_array
                mean_be = be.mean(axis=0, keepdims=True)
                booth_eff = np.zeros_like(logits)
                for i in range(n):
                    bi = booth_indices[i]
                    booth_eff[i] = be[bi] if (bi is not None and 0 <= bi < be.shape[0]) else mean_be
                logits = logits + booth_eff

            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 0.001, 0.999)
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = np.full((n, len(self.party_names)), 1.0 / max(1, len(self.party_names)))

        # Turnout
        if self._beta_T_array is not None and self._alpha0_value is not None:
            t_logits = (X @ self._beta_T_array) + self._alpha0_value
            if self._booth_effects_T_array is not None and self._booth_effects_T_array.size > 0:
                beT = self._booth_effects_T_array
                add = np.zeros(n, dtype=float)
                for i in range(n):
                    bi = booth_indices[i]
                    add[i] = beT[bi] if (bi is not None and 0 <= bi < beT.shape[0]) else float(beT.mean())
                t_logits = t_logits + add
            turnout = 1 / (1 + np.exp(-t_logits))
            turnout = np.clip(turnout, 0.01, 0.99)
        else:
            turnout = np.full(n, 0.75, dtype=float)

        # Package
        results = []
        for i in range(n):
            pvec = probs[i]
            party_prob_dict = dict(zip(self.party_names, pvec))
            pred_party = self.party_names[int(np.argmax(pvec))]
            results.append({
                "turnout_probability": float(turnout[i]),
                "party_probabilities": party_prob_dict,
                "predicted_party": pred_party
            })
        return results

    # -------------------------------
    # Debug helpers
    # -------------------------------
    def status(self):
        """Return dict of model/debug info for UI."""
        return {
            "feature_names": len(self.feature_names),
            "party_names": self.party_names,
            "beta_P_shape": None if self._beta_P_array is None else tuple(self._beta_P_array.shape),
            "beta_T_shape": None if self._beta_T_array is None else tuple(self._beta_T_array.shape),
            "gamma0_shape": None if self._gamma0_array is None else tuple(self._gamma0_array.shape),
            "alpha0": self._alpha0_value,
            "booth_P_shape": None if self._booth_effects_P_array is None else tuple(self._booth_effects_P_array.shape),
            "booth_T_len": None if self._booth_effects_T_array is None else self._booth_effects_T_array.size,
            "vectorizer_present": self.vectorizer is not None,
            "scaler_present": hasattr(self.scaler, 'transform'),
        }

    def debug_summary(self, sample_records=None, max_preview=5):
        """Return a detailed debug dict showing arrays, previews and feature-match diagnostics."""
        out = {}
        try:
            out['beta_P_shape'] = None if self._beta_P_array is None else tuple(self._beta_P_array.shape)
            if self._beta_P_array is not None:
                b = np.asarray(self._beta_P_array)
                out['beta_P_preview'] = b.reshape(-1)[:max_preview].tolist()
                out['beta_P_stats'] = {
                    'mean': float(np.mean(b)), 'std': float(np.std(b)), 'nonzero': int(np.count_nonzero(b))
                }
        except Exception:
            out['beta_P_preview'] = None

        try:
            out['gamma0_shape'] = None if self._gamma0_array is None else tuple(self._gamma0_array.shape)
            out['gamma0_preview'] = None if self._gamma0_array is None else np.asarray(self._gamma0_array).reshape(-1)[:max_preview].tolist()
        except Exception:
            out['gamma0_preview'] = None

        try:
            out['beta_T_shape'] = None if self._beta_T_array is None else tuple(self._beta_T_array.shape)
            out['beta_T_preview'] = None if self._beta_T_array is None else np.asarray(self._beta_T_array).reshape(-1)[:max_preview].tolist()
        except Exception:
            out['beta_T_preview'] = None

        out['alpha0'] = self._alpha0_value

        try:
            out['booth_P_shape'] = None if self._booth_effects_P_array is None else tuple(self._booth_effects_P_array.shape)
            out['booth_T_len'] = None if self._booth_effects_T_array is None else int(self._booth_effects_T_array.size)
        except Exception:
            out['booth_P_shape'] = None
            out['booth_T_len'] = None

        # feature matching diagnostics
        try:
            fv = list(self.feature_names) if self.feature_names is not None else []
            vec_feats = []
            if hasattr(self.vectorizer, 'feature_names_') and getattr(self.vectorizer, 'feature_names_', None) is not None:
                vec_feats = list(self.vectorizer.feature_names_)
            elif hasattr(self.vectorizer, 'get_feature_names_out'):
                vec_feats = list(self.vectorizer.get_feature_names_out())
            out['n_checkpoint_features'] = len(fv)
            out['n_vectorizer_features'] = len(vec_feats)
            # count how many checkpoint feature_names exist in vectorizer features
            vec_set = set([str(x).lower().strip() for x in vec_feats])
            matches = [f for f in fv if str(f).lower().strip() in vec_set]
            out['matched_feature_count'] = len(matches)
            out['matched_first_10'] = matches[:10]
            # mapping preview: for first N checkpoint features, show mapped vectorizer index/name if any
            mapping_preview = []
            for fname in fv[:50]:
                mapped = None
                mapped_name = None
                try:
                    # look up exact or normalized name
                    if fname in getattr(self.vectorizer, 'feature_names_', []) if hasattr(self.vectorizer, 'feature_names_') else False:
                        mapped = list(self.vectorizer.feature_names_).index(fname)
                        mapped_name = fname
                    else:
                        # try get_feature_names_out list
                        out_feats = []
                        if hasattr(self.vectorizer, 'get_feature_names_out'):
                            out_feats = list(self.vectorizer.get_feature_names_out())
                        elif hasattr(self.vectorizer, 'feature_names_'):
                            out_feats = list(self.vectorizer.feature_names_)
                        if out_feats:
                            nf = str(fname).lower().strip()
                            for i_of, ofn in enumerate(out_feats):
                                if str(ofn).lower().strip() == nf:
                                    mapped = i_of
                                    mapped_name = ofn
                                    break
                except Exception:
                    mapped = None
                    mapped_name = None
                mapping_preview.append({'feature_name': fname, 'vec_index': mapped, 'vec_feat_name': mapped_name})
            out['feature_mapping_preview'] = mapping_preview
        except Exception:
            out['n_checkpoint_features'] = None
            out['n_vectorizer_features'] = None
            out['matched_feature_count'] = None
        out['used_index_alignment'] = bool(getattr(self, '_used_index_alignment', False))

        # sample feature matrix diagnostics
        try:
            sample = sample_records or []
            if sample and not isinstance(sample, list):
                if isinstance(sample, pd.DataFrame):
                    sample = sample.to_dict('records')
                elif isinstance(sample, pd.Series):
                    sample = [sample.to_dict()]
                else:
                    sample = list(sample)
            if sample:
                # Build both the full feature matrix and also capture intermediate encodings
                Xs = self.preprocess_voter_data_vectorized(sample)
                out['X_shape'] = Xs.shape
                out['X_nonzero'] = int(np.count_nonzero(Xs))
                # per-row which feature indices are nonzero (first few)
                nz_idx = []
                for i in range(min(Xs.shape[0], 5)):
                    nz_idx.append(np.where(np.abs(Xs[i]) > 1e-8)[0].tolist())
                out['X_nonzero_indices_sample'] = nz_idx
                # build intermediate cat_dicts and X_num for the sample to show mapping
                try:
                    # replicate small portion of preprocessing to expose values
                    sample_records_list = sample if isinstance(sample, list) else list(sample)
                    cat_dicts = []
                    X_num_raw = []
                    for r in sample_records_list[:5]:
                        # build same tokens used in preprocess_voter_data_vectorized
                        age_val = get_any(r, 'age', default=0) or 0
                        try:
                            age_int = int(float(age_val))
                        except Exception:
                            age_int = 0
                        if 18 <= age_int <= 25: age_group = "Age_18-25"
                        elif 26 <= age_int <= 35: age_group = "Age_26-35"
                        elif 36 <= age_int <= 45: age_group = "Age_36-45"
                        elif 46 <= age_int <= 60: age_group = "Age_46-60"
                        else: age_group = "Age_60+"

                        caste_raw = str(get_any(r, 'caste', default='SC')).upper()
                        caste_map = {
                            "SC": "Caste_Sc", "ST": "Caste_St", "OBC": "Caste_Obc",
                            "BRAHMIN": "Caste_Brahmin", "KSHATRIYA": "Caste_Kshatriya",
                            "VAISHYA": "Caste_Vaishya", "NO CASTE SYSTEM": "Caste_No_caste_system"
                        }
                        caste_tok = caste_map.get(caste_raw, f"Caste_{caste_raw.title()}")

                        rel_raw = str(get_any(r, 'religion', default='HINDU')).upper()
                        rel_map = {
                            "HINDU": "Religion_Hindu", "MUSLIM": "Religion_Muslim",
                            "SIKH": "Religion_Sikh", "CHRISTIAN": "Religion_Christian",
                            "BUDDHIST": "Religion_Buddhist", "JAIN": "Religion_Jain"
                        }
                        rel_tok = rel_map.get(rel_raw, "Religion_Hindu")

                        econ_raw = str(get_any(r, 'economic_category', default='MIDDLE CLASS')).upper()
                        econ_to_income = {
                            "LOW INCOME AREAS": "income_low",
                            "LOWER MIDDLE CLASS": "income_low",
                            "MIDDLE CLASS": "income_middle",
                            "UPPER MIDDLE CLASS": "income_high",
                            "PREMIUM AREAS": "income_high"
                        }
                        income_tok = econ_to_income.get(econ_raw, "income_middle")
                        locality = str(get_any(r, 'Locality', 'locality', default=''))

                        cat = {
                            "age": age_group,
                            "caste": caste_tok,
                            "religion": rel_tok,
                            "economic": econ_raw,
                            "income": income_tok,
                            "locality": locality,
                            "econ_loc": f"{econ_raw} | {locality}"
                        }
                        cat_dicts.append(cat)

                        # numeric raw
                        lr = to_float_safe(get_any(r, 'land_rate_per_sqm', 'land_rate', default=0.0), default=0.0)
                        cc = to_float_safe(get_any(r, 'construction_cost_per_sqm', 'construction_cost', default=0.0), default=0.0)
                        pop = to_float_safe(get_any(r, 'population', default=0.0), default=0.0)
                        mfr = to_float_safe(get_any(r, 'MaleToFemaleRatio', 'male_female_ratio', default=1.0), default=1.0)
                        X_num_raw.append([lr, cc, pop, mfr])

                    out['sample_cat_dicts'] = cat_dicts
                    out['sample_X_num_raw'] = X_num_raw
                    # attempt to show scaled version
                    try:
                        X_num_arr = np.asarray(X_num_raw, dtype=float)
                        if hasattr(self.scaler, 'transform'):
                            X_num_scaled = self.scaler.transform(X_num_arr.copy())
                        else:
                            X_num_scaled = X_num_arr
                        out['sample_X_num_scaled'] = X_num_scaled.tolist()
                    except Exception:
                        out['sample_X_num_scaled'] = None
                    # also include the dense X rows for these samples
                    try:
                        out['sample_X_rows'] = [Xs[i].tolist() for i in range(min(Xs.shape[0], len(cat_dicts)))]
                    except Exception:
                        out['sample_X_rows'] = None
                except Exception:
                    out['sample_cat_dicts'] = None
                    out['sample_X_num_raw'] = None
                    out['sample_X_num_scaled'] = None
                    out['sample_X_rows'] = None
            else:
                out['X_shape'] = None
        except Exception as e:
            out['X_shape'] = None
            out['X_nonzero'] = None
        # raw model_data key/type preview + numeric diagnostics for common keys
        try:
            raw = self._raw_model_data or {}
            keys = list(raw.keys())
            out['raw_keys'] = keys[:50]
            raw_types = {}
            for k in keys[:50]:
                v = raw.get(k)
                t = type(v).__name__
                raw_types[k] = t
            out['raw_key_types_preview'] = raw_types
            # for some common numeric keys, show shape/min/max when possible
            numeric_info = {}
            for k in ['beta_P','beta_T','gamma0','alpha0','booth_effects_P','booth_effects_T']:
                if k in raw:
                    try:
                        arr = self._convert_to_numpy_like(raw[k])
                        if arr is None:
                            arr = self._to_numpy(raw[k])
                        a = np.asarray(arr)
                        numeric_info[k] = {
                            'shape': a.shape,
                            'min': float(np.min(a)) if a.size>0 else None,
                            'max': float(np.max(a)) if a.size>0 else None,
                            'nonzero': int(np.count_nonzero(a))
                        }
                    except Exception as e:
                        numeric_info[k] = {'error': str(e)}
            out['raw_numeric_preview'] = numeric_info
        except Exception:
            out['raw_keys'] = None
        return out

# -------------------------------
# UI pieces
# -------------------------------
def load_voter_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading voter data: {e}")
        return None

def display_voter_info(voter_row_dict):
    st.markdown('<div class="voter-card">', unsafe_allow_html=True)
    st.subheader("👤 Voter Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Name:** {get_any(voter_row_dict, 'name', default='N/A')}")
        st.write(f"**Age:** {get_any(voter_row_dict, 'age', default='N/A')}")
        st.write(f"**Gender:** {get_any(voter_row_dict, 'gender', default='N/A')}")
        st.write(f"**Booth No:** {get_any(voter_row_dict, 'partno', 'part_no', 'booth_no', default='N/A')}")
    with col2:
        st.write(f"**Religion:** {get_any(voter_row_dict, 'religion', default='N/A')}")
        st.write(f"**Caste:** {get_any(voter_row_dict, 'caste', default='N/A')}")
        st.write(f"**Economic Category:** {get_any(voter_row_dict, 'economic_category', default='N/A')}")
        st.write(f"**Locality:** {get_any(voter_row_dict, 'Locality', 'locality', default='N/A')}")
    with col3:
        st.write(f"**Address:** {get_any(voter_row_dict, 'full address', 'address', default='N/A')}")
        st.write(f"**House No:** {get_any(voter_row_dict, 'house number', 'house_number', 'house no', default='N/A')}")
        st.write(f"**Population:** {get_any(voter_row_dict, 'population', default='N/A')}")
        lr = get_any(voter_row_dict, 'land_rate_per_sqm', 'land_rate', default='N/A')
        st.write(f"**Land Rate:** ₹{lr}/sqm")
    st.markdown('</div>', unsafe_allow_html=True)

def display_predictions(pred):
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("🎯 Predictions")

    col1, col2 = st.columns(2)
    with col1:
        turnout_pct = pred['turnout_probability'] * 100
        emoji = "🟢" if turnout_pct > 70 else "🟡" if turnout_pct > 50 else "🔴"
        st.metric(
            label="Turnout Probability",
            value=f"{pred['turnout_probability']:.1%}",
            delta=f"{emoji} {'High' if turnout_pct > 70 else 'Medium' if turnout_pct > 50 else 'Low'} likelihood"
        )
        # Gauge
        fig_turnout = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=turnout_pct,
            title={'text': "Turnout Likelihood (%)"},
            delta={'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        fig_turnout.update_layout(height=300)
        st.plotly_chart(fig_turnout, use_container_width=True)

    with col2:
        party_probs = pred['party_probabilities']
        parties = list(party_probs.keys())
        probs = [p * 100 for p in party_probs.values()]
        max_prob = max(party_probs.values())
        conf_label = "High" if max_prob > 0.4 else "Medium" if max_prob > 0.25 else "Low"
        conf_emoji = "🎯" if max_prob > 0.4 else "🎲" if max_prob > 0.25 else "❓"

        st.metric(
            label="Most Likely Party",
            value=pred['predicted_party'],
            delta=f"{conf_emoji} {max_prob:.1%} confidence ({conf_label})"
        )

        party_colors = {
            'BJP': '#FF6B35',
            'Congress': '#4472C4',
            'AAP': '#70AD47',
            'Others': '#FFC000',
            'NOTA': '#C55A5A'
        }
        colors = [party_colors.get(p, '#888888') for p in parties]
        fig_party = go.Figure(data=[
            go.Bar(x=parties, y=probs, marker_color=colors,
                   text=[f'{p:.1f}%' for p in probs], textposition='auto',
                   hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>')
        ])
        fig_party.update_layout(
            title="Party Preference Probabilities",
            xaxis_title="Political Party", yaxis_title="Probability (%)",
            height=300, showlegend=False, yaxis=dict(range=[0, max(probs) * 1.2]),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_party, use_container_width=True)

    st.subheader("📊 Detailed Probability Breakdown")
    rows = []
    sorted_parties = sorted(party_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (party, p) in enumerate(sorted_parties):
        medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i] if i < 5 else f"{i+1}️⃣"
        rows.append({
            "Rank": f"{medal} {i+1}",
            "Party": party,
            "Probability": f"{p:.3f}",
            "Percentage": f"{p*100:.2f}%",
            "Confidence": "High" if p > 0.4 else "Medium" if p > 0.25 else "Low"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick interpretation
    st.subheader("🔍 Interpretation")
    c1, c2 = st.columns(2)
    with c1:
        if pred['turnout_probability'] > 0.75:
            st.success("🟢 Very likely to vote - highly engaged voter")
        elif pred['turnout_probability'] > 0.50:
            st.warning("🟡 Moderately likely to vote - average engagement")
        else:
            st.error("🔴 Less likely to vote - low engagement")
    with c2:
        max_prob = max(pred['party_probabilities'].values())
        if max_prob > 0.4:
            st.success(f"🎯 Strong preference for {pred['predicted_party']}")
        elif max_prob > 0.25:
            st.warning(f"🎲 Moderate preference for {pred['predicted_party']}")
        else:
            st.info("❓ Uncertain - voter could swing between parties")

    # Competitiveness
    vals = sorted([v for v in pred['party_probabilities'].values()], reverse=True)
    if len(vals) >= 2:
        margin = vals[0] - vals[1]
        if margin < 0.05:
            st.info("🔥 **Very close race!** Top two parties are within 5 percentage points.")
        elif margin < 0.10:
            st.info("⚡ **Competitive race!** Top two parties are within 10 percentage points.")

    st.markdown('</div>', unsafe_allow_html=True)

def _build_family_frame(voter_df, key_name, key_value):
    """Return subset of voter_df where key_name == key_value (handles NaNs and types)."""
    if key_value is None:
        return voter_df.iloc[0:0]  # empty
    # Use apply + get_any to be tolerant to Series inputs
    fam_df = voter_df[voter_df.apply(
        lambda r: get_any(r, key_name, default=None) == key_value, axis=1
    )]
    return fam_df

def _render_family_table(fam_df, predictor, title):
    st.markdown(f"**{title}**")
    if fam_df.empty:
        st.info("No members found.")
        return

    preds = predictor.predict_voters_vectorized(fam_df)
    rows = []
    for (idx, row), pred in zip(fam_df.iterrows(), preds):
        rows.append({
            "Name": get_any(row, 'name', default='N/A'),
            "Age": get_any(row, 'age', default='N/A'),
            # Removed "Relation"
            "Turnout Prob": f"{pred['turnout_probability']:.1%}",
            "Predicted Party": pred["predicted_party"],
            "Confidence": f"{max(pred['party_probabilities'].values()):.1%}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick aggregate view: mean party probabilities
    party_names = list(preds[0]['party_probabilities'].keys())
    party_means = {p: np.mean([pr['party_probabilities'][p] for pr in preds]) for p in party_names}
    fig = px.bar(
        x=list(party_means.keys()),
        y=[v*100 for v in party_means.values()],
        labels={'x': 'Party', 'y': 'Avg Probability (%)'},
        title=f"Average Party Support — {title}"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_family_block(selected_voter, voter_df, predictor):
    st.markdown('<div class="family-card">', unsafe_allow_html=True)
    st.subheader("👨‍👩‍👧‍👦 Family Analysis")

    # Get the IDs for both family systems from the selected voter
    core_id  = get_any(selected_voter, 'core_family_id', default=None)
    chain_id = get_any(selected_voter, 'family_by_chain_id', default=None)

    if core_id is None and chain_id is None:
        st.info("No family identifiers (`core_family_id` / `family_by_chain_id`) available for this voter.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Build frames
    core_df  = _build_family_frame(voter_df, 'core_family_id', core_id) if core_id is not None else voter_df.iloc[0:0]
    chain_df = _build_family_frame(voter_df, 'family_by_chain_id', chain_id) if chain_id is not None else voter_df.iloc[0:0]

    # Two columns side by side
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Core Family ID: `{core_id}`" if core_id is not None else "Core Family ID: N/A")
        _render_family_table(core_df, predictor, "Core Family")
    with c2:
        st.caption(f"Chain Family ID: `{chain_id}`" if chain_id is not None else "Chain Family ID: N/A")
        _render_family_table(chain_df, predictor, "Chain Family")

    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# Main App
# -------------------------------
def main():
    st.markdown('<h1 class="main-header">🗳️ Electoral Voter Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar uploads
    st.sidebar.header("📁 File Uploads")

    model_file = st.sidebar.file_uploader(
        "Upload Trained Model (.pth or .pkl)",
        type=['pth', 'pkl'],
        help="Upload your trained electoral prediction model"
    )
    voter_file = st.sidebar.file_uploader(
        "Upload Voter Data (Excel)",
        type=['xlsx', 'xls'],
        help="Upload Excel file with voter information"
    )

    predictor = VoterPredictor()

    # Load model
    if model_file is None:
        st.warning("Please upload a trained model file to begin.")
        return
    with st.spinner("Loading model..."):
        if not predictor.load_model(model_file):
            return
    st.sidebar.success("✅ Model loaded!")

    # Load voters
    if voter_file is None:
        st.info("Please upload voter data file to begin analysis.")
        return
    with st.spinner("Loading voter data..."):
        voter_df = load_voter_data(voter_file)
        if voter_df is None or voter_df.empty:
            st.error("No voter data found.")
            return

        # Defensive numeric clean for key columns
        for col in [
            'land_rate_per_sqm','land_rate',
            'construction_cost_per_sqm','construction_cost',
            'population','MaleToFemaleRatio','male_female_ratio','age','partno','part_no','booth_no'
        ]:
            if col in voter_df.columns:
                voter_df[col] = voter_df[col].apply(lambda x: to_float_safe(x, default=np.nan))

    st.sidebar.success(f"✅ Loaded {len(voter_df)} voter records")

    # ---------------- Model Debug Panel ----------------
    with st.expander("🔧 Model Debug"):
        status = predictor.status()
        st.write("**Checkpoint summary**")
        st.write({
            "n_feature_names": status["feature_names"],
            "party_names": status["party_names"],
            "beta_P_shape": status["beta_P_shape"],
            "beta_T_shape": status["beta_T_shape"],
            "gamma0_shape": status["gamma0_shape"],
            "alpha0": status["alpha0"],
            "booth_P_shape": status["booth_P_shape"],
            "booth_T_len": status["booth_T_len"],
            "vectorizer_present": status["vectorizer_present"],
            "scaler_present": status["scaler_present"],
        })

        # Small probe: take 5 rows (if exist), build X, show basic stats
        try:
            sample = voter_df.head(5).to_dict('records') if len(voter_df) > 0 else []
            if sample:
                Xs = predictor.preprocess_voter_data_vectorized(sample)
                st.write(f"Feature matrix shape: {Xs.shape}  ", f"<span class='code-badge'>nonzero={int(np.count_nonzero(Xs))}</span>", unsafe_allow_html=True)
                if predictor._beta_P_array is not None and predictor._gamma0_array is not None:
                    L = Xs @ predictor._beta_P_array + predictor._gamma0_array[np.newaxis, :]
                    st.write({
                        "logits_mean": float(np.mean(L)),
                        "logits_std": float(np.std(L)),
                        "logits_nonzero": int(np.count_nonzero(L))
                    })
                    st.caption("If logits_std ≈ 0 and logits_nonzero is tiny, the model isn’t getting signal (likely feature misalignment).")
                else:
                    st.warning("beta_P/gamma0 missing -> falling back to uniform party probabilities.")
        except Exception as e:
            st.warning(f"Debug probe failed: {e}")

        # Show first 20 feature names so you can eyeball alignment
        fn_preview = predictor.feature_names[:20]
        st.write("**First 20 feature_names from checkpoint**", fn_preview)

        # Detailed debug summary (previews array stats, feature matching, and sample X diagnostics)
        try:
            dbg = predictor.debug_summary(sample_records=sample, max_preview=10)
            st.write("**Detailed model debug summary**")
            st.json(dbg)
        except Exception as e:
            st.warning(f"Could not render detailed debug summary: {e}")

    # Selection UI
    st.sidebar.header("🔍 Voter Selection")
    method = st.sidebar.radio("Choose selection method:", ["Search by Voter ID", "Browse All Voters"])

    selected_voter = None
    if method == "Search by Voter ID":
        voter_id_input = st.sidebar.text_input("Enter Voter ID:", placeholder="e.g., KKFO0721605")
        if voter_id_input:
            candidates = voter_df[
                voter_df.apply(
                    lambda r: any(
                        str(get_any(r, fld, default="")).lower().__contains__(voter_id_input.lower())
                        for fld in ['voters id', 'voter id', 'voterid', 'id']
                    ), axis=1
                )
            ]
            if not candidates.empty:
                if len(candidates) == 1:
                    selected_voter = candidates.iloc[0]
                    st.sidebar.success(f"✅ Found voter: {get_any(selected_voter, 'name', default='N/A')}")
                else:
                    opts = {}
                    for i_en, (idx, r) in enumerate(candidates.iterrows()):
                        label = f"{get_any(r, 'name', default=f'Voter {i_en}')} ({get_any(r, 'voters id','voter id','id', default='NA')})"
                        opts[label] = idx
                    pick = st.sidebar.selectbox("Select from matches:", options=list(opts.keys()))
                    if pick:
                        selected_voter = candidates.loc[opts[pick]]
            else:
                st.sidebar.error("❌ No voter found with this ID")
    else:
        # Browse
        opts = {}
        for i_en, (idx, r) in enumerate(voter_df.iterrows()):
            label = f"{get_any(r, 'name', default=f'Voter {i_en}')} ({get_any(r, 'voters id','voter id','id', default='NA')})"
            opts[label] = idx
        pick = st.sidebar.selectbox("Select Voter:", options=[""] + list(opts.keys()))
        if pick and pick != "":
            selected_voter = voter_df.loc[opts[pick]]

    # If a voter is selected, run everything at once (individual + family)
    if selected_voter is not None:
        sv = selected_voter.to_dict()
        display_voter_info(sv)

        with st.spinner("Generating prediction…"):
            pred = predictor.predict_voters_vectorized([sv])[0]
        display_predictions(pred)

        display_family_block(selected_voter, voter_df, predictor)
    else:
        st.info("👆 Please select a voter using one of the methods in the sidebar.")

if __name__ == "__main__":
    main()
