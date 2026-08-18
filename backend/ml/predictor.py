import io
import logging
import pickle

import numpy as np
import pandas as pd

from backend.utils.numbers import get_any, to_float_safe

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    torch = None


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
                if '=' in fname:
                    k, v = fname.split('=', 1)
                    if k in rec and str(rec[k]) == v:
                        out[i, j] = 1.0
                elif any(str(v) == fname for v in rec.values()):
                    out[i, j] = 1.0
        return out

    def get_feature_names_out(self):
        return list(self.feature_names_)


class DummyScaler:
    def __init__(self, mean_, scale_):
        self.mean_ = np.asarray(mean_, dtype=float)
        self.scale_ = np.asarray(scale_, dtype=float)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        m = min(X.shape[1], self.mean_.size)
        X[:, :m] = (X[:, :m] - self.mean_[:m]) / (self.scale_[:m] + 1e-12)
        return X


def first_present(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


class VoterPredictor:
    def __init__(self):
        self.model_state_dict = None
        self.feature_names = None
        self.party_names = None
        self.scaler = None
        self.vectorizer = None
        self.booth_id_to_idx = None
        self._raw_model_data = None
        self._gamma0_array = None
        self._beta_P_array = None
        self._beta_T_array = None
        self._booth_effects_P_array = None
        self._booth_effects_T_array = None
        self._alpha0_value = None
        self.num_cols = ['land_rate', 'construction_cost', 'population', 'male_female_ratio']
        self.numeric_alias_to_index = {
            'land_rate': 0, 'land_rate_per_sqm': 0,
            'construction_cost': 1, 'construction_cost_per_sqm': 1,
            'population': 2,
            'male_female_ratio': 3, 'MaleToFemaleRatio': 3
        }
        self.model_loaded = False
        self.model_file_path = None
        self.alignment_thresholds = {"core": 0.7, "leaning": 0.4}

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
        try:
            if obj is None:
                return None
            if isinstance(obj, np.ndarray):
                return obj
            if hasattr(obj, 'detach') or hasattr(obj, 'numpy'):
                return VoterPredictor._to_numpy(obj)
            if isinstance(obj, (list, tuple)):
                return np.asarray(obj)
            if isinstance(obj, dict):
                keys = list(obj.keys())
                num_keys = []
                for k in keys:
                    try:
                        num_keys.append(int(k))
                    except Exception:
                        num_keys = None
                        break
                if num_keys is not None:
                    ordered = [obj[str(i)] if str(i) in obj else obj.get(i) for i in sorted(num_keys)]
                    return np.asarray(ordered)
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
        a = arr
        if hasattr(a, "detach"):
            a = a.detach()
        if hasattr(a, "cpu"):
            a = a.cpu()
        try:
            col = a[:, col_idx]
        except Exception:
            col = np.asarray(a)[:, col_idx]
        if hasattr(col, "toarray"):
            col = col.toarray().ravel()
        elif hasattr(col, "A1"):
            col = col.A1
        else:
            col = np.asarray(col).ravel()
        return col

    def _load_checkpoint_bytes(self, raw):
        model_data = None
        if torch is not None:
            try:
                md = torch.load(io.BytesIO(raw), map_location="cpu")
                if hasattr(md, 'state_dict'):
                    model_data = {'model_state_dict': md.state_dict()}
                elif isinstance(md, dict):
                    model_data = md
            except Exception:
                model_data = None
        if model_data is None:
            md = pickle.loads(raw)
            if hasattr(md, 'state_dict'):
                model_data = {'model_state_dict': md.state_dict()}
            elif isinstance(md, dict):
                model_data = md
            else:
                raise ValueError("Unsupported model format (.pth/.pkl expected)")
        if not isinstance(model_data, dict):
            raise ValueError("Model file did not contain a dict-like checkpoint")
        return model_data

    def load_model(self, model_file):
        try:
            if isinstance(model_file, (bytes, bytearray)):
                raw = bytes(model_file)
            else:
                raw = model_file.read() if hasattr(model_file, "read") else open(model_file, "rb").read()
            model_data = self._load_checkpoint_bytes(raw)
        except Exception as exc:
            logger.error("Could not read model file: %s", exc)
            self.model_loaded = False
            return False, f"Unsupported model format: {exc}"

        self._raw_model_data = model_data
        self.model_state_dict = first_present(model_data, ['model_state_dict', 'state_dict', 'weights', 'params'], {})
        self.feature_names = first_present(model_data, ['feature_names', 'features', 'feature_list'], [])
        self.party_names = first_present(model_data, ['party_names', 'classes', 'class_names'], ['BJP', 'Congress', 'AAP', 'Others', 'NOTA'])
        self.scaler = first_present(model_data, ['scaler', 'standardizer', 'preprocessor'], None)
        self.vectorizer = first_present(model_data, ['vectorizer', 'dict_vectorizer', 'dv'], None)
        self.booth_id_to_idx = first_present(model_data, ['booth_id_to_idx', 'booth_map', 'booth_index'], {})

        if isinstance(self.model_state_dict, dict):
            msd = self.model_state_dict
            norm = {}
            cand_beta_P = first_present(msd, ['beta_P', 'betaP', 'party_beta', 'W_party', 'linear_P.weight'], None)
            cand_beta_T = first_present(msd, ['beta_T', 'betaT', 'turnout_beta', 'W_turnout', 'linear_T.weight'], None)
            cand_gamma0 = first_present(msd, ['gamma0', 'party_bias', 'b_party', 'linear_P.bias'], None)
            cand_alpha0 = first_present(msd, ['alpha0', 'turnout_bias', 'b_turnout', 'linear_T.bias'], None)
            cand_boothP = first_present(msd, ['booth_effects_P', 'boothP', 'booth_party'], None)
            cand_boothT = first_present(msd, ['booth_effects_T', 'boothT', 'booth_turnout'], None)
            if cand_beta_P is not None:
                norm['beta_P'] = cand_beta_P
            if cand_beta_T is not None:
                norm['beta_T'] = cand_beta_T
            if cand_gamma0 is not None:
                norm['gamma0'] = cand_gamma0
            if cand_alpha0 is not None:
                norm['alpha0'] = cand_alpha0
            if cand_boothP is not None:
                norm['booth_effects_P'] = cand_boothP
            if cand_boothT is not None:
                norm['booth_effects_T'] = cand_boothT
            if norm:
                self.model_state_dict = norm

        if not self.model_state_dict or len(self.feature_names) == 0:
            self.model_loaded = False
            return False, "Missing model_state_dict or feature_names in checkpoint"

        if isinstance(self.vectorizer, dict):
            self.vectorizer = DummyVectorizer(self.vectorizer)
        if isinstance(self.scaler, dict):
            s = self.scaler
            mean_arr = self._convert_to_numpy_like(s.get('mean_', None))
            scale_arr = self._convert_to_numpy_like(s.get('scale_', None))
            n_in = int(s.get('n_features_in_', 4)) if isinstance(s.get('n_features_in_'), (int, float, np.number)) else 4
            if mean_arr is None or len(mean_arr) != n_in:
                mean_arr = np.zeros(n_in, dtype=float)
            if scale_arr is None or len(scale_arr) != n_in:
                scale_arr = np.ones(n_in, dtype=float)
            self.scaler = DummyScaler(mean_arr, scale_arr)

        self._preprocess_model_weights()
        self.model_loaded = True
        return True, f"Model loaded successfully with {len(self.feature_names)} features and {len(self.party_names)} parties"

    def _preprocess_model_weights(self):
        def _find_and_convert(key):
            val = None
            if isinstance(self.model_state_dict, dict) and key in self.model_state_dict:
                val = self.model_state_dict.get(key)
            if (val is None or (isinstance(val, dict) and not val)) and self._raw_model_data is not None:
                val = self._raw_model_data.get(key, val)
            if val is None:
                return None
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
            self._beta_P_array = np.asarray(_find_and_convert('beta_P'))
            if self._beta_P_array.ndim == 2 and self._beta_P_array.shape[0] == len(self.party_names) and self._beta_P_array.shape[1] == len(self.feature_names):
                self._beta_P_array = self._beta_P_array.T
        if _find_and_convert('beta_T') is not None:
            bT = np.asarray(_find_and_convert('beta_T'))
            if bT.ndim == 2:
                bT = bT.reshape(-1)
            self._beta_T_array = bT.reshape(-1)
        if _find_and_convert('booth_effects_P') is not None:
            self._booth_effects_P_array = np.asarray(_find_and_convert('booth_effects_P'))
        if _find_and_convert('booth_effects_T') is not None:
            self._booth_effects_T_array = np.asarray(_find_and_convert('booth_effects_T')).reshape(-1)
        if _find_and_convert('alpha0') is not None:
            a0 = np.asarray(_find_and_convert('alpha0'))
            try:
                self._alpha0_value = float(a0.reshape(())) if a0.size == 1 else float(np.ravel(a0)[0])
            except Exception:
                self._alpha0_value = None

    def preprocess_voter_data_vectorized(self, voter_rows):
        if isinstance(voter_rows, dict):
            voter_rows = [voter_rows]
        elif isinstance(voter_rows, pd.Series):
            voter_rows = [voter_rows.to_dict()]
        elif isinstance(voter_rows, pd.DataFrame):
            voter_rows = voter_rows.to_dict('records')

        cat_dicts = []
        X_num = np.zeros((len(voter_rows), len(self.num_cols)), dtype=float)

        for i, r in enumerate(voter_rows):
            age_val = get_any(r, 'age', default=None)
            age_group = None
            if age_val is not None and age_val != "":
                try:
                    age_int = int(float(age_val))
                    if age_int <= 25:
                        age_group = "Age_18-25"
                    elif age_int <= 35:
                        age_group = "Age_26-35"
                    elif age_int <= 45:
                        age_group = "Age_36-45"
                    elif age_int <= 60:
                        age_group = "Age_46-60"
                    else:
                        age_group = "Age_60+"
                except Exception:
                    age_group = None

            raw_rel = get_any(r, 'religion', default=None)
            rel_tok = None
            if raw_rel is not None:
                s = str(raw_rel).strip().upper()
                if s and s != "UNKNOWN":
                    if "HINDU" in s:
                        rel_tok = "Religion_Hindu"
                    elif "MUSLIM" in s:
                        rel_tok = "Religion_Muslim"
                    elif "SIKH" in s:
                        rel_tok = "Religion_Sikh"
                    elif "CHRISTIAN" in s:
                        rel_tok = "Religion_Christian"
                    elif "JAIN" in s:
                        rel_tok = "Religion_Jain"
                    elif "BUDDH" in s:
                        rel_tok = "Religion_Buddhist"

            raw_caste = get_any(r, 'caste', default=None)
            s_caste = str(raw_caste).strip().upper() if raw_caste is not None else ""
            s_rel = str(raw_rel).strip().upper() if raw_rel is not None else ""
            is_hindu = "HINDU" in s_rel
            caste_tok = None
            if not s_caste or s_caste in {"NA", "N/A", "NONE", "NULL", "UNKNOWN"}:
                if not is_hindu:
                    caste_tok = "Caste_No_caste_system"
            elif s_caste == "NO CASTE SYSTEM":
                caste_tok = "Caste_No_caste_system"
            elif s_caste == "OBC":
                caste_tok = "Caste_Obc"
            elif s_caste == "SC":
                caste_tok = "Caste_Sc"
            elif s_caste == "ST":
                caste_tok = "Caste_St"
            elif "BRAHMIN" in s_caste:
                caste_tok = "Caste_Brahmin"
            elif "KSHATRIYA" in s_caste:
                caste_tok = "Caste_Kshatriya"
            elif "VAISHYA" in s_caste:
                caste_tok = "Caste_Vaishya"
            elif not is_hindu:
                caste_tok = "Caste_No_caste_system"

            econ_raw = get_any(r, 'economic_category', default=None)
            econ_code = get_any(r, 'economic_category_code', default=None)
            econ_norm = str(econ_raw).strip().upper() if econ_raw else None
            income_tok = "income_middle"
            s_e = econ_norm or ""
            s_c = str(econ_code).strip().upper() if econ_code is not None else ""
            if "LOW INCOME" in s_e:
                income_tok = "income_low"
            elif "LOWER MIDDLE" in s_e or "MIDDLE CLASS" in s_e:
                income_tok = "income_middle"
            elif "UPPER MIDDLE" in s_e or "PREMIUM" in s_e:
                income_tok = "income_high"
            elif s_c == "L":
                income_tok = "income_low"
            elif s_c == "H":
                income_tok = "income_high"

            loc_raw = get_any(r, 'Locality', 'locality', default=None)
            locality = str(loc_raw).strip().upper() if loc_raw else None

            cat = {}
            if age_group is not None:
                cat["age"] = age_group
            if caste_tok is not None:
                cat["caste"] = caste_tok
            if rel_tok is not None:
                cat["religion"] = rel_tok
            if econ_norm is not None:
                cat["economic"] = econ_norm
            if income_tok is not None:
                cat["income"] = income_tok
            if locality:
                cat["locality"] = locality
            cat_dicts.append(cat)

            X_num[i, 0] = to_float_safe(get_any(r, 'land_rate_per_sqm', 'land_rate', default=0.0), default=0.0)
            X_num[i, 1] = to_float_safe(get_any(r, 'construction_cost_per_sqm', 'construction_cost', default=0.0), default=0.0)
            X_num[i, 2] = to_float_safe(get_any(r, 'population', default=0.0), default=0.0)
            X_num[i, 3] = to_float_safe(get_any(r, 'MaleToFemaleRatio', 'male_female_ratio', default=1.0), default=1.0)

        if self.vectorizer is None:
            X_cat = None
            vec_idx = {}
        else:
            X_cat = self.vectorizer.transform(cat_dicts)
            vec_feats = getattr(self.vectorizer, 'feature_names_', None)
            if vec_feats is None and hasattr(self.vectorizer, 'get_feature_names_out'):
                vec_feats = list(self.vectorizer.get_feature_names_out())
            vec_feats = vec_feats or []
            vec_idx = {n: i for i, n in enumerate(vec_feats)}
            feature_to_vec_idx = {}
            for fname in self.feature_names:
                if fname in vec_idx:
                    feature_to_vec_idx[fname] = vec_idx[fname]
            if len(feature_to_vec_idx) < len(self.feature_names):
                norm_vec = {str(n).lower().strip(): i for i, n in enumerate(vec_feats)}
                for fname in self.feature_names:
                    if fname in feature_to_vec_idx:
                        continue
                    nf = str(fname).lower().strip()
                    if nf in norm_vec:
                        feature_to_vec_idx[fname] = norm_vec[nf]
            missing = [
                fname for fname in self.feature_names
                if fname not in feature_to_vec_idx and fname not in self.numeric_alias_to_index
            ]
            if missing:
                raise ValueError(
                    f"DictVectorizer is missing {len(missing)} categorical feature(s) from checkpoint: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )
            vec_idx = feature_to_vec_idx

        if hasattr(self.scaler, 'transform'):
            try:
                X_num = self.scaler.transform(X_num)
            except Exception:
                pass

        X = np.zeros((len(voter_rows), len(self.feature_names)), dtype=np.float32)
        for j, fname in enumerate(self.feature_names):
            if fname in vec_idx and X_cat is not None:
                X[:, j] = self._extract_column(X_cat, vec_idx[fname])
            elif fname in self.numeric_alias_to_index:
                X[:, j] = X_num[:, self.numeric_alias_to_index[fname]]
        return X

    def predict_voters_vectorized(self, voter_rows):
        X = self.preprocess_voter_data_vectorized(voter_rows)
        if isinstance(voter_rows, (dict, pd.Series)):
            n_voters = 1
        elif isinstance(voter_rows, pd.DataFrame):
            n_voters = len(voter_rows)
        else:
            n_voters = len(voter_rows)

        results = []
        for i in range(n_voters):
            x_i = X[i:i + 1]
            if self._beta_P_array is not None and self._gamma0_array is not None:
                logits_P = x_i @ self._beta_P_array + self._gamma0_array[np.newaxis, :]
                logits_P = logits_P.ravel()
                exp_logits = np.exp(logits_P - np.max(logits_P))
                party_probs = exp_logits / np.sum(exp_logits)
            else:
                party_probs = np.ones(len(self.party_names)) / len(self.party_names)
            party_probabilities = {party: float(prob) for party, prob in zip(self.party_names, party_probs)}
            predicted_party = self.party_names[np.argmax(party_probs)]
            if self._beta_T_array is not None and self._alpha0_value is not None:
                logit_T = float(x_i @ self._beta_T_array) + self._alpha0_value
                turnout_prob = 1.0 / (1.0 + np.exp(-logit_T))
            else:
                turnout_prob = 0.75
            results.append({
                'turnout_probability': float(turnout_prob),
                'party_probabilities': party_probabilities,
                'predicted_party': predicted_party
            })
        return results

    def predict_voter(self, voter_data):
        try:
            results = self.predict_voters_vectorized([voter_data])
            if not results:
                return None, "Prediction failed - no results"
            pred = results[0]
            party_probs = pred.get('party_probabilities', {})
            predicted_party = pred.get('predicted_party', 'Unknown')
            turnout_prob = pred.get('turnout_probability', 0.5)
            confidence = max(party_probs.values()) if party_probs else 0.0
            if confidence > 0.7:
                confidence_level = "High"
            elif confidence > 0.5:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"
            alignment_category = 'swing'
            if confidence >= self.alignment_thresholds.get('core', 0.7):
                alignment_category = 'core'
            elif confidence >= self.alignment_thresholds.get('leaning', 0.4):
                alignment_category = 'leaning'
            age_val = voter_data.get('age', 0)
            try:
                age_int = int(float(age_val)) if age_val not in (None, '') else 0
                if 18 <= age_int <= 25:
                    age_group = "Age 18-25"
                elif 26 <= age_int <= 35:
                    age_group = "Age 26-35"
                elif 36 <= age_int <= 45:
                    age_group = "Age 36-45"
                elif 46 <= age_int <= 60:
                    age_group = "Age 46-60"
                else:
                    age_group = "Age 60+"
            except Exception:
                age_group = "Unknown age"
            religion = str(voter_data.get('religion', '')).strip().title()
            caste = str(voter_data.get('caste', '')).strip().title()
            econ_cat = str(voter_data.get('economic_category', '')).strip().upper()
            locality = str(voter_data.get('locality', '')).strip().title()
            return {
                'predicted_party': predicted_party,
                'party_probabilities': party_probs,
                'turnout_probability': turnout_prob,
                'confidence_level': confidence_level,
                'model_confidence': f"{confidence * 100:.1f}%",
                'alignment_category': alignment_category,
                'alignment_party': predicted_party,
                'alignment_confidence': confidence,
                'prediction_factors': {
                    'primary': f"{religion} - {caste}" if caste else religion,
                    'secondary': econ_cat if econ_cat else "Economic Status",
                    'tertiary': f"{age_group} | {locality}" if locality else age_group
                }
            }, None
        except Exception as exc:
            logger.exception("Prediction error")
            return None, str(exc)

    def status(self):
        return {
            "feature_names": len(self.feature_names or []),
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
