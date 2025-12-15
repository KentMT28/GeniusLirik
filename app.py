# ==========================================
# app.py - INTEGRATED LYRICS GENERATOR (v4.4 - SHAPE FIX)
# ==========================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import tensorflow di awal
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec # Digunakan untuk model ID Generic
import numpy as np
import json
import re
import os
import gc
from collections import defaultdict, Counter, OrderedDict # [ADDED] OrderedDict untuk paksa urutan
import pickle # Untuk memuat tokenizer/encoder

# Di awal app.py
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU ditemukan ({len(gpus)} device). Memory growth diaktifkan.")
    except RuntimeError as e:
        print(f"⚠️ Gagal mengkonfigurasi GPU: {e}")
else:
    print("ℹ️ Tidak ada GPU ditemukan oleh TensorFlow. Menggunakan CPU.")

    
# --- Prasyarat Bahasa Indonesia ---
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Pastikan Sastrawi terinstal: pip install Sastrawi
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    ID_STOPWORDS = set(stopwords.words('indonesian'))
    print("✅ Sastrawi loaded.")
except ImportError:
    print("⚠️ Sastrawi not installed. Indonesian preprocessing might fail. Run: pip install Sastrawi")
    stemmer = None
    ID_STOPWORDS = set()

# ------------------------------------
# --- Prasyarat Bahasa Inggris ---
try:
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet', quiet=True)
    lemmatizer_en = WordNetLemmatizer()
    print("✅ NLTK WordNet loaded for English.")
except ImportError:
    print("⚠️ NLTK WordNet Lemmatizer might be needed for English models.")
    lemmatizer_en = None

# ---------------------------------

app = Flask(__name__, static_folder='static')

# --- [PERBAIKAN UTAMA] ---
# 1. Matikan sort keys untuk Flask versi lama
app.config['JSON_SORT_KEYS'] = False 
# 2. Matikan sort keys untuk Flask versi baru (v2.3+)
app.json.sort_keys = False
# ------------------------------

CORS(app)

# ==========================================
# KONFIGURASI PATH
# ==========================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_PATH, "models")
INDONESIAN_MODELS_PATH = os.path.join(MODELS_PATH, "indonesian") # Artist specific
# Path ke model Generic V3 yang baru
INDONESIAN_GENERIC_MODELS_PATH = os.path.join(MODELS_PATH, "Bahasa Indonesia Generik V3") 
ENGLISH_ARTIST_MODELS_PATH = os.path.join(MODELS_PATH, "english_artist")
ENGLISH_GENERIC_MODELS_PATH = os.path.join(MODELS_PATH, "english_generic_v4")

# ==========================================
# KONSTANTA
# ==========================================
INDONESIAN_ARTIST_MAXLEN = 50
# [FIXED] Diubah ke 60 sesuai training di LirikGeneratorTerakhir.ipynb
INDONESIAN_GENERIC_MAXLEN = 60 
ENGLISH_ARTIST_MAXLEN = 50
ENGLISH_GENERIC_MAXLEN = 50 
INDONESIAN_GENERIC_VOCAB_SIZE = 3000
ENGLISH_GENERIC_VOCAB_SIZE = 15000 

# ==========================================
# PROFANITY FILTER BLACKLIST (ENGLISH) - RELAXED
# ==========================================
ENGLISH_PROFANITY_BLACKLIST = {
    'fuck', 'fucking', 'fucked', 'shit', 'bitch', 'cunt', 'motherfucker',
    'nigger', 'faggot',
    'murder', 'suicide', 'rape'
}

# ==========================================
# LOAD MODELS AT STARTUP
# ==========================================
print("="*70)
print("Loading models...")
print("="*70)

loaded_models = {} # Cache

# --- Load English Generic Model (v4) ---
print("📦 Loading English Generic Model (v4)...")
try:
    model_path = os.path.join(ENGLISH_GENERIC_MODELS_PATH, 'enhanced_english_lyrics_bilstm_backup.keras')
    tokenizer_path = os.path.join(ENGLISH_GENERIC_MODELS_PATH, 'english_tokenizer.pkl')
    label_encoder_path = os.path.join(ENGLISH_GENERIC_MODELS_PATH, 'english_label_encoder.pkl')

    missing_files_v4 = []
    if not os.path.exists(model_path): missing_files_v4.append('enhanced_english_lyrics_bilstm_backup.keras')
    if not os.path.exists(tokenizer_path): missing_files_v4.append('english_tokenizer.pkl')
    if not os.path.exists(label_encoder_path): missing_files_v4.append('english_label_encoder.pkl')

    if not missing_files_v4:
        model = load_model(model_path, compile=False)
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        w2i = tokenizer.word_index
        i2w = tokenizer.index_word 
        
        vocab_size = ENGLISH_GENERIC_VOCAB_SIZE 

        loaded_models['english_generic'] = {
            'model': model, 
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'w2i': w2i, 
            'i2w': i2w, 
            'maxlen': ENGLISH_GENERIC_MAXLEN, 
            'vocab_size': vocab_size
        }
        print("✅ English Generic Model (v4) loaded!")
        print(f"   Vocabulary (used by model): {vocab_size:,} words")
        print(f"   Genres supported: {list(label_encoder.classes_)}")
    else:
        print(f"   ⚠️ Files not found for v4: {', '.join(missing_files_v4)}, skipping.")
        loaded_models['english_generic'] = None
except Exception as e:
    print(f"   ⚠️ English Generic Model (v4) FAILED: {e}")
    import traceback
    traceback.print_exc()
    loaded_models['english_generic'] = None


# --- Load English Artist-Specific Models ---
print("\n📦 Loading English Artist-Specific Models...")
loaded_models['english_artist'] = {}
for genre in ['pop', 'rock', 'rap', 'indie']:
    try:
        model_file = os.path.join(ENGLISH_ARTIST_MODELS_PATH, f'model_{genre}.h5')
        map_file = os.path.join(ENGLISH_ARTIST_MODELS_PATH, f'mappings_{genre}.json')

        if os.path.exists(model_file) and os.path.exists(map_file):
            model = load_model(model_file, compile=False) 
            with open(map_file, 'r') as f:
                mappings = json.load(f)
            i2w = {int(k): v for k, v in mappings['i2w'].items()}
            w2i = mappings['w2i']
            loaded_models['english_artist'][genre] = {
                'model': model, 'w2i': w2i, 'i2w': i2w, 'maxlen': ENGLISH_ARTIST_MAXLEN
            }
            print(f"   ✅ {genre.upper()} model loaded (Vocab: {len(w2i)})")
        else:
            print(f"   ⚠️ {genre.upper()} files not found, skipping.")
    except Exception as e:
        print(f"   ⚠️ {genre.upper()} model FAILED: {e}")

# --- Load Indonesian Artist-Specific Models ---
print("\n📦 Loading Indonesian Artist-Specific Models...")
loaded_models['indonesian'] = {} 
for genre in ['pop', 'rock', 'dangdut', 'indie']:
    try:
        model_file = os.path.join(INDONESIAN_MODELS_PATH, f'model_id_{genre}.h5')
        map_file = os.path.join(INDONESIAN_MODELS_PATH, f'mappings_id_{genre}.json')

        if os.path.exists(model_file) and os.path.exists(map_file):
            model = load_model(model_file, compile=False) 
            with open(map_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            i2w = {int(k): v for k, v in mappings['i2w'].items()}
            w2i = mappings['w2i']
            loaded_models['indonesian'][genre] = {
                'model': model, 'w2i': w2i, 'i2w': i2w, 'maxlen': INDONESIAN_ARTIST_MAXLEN
            }
            print(f"   ✅ ID {genre.upper()} model loaded (Vocab: {len(w2i)})")
        else:
            print(f"   ⚠️ ID {genre.upper()} files not found, skipping.")
    except Exception as e:
        print(f"   ⚠️ ID {genre.upper()} model FAILED: {e}")

# --- Load Indonesian Generic Model ---
print("\n📦 Loading Indonesian Generic Model...")
tokenizer_id_generic = None
label_encoder_id_generic = None

tokenizer_pkl_path = os.path.join(INDONESIAN_GENERIC_MODELS_PATH, 'tokenizer.pkl')
label_encoder_pkl_path = os.path.join(INDONESIAN_GENERIC_MODELS_PATH, 'label_encoder.pkl')
# Nama file model V3 yang baru
keras_model_path = os.path.join(INDONESIAN_GENERIC_MODELS_PATH, 'best_model_epoch_42_val_loss_4.4271.keras')
w2v_model_path = os.path.join(INDONESIAN_GENERIC_MODELS_PATH, 'word2vec_indonesian_optimized.model')

missing_files = []

try:
    if os.path.exists(tokenizer_pkl_path):
        with open(tokenizer_pkl_path, 'rb') as handle:
            tokenizer_id_generic = pickle.load(handle)
        print("   ✅ Tokenizer ID Generic loaded from tokenizer.pkl")
    else:
        missing_files.append('tokenizer.pkl')

    if os.path.exists(label_encoder_pkl_path):
        with open(label_encoder_pkl_path, 'rb') as handle:
            label_encoder_id_generic = pickle.load(handle)
        print("   ✅ Label Encoder ID Generic loaded from label_encoder.pkl")
    else:
        missing_files.append('label_encoder.pkl')

    if not os.path.exists(keras_model_path):
        missing_files.append('best_model_epoch_42_val_loss_4.4271.keras')

    if not os.path.exists(w2v_model_path):
        missing_files.append('word2vec_indonesian_optimized.model')

    if not missing_files:
        model_id_generic = load_model(keras_model_path, compile=False)
        w2v_id_generic = Word2Vec.load(w2v_model_path)

        w2i_id_generic = tokenizer_id_generic.word_index
        i2w_id_generic = {i: w for w, i in w2i_id_generic.items()}
        vocab_size_id_generic = INDONESIAN_GENERIC_VOCAB_SIZE

        loaded_models['indonesian_generic'] = {
            'model': model_id_generic,
            'tokenizer': tokenizer_id_generic,
            'w2i': w2i_id_generic,
            'i2w': i2w_id_generic,
            'label_encoder': label_encoder_id_generic,
            'maxlen': INDONESIAN_GENERIC_MAXLEN,
            'w2v': w2v_id_generic,
            'vocab_size': vocab_size_id_generic 
        }
        print("✅ Indonesian Generic Model loaded!")
        print(f"   Vocabulary (used by model): {vocab_size_id_generic:,} words")
        print(f"   Genres supported: {list(label_encoder_id_generic.classes_)}")
    else:
        print(f"   ⚠️ Files not found or missing for Indonesian Generic: {', '.join(missing_files)}. Skipping.")
        loaded_models['indonesian_generic'] = None

except Exception as e:
    print(f"   ⚠️ Indonesian Generic Model FAILED to load: {e}")
    import traceback
    traceback.print_exc()
    loaded_models['indonesian_generic'] = None

print("\n📊 Models Ready:")
print(f"   English Generic: {'✅' if loaded_models.get('english_generic') else '❌'} (v4)") 
print(f"   English Artist: {len(loaded_models.get('english_artist', {}))}/4 genres")
print(f"   Indonesian Artist: {len(loaded_models.get('indonesian', {}))}/4 genres")
print(f"   Indonesian Generic: {'✅' if loaded_models.get('indonesian_generic') else '❌'}")
print("="*70)

# ==========================================
# FUNGSI PREPROCESSING
# ==========================================
def preprocess_text_id(text: str):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    toks = word_tokenize(text)
    toks = [t for t in toks if len(t) > 1 and t not in ID_STOPWORDS]

    if stemmer:
        try:
            stemmed_toks = [stemmer.stem(t) for t in toks]
            toks = [t for t in stemmed_toks if len(t) > 1]
        except Exception as e:
            print(f"   Stemming error: {e}")
            toks = [t for t in toks if len(t) > 1]
    else:
        toks = [t for t in toks if len(t) > 1]

    slang_dict = { 'gue': 'saya', 'gw': 'saya', 'aku': 'saya', 'lo': 'kamu', 'lu': 'kamu', 'udah': 'sudah', 'udh': 'sudah', 'klo': 'kalau', 'kalo': 'kalau', 'gimana': 'bagaimana', 'gmn': 'bagaimana', 'bgt': 'banget', 'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak', 'dlm': 'dalam', 'dgn': 'dengan', 'utk': 'untuk' }
    toks = [slang_dict.get(word, word) for word in toks]
    toks = [t for t in toks if len(t) > 1]

    return toks

def preprocess_text_en(text: str):
    if not isinstance(text, str) or not lemmatizer_en:
        if not isinstance(text, str): return []
        print("   (Preprocessing EN fallback: Lemmatizer not loaded)")
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        toks = word_tokenize(text)
        return [t for t in toks if len(t) > 1]

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    toks = word_tokenize(text)
    toks = [lemmatizer_en.lemmatize(t) for t in toks if len(t) > 1]
    return toks

# ==========================================
# FUNGSI FILTER & UTILS
# ==========================================
def filter_profanity_en(word):
    word_lower = word.lower()
    if word_lower in ENGLISH_PROFANITY_BLACKLIST:
        return False
    return True

def is_valid_word(word):
    if not word or word == '': return False
    if word.startswith('<') and word.endswith('>'): return False
    if '<UNK>' in word.upper(): return False
    if len(word) < 2: return False
    return True

# ==========================================
# FUNGSI GENERASI SECTION (ENGLISH GENERIC)
# ==========================================
def generate_section_generic_en(seed_text, genre, num_words, temperature,
                                repetition_penalty=1.5, top_p=0.92):
    res = loaded_models.get('english_generic')
    if not res: return []

    model = res['model']
    tokenizer = res['tokenizer']
    w2i = res['w2i']
    i2w = res['i2w']
    label_encoder = res['label_encoder']
    maxlen = res['maxlen']
    vocab_size = res['vocab_size']

    sequence = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    ids = [idx for idx in sequence if 0 < idx < vocab_size]

    if not ids:
        common_words = ['love', 'heart', 'night', 'dream', 'i', 'you']
        for word in common_words:
            if word in w2i and 0 < w2i[word] < vocab_size:
                ids = [w2i[word]]
                print(f"   ⚠️ No valid seed tokens found. Starting with fallback word: '{word}'")
                break

    if not ids:
        print("   ⚠️ Error: Cannot start generation. No valid seed or fallback tokens found.")
        return [] 

    genre_input_array = None
    try:
        genre_capitalized = genre.title() 
        genre_encoded = label_encoder.transform([genre_capitalized])[0] 
        genre_input_array = np.array([genre_encoded])
    except ValueError:
        try:
            genre_encoded = label_encoder.transform(['Pop'])[0] 
            genre_input_array = np.array([genre_encoded])
        except ValueError:
            return []

    history = Counter(ids)
    generated_words = []
    
    attempts = 0
    max_attempts = num_words * 20 

    while len(generated_words) < num_words and attempts < max_attempts:
        attempts += 1
        x_text = pad_sequences([ids], maxlen=maxlen, padding='post', truncating='post') 
        model_input = [x_text, genre_input_array] 

        try:
            probs = model.predict(model_input, verbose=0)[0]
        except Exception as e:
            print(f"\n   Prediction error (EN Generic v4): {e}")
            break

        if len(probs) != vocab_size: break

        probs = np.asarray(probs).astype('float64') + 1e-10 
        logits = np.log(probs) / temperature

        for idx, count in history.items():
            if 0 <= idx < vocab_size and count > 2: 
                logits[idx] -= np.log(repetition_penalty ** (count - 1))

        sorted_indices = np.argsort(logits)[::-1]
        valid_sorted_indices = [idx for idx in sorted_indices if 0 < idx < vocab_size]

        if not valid_sorted_indices: continue 

        valid_logits = logits[valid_sorted_indices]
        valid_probs = np.exp(valid_logits - np.max(valid_logits)) 
        sum_valid_probs = np.sum(valid_probs)

        if sum_valid_probs == 0 or np.isnan(sum_valid_probs):
            next_id = valid_sorted_indices[0] 
        else:
            valid_probs /= sum_valid_probs 
            cumsum_probs = np.cumsum(valid_probs)
            nucleus_mask = cumsum_probs <= top_p
            if not np.any(nucleus_mask): nucleus_mask[0] = True

            nucleus_local_indices = np.where(nucleus_mask)[0]
            nucleus_indices = [valid_sorted_indices[i] for i in nucleus_local_indices]
            nucleus_probs = valid_probs[nucleus_mask]
            sum_nucleus_probs = np.sum(nucleus_probs)

            if sum_nucleus_probs == 0 or np.isnan(sum_nucleus_probs):
                next_id = nucleus_indices[0]
            else:
                nucleus_probs /= sum_nucleus_probs 
                next_id = np.random.choice(nucleus_indices, p=nucleus_probs)

        if not (0 < next_id < vocab_size): continue

        word = i2w.get(next_id) 

        if not word or not is_valid_word(word): continue
        if not filter_profanity_en(word): continue

        if len(generated_words) >= 2 and word == generated_words[-1] == generated_words[-2]:
            continue

        generated_words.append(word)
        ids.append(next_id)
        history[next_id] += 1
        
        if len(ids) > maxlen:
            ids = ids[-maxlen:] 

    return generated_words

# ==========================================
# FUNGSI GENERASI SECTION (INDONESIAN GENERIC)
# ==========================================
def generate_section_generic_id(seed_text, genre, num_words, temperature,
                                repetition_penalty=1.5, top_p=0.92):
    res = loaded_models.get('indonesian_generic')
    if not res: return []

    model = res['model']
    tokenizer = res['tokenizer']
    w2i = res['w2i']
    i2w = res['i2w']
    label_encoder = res['label_encoder']
    maxlen = res['maxlen']
    vocab_size = res['vocab_size']

    tokens = preprocess_text_id(seed_text)
    ids = [w2i[t] for t in tokens if t in w2i and w2i[t] < vocab_size] 

    if not ids:
        common_words = ['cinta', 'hati', 'rindu', 'sayang', 'aku', 'kau']
        for word in common_words:
            if word in w2i and w2i[word] < vocab_size:
                ids = [w2i[word]]
                print(f"   ⚠️ No valid seed tokens found. Starting with fallback word: '{word}'")
                break

    if not ids: return []

    genre_input_array = None
    if label_encoder:
        try:
            genre_capitalized = genre.capitalize()
            genre_encoded = label_encoder.transform([genre_capitalized])[0]
            genre_input_array = np.array([genre_encoded])
        except ValueError:
            try:
                genre_encoded = label_encoder.transform(['Pop'])[0]
                genre_input_array = np.array([genre_encoded])
            except ValueError:
                return []
    elif len(model.inputs) > 1:
        return []

    history = Counter(ids)
    generated_words = []
    
    attempts = 0
    max_attempts = num_words * 20 

    while len(generated_words) < num_words and attempts < max_attempts:
        attempts += 1
        
        # [FIXED] Menggunakan maxlen (60) tanpa -1
        current_valid_ids = [idx for idx in ids[-maxlen:] if idx < vocab_size]

        if not current_valid_ids: break 

        # [FIXED] Menggunakan maxlen (60) tanpa -1
        x_text = pad_sequences([current_valid_ids], maxlen=maxlen, padding='pre') 

        model_input = [x_text]
        if genre_input_array is not None:
            model_input.append(genre_input_array)
        elif len(model.inputs) > 1:
            break 

        try:
            probs = model.predict(model_input, verbose=0)[0]
        except Exception as e:
            print(f"\n   Prediction error (ID Generic): {e}")
            break 

        if len(probs) != vocab_size: break 

        probs = np.asarray(probs).astype('float64') + 1e-10 
        logits = np.log(probs) / temperature

        for idx, count in history.items():
            if 0 <= idx < vocab_size and count > 2: 
                logits[idx] -= np.log(repetition_penalty ** (count - 1))

        sorted_indices = np.argsort(logits)[::-1]
        valid_sorted_indices = [idx for idx in sorted_indices if 0 <= idx < vocab_size]

        if not valid_sorted_indices: continue 

        valid_logits = logits[valid_sorted_indices]
        valid_probs = np.exp(valid_logits - np.max(valid_logits)) 
        sum_valid_probs = np.sum(valid_probs)

        if sum_valid_probs == 0 or np.isnan(sum_valid_probs):
            next_id = valid_sorted_indices[0] 
        else:
            valid_probs /= sum_valid_probs 
            cumsum_probs = np.cumsum(valid_probs)
            nucleus_mask = cumsum_probs <= top_p
            if not np.any(nucleus_mask): nucleus_mask[0] = True

            nucleus_local_indices = np.where(nucleus_mask)[0]
            nucleus_indices = [valid_sorted_indices[i] for i in nucleus_local_indices]
            nucleus_probs = valid_probs[nucleus_mask]
            sum_nucleus_probs = np.sum(nucleus_probs)

            if sum_nucleus_probs == 0 or np.isnan(sum_nucleus_probs):
                next_id = nucleus_indices[0] 
            else:
                nucleus_probs /= sum_nucleus_probs 
                next_id = np.random.choice(nucleus_indices, p=nucleus_probs)

        if not (0 <= next_id < vocab_size): continue

        word = i2w.get(next_id) 
        if not word or not is_valid_word(word): continue

        if len(generated_words) >= 2 and word == generated_words[-1] == generated_words[-2]:
            continue

        generated_words.append(word)
        ids.append(next_id)
        history[next_id] += 1

    return generated_words


# ==========================================
# DYNAMIC SONG STRUCTURE GENERATOR
# ==========================================
def get_dynamic_structure(genre, total_words, is_indonesian=False):
    if is_indonesian:
        structures = {
            'pop': [('🎵 [Intro]', 0.05), ('🎤 [Bait 1]', 0.20), ('🌟 [Pre-Reff]', 0.10), ('🎶 [Reff]', 0.15), ('🎤 [Bait 2]', 0.20), ('🎶 [Reff]', 0.15), ('🌉 [Bridge]', 0.10), ('🎵 [Outro]', 0.05)],
            'rock': [('⚡ [Intro]', 0.06), ('🎸 [Bait 1]', 0.22), ('🔥 [Reff]', 0.18), ('🎸 [Bait 2]', 0.22), ('🔥 [Reff]', 0.18), ('⚡ [Bridge]', 0.08), ('🎸 [Outro]', 0.06)],
            'dangdut': [('🎺 [Intro]', 0.05), ('🎤 [Bait 1]', 0.22), ('💃 [Reff]', 0.18), ('🎤 [Bait 2]', 0.22), ('💃 [Reff]', 0.18), ('🎼 [Interlude]', 0.08), ('🎺 [Outro]', 0.07)],
            'indie': [('🌙 [Intro]', 0.07), ('🎼 [Bait 1]', 0.23), ('🎶 [Reff]', 0.17), ('🎼 [Bait 2]', 0.23), ('🎶 [Reff]', 0.17), ('🌊 [Interlude]', 0.06), ('🌙 [Outro]', 0.07)]
        }
    else:
        structures = {
            'pop': [('🎵 [Intro]', 0.05), ('🎤 [Verse 1]', 0.20), ('🌟 [Pre-Chorus]', 0.10), ('🎶 [Chorus]', 0.15), ('🎤 [Verse 2]', 0.20), ('🎶 [Chorus]', 0.15), ('🌉 [Bridge]', 0.10), ('🎵 [Outro]', 0.05)],
            'rock': [('⚡ [Intro]', 0.06), ('🎸 [Verse 1]', 0.22), ('🔥 [Chorus]', 0.18), ('🎸 [Verse 2]', 0.22), ('🔥 [Chorus]', 0.18), ('⚡ [Bridge]', 0.08), ('🎸 [Outro]', 0.06)],
            'rap': [('🎤 [Intro]', 0.04), ('📢 [Verse 1]', 0.26), ('🎶 [Hook]', 0.14), ('📢 [Verse 2]', 0.26), ('🎶 [Hook]', 0.14), ('🌉 [Bridge]', 0.10), ('🎵 [Outro]', 0.06)],
            'indie': [('🌙 [Intro]', 0.07), ('🎼 [Verse 1]', 0.23), ('🎶 [Chorus]', 0.17), ('🎼 [Verse 2]', 0.23), ('🎶 [Chorus]', 0.17), ('🌊 [Interlude]', 0.06), ('🌙 [Outro]', 0.07)]
        }

    structure_template = structures.get(genre, structures['pop'])

    dynamic_structure = []
    total_proportion = sum(p for _, p in structure_template) 
    remaining_words = total_words
    for i, (label, proportion) in enumerate(structure_template):
        if i == len(structure_template) - 1: 
            word_count = max(4, remaining_words)
        else:
            calculated_words = int(total_words * (proportion / total_proportion)) if total_proportion > 0 else int(total_words / len(structure_template))
            word_count = max(4, calculated_words) 
            remaining_words -= word_count

        if remaining_words < 0: remaining_words = 0

        if 'Intro' in label or 'Outro' in label: temp_mult = 0.75
        elif 'Bridge' in label or 'Interlude' in label or 'Solo' in label: temp_mult = 0.95
        elif 'Chorus' in label or 'Reff' in label or 'Hook' in label: temp_mult = 0.80
        elif 'Pre' in label: temp_mult = 0.90
        else: temp_mult = 0.85

        dynamic_structure.append((label, word_count, temp_mult))

    current_total = sum(wc for _, wc, _ in dynamic_structure)
    diff = total_words - current_total
    if diff > 0:
        for i in range(len(dynamic_structure)):
            label, wc, tm = dynamic_structure[i]
            if 'Verse' in label or 'Bait' in label or 'Chorus' in label or 'Reff' in label:
                add_words = int(np.ceil(diff / 2)) 
                dynamic_structure[i] = (label, wc + add_words, tm)
                diff -= add_words
                if diff <= 0: break
        if diff > 0:
             label, wc, tm = dynamic_structure[-1]
             dynamic_structure[-1] = (label, wc + diff, tm)

    return dynamic_structure

# ==========================================
# FUNGSI GENERASI (GENERIC ENGLISH) - DENGAN DISPLAY INPUT
# ==========================================
def generate_lyrics_generic_en(seed_text, genre, temperature=0.8, num_words=200):
    """
    Generate English Generic dengan struktur lagu lengkap dan num_words control
    MODIFIED: Menggunakan OrderedDict untuk memastikan urutan JSON.
    """
    structure = get_dynamic_structure(genre, num_words, is_indonesian=False)
    
    # [FIX] Gunakan OrderedDict, bukan dict biasa
    song_parts = OrderedDict()
    
    current_full_text = seed_text 

    print(f"\n🎵 Generating English Generic Song ({genre}) - Theme: '{seed_text}' - Total Words: ~{num_words}")
    print("-" * 70)

    for i, (label, words, temp_multiplier) in enumerate(structure): # Gunakan enumerate
        section_seed = ' '.join(current_full_text.split()[-10:]) 
        if not section_seed: section_seed = seed_text 

        print(f"   Generating {label} (~{words} words) from context '{section_seed}'...", end=' ')

        lyrics_words = generate_section_generic_en( 
            section_seed, genre, words,
            temperature=temp_multiplier * temperature
        )

        if isinstance(lyrics_words, str): 
            print(f"❌ Error: {lyrics_words}")
            song_parts[label] = f"[Error: {lyrics_words}]"
        elif lyrics_words and len(lyrics_words) > 0:
            section_text = ' '.join(lyrics_words)
            print(f"✅ {len(lyrics_words)} words generated")
            
            formatted_lines = []
            current_line = []
            words_per_line = 8 

            for word in lyrics_words:
                current_line.append(word)
                if len(current_line) >= words_per_line:
                    formatted_lines.append(' '.join(current_line))
                    current_line = []

            if current_line:
                formatted_lines.append(' '.join(current_line))

            final_section_text = '\n'.join(formatted_lines)
            
            # --- [MODIFIKASI] Tampilkan Seed Text di bagian pertama ---
            if i == 0:
                final_section_text = f"{seed_text}\n{final_section_text}"
            # ---------------------------------------------------------

            song_parts[label] = final_section_text
            current_full_text += " " + section_text 
        else:
            print("❌ No words generated")
            song_parts[label] = "[Generation produced no words]"

        gc.collect() 

    print("-" * 70)
    print("✅ Generation Complete.")

    return song_parts


# ==========================================
# FUNGSI GENERASI (INDONESIAN GENERIC) - DENGAN DISPLAY INPUT
# ==========================================
def generate_lyrics_indonesian_generic(seed_text, genre, temperature=1.0, num_words=200):
    """
    Generate Indonesian Generic dengan struktur lagu lengkap dan num_words control
    MODIFIED: Menggunakan OrderedDict untuk memastikan urutan JSON.
    """
    structure = get_dynamic_structure(genre, num_words, is_indonesian=True)
    
    # [FIX] Gunakan OrderedDict
    song_parts = OrderedDict()
    
    current_full_text = seed_text 

    print(f"\n🎵 Generating Indonesian Generic Song ({genre}) - Tema: '{seed_text}' - Total Words: ~{num_words}")
    print("-" * 70)

    for i, (label, words, temp_multiplier) in enumerate(structure): # Gunakan enumerate
        section_seed = ' '.join(current_full_text.split()[-10:]) 
        if not section_seed: section_seed = seed_text 

        print(f"   Generating {label} (~{words} words) from context '{section_seed}'...", end=' ')

        lyrics_words = generate_section_generic_id(
            section_seed, genre, words,
            temperature=temp_multiplier * temperature
        )

        if isinstance(lyrics_words, list) and lyrics_words:
            section_text = ' '.join(lyrics_words)
            print(f"✅ {len(lyrics_words)} words generated")
            
            formatted_lines = []
            current_line = []
            words_per_line = 8 

            for word in lyrics_words:
                current_line.append(word)
                if len(current_line) >= words_per_line:
                    formatted_lines.append(' '.join(current_line))
                    current_line = []

            if current_line:
                formatted_lines.append(' '.join(current_line))

            final_section_text = '\n'.join(formatted_lines)

            # --- [MODIFIKASI] Tampilkan Seed Text di bagian pertama ---
            if i == 0:
                final_section_text = f"{seed_text}\n{final_section_text}"
            # ---------------------------------------------------------

            song_parts[label] = final_section_text
            current_full_text += " " + section_text 
        elif isinstance(lyrics_words, str): 
            print(f"❌ Error during generation: {lyrics_words}")
            song_parts[label] = f"[Error: {lyrics_words}]"
        else: 
            print("❌ No words generated")
            song_parts[label] = "[Generation produced no words]"

        gc.collect() 

    print("-" * 70)
    print("✅ Generation Complete.")

    return song_parts


# ==========================================
# FUNGSI GENERASI (ARTIST ENGLISH) - DENGAN DISPLAY INPUT
# ==========================================
def generate_lyrics_artist_en(genre, artist, seed_text, num_words=120, temperature=0.85):
    res_genre = loaded_models.get('english_artist', {}).get(genre)
    if not res_genre: return f"Model/mappings for English {genre} not loaded"

    model, w2i, i2w, maxlen = res_genre['model'], res_genre['w2i'], res_genre['i2w'], res_genre['maxlen']

    artist_token = f"<{artist.lower().replace(' ','_')}>"
    if artist_token not in w2i:
        return f"Artist token '{artist_token}' not found in English {genre} vocab."

    in_tokens = [artist_token] + preprocess_text_en(seed_text)
    generated_lyrics = []

    valid_in_tokens_indices = [w2i[t] for t in in_tokens if t in w2i]
    if not valid_in_tokens_indices:
        print(f"   ⚠️ Warning: No valid tokens from seed '{seed_text}' and artist '{artist}'. Cannot generate.")
        return "Error: Invalid seed or artist for this genre."

    in_tokens = [t for t in in_tokens if t in w2i] 
    history = Counter(valid_in_tokens_indices) 

    repetition_penalty = 1.5  
    top_p = 0.92

    print(f"   Generating {num_words} words (EN Artist: {artist})...", end='')

    attempts = 0
    max_attempts = num_words * 15  

    while len(generated_lyrics) < num_words and attempts < max_attempts:
        attempts += 1
        token_list = [w2i[w] for w in in_tokens[-maxlen:] if w in w2i] 
        if not token_list: break

        seq = pad_sequences([token_list], maxlen=maxlen, padding='post', dtype='int32') 

        try:
            probs = model.predict(seq, verbose=0)[0]
        except Exception as e:
            print(f"\n   Prediction error (Artist EN): {e}")
            break

        if len(probs) != len(w2i): break

        probs = np.asarray(probs).astype('float64') + 1e-10
        logits = np.log(probs) / temperature

        for idx, count in history.items():
            if 0 <= idx < len(logits) and count > 2: 
                logits[idx] -= np.log(repetition_penalty ** (count - 1))

        sorted_indices = np.argsort(logits)[::-1]
        valid_sorted_indices = [idx for idx in sorted_indices if 0 <= idx < len(w2i)] 

        if not valid_sorted_indices: continue

        valid_logits = logits[valid_sorted_indices]
        valid_probs = np.exp(valid_logits - np.max(valid_logits))
        sum_valid_probs = np.sum(valid_probs)

        if sum_valid_probs == 0 or np.isnan(sum_valid_probs):
            next_id = valid_sorted_indices[0]
        else:
            valid_probs /= sum_valid_probs 
            cumsum_probs = np.cumsum(valid_probs)
            nucleus_mask = cumsum_probs <= top_p
            if not np.any(nucleus_mask): nucleus_mask[0] = True

            nucleus_local_indices = np.where(nucleus_mask)[0]
            nucleus_indices = [valid_sorted_indices[i] for i in nucleus_local_indices]
            nucleus_probs = valid_probs[nucleus_mask]
            sum_nucleus_probs = np.sum(nucleus_probs)

            if sum_nucleus_probs == 0 or np.isnan(sum_nucleus_probs):
                next_id = nucleus_indices[0]
            else:
                nucleus_probs /= sum_nucleus_probs 
                next_id = np.random.choice(nucleus_indices, p=nucleus_probs)

        if not (0 <= next_id < len(w2i)): continue

        word = i2w.get(next_id) 

        if not word or not is_valid_word(word): continue
        if len(in_tokens) >= 2 and word == in_tokens[-1] == in_tokens[-2]: continue

        in_tokens.append(word) 
        generated_lyrics.append(word)
        history[next_id] += 1 

        if len(generated_lyrics) % 10 == 0: print('.', end='', flush=True)

    print(" Done.")

    formatted_lines = []
    current_line = []
    words_per_line = 8

    for word in generated_lyrics:
        current_line.append(word)
        if len(current_line) >= words_per_line:
            formatted_lines.append(' '.join(current_line))
            current_line = []

    if current_line:
        formatted_lines.append(' '.join(current_line))
    
    final_text = '\n'.join(formatted_lines)
    
    # --- [MODIFIKASI] Gabungkan seed text di paling awal ---
    return f"{seed_text}\n{final_text}"
    # -----------------------------------------------------


# ============================================================
# FUNGSI GENERASI BAGIAN (INDONESIAN ARTIST)
# ============================================================
def generate_section_v2_id(genre, artist, theme, wordcount, temperature=0.90,
                           repetition_penalty=1.5, top_p=0.92):
    res_genre = loaded_models.get('indonesian', {}).get(genre)
    if not res_genre:
        return f"Gagal memuat model/mappings untuk Bahasa Indonesia {genre}"

    model, w2i, i2w, maxlen = res_genre['model'], res_genre['w2i'], res_genre['i2w'], res_genre['maxlen']
    vocab_size_artist = len(w2i) 

    artist_tok = f"<{artist.lower().replace(' ', '_')}>"
    if artist_tok not in w2i:
        print(f"   Warning: Artis '{artist}' tidak ada di vocab {genre}. Memulai hanya dengan tema.")
        tokens = preprocess_text_id(theme)
    else:
        tokens = [artist_tok] + preprocess_text_id(theme)

    ids = [w2i[t] for t in tokens if t in w2i] 

    if not ids:
        common_words = ['cinta', 'hati', 'rindu', 'sayang', 'aku', 'kau']
        for word in common_words:
            if word in w2i:
                ids = [w2i[word]]
                print(f"   Starting with fallback word: '{word}'")
                break
        if not ids and artist_tok in w2i:
            ids = [w2i[artist_tok]]
            print(f"   Starting with only artist token: '{artist_tok}'")

        if not ids: 
            return "Error: Tema dan Artis tidak valid untuk memulai generasi."

    history = Counter(ids)
    generated_words = []

    attempts = 0
    max_attempts = wordcount * 15  

    print(f"   Generating section (~{wordcount} words)...", end='')

    while len(generated_words) < wordcount and attempts < max_attempts:
        attempts += 1
        current_valid_ids = [idx for idx in ids[-maxlen:] if 0 <= idx < vocab_size_artist]
        if not current_valid_ids: break

        x = pad_sequences([current_valid_ids], maxlen=maxlen, padding='post') 

        try:
            probs = model.predict(x, verbose=0)[0]
        except Exception as e:
            print(f"\n   Prediction error (Artist ID Section): {e}")
            break

        if len(probs) != vocab_size_artist: break

        probs = np.asarray(probs, dtype='float64') + 1e-10
        logits = np.log(probs) / temperature

        for wid, count in history.items():
            if 0 <= wid < vocab_size_artist and count > 2: 
                penalty_factor = repetition_penalty ** (count - 1)
                logits[wid] -= np.log(penalty_factor)

        sorted_indices = np.argsort(logits)[::-1]
        valid_sorted_indices = [idx for idx in sorted_indices if 0 <= idx < vocab_size_artist]

        if not valid_sorted_indices: continue

        valid_logits = logits[valid_sorted_indices]
        valid_probs = np.exp(valid_logits - np.max(valid_logits))
        sum_valid_probs = np.sum(valid_probs)

        if sum_valid_probs == 0 or np.isnan(sum_valid_probs):
            next_id = valid_sorted_indices[0]
        else:
            valid_probs /= sum_valid_probs 
            cumsum_probs = np.cumsum(valid_probs)
            nucleus_mask = cumsum_probs <= top_p
            if not np.any(nucleus_mask): nucleus_mask[0] = True

            nucleus_local_indices = np.where(nucleus_mask)[0]
            nucleus_indices = [valid_sorted_indices[i] for i in nucleus_local_indices]
            nucleus_probs = valid_probs[nucleus_mask]
            sum_nucleus_probs = np.sum(nucleus_probs)

            if sum_nucleus_probs == 0 or np.isnan(sum_nucleus_probs):
                next_id = nucleus_indices[0]
            else:
                nucleus_probs /= sum_nucleus_probs 
                next_id = np.random.choice(nucleus_indices, p=nucleus_probs)

        if not (0 <= next_id < vocab_size_artist): continue

        word = i2w.get(next_id)
        if not word or not is_valid_word(word): continue

        if len(generated_words) >= 2 and word == generated_words[-1] == generated_words[-2]:
            continue

        generated_words.append(word)
        ids.append(next_id) 
        history[next_id] += 1

        if len(generated_words) % 10 == 0: print('.', end='', flush=True)

    print(f" Done. ({len(generated_words)} words)")

    formatted_lines = []
    current_line = []
    line_break_len = 8

    for word in generated_words:
        current_line.append(word)
        if len(current_line) >= line_break_len:
            formatted_lines.append(' '.join(current_line))
            current_line = []

    if current_line:
        formatted_lines.append(' '.join(current_line))

    return '\n'.join(formatted_lines)


# ============================================================
# STRUKTUR LAGU (ID ARTIST)
# ============================================================
def get_indonesian_artist_structure(genre, total_words):
    return get_dynamic_structure(genre, total_words, is_indonesian=True)

# ============================================================
# FUNGSI PEMBUAT LAGU LENGKAP (ID ARTIST) - DENGAN DISPLAY INPUT
# ============================================================
def create_improved_song_id(genre, artist, theme, user_temperature=0.85, num_words=200):
    """
    Generate lagu lengkap untuk Indonesian Artist dengan temperature dan num_words control
    MODIFIED: Menggunakan OrderedDict.
    """
    structure = get_indonesian_artist_structure(genre, num_words)
    
    # [FIX] Gunakan OrderedDict
    song_parts = OrderedDict()
    
    current_full_text_list = theme.split() 

    print(f"\n🇮🇩 Generating Indonesian Song for '{artist}' ({genre}) - Theme: '{theme}'")
    print(f"   Base Temperature: {user_temperature}, Total Words: ~{num_words}")
    print("-" * 70)

    for i, (label, words, temp_multiplier) in enumerate(structure): # Gunakan enumerate
        section_seed = ' '.join(current_full_text_list[-10:]) 
        if not section_seed: section_seed = theme 

        section_theme_variation = theme
        if "Bait 2" in label or "Bait 3" in label or "Verse 2" in label or "Verse 3" in label:
            section_theme_variation = f"{theme} cerita berlanjut"
        elif "Bridge" in label or "Interlude" in label or "Reff" in label or "Chorus" in label:
            section_theme_variation = f"{theme} inti perasaan"

        combined_seed = f"{section_seed} {section_theme_variation}"

        print(f"   Generating {label} (~{words} words) from context '{section_seed}'...", end=' ')

        final_temperature = temp_multiplier * user_temperature

        lyrics_section_text = generate_section_v2_id(
            genre, artist, combined_seed, words, 
            temperature=final_temperature,
            repetition_penalty=1.5,  
            top_p=0.92
        )

        if isinstance(lyrics_section_text, str) and (lyrics_section_text.startswith("Error") or lyrics_section_text.startswith("Gagal")):
            print(f"❌ Error: {lyrics_section_text}")
            song_parts[label] = f"[{label} - Error: {lyrics_section_text}]"
        elif isinstance(lyrics_section_text, str) and lyrics_section_text.strip():
            
            # --- [MODIFIKASI] Tampilkan Seed Text/Theme di bagian pertama ---
            if i == 0:
                lyrics_section_text = f"{theme}\n{lyrics_section_text}"
            # -------------------------------------------------------------
            
            song_parts[label] = lyrics_section_text 
            current_full_text_list.extend(lyrics_section_text.split()) 
        else: 
            print("❌ No lyrics generated")
            song_parts[label] = "[Generation produced no lyrics]"

        gc.collect()

    print("-" * 70)
    print("✅ Generation Complete.")

    return song_parts


# ==========================================
# FUNGSI STRUKTUR LAGU (ENGLISH ARTIST)
# ==========================================
def structure_lyrics_en(raw_lyrics):
    words = raw_lyrics.split()
    total_words = len(words)

    if total_words < 40:
        return {'generated_text': raw_lyrics} 

    verse1_end = int(total_words * 0.30)
    chorus1_end = verse1_end + int(total_words * 0.25)
    verse2_end = chorus1_end + int(total_words * 0.30)

    verse1 = " ".join(words[:verse1_end])
    chorus1 = " ".join(words[verse1_end:chorus1_end])
    verse2 = " ".join(words[chorus1_end:verse2_end])
    outro = " ".join(words[verse2_end:]) 

    # [FIX] Gunakan OrderedDict
    structured = OrderedDict()
    structured['[Verse 1]'] = verse1
    structured['[Chorus]'] = chorus1
    structured['[Verse 2]'] = verse2
    structured['[Outro]'] = outro

    ordered_keys = ['[Verse 1]', '[Chorus]', '[Verse 2]', '[Outro]']
    final_structured = OrderedDict()
    for k in ordered_keys:
        if structured.get(k, "").strip():
            final_structured[k] = structured[k].strip()

    if not final_structured:
        return {'generated_text': raw_lyrics}

    return final_structured


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/generate', methods=['POST'])
def generate_en_generic():
    try:
        data = request.json
        seed_text = data.get('seedText', '').strip()
        genre = data.get('genre', 'pop')
        temperature = float(data.get('temperature', 0.8))
        num_words = int(data.get('num_words', 200)) 

        if not seed_text:
            return jsonify({'error': 'Seed text required'}), 400
        
        if not loaded_models.get('english_generic'):
            return jsonify({'error': 'English Generic Model (v4) is not loaded or failed to load. Check server logs.'}), 503

        num_words = max(50, min(num_words, 500)) 

        print(f"\n🎵 [EN Generic v4] Gen: '{seed_text}'|{genre}|temp={temperature}|words={num_words}")

        song_parts = generate_lyrics_generic_en(seed_text, genre, temperature, num_words)

        has_errors = any("[Error:" in content for content in song_parts.values())
        if has_errors:
            print(f"⚠️ Generated structured song with errors")
        elif not song_parts or all(not content or "[Generation produced no words]" in content for content in song_parts.values()):
            print(f"⚠️ Generation produced no significant content (EN Generic v4)")
            return jsonify({'error': 'Generation produced no lyrics', 'lyrics': song_parts}), 500
        else:
            print(f"✅ Generated structured song")

        return jsonify({'success': True, 'lyrics': song_parts})

    except Exception as e:
        print(f"❌ Error [EN Generic]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/generate-artist', methods=['POST'])
def generate_en_artist():
    try:
        data = request.json
        genre = data.get('genre', 'pop')
        artist = data.get('artist', '').strip()
        seed_text = data.get('seedText', '').strip()
        temperature = float(data.get('temperature', 0.85))
        num_words = int(data.get('num_words', 150))

        if not artist or not seed_text:
            return jsonify({'error': 'Artist and seed text required'}), 400

        num_words = max(20, min(num_words, 400)) 

        print(f"\n🎤 [EN Artist] {artist}|{genre}|'{seed_text}'|temp={temperature}|words={num_words}")

        raw_lyrics = generate_lyrics_artist_en(genre, artist, seed_text, num_words, temperature)

        if isinstance(raw_lyrics, str) and (raw_lyrics.startswith("Model") or raw_lyrics.startswith("Artist") or raw_lyrics.startswith("Error:")):
            print(f"⚠️ Generation failed: {raw_lyrics}")
            return jsonify({'error': raw_lyrics}), 400
        elif not raw_lyrics or not raw_lyrics.strip():
            print(f"⚠️ Generation produced empty result.")
            return jsonify({'error': 'Generation produced no lyrics'}), 500

        structured_lyrics = structure_lyrics_en(raw_lyrics)

        print(f"✅ Generated {len(raw_lyrics.split())} words")

        return jsonify({'success': True, 'lyrics': structured_lyrics})

    except Exception as e:
        print(f"❌ Error [EN Artist]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/artists/<genre>', methods=['GET'])
def get_artists_en(genre):
    try:
        res_genre = loaded_models.get('english_artist', {}).get(genre)
        if not res_genre:
            return jsonify({'error': f'EN model {genre} not found'}), 404

        w2i = res_genre['w2i']
        artists = sorted([k.strip('<>').replace('_', ' ').title()
                          for k in w2i.keys()
                          if k.startswith('<') and k.endswith('>') and '_' in k and k.lower() != f'<{genre.lower()}>'])


        return jsonify({'artists': artists})

    except Exception as e:
        print(f"❌ Error getting EN artists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/artists-indonesian/<genre>', methods=['GET'])
def get_artists_id(genre):
    try:
        res_genre = loaded_models.get('indonesian', {}).get(genre)
        if not res_genre:
            return jsonify({'error': f'ID model {genre} not found'}), 404

        w2i = res_genre['w2i']
        artists = sorted([k.strip('<>').replace('_', ' ').title()
                          for k in w2i.keys()
                          if k.startswith('<') and k.endswith('>') and '_' in k and k.lower() != f'<{genre.lower()}>'])


        return jsonify({'artists': artists})

    except Exception as e:
        print(f"❌ Error getting ID artists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-indonesian', methods=['POST'])
def generate_id_artist():
    try:
        data = request.json
        genre = data.get('genre', 'pop').lower()
        artist = data.get('artist', '').strip()
        theme = data.get('theme', '').strip()
        temperature = float(data.get('temperature', 0.85))
        num_words = int(data.get('num_words', 200)) 

        if not artist or not theme:
            return jsonify({'error': 'Artis dan Tema diperlukan'}), 400

        if genre not in loaded_models.get('indonesian', {}):
            return jsonify({'error': f'ID Artist model {genre} not loaded'}), 503

        num_words = max(50, min(num_words, 500)) 

        print(f"\n🇮🇩 [ID Artist] {artist}|{genre}|'{theme}'|temp={temperature}|words={num_words}")

        structured_lyrics_dict = create_improved_song_id(genre, artist, theme, user_temperature=temperature, num_words=num_words)

        if structured_lyrics_dict is None:
            raise Exception("ID Artist generation failed")

        has_errors = any("[Error:" in content for content in structured_lyrics_dict.values())
        if has_errors:
            print(f"⚠️ Generated ID lyrics for {artist} with errors")
        else:
            print(f"✅ Generated ID lyrics for {artist}")

        return jsonify({'success': True, 'lyrics': structured_lyrics_dict})

    except Exception as e:
        print(f"❌ Error [ID Artist]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

@app.route('/api/generate-indonesian-generic', methods=['POST'])
def generate_id_generic():
    try:
        data = request.json
        seed_text = data.get('seedText', '').strip()
        temperature = float(data.get('temperature', 1.0))
        genre = data.get('genre', 'pop')
        num_words = int(data.get('num_words', 200)) 

        if not seed_text:
            return jsonify({'error': 'Teks awal diperlukan'}), 400

        if not loaded_models.get('indonesian_generic'):
            return jsonify({'error': 'ID Generic model not loaded'}), 503

        num_words = max(50, min(num_words, 500)) 

        print(f"\n🎵 [ID Generic] Gen: '{seed_text}'|genre={genre}|temp={temperature}|words={num_words}")

        song_parts = generate_lyrics_indonesian_generic(seed_text, genre, temperature, num_words)

        has_errors = any("[Error:" in content for content in song_parts.values())
        if has_errors:
            print(f"⚠️ Generated structured song (ID Generic) with errors")
        elif not song_parts or all(not content or "[Generation produced no words]" in content for content in song_parts.values()):
            print(f"⚠️ Generation produced no significant content (ID Generic)")
            return jsonify({'error': 'Generation produced no lyrics', 'lyrics': song_parts}), 500
        else:
            print(f"✅ Generated structured song (ID Generic)")

        return jsonify({'success': True, 'lyrics': song_parts})

    except Exception as e:
        print(f"❌ Error [ID Generic]: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 SERVER STARTING...")
    print("="*70)
    print(f"   Base Path: {BASE_PATH}")
    print(f"   Models Path: {MODELS_PATH}")
    print(f"   Serving static files from: {app.static_folder}")
    print("   Access UI: http://localhost:5000 or http://<your-ip>:5000")
    print("   API Endpoints:")
    print("    - /api/generate (POST - EN Generic)")
    print("    - /api/generate-artist (POST - EN Artist)")
    print("    - /api/generate-indonesian (POST - ID Artist Based)")
    print("    - /api/generate-indonesian-generic (POST - ID Generic)")
    print("    - /api/artists/<genre> (GET - EN Artists)")
    print("    - /api/artists-indonesian/<genre> (GET - ID Artists)")
    print("="*70)
    print("\nPress CTRL+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)