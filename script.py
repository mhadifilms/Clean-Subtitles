#!/usr/bin/env python3
"""
script.py – Whisper-large-v3 CLI with progress bar + smart cleaning

usage: python script.py /path/to/file.mp4
"""

import argparse, subprocess, tempfile, sys, os, logging, json, re, regex, srt, shutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from faster_whisper import WhisperModel

# -------------- settings -----------------
SETTINGS_FILE = Path("settings.json")

DEFAULT_SETTINGS = {
    "model": "large-v2",
    "lower_case": True,
    "simplify_punctuation": True,
    "single_lines": True,
    "max_chars": 25,
    "min_duration": 1.2,
    "gap_frames": 0
}

def load_settings() -> Dict:
    """Load settings from file or create with defaults"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Merge with defaults to handle missing keys
                return {**DEFAULT_SETTINGS, **settings}
        except:
            print("⚠️  Error reading settings.json, using defaults")
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict):
    """Save settings to file"""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

def create_settings_file():
    """Create initial settings file with defaults"""
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        print(f"📝 Created {SETTINGS_FILE} with default settings")
        print("💡 Edit this file to customize your preferences")

# -------------- logging -----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# -------------- helpers -----------------
def ffprobe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration", "-of",
           "default=noprint_wrappers=1:nokey=1", str(path)]
    return float(subprocess.check_output(cmd).decode().strip())

def extract_audio(src: Path) -> Path:
    dst = Path(tempfile.mktemp(suffix=".wav"))
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
           "-loglevel", "error", str(dst)]
    subprocess.run(cmd, check=True)
    return dst

def ask_yes_no(q: str, default: bool) -> bool:
    prompt = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{q} [{prompt}]: ").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in {"y","yes"}: return True
        if ans in {"n","no"}: return False
        print("Type y or n")

def ask_choice(q: str, choices: List[str], default: str) -> str:
    disp = "/".join(c.upper() if c==default else c for c in choices)
    while True:
        ans = input(f"{q} [{disp}]: ").strip().lower()
        if ans=="": return default
        if ans in choices: return ans
        print(f"Choose one of {choices}")

def ask_float(q:str, default:float, lo:float, hi:float) -> float:
    while True:
        ans = input(f"{q} ({lo}-{hi}) [{default}]: ").strip()
        if ans=="": return default
        try:
            val=float(ans); 
            if lo<=val<=hi: return val
        except: pass
        print("Enter number in range")

def ask_int(q:str, default:int, lo:int, hi:int) -> int:
    return int(ask_float(q, default, lo, hi))

# --- punctuation presets ---
ALLOWED_DIALOGUE = r"[^,\.\'\?\!\u2026\-–—“”\" \p{L}\p{N}\s]"
ALLOWED_FULL     = r"[^\[\]\(\):,.\'\?\!\u2026\-–—“”\" \p{L}\p{N}\s]"

def sanitize(txt:str, preset:str) -> str:
    txt = re.sub(r"<[^>]+>","",txt)          # HTML tags
    txt = re.sub(r"\([^)]*?\)|\[[^\]]*?\]","",txt)  # remove bracketed if not keeping
    txt = re.sub(r"[\u266A-\u266C]","",txt)  # music notes
    pattern = ALLOWED_DIALOGUE if preset=="dialogue" else ALLOWED_FULL
    txt = regex.sub(pattern,"",txt)
    return re.sub(r"\s+"," ",txt).strip()

# --- segment post-processing ---
def merge_segments(segs: List[Dict], min_d: float, gap: float) -> List[Dict]:
    if not segs: return []
    merged=[dict(segs[0])]
    for s in segs[1:]:
        prev=merged[-1]
        if (prev["end"]-prev["start"]<min_d) or (s["start"]-prev["end"]<=gap):
            prev["text"] += " " + s["text"]; prev["end"]=s["end"]
        else: merged.append(dict(s))
    return merged

def split_chars(seg:Dict, max_c:int) -> List[Dict]:
    words = seg["text"].split()
    parts = []
    current_line = []
    current_len = 0
    for w in words:
        # If word itself is longer than max_c, put it on its own line
        if len(w) > max_c:
            if current_line:
                parts.append(" ".join(current_line))
                current_line = []
                current_len = 0
            parts.append(w)
            continue
        # If adding this word would exceed max_c, start new line
        if current_len + len(w) + (1 if current_line else 0) > max_c:
            if current_line:
                parts.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + (1 if current_line[:-1] else 0)
    if current_line:
        parts.append(" ".join(current_line))
    # Distribute timing proportionally
    dur = (seg["end"] - seg["start"]) / max(1, len(parts))
    out, t0 = [], seg["start"]
    for p in parts:
        t1 = t0 + dur
        out.append({"start": t0, "end": t1, "text": p})
        t0 = t1
    return out

def balance_lines(txt:str)->str:
    w=txt.split(); 
    return txt if len(w)<4 else " ".join(w[:len(w)//2])+"\n"+" ".join(w[len(w)//2:])

def build_srt(segs:List[Dict], lower:bool, double:bool, preset:str)->str:
    out=[]
    for i,s in enumerate(segs,1):
        text=s["text"]
        if lower: text=text.lower()
        text=sanitize(text,preset)
        if double: text=balance_lines(text)
        out.append(srt.Subtitle(index=i,
                    start=srt.timedelta(seconds=s["start"]),
                    end=srt.timedelta(seconds=s["end"]),
                    content=text))
    return srt.compose(out)

def validate_model(model_name: str) -> str:
    """Validate and return a working model name, with fallback"""
    available_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2']
    
    # If user specified a model that might not work, try it first
    if model_name not in available_models:
        try:
            print(f"🔄 Testing {model_name}...")
            test_model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print(f"✅ {model_name} works!")
            return model_name
        except Exception as e:
            print(f"⚠️  {model_name} failed to load: {e}")
            print(f"🔄 Falling back to large-v2...")
            return "large-v2"
    
    return model_name

# -------------- main -----------------
def main():
    p=argparse.ArgumentParser(); p.add_argument("media", nargs='+', help="Path to audio/video file")
    args=p.parse_args()
    
    # Create settings file if it doesn't exist
    create_settings_file()
    
    # Join all arguments to handle paths with spaces
    media_path = " ".join(args.media)
    media = Path(media_path)
    
    if not media.exists(): 
        sys.exit(f"❌ File not found: {media}")

    wav=extract_audio(media)
    duration=ffprobe_duration(wav)
    
    # Load settings first
    settings = load_settings()
    model_name = validate_model(settings["model"])
    
    # Use CPU for cross-platform compatibility
    device = "cpu"
    model=WhisperModel(model_name, device=device, compute_type="int8")

    log.info("Transcribing with faster-whisper …")
    bar=tqdm(total=duration, unit="s", desc="progress", ncols=70)

    segments_data=[]
    for seg in model.transcribe(str(wav), beam_size=5, vad_filter=True, word_timestamps=False)[0]:
        segments_data.append({"start":seg.start, "end":seg.end, "text":seg.text})
        bar.update(seg.end-seg.start)
    bar.close(); print("✅ transcription complete\n")

    fps=30; gap_s=settings["gap_frames"]/fps

    # Apply settings
    lower = settings["lower_case"]
    simplify_punct = settings["simplify_punctuation"]
    single_lines = settings["single_lines"]
    max_c = settings["max_chars"]
    min_d = settings["min_duration"]
    
    # Determine punctuation preset based on simplify_punctuation setting
    preset = "dialogue" if simplify_punct else "full"
    
    segs=merge_segments(segments_data, min_d, gap_s)
    final=[]; [final.extend(split_chars(s,max_c)) for s in segs]

    srt_str=build_srt(final,lower,not single_lines,preset)

    base=media.with_suffix("")
    srt_path = Path(f"{base}.srt")
    srt_path.write_text(srt_str,encoding="utf-8")
    print(f"\n🎉 SRT file ready! Saved as: {srt_path}")

if __name__=="__main__":
    main()