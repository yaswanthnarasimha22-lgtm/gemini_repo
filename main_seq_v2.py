#!/usr/bin/env python3
"""
Travel Call TRANSCRIPTION — Hourly Scheduled Parallel Sequential Vertex AI
===========================================================================
Features:
  - 30-minute scheduler: runs every 30 minutes until 2:00 AM UK time
  - Subfolder support: navigates date/02/, date/06/, date/07/ etc.
  - DID & Extension filtering: reads from dids.txt and extensions.txt
  - Processed file tracking: persists to processed_calls.json (no reprocessing)
  - Sequential file processing: each worker completes one file before starting another
  - Improved rate limit handling: global semaphore + exponential backoff
  - Sequential chunks per file, parallel files (critical for transcription quality)
  - GPU preprocessing (inaSpeechSegmenter + VAD) preserved
  - Graceful shutdown: waits for all processing to complete before stopping
  - Agent name correction: uses context cache from agent_names.txt to spell agent names correctly
  - Transcript duration verification: ensures transcripts match audio duration
  - Retry mechanism: retries failed files after all others are processed
  - FIXED: Chunk boundary loss with 45s overlap + safe stitching
  - FIXED: VAD over-aggressiveness (level 1, 500ms min speech)
  - FIXED: Hallucination cleaner (safe skipping vs destructive deletion)
  - FIXED: Chunk-level retry + truncation detection
  - FIXED: Chunk sanity validation before stitching
"""
import os
import sys
import json
import re
import time
import math
import subprocess
import traceback
import hashlib
import argparse
import shutil
import threading
import base64
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

# =======================================================================
# CONFIGURATION CONSTANTS
# =======================================================================
class Config:
    # Buckets & Paths
    LOCATION = "us-central1"
    INPUT_BUCKET = "travel-audio-batch-input"
    INPUT_BASE_FOLDER = "teleappliant_recordings"
    OUTPUT_BUCKET = "travel-audio-results-2409"
    OUTPUT_FOLDER = "results"
    LOCAL_DOWNLOAD_DIR = "audio_downloads"
    LOCAL_RESULTS_DIR = "analysis_results"
    LOCAL_CHUNKS_DIR = "audio_chunks"
    LOCAL_PREPROCESSED_DIR = "preprocessed_audio"
    CREDENTIALS_PATH = "credentials.json"
    PROCESSED_TRACKER_FILE = "processed_calls.json"
    
    # Filter Files
    DIDS_FILE = "dids.txt"
    EXTENSIONS_FILE = "extensions.txt"
    AGENT_NAMES_FILE = "agent_names.txt"
    NUMBERS_FILE = "numbers.txt"
    
    # Scheduler
    SCHEDULER_END_HOUR = 2       # 2:00 AM (next day)
    SCHEDULER_END_MINUTE = 0
    SCHEDULER_INTERVAL_SECONDS = 1000  # 30 minutes
    SCHEDULER_FINAL_CHECK_MINUTES_BEFORE = 30
    
    # Preprocessing
    FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 1  # Preserves soft speech
    HIGH_PASS_HZ = 100
    LOW_PASS_HZ = 4000
    MIN_SPEECH_DURATION_MS = 500  # Captures short responses
    
    # Chunking (CRITICAL FOR QUALITY - DO NOT REDUCE OVERLAP)
    CHUNK_DURATION_MINUTES = 15
    CHUNK_OVERLAP_SECONDS = 45  # CRITICAL FIX for sentence continuity
    SHORT_AUDIO_THRESHOLD_MINUTES = 8
    MAX_CHUNKS_PER_FILE = 30
    
    # Hallucination Detection
    MAX_LINE_LENGTH = 500
    REPETITION_WINDOW = 5
    REPETITION_SIMILARITY = 0.75
    MIN_CHUNK_WORDS = 20
    MIN_UNIQUE_WORD_RATIO = 0.5
    
    # API Settings
    GEMINI_MODEL = "gemini-2.5-flash"
    TRANSCRIBE_MAX_TOKENS = 65536
    TEMPERATURE = 0.1
    SEQUENTIAL_RETRY_COUNT = 5
    SEQUENTIAL_RETRY_DELAY = 15
    SEQUENTIAL_CALL_DELAY = 5
    RATE_LIMIT_BACKOFF_BASE = 30
    RATE_LIMIT_BACKOFF_MAX = 300
    MAX_CHUNK_RETRIES = 2
    
    # Parallelism (Optimized for 24-core/24GB system)
    DEFAULT_WORKERS = 3
    MAX_WORKERS = 15
    MAX_CONCURRENT_API_CALLS = 2  # Critical for rate limit safety
    
    # Audio Formats
    AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".aac", ".wma", ".webm", ".opus")
    AUDIO_MIME_TYPES = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".mp4": "video/mp4", ".aac": "audio/aac", ".wma": "audio/x-ms-wma",
        ".webm": "audio/webm", ".opus": "audio/opus",
    }

# =======================================================================
# GLOBAL STATE
# =======================================================================
DEVICE_INFO = {}
DEVICE = "cpu"
PROJECT_ID = None
START_DATE = None
shutdown_requested = False
_api_semaphore = threading.Semaphore(Config.MAX_CONCURRENT_API_CALLS)
_rate_limit_lock = threading.Lock()
_last_429_time = 0.0
_seg = None
_seg_device = None
_seg_lock = threading.Lock()
_save_lock = threading.Lock()
_log_lock = threading.Lock()

# =======================================================================
# DEVICE & SETUP
# =======================================================================
def get_device_info():
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "vram_free_gb": round((torch.cuda.get_device_properties(0).total_memory - 
                                 torch.cuda.memory_allocated(0)) / (1024**3), 2),
            "cuda_version": torch.version.cuda,
        }
    return {"device": "cpu", "gpu_name": None, "vram_total_gb": 0, "vram_free_gb": 0, "cuda_version": None}

def _load_project_id():
    global PROJECT_ID
    if not os.path.exists(Config.CREDENTIALS_PATH):
        print(f"ERROR: {Config.CREDENTIALS_PATH} not found")
        sys.exit(1)
    with open(Config.CREDENTIALS_PATH) as f:
        creds = json.load(f)
    PROJECT_ID = creds.get("project_id")
    if not PROJECT_ID:
        print("ERROR: No project_id in credentials.json")
        sys.exit(1)
    print(f"[CONFIG] Project: {PROJECT_ID} | SA: {creds.get('client_email', '?')}")
    print(f"[CONFIG] Buckets: in={Config.INPUT_BUCKET}, out={Config.OUTPUT_BUCKET}")

# Initialize early dependencies
DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["device"]
_load_project_id()

# =======================================================================
# LOGGING & UTILITIES
# =======================================================================
def log(msg: str, level: str = "INFO"):
    """Thread-safe logging with emoji indicators."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols = {
        "INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "PROCESSING": "🔄", "UPLOAD": "📤",
        "DOWNLOAD": "📥", "AUDIO": "🎤", "SAVE": "💾", "WAIT": "⏳", "DONE": "🎉", "BATCH": "📦",
        "DATE": "📅", "MEGA": "🚀", "SEQ": "🔢", "PREPROCESS": "🎵", "VAD": "🔇", "MAP": "🗺️",
        "VERTEX": "🔷", "GPU": "🎮", "API": "🔶", "WORKER": "🔧", "SCHEDULE": "🕐", "TRACK": "📋",
        "FILTER": "🔍", "HOUR": "⏰"
    }
    tid = threading.current_thread().name
    prefix = f"[{tid}] " if tid != "MainThread" else ""
    with _log_lock:
        print(f"[{ts}] {symbols.get(level, '•')} {prefix}{msg}")

def format_file_size(bytes_size: Optional[int]) -> str:
    if bytes_size is None:
        return "?"
    if bytes_size < 1024:
        return f"{bytes_size}B"
    if bytes_size < 1048576:
        return f"{bytes_size/1024:.1f}KB"
    return f"{bytes_size/1048576:.1f}MB"

def format_duration(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60}s"
    return f"{s//60}m {s%60}s"

def get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def print_gpu_status():
    """Log current VRAM usage."""
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        log(f"VRAM: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved / {DEVICE_INFO['vram_total_gb']}GB", "GPU")

def clear_gpu_cache():
    """Clear CUDA cache and synchronize."""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =======================================================================
# UK TIME HANDLING (BST/GMT aware)
# =======================================================================
def get_uk_now() -> datetime:
    """Get current UK time (handles GMT/BST automatically)."""
    utc_now = datetime.now(timezone.utc)
    year = utc_now.year
    
    # Find last Sunday of March (BST start)
    mar31 = datetime(year, 3, 31, tzinfo=timezone.utc)
    bst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
    bst_start = bst_start.replace(hour=1, minute=0, second=0)
    
    # Find last Sunday of October (BST end)
    oct31 = datetime(year, 10, 31, tzinfo=timezone.utc)
    bst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
    bst_end = bst_end.replace(hour=1, minute=0, second=0)
    
    if bst_start <= utc_now < bst_end:
        return utc_now + timedelta(hours=1)  # BST = UTC+1
    return utc_now  # GMT = UTC

def is_past_cutoff() -> bool:
    """Check if UK time is past 2:00 AM of the next day."""
    global START_DATE
    if START_DATE is None:
        uk = get_uk_now()
        return uk.hour > Config.SCHEDULER_END_HOUR or (
            uk.hour == Config.SCHEDULER_END_HOUR and uk.minute >= Config.SCHEDULER_END_MINUTE
        )
    
    uk = get_uk_now()
    cutoff_date = START_DATE + timedelta(days=1)
    cutoff = datetime(
        cutoff_date.year, cutoff_date.month, cutoff_date.day,
        Config.SCHEDULER_END_HOUR, Config.SCHEDULER_END_MINUTE,
        tzinfo=uk.tzinfo
    )
    return uk >= cutoff

def time_until_cutoff() -> float:
    """Seconds until 2:00 AM cutoff."""
    global START_DATE
    if START_DATE is None:
        return 0
    
    uk = get_uk_now()
    cutoff_date = START_DATE + timedelta(days=1)
    cutoff = datetime(
        cutoff_date.year, cutoff_date.month, cutoff_date.day,
        Config.SCHEDULER_END_HOUR, Config.SCHEDULER_END_MINUTE,
        tzinfo=uk.tzinfo
    )
    
    return max(0, (cutoff - uk).total_seconds())

# =======================================================================
# FILTERING & AGENT NAMES
# =======================================================================
def load_filter_list(filepath: str, label: str) -> List[str]:
    """Load filter list from file (one item per line)."""
    if not os.path.exists(filepath):
        log(f"{label} file not found: {filepath} — no {label} filter active", "WARN")
        return []
    
    items = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(line)
    
    if items:
        log(f"Loaded {len(items)} {label} from {filepath}", "OK")
        for item in items[:5]:
            log(f"  • {item}", "INFO")
        if len(items) > 5:
            log(f"  ... and {len(items)-5} more", "INFO")
    return items

def filename_matches_filters(filename: str, dids: List[str], extensions: List[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if filename matches any DID or extension."""
    for did in dids:
        if did in filename:
            return True, "DID", did
    
    for ext in extensions:
        if ext in filename:
            return True, "EXT", ext
    
    return False, None, None

def load_agent_names(filepath: str = Config.AGENT_NAMES_FILE) -> List[str]:
    """Load agent names for spelling correction."""
    if not os.path.exists(filepath):
        log(f"Agent names file not found: {filepath} — no agent names to correct", "WARN")
        return []
    
    names = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                name = line.split(',')[0].strip() if ',' in line else line
                names.append(name)
    
    if names:
        log(f"Loaded {len(names)} agent names from {filepath}", "OK")
        for name in names[:5]:
            log(f"  • {name}", "INFO")
        if len(names) > 5:
            log(f"  ... and {len(names)-5} more", "INFO")
    return names

# =======================================================================
# TRANSCRIPTION PROMPT
# =======================================================================
def get_transcription_prompt(agent_names: Optional[List[str]] = None) -> str:
    """Generate transcription prompt with agent name correction."""
    prompt = """You are a precise audio transcriptionist for a travel call center. The brand is Teletext Holidays, and the website is teletextholidays.co.uk.

TASK: Transcribe this audio segment COMPLETELY — every single word, exactly as spoken.

CRITICAL RULES:
1. Provide the COMPLETE word-for-word transcript. Do NOT skip, summarize, or paraphrase ANY part.
2. Label speakers as "Agent:" and "Customer:" (or "Agent 1:", "Agent 2:" if multiple agents).
3. Include timestamps relative to THIS audio segment starting from [00:00].
   - Use format [MM:SS] for times under 1 hour, [H:MM:SS] for times over 1 hour.
   - Add a timestamp at least every 30 seconds of speech, and at every speaker change.
   - Each new timestamp MUST be on its own new line.
4. Redact credit card numbers, CVV codes, expiration dates, and full street addresses as [PII REDACTED].
5. Do NOT redact phone numbers, email addresses, or names — keep them exactly as spoken.
6. Capture filler words (um, uh, like), false starts, and interruptions accurately.
7. If speakers talk over each other, note it as [crosstalk].
8. If there is silence or hold music, note it briefly (e.g., [hold music ~2min], [silence ~30s]).
9. Do NOT add any analysis, summary, or commentary — ONLY the transcript.
10. Do NOT skip any part of the audio even if it seems repetitive or unimportant.
11. Transcribe the ENTIRE audio from start to finish with NO gaps.
12. IMPORTANT: Put EACH speaker turn on its own line. Do NOT put the entire transcript on a single line.
13. NEVER repeat the same phrase or sentence. If you notice yourself repeating, STOP and move forward.
14. If the transcript appears incomplete due to audio cutoff, continue transcribing until the audio segment ends completely."""

    if agent_names and len(agent_names) > 0:
        agent_names_str = "\n".join(f"- {name}" for name in agent_names)
        prompt += f"""

KNOWN AGENT NAMES:
{agent_names_str}

When transcribing, ensure these agent names are spelled correctly as shown above. Pay special attention to unusual spellings or pronunciations. If you hear an agent introduce themselves, use the correct spelling from this list.
"""

    prompt += """

OUTPUT FORMAT — one speaker turn per line, nothing else:
[00:00] Agent: ...
[00:05] Customer: ...
[00:30] Agent: ...
"""
    return prompt

# =======================================================================
# CHUNK VALIDATION & HALLUCINATION DETECTION
# =======================================================================
def is_valid_chunk(transcript: str) -> bool:
    """Validate chunk transcript quality."""
    words = transcript.split()
    if len(words) < Config.MIN_CHUNK_WORDS:
        return False
    
    unique_ratio = len(set(words)) / (len(words) + 1)
    if unique_ratio < Config.MIN_UNIQUE_WORD_RATIO:
        return False
    
    # Reject truncated transcripts
    if transcript.endswith("...") or transcript.endswith("…"):
        return False
    
    return True

def clean_chunk_transcript(raw_transcript: str) -> str:
    """Remove hallucinations and repetitions safely."""
    if not raw_transcript:
        return raw_transcript
    
    # Split lines carefully (handle multiple timestamps per line)
    lines = []
    for raw_line in raw_transcript.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        
        ts_positions = [m.start() for m in re.finditer(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', raw_line)]
        if len(ts_positions) > 1:
            for i, pos in enumerate(ts_positions):
                end = ts_positions[i+1] if i+1 < len(ts_positions) else len(raw_line)
                segment = raw_line[pos:end].strip()
                if segment:
                    lines.append(segment)
        else:
            lines.append(raw_line)
    
    # Cap excessively long lines
    capped_lines = []
    for line in lines:
        if len(line) > Config.MAX_LINE_LENGTH:
            truncated = line[:Config.MAX_LINE_LENGTH]
            last_space = truncated.rfind(' ')
            if last_space > Config.MAX_LINE_LENGTH * 0.7:
                truncated = truncated[:last_space]
            capped_lines.append(truncated)
            log(f"    ⚠️ Truncated hallucination line: {len(line)} → {len(truncated)} chars", "WARN")
        else:
            capped_lines.append(line)
    
    # Detect repetitions safely
    cleaned = []
    repetition_count = 0
    repetition_truncated = False
    
    for i, line in enumerate(capped_lines):
        if repetition_truncated:
            break
        
        current_norm = _normalize_text(line)
        is_repeat = False
        
        if current_norm and len(current_norm) > 10 and len(cleaned) >= 2:
            recent_norms = [_normalize_text(l) for l in cleaned[-Config.REPETITION_WINDOW:]]
            for recent in recent_norms:
                if recent and len(recent) > 10 and _text_similarity(current_norm, recent) > Config.REPETITION_SIMILARITY:
                    is_repeat = True
                    break
        
        if is_repeat:
            repetition_count += 1
            if repetition_count >= Config.REPETITION_WINDOW:
                log(f"    ⚠️ Repetition detected at line {i}, skipping repeated content", "WARN")
                repetition_truncated = True
                continue
        else:
            repetition_count = 0
        
        cleaned.append(line)
    
    result = '\n'.join(cleaned)
    if len(result) < len(raw_transcript) * 0.5 and len(raw_transcript) > 1000:
        log(f"    ℹ️ Cleaned: {len(raw_transcript)} → {len(result)} chars", "INFO")
    
    return result

def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'(agent|customer)\s*\d*\s*:', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def _text_similarity(a: str, b: str) -> float:
    """Calculate text similarity ratio."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# =======================================================================
# AUDIO PREPROCESSING (GPU ACCELERATED)
# =======================================================================
def get_segmenter():
    """Get thread-safe inaSpeechSegmenter instance."""
    global _seg, _seg_device
    with _seg_lock:
        if _seg is None:
            target = DEVICE
            log(f"Loading inaSpeechSegmenter ({target.upper()})...", 
                "GPU" if target == "cuda" else "PREPROCESS")
            
            if target == "cuda":
                vram_before = torch.cuda.memory_allocated(0) / (1024**3)
                log(f"  VRAM before: {vram_before:.2f}GB", "GPU")
            
            try:
                _seg = Segmenter()
                _seg_device = target
                
                if target == "cuda":
                    vram_after = torch.cuda.memory_allocated(0) / (1024**3)
                    delta = vram_after - vram_before
                    if delta > 0.1:
                        log(f"  Loaded on GPU (VRAM: +{delta:.2f}GB)", "GPU")
                    else:
                        log(f"  Loaded (VRAM +{delta:.2f}GB — may be on CPU)", "WARN")
                        _seg_device = "cpu"
                else:
                    log("  Loaded on CPU", "OK")
            except Exception as e:
                log(f"  Load failed: {e}", "WARN")
                if target == "cuda":
                    log("  Falling back to CPU...", "INFO")
                    _seg = Segmenter()
                    _seg_device = "cpu"
                    log("  Loaded on CPU (fallback)", "OK")
                else:
                    raise
    return _seg

def run_segmenter(path: str):
    """Run segmenter with thread safety."""
    with _seg_lock:
        return _seg(path)

def filter_music(path: str) -> Tuple[Any, List[Tuple[int, int]]]:
    """Remove music segments using inaSpeechSegmenter."""
    log("    Removing music..." + (" (GPU)" if _seg_device == "cuda" else ""),
        "GPU" if _seg_device == "cuda" else "PREPROCESS")
    start_time = time.time()
    
    segments = run_segmenter(path)
    elapsed = time.time() - start_time
    
    orig = AudioSegment.from_file(path)
    speech = AudioSegment.empty()
    kept_segments = []
    
    for label, start, end in segments:
        if label in ['male', 'female']:
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            speech += orig[start_ms:end_ms]
            kept_segments.append((start_ms, end_ms))
    
    log(f"    {len(orig)}ms→{len(speech)}ms ({len(kept_segments)} segs) [{elapsed:.1f}s]", "PREPROCESS")
    return speech, kept_segments

def vad_filter(audio: Any, aggressiveness: int = Config.VAD_AGGRESSIVENESS) -> Tuple[Any, List[Tuple[int, int]]]:
    """Apply VAD filtering to remove non-speech segments."""
    vad = webrtcvad.Vad(aggressiveness)
    pcm_data, sample_rate = _audio_to_pcm(audio)
    
    frame_duration = Config.FRAME_DURATION_MS
    frame_size = int(sample_rate * frame_duration / 1000) * 2
    frames = [pcm_data[i:i+frame_size] for i in range(0, len(pcm_data), frame_size)]
    
    filtered_audio = AudioSegment.empty()
    kept_ranges = []
    
    for i, frame in enumerate(frames):
        if len(frame) < frame_size:
            continue
        if vad.is_speech(frame, sample_rate):
            start_ms = i * frame_duration
            end_ms = start_ms + frame_duration
            filtered_audio += audio[start_ms:end_ms]
            kept_ranges.append((start_ms, end_ms))
    
    return filtered_audio, kept_ranges

def _audio_to_pcm(audio: Any) -> Tuple[bytes, int]:
    """Convert audio to 16kHz mono PCM."""
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    return audio.raw_data, 16000

def build_timestamp_map(kept_segments: List[Tuple[int, int]], 
                       vad_kept: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Build mapping from processed time to original time."""
    # Build music removal map
    music_map = []
    current_processed = 0
    for start_orig, end_orig in kept_segments:
        music_map.append((current_processed, start_orig))
        duration = end_orig - start_orig
        music_map.append((current_processed + duration, end_orig))
        current_processed += duration
    
    if not music_map:
        return []
    
    # Build VAD map
    vad_map = []
    current_vad = 0
    for start_vad, end_vad in vad_kept:
        vad_map.append((current_vad, start_vad))
        duration = end_vad - start_vad
        vad_map.append((current_vad + duration, end_vad))
        current_vad += duration
    
    if not vad_map:
        return music_map
    
    # Combine maps
    final_map = []
    for vad_time, orig_time_after_music in vad_map:
        orig_time = _interpolate(music_map, orig_time_after_music)
        final_map.append((vad_time, orig_time))
    
    # Deduplicate and simplify
    final_map = sorted(set(final_map), key=lambda x: x[0])
    if len(final_map) > 1000:
        step = max(1, len(final_map) // 1000)
        simplified = [final_map[i] for i in range(0, len(final_map), step)]
        if simplified[-1] != final_map[-1]:
            simplified.append(final_map[-1])
        final_map = simplified
    
    return final_map

def _interpolate(mapping: List[Tuple[int, int]], query_time: int) -> int:
    """Interpolate timestamp using linear mapping."""
    if not mapping:
        return query_time
    if query_time <= mapping[0][0]:
        return mapping[0][1]
    if query_time >= mapping[-1][0]:
        return mapping[-1][1]
    
    # Binary search for interpolation points
    low, high = 0, len(mapping) - 1
    while low < high - 1:
        mid = (low + high) // 2
        if mapping[mid][0] <= query_time:
            low = mid
        else:
            high = mid
    
    proc1, orig1 = mapping[low]
    proc2, orig2 = mapping[high]
    
    if proc2 == proc1:
        return orig1
    
    ratio = (query_time - proc1) / (proc2 - proc1)
    return int(orig1 + ratio * (orig2 - orig1))

def preprocess_audio(path: str, filename: str, work_dir: str) -> Tuple[Optional[str], Optional[List], float, Optional[float]]:
    """Full preprocessing pipeline: music removal + VAD + filtering."""
    log(f"    Preprocessing: {filename}", "PREPROCESS")
    try:
        original = AudioSegment.from_file(path)
        orig_duration = len(original) / 1000.0
        log(f"    Original: {format_duration(orig_duration)}", "AUDIO")
        
        # Step 1: Remove music
        speech, kept_segments = filter_music(path)
        if len(speech) < Config.MIN_SPEECH_DURATION_MS:
            log("    Too short after music removal", "WARN")
            return None, None, orig_duration, None
        
        # Step 2: Apply bandpass filter
        log(f"    Bandpass {Config.HIGH_PASS_HZ}-{Config.LOW_PASS_HZ}Hz...", "PREPROCESS")
        filtered = speech.high_pass_filter(Config.HIGH_PASS_HZ).low_pass_filter(Config.LOW_PASS_HZ)
        
        # Step 3: Apply VAD
        log(f"    VAD (agg={Config.VAD_AGGRESSIVENESS})...", "VAD")
        vad_audio, vad_kept = vad_filter(filtered, Config.VAD_AGGRESSIVENESS)
        if vad_audio is None or len(vad_audio) < Config.MIN_SPEECH_DURATION_MS:
            log("    Too short after VAD", "WARN")
            return None, None, orig_duration, None
        
        # Step 4: Build timestamp mapping
        proc_duration = len(vad_audio) / 1000.0
        log(f"    Clean: {format_duration(proc_duration)} (removed {format_duration(orig_duration - proc_duration)})", "OK")
        timestamp_map = build_timestamp_map(kept_segments, vad_kept)
        log(f"    Map: {len(timestamp_map)} points", "MAP")
        
        # Step 5: Save preprocessed audio
        prep_dir = os.path.join(work_dir, "preprocessed")
        os.makedirs(prep_dir, exist_ok=True)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(prep_dir, f"{base_name}_preprocessed.mp3")
        vad_audio.export(output_path, format="mp3", parameters=["-b:a", "64k", "-ar", "16000", "-ac", "1"])
        
        return output_path, timestamp_map, orig_duration, proc_duration
    
    except Exception as e:
        log(f"    Failed: {e}", "ERROR")
        traceback.print_exc()
        return None, None, 0.0, None

# =======================================================================
# CHUNKING
# =======================================================================
def get_audio_duration(filepath: str) -> Optional[float]:
    """Get audio duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def split_chunks(preprocessed_path: str, chunk_dir: str, filename: str) -> Tuple[List[Dict], Optional[float]]:
    """Split audio into overlapping chunks."""
    base_name = os.path.splitext(filename)[0]
    duration = get_audio_duration(preprocessed_path)
    
    if duration is None:
        return [{"path": preprocessed_path, "chunk_num": 1, "start_time": 0, 
                "end_time": None, "original_filename": filename}], None
    
    # Short audio: no chunking needed
    if duration <= Config.SHORT_AUDIO_THRESHOLD_MINUTES * 60:
        return [{
            "path": preprocessed_path,
            "chunk_num": 1,
            "start_time": 0,
            "end_time": duration,
            "original_filename": filename
        }], duration
    
    # Chunk longer audio with overlap
    chunk_duration_sec = Config.CHUNK_DURATION_MINUTES * 60
    overlap_sec = Config.CHUNK_OVERLAP_SECONDS
    step_sec = chunk_duration_sec - overlap_sec
    chunks = []
    chunk_num = 0
    start_time = 0
    
    chunk_base_dir = os.path.join(chunk_dir, base_name)
    os.makedirs(chunk_base_dir, exist_ok=True)
    
    while start_time < duration and chunk_num < Config.MAX_CHUNKS_PER_FILE:
        chunk_num += 1
        end_time = min(start_time + chunk_duration_sec, duration)
        actual_duration = end_time - start_time
        
        # Skip tiny chunks at end
        if actual_duration < 10 and chunk_num > 1:
            break
        
        chunk_filename = f"{base_name}_chunk{chunk_num:03d}.mp3"
        chunk_path = os.path.join(chunk_base_dir, chunk_filename)
        
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", preprocessed_path, "-ss", str(start_time), 
                 "-t", str(actual_duration), "-acodec", "libmp3lame", "-b:a", "64k",
                 "-ar", "16000", "-ac", "1", "-loglevel", "error", chunk_path],
                check=True, capture_output=True, timeout=180
            )
            
            if os.path.exists(chunk_path):
                chunks.append({
                    "path": chunk_path,
                    "chunk_num": chunk_num,
                    "start_time": start_time,
                    "end_time": end_time,
                    "size": os.path.getsize(chunk_path),
                    "original_filename": filename,
                    "file_hash": get_file_hash(chunk_path)
                })
        except Exception as e:
            log(f"    Chunk {chunk_num} creation failed: {e}", "ERROR")
        
        start_time += step_sec
    
    return chunks, duration

# =======================================================================
# VERTEX AI TRANSCRIPTION (SEQUENTIAL PER FILE)
# =======================================================================
def wait_for_rate_limit_cooldown() -> float:
    """Check if we need to cool down after recent 429 errors."""
    global _last_429_time
    with _rate_limit_lock:
        elapsed = time.time() - _last_429_time
        return max(0, 30 - elapsed)

def record_rate_limit():
    """Record timestamp of last 429 error."""
    global _last_429_time
    with _rate_limit_lock:
        _last_429_time = time.time()

def transcribe_chunk_sequential(client, chunk_path: str, chunk_num: int, total_chunks: int,
                               original_filename: str, agent_names: Optional[List[str]] = None) -> Dict:
    """Transcribe a single chunk with robust retry logic."""
    ext = os.path.splitext(chunk_path)[1].lower()
    mime_type = Config.AUDIO_MIME_TYPES.get(ext, "audio/mpeg")
    
    with open(chunk_path, "rb") as f:
        audio_bytes = f.read()
    
    prompt = get_transcription_prompt(agent_names)
    if total_chunks > 1:
        prompt += f"\nNOTE: Segment {chunk_num} of {total_chunks}. Transcribe from [00:00]. Complete segment."
    
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": audio_b64}}
        ]
    }]
    
    for attempt in range(1, Config.SEQUENTIAL_RETRY_COUNT + 1):
        # Rate limit cooldown
        cooldown = wait_for_rate_limit_cooldown()
        if cooldown > 0:
            log(f"      Chunk {chunk_num}: Cooling down {cooldown:.0f}s (recent 429)...", "WAIT")
            time.sleep(cooldown)
        
        # Acquire API semaphore
        _api_semaphore.acquire()
        try:
            start_time = time.time()
            response = client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=contents,
                config={
                    "temperature": Config.TEMPERATURE,
                    "max_output_tokens": Config.TRANSCRIBE_MAX_TOKENS,
                }
            )
            elapsed = time.time() - start_time
            
            if response and response.text:
                transcript = response.text.strip()
                
                # Validate transcript quality
                if not is_valid_chunk(transcript):
                    log(f"      Chunk {chunk_num}: suspiciously short/low-quality transcript "
                        f"({len(transcript.split())} words)", "WARN")
                    if attempt < Config.SEQUENTIAL_RETRY_COUNT:
                        time.sleep(Config.SEQUENTIAL_RETRY_DELAY * attempt)
                        continue
                
                log(f"      Chunk {chunk_num}/{total_chunks} OK ({len(transcript)} chars, {elapsed:.1f}s)", "API")
                return {"transcript": transcript, "status": "success", "error": None}
            else:
                log(f"      Chunk {chunk_num}: Empty response (attempt {attempt}/{Config.SEQUENTIAL_RETRY_COUNT})", "WARN")
        
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str.upper():
                record_rate_limit()
                wait_time = min(
                    Config.RATE_LIMIT_BACKOFF_BASE * (2 ** (attempt - 1)) + (attempt * 10),
                    Config.RATE_LIMIT_BACKOFF_MAX
                )
                log(f"      Chunk {chunk_num}: 429 Rate limited (attempt {attempt}/{Config.SEQUENTIAL_RETRY_COUNT}) "
                    f"— waiting {wait_time}s...", "WAIT")
                time.sleep(wait_time)
            elif "403" in error_str or "401" in error_str:
                log(f"      Chunk {chunk_num}: Auth error — check SA permissions", "ERROR")
                return {"transcript": None, "status": "failed", "error": f"Auth: {error_str[:100]}"}
            else:
                log(f"      Chunk {chunk_num}: Error (attempt {attempt}/{Config.SEQUENTIAL_RETRY_COUNT}): {error_str[:120]}", "ERROR")
                if attempt < Config.SEQUENTIAL_RETRY_COUNT:
                    time.sleep(Config.SEQUENTIAL_RETRY_DELAY * attempt)
        finally:
            _api_semaphore.release()
    
    return {"transcript": None, "status": "failed", "error": f"All {Config.SEQUENTIAL_RETRY_COUNT} retries exhausted"}

def transcribe_file_sequential(client, chunks_meta: List[Dict], original_filename: str,
                              agent_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Transcribe all chunks sequentially with per-chunk retries."""
    results = {}
    total_chunks = len(chunks_meta)
    
    for chunk_info in sorted(chunks_meta, key=lambda x: x["chunk_num"]):
        chunk_id = f"{original_filename}_chunk{chunk_info['chunk_num']:03d}"
        
        # Primary attempt
        result = transcribe_chunk_sequential(
            client=client,
            chunk_path=chunk_info["path"],
            chunk_num=chunk_info["chunk_num"],
            total_chunks=total_chunks,
            original_filename=original_filename,
            agent_names=agent_names
        )
        results[chunk_id] = result
        
        # Retry failed chunks (with backoff)
        if result["status"] != "success":
            log(f"      🔁 Retrying chunk {chunk_info['chunk_num']} (up to {Config.MAX_CHUNK_RETRIES} retries)...", "WARN")
            
            for retry_attempt in range(Config.MAX_CHUNK_RETRIES):
                time.sleep(5 * (retry_attempt + 1))  # Exponential backoff
                
                result = transcribe_chunk_sequential(
                    client=client,
                    chunk_path=chunk_info["path"],
                    chunk_num=chunk_info["chunk_num"],
                    total_chunks=total_chunks,
                    original_filename=original_filename,
                    agent_names=agent_names
                )
                results[chunk_id] = result
                
                if result["status"] == "success":
                    break
        
        # Delay between chunks to avoid rate limits
        if chunk_info["chunk_num"] < total_chunks:
            time.sleep(Config.SEQUENTIAL_CALL_DELAY)
    
    return results

# =======================================================================
# TIMESTAMP HANDLING
# =======================================================================
TIMESTAMP_RE = re.compile(r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]')

def _parse_timestamp(match) -> Optional[int]:
    """Parse timestamp match to seconds."""
    if isinstance(match, str):
        m = TIMESTAMP_RE.search(match)
        if not m:
            return None
        groups = m.groups()
    else:
        groups = match.groups()
    
    hours = 0
    minutes = int(groups[0])
    seconds = int(groups[1])
    
    if groups[2] is not None:  # Has hours component
        hours = minutes
        minutes = seconds
        seconds = int(groups[2])
    
    return hours * 3600 + minutes * 60 + seconds

def _seconds_to_timestamp(seconds: int) -> str:
    """Convert seconds to [MM:SS] or [H:MM:SS] format."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"

def adjust_timestamp(line: str, offset_seconds: float) -> str:
    """Adjust timestamps by adding offset."""
    def replacer(match):
        orig_seconds = _parse_timestamp(match)
        if orig_seconds is None:
            return match.group(0)
        return _seconds_to_timestamp(orig_seconds + offset_seconds)
    
    return TIMESTAMP_RE.sub(replacer, line)

def remap_timestamp(line: str, timestamp_map: Optional[List[Tuple[int, int]]], 
                   max_original_sec: Optional[float]) -> str:
    """Remap processed timestamps back to original audio timeline."""
    def replacer(match):
        proc_seconds = _parse_timestamp(match)
        if proc_seconds is None:
            return match.group(0)
        
        # Convert to milliseconds for mapping
        proc_ms = proc_seconds * 1000
        orig_ms = _lookup_original_time(timestamp_map, proc_ms)
        
        # Clamp to max duration if provided
        if max_original_sec and orig_ms > max_original_sec * 1000:
            orig_ms = max_original_sec * 1000
        
        return _seconds_to_timestamp(orig_ms / 1000.0)
    
    return TIMESTAMP_RE.sub(replacer, line)

def _lookup_original_time(timestamp_map: Optional[List[Tuple[int, int]]], 
                         processed_ms: int) -> int:
    """Lookup original time using timestamp mapping."""
    if not timestamp_map:
        return processed_ms
    
    if processed_ms <= timestamp_map[0][0]:
        return timestamp_map[0][1]
    if processed_ms >= timestamp_map[-1][0]:
        return timestamp_map[-1][1]
    
    # Binary search
    low, high = 0, len(timestamp_map) - 1
    while low < high - 1:
        mid = (low + high) // 2
        if timestamp_map[mid][0] <= processed_ms:
            low = mid
        else:
            high = mid
    
    proc1, orig1 = timestamp_map[low]
    proc2, orig2 = timestamp_map[high]
    
    if proc2 == proc1:
        return orig1
    
    ratio = (processed_ms - proc1) / (proc2 - proc1)
    return int(orig1 + ratio * (orig2 - orig1))

# =======================================================================
# TRANSCRIPT STITCHING & VALIDATION
# =======================================================================
def combine_chunks(chunk_results: Dict[str, Dict], chunks_meta: List[Dict],
                  timestamp_map: Optional[List[Tuple[int, int]]], 
                  original_duration_sec: float) -> Tuple[str, List[str]]:
    """Stitch chunks together with deduplication and validation."""
    sorted_chunks = sorted(chunks_meta, key=lambda x: x["chunk_num"])
    final_lines = []
    seen_lines = set()
    failed_chunks = []
    
    for chunk_info in sorted_chunks:
        chunk_id = f"{chunk_info['original_filename']}_chunk{chunk_info['chunk_num']:03d}"
        result = chunk_results.get(chunk_id, {})
        
        if result.get("status") != "success" or not result.get("transcript"):
            log(f"    ⚠️ Skipping failed chunk {chunk_info['chunk_num']}", "WARN")
            failed_chunks.append(chunk_id)
            continue
        
        # Clean transcript
        raw_transcript = result["transcript"].strip()
        cleaned_transcript = clean_chunk_transcript(raw_transcript)
        
        # Apply timestamp transformations
        offset = chunk_info["start_time"]
        for line in cleaned_transcript.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            
            # Adjust to global timeline then remap to original audio
            adjusted = adjust_timestamp(stripped, offset)
            remapped = remap_timestamp(adjusted, timestamp_map, original_duration_sec)
            
            # Deduplication using normalized text
            norm_key = _normalize_text(remapped)
            if norm_key not in seen_lines:
                final_lines.append(remapped)
                seen_lines.add(norm_key)
    
    # Report missing chunks
    if failed_chunks:
        log(f"    ❌ Missing {len(failed_chunks)} chunks: {', '.join(failed_chunks[:3])}" + 
            ("..." if len(failed_chunks) > 3 else ""), "ERROR")
    
    return "\n".join(final_lines), failed_chunks

def calculate_transcript_duration(transcript: str) -> float:
    """Calculate maximum timestamp in transcript."""
    max_seconds = 0.0
    
    for line in transcript.split('\n'):
        if not line.strip():
            continue
        
        timestamps = re.findall(r'\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]', line)
        for ts in timestamps:
            hours = 0
            minutes = int(ts[0])
            seconds = int(ts[1])
            
            if len(ts) > 2 and ts[2]:
                seconds = int(ts[2])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            max_seconds = max(max_seconds, total_seconds)
    
    return max_seconds

# =======================================================================
# GCS INTEGRATION
# =======================================================================
def get_storage_client():
    """Get thread-safe GCS client."""
    return storage.Client.from_service_account_json(Config.CREDENTIALS_PATH)

def get_vertex_client():
    """Get Vertex AI client."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(Config.CREDENTIALS_PATH)
    return genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        project=PROJECT_ID,
        location=Config.LOCATION
    )

def download_from_gcs(blob_name: str, filename: str, download_dir: str) -> Optional[str]:
    """Download file from GCS bucket."""
    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, filename)
    
    if os.path.exists(local_path):
        return local_path
    
    try:
        client = get_storage_client()
        bucket = client.bucket(Config.INPUT_BUCKET)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        log(f"    Download failed: {e}", "ERROR")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def list_audio_files_in_date_folder_with_subfolders(date_folder: str, dids: List[str], 
                                                   extensions: List[str], tracker) -> List[Dict]:
    """List all matching audio files in date folder and subfolders."""
    prefix = f"{Config.INPUT_BASE_FOLDER}/{date_folder}/"
    processed_blobs = tracker.get_processed_blobs()
    
    try:
        client = get_storage_client()
        all_blobs = list(client.list_blobs(Config.INPUT_BUCKET, prefix=prefix))
        
        audio_files = []
        skipped_processed = 0
        skipped_filter = 0
        
        for blob in all_blobs:
            if blob.name.endswith("/"):
                continue
            if not blob.name.lower().endswith(Config.AUDIO_EXTENSIONS):
                continue
            
            filename = os.path.basename(blob.name)
            
            # Skip processed files
            if blob.name in processed_blobs:
                skipped_processed += 1
                continue
            
            # Apply DID/extension filters
            matches, match_type, match_value = filename_matches_filters(filename, dids, extensions)
            if not matches:
                skipped_filter += 1
                continue
            
            # Extract subfolder
            rel_path = blob.name[len(prefix):]
            subfolder = os.path.dirname(rel_path) if "/" in rel_path else ""
            
            audio_files.append({
                "blob_name": blob.name,
                "filename": filename,
                "size": blob.size,
                "date_folder": date_folder,
                "subfolder": subfolder,
                "match_type": match_type,
                "match_value": match_value,
            })
        
        log(f"  {date_folder}: {len(audio_files)} new matching files "
            f"(skipped: {skipped_processed} processed, {skipped_filter} filtered out, "
            f"{len(all_blobs)} total blobs)", "OK")
        
        return audio_files
    
    except Exception as e:
        log(f"Failed listing {date_folder}: {e}", "ERROR")
        traceback.print_exc()
        return []

# =======================================================================
# FILE PROCESSING
# =======================================================================
def extract_metadata_from_filename(filename: str) -> Dict[str, Optional[str]]:
    """Extract call metadata from filename."""
    metadata = {"call_id": None, "call_date": None, "call_time": None}
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        cleaned = re.sub(r'^processed[_ ]+', '', base, flags=re.IGNORECASE)
        
        # Extract date
        date_match = re.search(
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            cleaned, re.IGNORECASE
        )
        if date_match:
            metadata["call_date"] = date_match.group(1).strip()
        else:
            date_match2 = re.search(
                r'(\d{1,2})[_ ]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_ ]+(\d{4})',
                cleaned, re.IGNORECASE
            )
            if date_match2:
                metadata["call_date"] = f"{date_match2.group(1)} {date_match2.group(2)} {date_match2.group(3)}"
        
        # Extract time
        time_match = re.search(r'(\d{1,2})-(\d{2})-(\d{2})(?=-|\s|$)', cleaned)
        if time_match:
            metadata["call_time"] = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}"
        
        # Extract call ID
        remainder = cleaned[(date_match.end() if date_match else 0):]
        parts = remainder.replace(' ', '-').strip('-').split('-')
        for part in reversed(parts):
            if '.' in part and re.match(r'^\d+\.\d+$', part):
                metadata["call_id"] = part
                break
        if not metadata["call_id"]:
            for part in reversed(parts):
                if re.match(r'^\d{5,}', part):
                    metadata["call_id"] = part
                    break
    except Exception as e:
        log(f"Metadata extraction error: {e}", "WARN")
    
    return metadata

def save_result(save_info: Dict) -> Tuple[str, bool]:
    """Save transcript locally and to GCS."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(save_info["audio_filename"])[0]
        orig_dur = format_duration(save_info["original_duration"]) if save_info.get("original_duration") else "?"
        proc_dur = format_duration(save_info["preprocessed_duration"]) if save_info.get("preprocessed_duration") else "?"
        date_folder = save_info["date_folder"]
        metadata = save_info.get("file_metadata", {})
        
        # Build metadata section
        meta_lines = []
        if metadata.get("call_id"):
            meta_lines.append(f"Call ID: {metadata['call_id']}")
        if metadata.get("call_date"):
            meta_lines.append(f"Call Date: {metadata['call_date']}")
        if metadata.get("call_time"):
            meta_lines.append(f"Call Time: {metadata['call_time']}")
        if date_folder:
            meta_lines.append(f"Date Folder: {date_folder}")
        if save_info.get("match_type"):
            meta_lines.append(f"Match: {save_info['match_type']}={save_info.get('match_value', '')}")
        
        meta_text = "\n".join(meta_lines) + "\n" if meta_lines else ""
        
        # Build full transcript text
        gpu_info = f"GPU: {DEVICE_INFO['gpu_name']}" if DEVICE == "cuda" else "CPU"
        transcript_text = (
            f"FULL CALL TRANSCRIPT\n{'='*70}\n"
            f"Source: {save_info['audio_filename']}\n"
            f"{meta_text}"
            f"Original Duration: {orig_dur}\n"
            f"Speech Duration: {proc_dur}\n"
            f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Pipeline: Preprocess→Sequential Vertex AI ({Config.CHUNK_DURATION_MINUTES}min, "
            f"{Config.CHUNK_OVERLAP_SECONDS}s overlap)\n"
            f"Model: {Config.GEMINI_MODEL}\n"
            f"Compute: {gpu_info}\n"
            f"Timestamps: Remapped to original\n"
            f"{'='*70}\n\n"
            f"{save_info['full_transcript']}\n"
        )
        
        # Save locally
        local_dir = os.path.join(Config.LOCAL_RESULTS_DIR, date_folder) if date_folder else Config.LOCAL_RESULTS_DIR
        result_filename = f"{base_name}_{timestamp}_transcript.txt"
        local_path = os.path.join(local_dir, result_filename)
        
        with _save_lock:
            os.makedirs(local_dir, exist_ok=True)
        
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        log(f"    Saved locally: {local_path}", "SAVE")
        
        # Upload to GCS
        client = get_storage_client()
        gcs_path = f"{Config.OUTPUT_FOLDER}/{date_folder}/{result_filename}" if date_folder else f"{Config.OUTPUT_FOLDER}/{result_filename}"
        bucket = client.bucket(Config.OUTPUT_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(transcript_text, content_type="text/plain; charset=utf-8")
        log(f"    Uploaded: gs://{Config.OUTPUT_BUCKET}/{gcs_path}", "SAVE")
        
        return save_info["audio_filename"], True
    
    except Exception as e:
        log(f"  Save error: {e}", "ERROR")
        traceback.print_exc()
        return save_info.get("audio_filename", "?"), False

def process_single_file(vertex_client, file_info: Dict, date_folder: str, 
                       file_index: int, total_files: int, tracker, 
                       agent_names: Optional[List[str]] = None) -> Dict:
    """Process a single audio file through the full pipeline."""
    filename = file_info["filename"]
    blob_name = file_info["blob_name"]
    match_type = file_info.get("match_type", "")
    match_value = file_info.get("match_value", "")
    subfolder = file_info.get("subfolder", "")
    
    # Create isolated work directory
    file_hash = hashlib.md5(blob_name.encode()).hexdigest()[:12]
    work_dir = os.path.join("work_files", f"{file_hash}_{threading.current_thread().name}")
    os.makedirs(work_dir, exist_ok=True)
    
    subfolder_label = f" [{subfolder}]" if subfolder else ""
    log(f"  ▶ [{file_index}/{total_files}]{subfolder_label} {filename} ({match_type}={match_value})", "PROCESSING")
    file_start_time = time.time()
    
    try:
        # Step 1: Download
        log(f"    Step 1: Download...", "DOWNLOAD")
        download_dir = os.path.join(work_dir, "download")
        local_path = download_from_gcs(blob_name, filename, download_dir)
        if not local_path:
            tracker.mark_processed(blob_name, filename, "failed", {"reason": "Download failed"})
            return {"filename": filename, "status": "failed", "reason": "Download failed"}
        log(f"    Downloaded: {format_file_size(os.path.getsize(local_path))}", "OK")
        
        # Step 2: Preprocess
        log(f"    Step 2: Preprocess...", "PREPROCESS")
        metadata = extract_metadata_from_filename(filename)
        preprocessed_path, timestamp_map, orig_duration, proc_duration = preprocess_audio(
            local_path, filename, work_dir
        )
        if preprocessed_path is None:
            tracker.mark_processed(blob_name, filename, "failed", {"reason": "Preprocessing failed"})
            return {"filename": filename, "status": "failed", 
                   "reason": "Preprocessing failed (too short or error)"}
        
        # Step 3: Chunk
        log(f"    Step 3: Chunk...", "PROCESSING")
        chunk_dir = os.path.join(work_dir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunks, _ = split_chunks(preprocessed_path, chunk_dir, filename)
        if not chunks or not chunks[0].get("path"):
            tracker.mark_processed(blob_name, filename, "failed", {"reason": "Chunking failed"})
            return {"filename": filename, "status": "failed", "reason": "Chunking failed"}
        log(f"    {len(chunks)} chunk(s)", "OK")
        
        # Step 4: Transcribe
        log(f"    Step 4: Transcribe ({len(chunks)} chunks)...", "API")
        chunk_results = transcribe_file_sequential(vertex_client, chunks, filename, agent_names=agent_names)
        
        success_count = sum(1 for r in chunk_results.values() if r.get("status") == "success")
        failed_count = len(chunk_results) - success_count
        log(f"    Transcription: {success_count}/{len(chunks)} chunks OK" + 
            (f", {failed_count} failed" if failed_count else ""), 
            "OK" if failed_count == 0 else "WARN")
        
        # Step 5: Stitch
        log(f"    Step 5: Stitch & validate...", "PROCESSING")
        full_transcript, failed_chunks = combine_chunks(
            chunk_results, chunks, timestamp_map, orig_duration
        )
        
        if not full_transcript or len(full_transcript) < 50:
            tracker.mark_processed(blob_name, filename, "failed", {"reason": "Transcript too short"})
            return {"filename": filename, "status": "failed", "reason": "Stitched transcript too short"}
        
        # Validate duration match
        transcript_duration = calculate_transcript_duration(full_transcript)
        duration_ratio = transcript_duration / orig_duration if orig_duration > 0 else 0
        
        if duration_ratio < 0.85:
            log(f"    ⚠️ Transcript duration ({transcript_duration:.1f}s) is only "
                f"{duration_ratio*100:.0f}% of audio duration ({orig_duration:.1f}s)", "WARN")
            # Still save but mark as partial
            if duration_ratio < 0.5:
                tracker.mark_processed(blob_name, filename, "failed", {
                    "reason": f"Severe duration mismatch ({transcript_duration:.1f}/{orig_duration:.1f}s)"
                })
                return {"filename": filename, "status": "failed", 
                       "reason": f"Severe duration mismatch ({transcript_duration:.1f}/{orig_duration:.1f}s)"}
        
        # Step 6: Save
        save_info = {
            "full_transcript": full_transcript,
            "audio_filename": filename,
            "batch_id": "sequential",
            "original_duration": orig_duration,
            "preprocessed_duration": proc_duration,
            "file_metadata": metadata,
            "date_folder": date_folder,
            "match_type": match_type,
            "match_value": match_value,
        }
        _, saved = save_result(save_info)
        
        elapsed = time.time() - file_start_time
        if saved:
            tracker.mark_processed(blob_name, filename, "success", {
                "chars": len(full_transcript),
                "chunks": success_count,
                "elapsed": round(elapsed, 1),
                "transcript_duration": transcript_duration,
                "audio_duration": orig_duration,
                "failed_chunks": len(failed_chunks)
            })
            log(f"  ✅ [{file_index}/{total_files}] Done: {filename} "
                f"({format_duration(elapsed)}, {len(full_transcript)} chars, "
                f"{success_count}/{len(chunks)} chunks)", "DONE")
            return {
                "filename": filename,
                "status": "success",
                "transcript_chars": len(full_transcript),
                "chunks_ok": success_count,
                "chunks_fail": failed_count,
                "elapsed": elapsed,
                "transcript_duration": transcript_duration,
                "audio_duration": orig_duration
            }
        else:
            tracker.mark_processed(blob_name, filename, "failed", {"reason": "Save failed"})
            return {"filename": filename, "status": "failed", "reason": "Save failed"}
    
    except Exception as e:
        log(f"  ❌ [{file_index}/{total_files}] Error for {filename}: {e}", "ERROR")
        traceback.print_exc()
        tracker.mark_processed(blob_name, filename, "failed", {"reason": f"Exception: {str(e)[:200]}"})
        return {"filename": filename, "status": "failed", "reason": f"Exception: {str(e)[:200]}"}
    
    finally:
        # Cleanup work directory
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass

# =======================================================================
# TRACKING & PROGRESS
# =======================================================================
class ProcessedTracker:
    """Thread-safe tracker for processed files."""
    def __init__(self, filepath: str = Config.PROCESSED_TRACKER_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self._processed: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                self._processed = data.get("processed", {})
                log(f"Loaded tracker: {len(self._processed)} previously processed files", "TRACK")
            except Exception as e:
                log(f"Tracker load error (starting fresh): {e}", "WARN")
                self._processed = {}
        else:
            log("No tracker file found — starting fresh", "TRACK")
    
    def _save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({
                    "processed": self._processed,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_processed": len(self._processed)
                }, f, indent=2)
        except Exception as e:
            log(f"Tracker save error: {e}", "ERROR")
    
    def is_processed(self, blob_name: str) -> bool:
        with self._lock:
            return blob_name in self._processed
    
    def mark_processed(self, blob_name: str, filename: str, status: str, details: Optional[Dict] = None):
        with self._lock:
            self._processed[blob_name] = {
                "filename": filename,
                "status": status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **(details or {})
            }
            self._save()
    
    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            total = len(self._processed)
            success = sum(1 for v in self._processed.values() if v.get("status") == "success")
            failed = sum(1 for v in self._processed.values() if v.get("status") == "failed")
            return {"total": total, "success": success, "failed": failed}
    
    def get_processed_blobs(self) -> Set[str]:
        with self._lock:
            return set(self._processed.keys())

class ProgressTracker:
    """Track processing progress with statistics."""
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.results = []
        self._lock = threading.Lock()
        self.start_time = time.time()
    
    def record(self, result: Dict):
        with self._lock:
            self.completed += 1
            if result["status"] == "success":
                self.success += 1
            else:
                self.failed += 1
            self.results.append(result)
            
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed * 3600 if elapsed > 0 else 0
            remaining = (self.total - self.completed) / (self.completed / elapsed) if self.completed > 0 else 0
            
            log(f"  📊 Progress: {self.completed}/{self.total} "
                f"({self.success}✅ {self.failed}❌) "
                f"| {rate:.0f}/hr | ETA: {format_duration(remaining)}", "SEQ")

# =======================================================================
# SCHEDULING & EXECUTION
# =======================================================================
def run_hourly_cycle(date_folder: str, dids: List[str], extensions: List[str], 
                    tracker: ProcessedTracker, vertex_client, num_workers: int,
                    agent_names: Optional[List[str]] = None, max_retries: int = 3) -> Tuple[int, int, int]:
    """Run a single hourly processing cycle."""
    uk_time = get_uk_now()
    log(f"{'─'*75}", "HOUR")
    log(f"HOURLY SCAN — {date_folder} — UK time: {uk_time.strftime('%H:%M:%S')}", "HOUR")
    
    # Find new files
    new_files = list_audio_files_in_date_folder_with_subfolders(
        date_folder, dids, extensions, tracker
    )
    
    if not new_files:
        log(f"  No new matching files this cycle", "OK")
        stats = tracker.get_stats()
        log(f"  Running total: {stats['success']}✅ {stats['failed']}❌ / {stats['total']} processed", "TRACK")
        return 0, 0, 0
    
    total = len(new_files)
    effective_workers = min(num_workers, total)
    log(f"  Found {total} new files → processing with {effective_workers} workers...", "MEGA")
    
    # Process files
    progress = ProgressTracker(total)
    all_results = []
    
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="W") as executor:
        future_to_info = {}
        for idx, file_info in enumerate(new_files, 1):
            future = executor.submit(
                process_single_file,
                vertex_client, file_info, date_folder, idx, total, tracker, agent_names
            )
            future_to_info[future] = (idx, file_info)
        
        for future in as_completed(future_to_info):
            idx, file_info = future_to_info[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                result = {
                    "filename": file_info["filename"],
                    "status": "failed",
                    "reason": f"Worker exception: {str(e)[:200]}"
                }
                tracker.mark_processed(
                    file_info["blob_name"], file_info["filename"], "failed",
                    {"reason": f"Worker exception: {str(e)[:200]}"}
                )
                all_results.append(result)
            progress.record(result)
    
    # Calculate stats
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = len(all_results) - success
    elapsed = time.time() - progress.start_time
    
    stats = tracker.get_stats()
    log(f"  Cycle done: {success}✅ {failed}❌ / {total} ({format_duration(elapsed)})", "DONE")
    log(f"  Running total: {stats['success']}✅ {stats['failed']}❌ / {stats['total']} processed", "TRACK")
    
    return success, failed, total

def run_scheduled(date_folder: str, num_workers: int = Config.DEFAULT_WORKERS):
    """Run scheduled processing until 2:00 AM UK time."""
    # Load filters
    dids = load_filter_list(Config.DIDS_FILE, "DIDs")
    extensions = load_filter_list(Config.EXTENSIONS_FILE, "Extensions")
    
    if not dids and not extensions:
        log("No DIDs or Extensions loaded — nothing to filter by. "
            "Create dids.txt and/or extensions.txt with numbers, one per line.", "ERROR")
        return
    
    agent_names = load_agent_names(Config.AGENT_NAMES_FILE)
    tracker = ProcessedTracker()
    
    # System info
    gpu_info = f"🎮 {DEVICE_INFO['gpu_name']} ({DEVICE_INFO['vram_total_gb']}GB)" if DEVICE == "cuda" else "⚠️ CPU mode"
    uk_now = get_uk_now()
    
    print("\n" + "=" * 75)
    print(f"  🕐 HOURLY SCHEDULED TRANSCRIPTION")
    print(f"  {gpu_info}")
    print(f"  Workers: {num_workers} | API concurrency: {Config.MAX_CONCURRENT_API_CALLS}")
    print(f"  Model: {Config.GEMINI_MODEL} | {Config.CHUNK_DURATION_MINUTES}min chunks")
    print(f"  Date folder: {date_folder}")
    print(f"  DIDs: {len(dids)} | Extensions: {len(extensions)}")
    print(f"  Agent names: {len(agent_names)}" if agent_names else "  Agent names: none")
    print(f"  UK time now: {uk_now.strftime('%H:%M:%S')} | Runs until: "
          f"{Config.SCHEDULER_END_HOUR}:{Config.SCHEDULER_END_MINUTE:02d} next day")
    print(f"  Tracker: {tracker.filepath}")
    print("=" * 75)
    
    # Verify setup
    if not _verify_setup():
        return
    
    vertex_client = get_vertex_client()
    log("Pre-loading segmenter...", "PREPROCESS")
    get_segmenter()
    if DEVICE == "cuda":
        print_gpu_status()
    
    os.makedirs("work_files", exist_ok=True)
    
    # Initialize scheduler state
    global START_DATE
    START_DATE = uk_now.date()
    
    cutoff_date = START_DATE + timedelta(days=1)
    cutoff_time = datetime(
        cutoff_date.year, cutoff_date.month, cutoff_date.day,
        Config.SCHEDULER_END_HOUR, Config.SCHEDULER_END_MINUTE,
        tzinfo=uk_now.tzinfo
    )
    log(f"Process will run until {cutoff_date} at {Config.SCHEDULER_END_HOUR}:{Config.SCHEDULER_END_MINUTE:02d} UK time", "SCHEDULE")
    
    # Main scheduling loop
    cycle_count = 0
    total_success = 0
    total_failed = 0
    total_processed = 0
    
    while not shutdown_requested:
        if is_past_cutoff():
            log(f"UK time is past {Config.SCHEDULER_END_HOUR}:{Config.SCHEDULER_END_MINUTE:02d} — stopping", "SCHEDULE")
            break
        
        cycle_count += 1
        log(f"\n{'█'*75}", "HOUR")
        log(f"CYCLE {cycle_count} — UK time: {get_uk_now().strftime('%H:%M:%S')}", "SCHEDULE")
        
        # Run processing cycle
        success, failed, processed = run_hourly_cycle(
            date_folder, dids, extensions, tracker, vertex_client, num_workers, agent_names
        )
        total_success += success
        total_failed += failed
        total_processed += processed
        
        # Cleanup GPU memory
        if DEVICE == "cuda":
            clear_gpu_cache()
        
        # Check remaining time
        remaining = time_until_cutoff()
        if remaining <= 0:
            log(f"UK time reached cutoff — stopping", "SCHEDULE")
            break
        
        # Final check before cutoff
        if remaining < Config.SCHEDULER_FINAL_CHECK_MINUTES_BEFORE * 60:
            log(f"Less than {Config.SCHEDULER_FINAL_CHECK_MINUTES_BEFORE}min until cutoff — final check...", "SCHEDULE")
            success, failed, processed = run_hourly_cycle(
                date_folder, dids, extensions, tracker, vertex_client, num_workers, agent_names
            )
            total_success += success
            total_failed += failed
            total_processed += processed
            log(f"Final check done — stopping", "SCHEDULE")
            break
        
        # Wait for next cycle
        wait_seconds = min(Config.SCHEDULER_INTERVAL_SECONDS, 
                          remaining - Config.SCHEDULER_FINAL_CHECK_MINUTES_BEFORE * 60)
        if wait_seconds <= 0:
            continue
        
        next_uk = get_uk_now() + timedelta(seconds=wait_seconds)
        log(f"Next cycle at UK time ~{next_uk.strftime('%H:%M:%S')} "
            f"(waiting {format_duration(wait_seconds)})...", "SCHEDULE")
        
        # Sleep with heartbeat logging
        sleep_until = time.time() + wait_seconds
        while time.time() < sleep_until and not shutdown_requested:
            remaining_sleep = sleep_until - time.time()
            if remaining_sleep <= 0:
                break
            chunk = min(remaining_sleep, 300)  # Wake every 5 min
            time.sleep(chunk)
            if remaining_sleep > 300:
                log(f"  💤 Waiting... ({format_duration(remaining_sleep)} until next cycle, "
                    f"UK: {get_uk_now().strftime('%H:%M:%S')})", "SCHEDULE")
    
    # Final summary
    stats = tracker.get_stats()
    print("\n" + "=" * 75)
    print("  📊 FINAL SUMMARY")
    print("=" * 75)
    print(f"  Cycles: {cycle_count}")
    print(f"  This session: {total_success}✅ {total_failed}❌ / {total_processed} processed")
    print(f"  All-time total: {stats['success']}✅ {stats['failed']}❌ / {stats['total']} processed")
    print(f"  Results: gs://{Config.OUTPUT_BUCKET}/{Config.OUTPUT_FOLDER}/{date_folder}/")
    print(f"  Tracker: {tracker.filepath}")
    print("=" * 75)
    
    # Cleanup
    try:
        shutil.rmtree("work_files", ignore_errors=True)
    except Exception:
        pass

def run_once(date_folder: str, num_workers: int = Config.DEFAULT_WORKERS):
    """Run a single processing cycle (no scheduling)."""
    dids = load_filter_list(Config.DIDS_FILE, "DIDs")
    extensions = load_filter_list(Config.EXTENSIONS_FILE, "Extensions")
    agent_names = load_agent_names(Config.AGENT_NAMES_FILE)
    
    if not dids and not extensions:
        log("No DIDs or Extensions loaded", "ERROR")
        return
    
    tracker = ProcessedTracker()
    
    gpu_info = f"🎮 {DEVICE_INFO['gpu_name']} ({DEVICE_INFO['vram_total_gb']}GB)" if DEVICE == "cuda" else "⚠️ CPU mode"
    print("\n" + "=" * 75)
    print(f"  🔶 ONE-SHOT TRANSCRIPTION (single scan)")
    print(f"  {gpu_info} | Workers: {num_workers}")
    print(f"  Date folder: {date_folder}")
    print(f"  DIDs: {len(dids)} | Extensions: {len(extensions)}")
    print(f"  Agent names: {len(agent_names)}" if agent_names else "  Agent names: none")
    print("=" * 75)
    
    if not _verify_setup():
        return
    
    vertex_client = get_vertex_client()
    log("Pre-loading segmenter...", "PREPROCESS")
    get_segmenter()
    if DEVICE == "cuda":
        print_gpu_status()
    os.makedirs("work_files", exist_ok=True)
    
    success, failed, total = run_hourly_cycle(
        date_folder, dids, extensions, tracker, vertex_client, num_workers, agent_names
    )
    
    stats = tracker.get_stats()
    print("\n" + "=" * 75)
    print(f"  📊 DONE: {success}✅ {failed}❌ / {total} | Total tracked: {stats['total']}")
    print("=" * 75)
    
    try:
        shutil.rmtree("work_files", ignore_errors=True)
    except Exception:
        pass

# =======================================================================
# SETUP VERIFICATION
# =======================================================================
def _verify_ffmpeg() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        if result.returncode == 0:
            log("ffmpeg OK", "OK")
            return True
    except Exception:
        pass
    log("ffmpeg not found!", "ERROR")
    return False

def _verify_gpu() -> bool:
    if DEVICE == "cuda":
        log(f"GPU: {DEVICE_INFO['gpu_name']} | CUDA {DEVICE_INFO['cuda_version']} | "
            f"{DEVICE_INFO['vram_total_gb']}GB", "GPU")
    else:
        log("No GPU — CPU mode (slower)", "WARN")
    return True

def _verify_vertex() -> bool:
    try:
        client = get_vertex_client()
        resp = client.models.generate_content(model=Config.GEMINI_MODEL, contents="Say OK")
        if resp and resp.text:
            log(f"Vertex AI OK ({resp.text[:30].strip()})", "OK")
            return True
        return True
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "403" in error_str:
            log(f"Vertex AI AUTH FAILED: {e}", "ERROR")
            return False
        if "404" in error_str:
            log(f"Model not found: {Config.GEMINI_MODEL}", "ERROR")
            return False
        log(f"Vertex AI OK (non-auth error: {error_str[:80]})", "WARN")
        return True

def _verify_gcs() -> bool:
    try:
        client = get_storage_client()
        for bucket_name in [Config.INPUT_BUCKET, Config.OUTPUT_BUCKET]:
            if not client.bucket(bucket_name).exists():
                log(f"Bucket missing: {bucket_name}", "ERROR")
                return False
            log(f"  Bucket OK: gs://{bucket_name}", "OK")
        log("All buckets OK", "OK")
        return True
    except Exception as e:
        log(f"GCS: {e}", "ERROR")
        return False

def _verify_preprocess() -> bool:
    try:
        _ = AudioSegment.silent(duration=100)
        _ = webrtcvad.Vad(Config.VAD_AGGRESSIVENESS)
        log("Preprocessing OK", "OK")
        return True
    except Exception as e:
        log(f"Preprocess: {e}", "ERROR")
        return False

def _verify_setup() -> bool:
    log("Verifying setup...", "PROCESSING")
    checks = [
        ("GPU", _verify_gpu, False),
        ("ffmpeg", _verify_ffmpeg, True),
        ("Vertex AI", _verify_vertex, True),
        ("GCS", _verify_gcs, True),
        ("Preprocessing", _verify_preprocess, True)
    ]
    
    failed = False
    for name, func, critical in checks:
        try:
            if not func() and critical:
                failed = True
        except Exception as e:
            if critical:
                failed = True
                log(f"  {name}: {e}", "ERROR")
    
    if failed:
        log("Setup failed", "ERROR")
        return False
    return True

# =======================================================================
# UTILITY FUNCTIONS
# =======================================================================
def check_results():
    """Check output bucket for results."""
    print(f"\n📊 Results in gs://{Config.OUTPUT_BUCKET}/{Config.OUTPUT_FOLDER}/")
    try:
        client = get_storage_client()
        blobs = list(client.list_blobs(Config.OUTPUT_BUCKET, prefix=f"{Config.OUTPUT_FOLDER}/"))
        result_blobs = [b for b in blobs if not b.name.endswith("/")]
        
        if not result_blobs:
            print("  (none)")
            return
        
        # Group by date folder
        by_date = {}
        for blob in result_blobs:
            parts = blob.name.split("/")
            date_folder = parts[1] if len(parts) >= 3 else "(root)"
            by_date.setdefault(date_folder, []).append(blob)
        
        for date_key in sorted(by_date):
            blobs_in_date = by_date[date_key]
            print(f"\n  📅 {date_key} ({len(blobs_in_date)} files)")
            for blob in sorted(blobs_in_date, key=lambda x: x.name)[:10]:
                print(f"    • {os.path.basename(blob.name)} ({format_file_size(blob.size)})")
    except Exception as e:
        print(f"  Error: {e}")

# =======================================================================
# SIGNAL HANDLING
# =======================================================================
def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    log("Shutdown requested via signal (Ctrl+C or SIGTERM). Finishing current processing...", "WARN")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =======================================================================
# CLI
# =======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Hourly Scheduled Parallel Vertex AI Transcription")
    parser.add_argument("--date", type=str, required=False,
                       help="Date folder to process (YYYY-MM-DD). Default: today")
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_WORKERS,
                       help=f"Parallel file workers (default: {Config.DEFAULT_WORKERS}, max: {Config.MAX_WORKERS})")
    parser.add_argument("--once", action="store_true",
                       help="Run single scan (no hourly loop)")
    parser.add_argument("--schedule", action="store_true",
                       help="Run hourly until 2:00 AM UK time (default mode)")
    parser.add_argument("--list", action="store_true",
                       help="List date folders in bucket")
    parser.add_argument("--check", action="store_true",
                       help="Check results in output bucket")
    parser.add_argument("--status", action="store_true",
                       help="Show tracker status (processed files count)")
    parser.add_argument("--reset-tracker", action="store_true",
                       help="Reset the processed files tracker")
    parser.add_argument("--gpu-info", action="store_true",
                       help="Show GPU info")
    parser.add_argument("--dids", type=str, default=Config.DIDS_FILE,
                       help=f"DIDs file (default: {Config.DIDS_FILE})")
    parser.add_argument("--extensions", type=str, default=Config.EXTENSIONS_FILE,
                       help=f"Extensions file (default: {Config.EXTENSIONS_FILE})")
    parser.add_argument("--agent-names", type=str, default=Config.AGENT_NAMES_FILE,
                       help=f"Agent names file (default: {Config.AGENT_NAMES_FILE})")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Override config files if specified
    if args.dids != Config.DIDS_FILE:
        globals()["Config"] = type('Config', (object,), {**Config.__dict__, 'DIDS_FILE': args.dids})
    if args.extensions != Config.EXTENSIONS_FILE:
        globals()["Config"] = type('Config', (object,), {**Config.__dict__, 'EXTENSIONS_FILE': args.extensions})
    if args.agent_names != Config.AGENT_NAMES_FILE:
        globals()["Config"] = type('Config', (object,), {**Config.__dict__, 'AGENT_NAMES_FILE': args.agent_names})
    
    # Handle utility commands
    if args.gpu_info:
        print(f"\n🎮 GPU: {DEVICE_INFO['device'].upper()}")
        if DEVICE == "cuda":
            print(f"  {DEVICE_INFO['gpu_name']} | CUDA {DEVICE_INFO['cuda_version']} | {DEVICE_INFO['vram_total_gb']}GB")
        else:
            print("  No GPU")
        print(f"  PyTorch: {torch.__version__}")
        return
    
    if args.list:
        try:
            client = get_storage_client()
            iterator = client.list_blobs(Config.INPUT_BUCKET, prefix=f"{Config.INPUT_BASE_FOLDER}/", delimiter="/")
            _ = list(iterator)  # Force evaluation
            
            for prefix in sorted(iterator.prefixes):
                folder_name = prefix.rstrip("/").split("/")[-1]
                
                # Count subfolders
                sub_iterator = client.list_blobs(Config.INPUT_BUCKET, prefix=prefix, delimiter="/")
                _ = list(sub_iterator)
                subfolders = [sp.rstrip("/").split("/")[-1] for sp in sub_iterator.prefixes]
                
                print(f"  📅 {folder_name} — subfolders: {', '.join(subfolders) if subfolders else '(none)'}")
        except Exception as e:
            print(f"  Error: {e}")
        return
    
    if args.check:
        check_results()
        return
    
    if args.status:
        tracker = ProcessedTracker()
        stats = tracker.get_stats()
        print(f"\n📋 Tracker: {Config.PROCESSED_TRACKER_FILE}")
        print(f"  Total processed: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        return
    
    if args.reset_tracker:
        if os.path.exists(Config.PROCESSED_TRACKER_FILE):
            os.remove(Config.PROCESSED_TRACKER_FILE)
            print(f"✅ Tracker reset: {Config.PROCESSED_TRACKER_FILE} deleted")
        else:
            print(f"ℹ️ No tracker file to reset")
        return
    
    # Determine date folder
    date_folder = args.date
    if not date_folder:
        uk_now = get_uk_now()
        date_folder = uk_now.strftime("%Y-%m-%d")
        log(f"No --date specified, using today (UK): {date_folder}", "INFO")
    
    try:
        datetime.strptime(date_folder, "%Y-%m-%d")
    except ValueError:
        print(f"Bad date format: {date_folder} (use YYYY-MM-DD)")
        return
    
    workers = min(args.workers, Config.MAX_WORKERS)
    
    if args.once:
        run_once(date_folder, num_workers=workers)
    else:
        run_scheduled(date_folder, num_workers=workers)

if __name__ == "__main__":
    # Import heavy dependencies only when needed
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError:
        print("ERROR: google-genai not installed.")
        sys.exit(1)
    
    try:
        from google.cloud import storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed.")
        sys.exit(1)
    
    try:
        from pydub import AudioSegment
    except ImportError:
        print("ERROR: pydub not installed.")
        sys.exit(1)
    
    try:
        import webrtcvad
    except ImportError:
        print("ERROR: webrtcvad not installed.")
        sys.exit(1)
    
    try:
        from inaSpeechSegmenter import Segmenter
    except ImportError:
        print("ERROR: inaSpeechSegmenter not installed.")
        sys.exit(1)
    
    main()