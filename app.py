"""
app.py — Advanced yt-dlp FastAPI server

Endpoints:
  GET  /                          — health check
  GET  /info?url=                 — full video metadata + all formats
  GET  /formats?url=              — formats only (lightweight)
  GET  /transcript?url=&lang=     — transcript/subtitles as structured JSON
  POST /download                  — download video/audio, returns the file
  GET  /download/file?path=       — serve a previously downloaded file

Design:
  - All yt-dlp work runs in a ThreadPoolExecutor (yt-dlp is sync/blocking)
    so FastAPI's async event loop is never blocked
  - Downloads are saved to ./downloads/, served back via FileResponse
  - Transcripts are extracted in-memory (no file written to disk)
  - Format classification mirrors what you'd expect: combined, video-only, audio-only
"""

import os
import re
import asyncio
import tempfile
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Setup ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="yt-dlp server", version="2.0.0")
# Set the path to your cookies file
COOKIES_FILE = Path(os.environ.get("COOKIES_FILE", "./cookies.txt"))
DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# yt-dlp is CPU/IO-bound and synchronous — run it in a thread pool
executor = ThreadPoolExecutor(max_workers=4)


# ── Models ─────────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None        # specific format_id from /formats
    quality: Optional[str] = "best"        # best | 1080p | 720p | 480p | 360p | audio
    ext: Optional[str] = "mp4"             # mp4 | webm | m4a | mp3

class QuickDownloadRequest(BaseModel):
    url: str
    # No format constraints — yt-dlp decides everything.
    # Optional: prefer_audio=True to favour audio-only output
    # prefer_audio: Optional[bool] = False
    # audio_format: Optional[str] = "mp3"  # mp3 | m4a | best (only used when prefer_audio=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "noplaylist": True,
    "socket_timeout": 15,
    "cookiefile": str(COOKIES_FILE),
    "extractor_args": {
        "youtube": {
            "player_client": ["web_safari"],  # Forces pre-merged formats
        }
    },
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    },
}


def classify_format(f: dict) -> dict | None:
    """
    Converts a raw yt-dlp format dict into a clean, classified object.
    Returns None for storyboard/thumbnail formats that aren't playable.
    """
    vcodec = f.get("vcodec", "none")
    acodec = f.get("acodec", "none")
    has_video = vcodec not in (None, "none")
    has_audio = acodec not in (None, "none")

    if not has_video and not has_audio:
        return None

    if has_video and has_audio:
        kind = "video+audio"
    elif has_video:
        kind = "video-only"
    else:
        kind = "audio-only"

    filesize = f.get("filesize") or f.get("filesize_approx")

    return {
        "format_id": f.get("format_id"),
        "ext": f.get("ext"),
        "type": kind,
        "resolution": f.get("resolution") or (
            f"{f['height']}p" if f.get("height") else "audio only"
        ),
        "width": f.get("width"),
        "height": f.get("height"),
        "fps": f.get("fps"),
        "vcodec": vcodec if has_video else None,
        "acodec": acodec if has_audio else None,
        "tbr": f.get("tbr"),                    # total bitrate kbps
        "vbr": f.get("vbr"),                    # video bitrate kbps
        "abr": f.get("abr"),                    # audio bitrate kbps
        "asr": f.get("asr"),                    # audio sample rate hz
        "filesize": filesize,
        "filesize_human": _human_bytes(filesize),
        "format_note": f.get("format_note"),
        "protocol": f.get("protocol"),
        "dynamic_range": f.get("dynamic_range"),  # SDR, HDR10, HLG etc.
    }


def _human_bytes(b: int | None) -> str:
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def _extract_video_id(url: str) -> str | None:
    """Extracts YouTube video ID from any known URL format."""
    patterns = [
        r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _resolve_format_selector(
    format_id: str | None,
    quality: str | None,
    ext: str | None,
) -> str:
    """
    Build a yt-dlp format selector string with deep fallback chains.

    The key insight: YouTube increasingly serves video-only streams in vp9/webm
    and may not have mp4 available at every resolution. Constraining to [ext=mp4]
    causes hard failures. Instead we:
      1. Try to get the preferred codec/container combo
      2. Fall back to any codec at that resolution
      3. Fall back to best combined format at any resolution
      4. As last resort, let yt-dlp pick freely ("b")

    This ensures the download ALWAYS works even when YouTube is serving limited
    format sets (SABR enforcement, datacenter IP restrictions, etc.).

    Selector syntax reminder:
      /   = fallback: try left, if not available try right
      *   = "if no audio, merge with best audio" wildcard
      bv  = bestvideo (alias)
      ba  = bestaudio (alias)
      b   = best combined (single file, no merge needed)
    """

    # Explicit format_id takes top priority — caller knows exactly what they want.
    # Still add "+bestaudio/format_id" fallback in case it's video-only.
    if format_id:
        return f"{format_id}+bestaudio/{format_id}"

    q = (quality or "best").lower()
    e = (ext or "mp4").lower()

    # Audio-only path — no video needed, just best audio + optional post-process
    if q == "audio":
        # bestaudio in preferred container, fallback to any audio, fallback to best
        if e == "mp3":
            # mp3 always comes from post-processing bestaudio — selector is simple
            return "bestaudio/best"
        elif e == "m4a":
            return "bestaudio[ext=m4a]/bestaudio/best"
        else:
            return "bestaudio/best"

    # Video path — build height-constrained selectors with generous fallbacks.
    # Pattern:
    #   bv*[height<=H]+ba/bv[height<=H]+ba/b[height<=H]/b
    #
    # bv*  = bestvideo that already has audio, or best video-only (merge if needed)
    # /b   = last resort: best single-file combined stream yt-dlp can find

    height_map = {
        "1080p": 1080,
        "720p": 720,
        "480p": 480,
        "360p": 360,
    }

    if q in height_map:
        h = height_map[q]
        # Tier 1: best video (any codec) at or below height + best audio → merged
        # Tier 2: best combined stream at or below height (no merge)
        # Tier 3: unconstrained best (ignores height preference but never fails)
        return (
            f"bestvideo[height<={h}]+bestaudio"
            f"/bestvideo[height<={h}]+bestaudio[ext=m4a]"
            f"/best[height<={h}]"
            f"/bestvideo+bestaudio"
            f"/best"
        )

    # "best" quality — no height constraint, just get the best available
    # Prefer h264+aac in mp4 container for maximum compatibility, but don't
    # hard-require it — fall back all the way to "b".
    if e in ("mp4", "m4a"):
        return (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4]+bestaudio"
            "/bestvideo+bestaudio"
            "/best"
        )
    elif e == "webm":
        return (
            "bestvideo[ext=webm]+bestaudio[ext=webm]"
            "/bestvideo[ext=webm]+bestaudio"
            "/bestvideo+bestaudio"
            "/best"
        )
    else:
        # mkv or any other container: no ext constraint, rely on merge_output_format
        return "bestvideo+bestaudio/best"


# ── Sync worker functions (run in thread pool) ─────────────────────────────────

def _fetch_info(url: str) -> dict:
    opts = {
        **BASE_OPTS,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

    formats = [classify_format(f) for f in raw.get("formats", [])]
    formats = [f for f in formats if f is not None]

    thumbnails = sorted(
        raw.get("thumbnails") or [],
        key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
        reverse=True,
    )

    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "description": (raw.get("description") or "")[:500] or None,
        "uploader": raw.get("uploader"),
        "uploader_url": raw.get("uploader_url"),
        "channel_id": raw.get("channel_id"),
        "upload_date": raw.get("upload_date"),           # YYYYMMDD string
        "timestamp": raw.get("timestamp"),               # unix epoch
        "duration": raw.get("duration"),                 # seconds
        "duration_string": raw.get("duration_string"),
        "view_count": raw.get("view_count"),
        "like_count": raw.get("like_count"),
        "comment_count": raw.get("comment_count"),
        "age_limit": raw.get("age_limit"),
        "categories": raw.get("categories"),
        "tags": (raw.get("tags") or [])[:20],            # cap at 20 tags
        "is_live": raw.get("is_live"),
        "was_live": raw.get("was_live"),
        "chapters": raw.get("chapters"),
        "thumbnail": raw.get("thumbnail"),
        "thumbnails": thumbnails[:5],                    # top 5 by resolution
        "webpage_url": raw.get("webpage_url"),
        "playability_status": raw.get("availability"),
        "has_subtitles": bool(raw.get("subtitles")),
        "has_auto_captions": bool(raw.get("automatic_captions")),
        "subtitle_languages": list((raw.get("subtitles") or {}).keys()),
        "auto_caption_languages": list(
            (raw.get("automatic_captions") or {}).keys()
        )[:10],
        "format_count": len(formats),
        "formats": formats,
        "formats_grouped": {
            "combined": [f for f in formats if f["type"] == "video+audio"],
            "video_only": [f for f in formats if f["type"] == "video-only"],
            "audio_only": [f for f in formats if f["type"] == "audio-only"],
        },
    }


def _fetch_transcript(url: str, lang: str) -> dict:
    """
    Extracts subtitle/transcript data in-memory using yt-dlp.
    Returns structured segments with start, duration, text.
    Falls back to auto-captions if manual subtitles are unavailable.
    """
    opts = {
        **BASE_OPTS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",     # json3 gives structured start/duration/text
    }

    transcript_data = {}

    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

        # Check manual subtitles first, then fall back to auto-captions
        subtitles = raw.get("subtitles") or {}
        auto_captions = raw.get("automatic_captions") or {}

        source = None
        source_type = None

        if lang in subtitles:
            source = subtitles[lang]
            source_type = "manual"
        elif lang in auto_captions:
            source = auto_captions[lang]
            source_type = "auto"
        else:
            # Try finding any English variant (en-US, en-GB etc.)
            for key in list(subtitles.keys()) + list(auto_captions.keys()):
                if key.startswith(lang):
                    source = subtitles.get(key) or auto_captions.get(key)
                    source_type = "manual" if key in subtitles else "auto"
                    lang = key
                    break

        if not source:
            available = list(subtitles.keys()) + list(auto_captions.keys())
            raise ValueError(
                f"No transcript for lang '{lang}'. "
                f"Available: {available or 'none'}"
            )

        # Find the json3 format entry (has structured data)
        json3_entry = next(
            (s for s in source if s.get("ext") == "json3"), None
        )

        if not json3_entry or not json3_entry.get("url"):
            raise ValueError("Transcript URL not found in yt-dlp response")

        # Fetch the actual json3 subtitle file
        with urllib.request.urlopen(json3_entry["url"], timeout=10) as resp:
            raw_json = json.loads(resp.read().decode("utf-8"))

        # Parse json3 format: events -> segs -> utf8
        segments = []
        for event in raw_json.get("events", []):
            if "segs" not in event:
                continue
            start_ms = event.get("tStartMs", 0)
            duration_ms = event.get("dDurationMs", 0)
            text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
            if text and text != "\n":
                segments.append({
                    "start": round(start_ms / 1000, 2),
                    "duration": round(duration_ms / 1000, 2),
                    "text": text,
                })

        # Also build a plain text version
        full_text = " ".join(s["text"] for s in segments)

        transcript_data = {
            "video_id": raw.get("id"),
            "title": raw.get("title"),
            "language": lang,
            "type": source_type,
            "segment_count": len(segments),
            "full_text": full_text,
            "segments": segments,
        }

    return transcript_data


def _run_download(url: str, format_id: str | None, quality: str, ext: str) -> dict:
    """
    Downloads with the robust fallback selector from _resolve_format_selector.
    merge_output_format is always set so FFmpeg re-muxes when needed.
    For mp3/audio requests, FFmpegExtractAudio post-processor is added.
    """
    selector = _resolve_format_selector(format_id, quality, ext)

    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s")

    final_path: dict[str, str | None] = {"value": None}

    def progress_hook(d: dict):
        if d["status"] == "finished":
            final_path["value"] = d.get("filename") or d.get("info_dict", {}).get("filepath")

    is_audio_extract = ext == "mp3" or quality == "audio"
    preferred_codec = ext if ext in ("mp3", "m4a") else "mp3"

    merge_fmt = ext if ext in ("mp4", "mkv", "webm") else "mp4"

    opts: dict = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "merge_output_format": merge_fmt,
    }

    if is_audio_extract:
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": preferred_codec,
            "preferredquality": "192",
        }]

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)

    if not final_path["value"]:
        matches = list(DOWNLOADS_DIR.glob(f"*{uid}*"))
        matches = [f for f in matches if f.suffix not in (".part", ".ytdl", ".tmp")]
        if matches:
            final_path["value"] = str(matches[0])

    if not final_path["value"] or not Path(final_path["value"]).exists():
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = Path(final_path["value"])
    return {
        "title": info.get("title"),
        "file_path": str(file_path),
        "filename": file_path.name,
        "ext": file_path.suffix.lstrip("."),
        "filesize": file_path.stat().st_size,
        "filesize_human": _human_bytes(file_path.stat().st_size),
        "format_selector": selector,
        "format_id_used": info.get("format_id"),
        "resolution": info.get("resolution") or info.get("format_note"),
    }

def _run_quick_download(url: str, prefer_audio: bool, audio_format: str) -> dict:
    """
    Zero-config download — yt-dlp chooses the best format it can get.

    Uses format="b" (best available single file) for video, or
    "bestaudio/best" for audio-only. No ext/codec constraints.
    This is the most reliable path when the standard selectors fail
    due to YouTube SABR enforcement or IP-based format restrictions.

    The output container may be webm/mkv if that's what yt-dlp picks —
    that's intentional. The goal is "something that plays", not "mp4 or bust".
    """
    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(title)s [quick-{uid}].%(ext)s")
    final_path: dict[str, str | None] = {"value": None}

    def progress_hook(d: dict):
        if d["status"] == "finished":
            final_path["value"] = d.get("filename") or d.get("info_dict", {}).get("filepath")

    if prefer_audio:
        selector = "bestaudio/best"
        postprocessors = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format if audio_format in ("mp3", "m4a") else "mp3",
            "preferredquality": "192",
        }]
        merge_fmt = "mp4"
    else:
        # "b" = yt-dlp's shorthand for best single-file format.
        # This avoids triggering the merge path entirely — if a combined
        # stream exists, it uses it; otherwise it falls back gracefully.
        selector = "b"
        postprocessors = []
        merge_fmt = "mp4"

    opts: dict = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "merge_output_format": merge_fmt,
    }

    if postprocessors:
        opts["postprocessors"] = postprocessors

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)

    if not final_path["value"]:
        matches = list(DOWNLOADS_DIR.glob(f"*quick-{uid}*"))
        matches = [f for f in matches if f.suffix not in (".part", ".ytdl", ".tmp")]
        if matches:
            final_path["value"] = str(matches[0])

    if not final_path["value"] or not Path(final_path["value"]).exists():
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = Path(final_path["value"])
    return {
        "title": info.get("title"),
        "file_path": str(file_path),
        "filename": file_path.name,
        "ext": file_path.suffix.lstrip("."),
        "filesize": file_path.stat().st_size,
        "filesize_human": _human_bytes(file_path.stat().st_size),
        "format_selector": selector,
        "format_id_used": info.get("format_id"),
        "resolution": info.get("resolution") or info.get("format_note"),
        "vcodec": info.get("vcodec"),
        "acodec": info.get("acodec"),
        "quick": True,
    }

async def download_video(url: str) -> dict:
    output_template = "%(title)s.%(ext)s"
    file_path = {"value": None}

    def hook(d):
        if d["status"] == "finished":
            file_path["value"] = d.get("filename")

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not file_path["value"]:
        raise RuntimeError("Download failed or file not found")

    file = Path(file_path["value"])

    return {
        "title": info.get("title"),
        "file_path": str(file),
        "filename": file.name,
        "ext": file.suffix.lstrip("."),
        "filesize": file.stat().st_size,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "yt-dlp-server", "version": "2.0.0"}


@app.post("/quick")
async def quick_download(req: QuickDownloadRequest):
    try:
        result = await asyncio.to_thread(download_video, req.url)

        return {
            "success": True,
            "data": {
                **result,
                "fetch_url": f"/download/file?path={result['file_path']}",
            },
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def video_info(url: str):
    """
    Returns full video metadata + all classified formats.

    Browser call:
      GET /info?url=https://www.youtube.com/watch?v=VIDEO_ID
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {"success": True, "data": data}
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/formats")
async def video_formats(url: str):
    """
    Lightweight — returns only the formats list without full metadata.
    Useful when you already have metadata and just need formats.

    Browser call:
      GET /formats?url=https://www.youtube.com/watch?v=VIDEO_ID
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {
            "success": True,
            "data": {
                "id": data["id"],
                "title": data["title"],
                "format_count": data["format_count"],
                "formats": data["formats"],
                "formats_grouped": data["formats_grouped"],
            },
        }
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/transcript")
async def video_transcript(url: str, lang: str = "en"):
    """
    Returns the transcript/subtitles as structured JSON.
    Tries manual subtitles first, falls back to auto-captions.

    Browser call:
      GET /transcript?url=https://www.youtube.com/watch?v=VIDEO_ID
      GET /transcript?url=...&lang=fr
    """
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            executor, _fetch_transcript, url, lang
        )
        return {"success": True, "data": data}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download")
async def download_video(req: DownloadRequest):
    """
    Downloads a video or audio file to the server's downloads directory.
    Returns the filename and a path you can use to fetch the file.

    Body:
      { "url": "...", "quality": "720p", "ext": "mp4" }
      { "url": "...", "format_id": "137" }
      { "url": "...", "quality": "audio", "ext": "mp3" }

    After download, fetch the file via:
      GET /download/file?path=<filename>
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            _run_download,
            req.url,
            req.format_id,
            req.quality or "best",
            req.ext or "mp4",
        )
        return {
            "success": True,
            "data": {
                **result,
                "fetch_url": f"/download/file?path={result['filename']}",
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download/file")
async def serve_file(path: str):
    """
    Serves a previously downloaded file by filename.

    Browser call:
      GET /download/file?path=Some+Title+[abc1].mp4
    """
    file_path = DOWNLOADS_DIR / path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")

    # Security: ensure it's inside downloads dir
    if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
        raise HTTPException(403, "Access denied")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
