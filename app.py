from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp

app = FastAPI()


class VideoRequest(BaseModel):
    url: str


def extract_video_info(url: str):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "format": "best",
        "socket_timeout": 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            info = ydl.sanitize_info(info)

            formats = [
                f for f in info.get("formats", [])
                if f.get("vcodec") != "none" and f.get("acodec") != "none"
            ][:10]

            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "thumbnail": info.get("thumbnail"),
                "uploader": info.get("uploader"),
                "view_count": info.get("view_count"),
                "formats": [
                    {
                        "format_id": f.get("format_id"),
                        "ext": f.get("ext"),
                        "resolution": f.get("resolution") or f"{f.get('height')}p",
                        "filesize": f.get("filesize"),
                        "url": f.get("url"),
                    }
                    for f in formats
                ],
            }

    except Exception:
        raise HTTPException(status_code=400, detail="Failed to extract video info")


@app.get("/extract")
def extract_video(data: VideoRequest):
    return extract_video_info(data.url)


@app.get("/")
def health():
    return {"status": "ok"}
