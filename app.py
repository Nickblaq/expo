
# app.py

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
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            info = ydl.sanitize_info(info)

            # Extract only what you actually need (important for performance)
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
                        "resolution": f.get("resolution"),
                        "filesize": f.get("filesize"),
                        "url": f.get("url"),
                    }
                    for f in info.get("formats", [])
                ],
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract")
def extract_video(data: VideoRequest):
    return extract_video_info(data.url)


@app.get("/")
def health():
    return {"status": "ok"}
