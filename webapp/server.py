"""Starlette app serving the Moments UI and its local JSON API."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import unquote

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .library import VIDEO_EXTENSIONS, LibraryError, NotFound
from .service import Service

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 ** 3
LOCAL_HOSTS = ["127.0.0.1", "localhost", "[::1]"]

# Windows registries sometimes map .js to text/plain, which browsers refuse for modules.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class RequireAppHeader:
    """Mutating API calls must send X-Moments: 1.

    Browsers only let another site send a custom header after a CORS preflight,
    which this server never approves, so web pages cannot drive the local API.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope["method"] not in ("GET", "HEAD", "OPTIONS")
            and scope["path"].startswith("/api/")
            and dict(scope["headers"]).get(b"x-moments") != b"1"
        ):
            await JSONResponse({"error": "Missing X-Moments header."}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app(service=None, allowed_hosts=None):
    service = service or Service()
    library = service.library

    async def body_json(request):
        try:
            data = await request.json()
        except ValueError:
            raise LibraryError("Request body must be JSON.")
        if not isinstance(data, dict):
            raise LibraryError("Request body must be a JSON object.")
        return data

    async def index(request):
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    async def system(request):
        return JSONResponse(await run_in_threadpool(service.system))

    async def settings(request):
        if request.method == "PUT":
            return JSONResponse(asdict(service.save_settings(await body_json(request))))
        return JSONResponse(asdict(service.settings()))

    async def videos(request):
        if request.method == "GET":
            return JSONResponse({"videos": service.videos()})
        name = unquote(request.headers.get("x-filename", ""))
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise LibraryError(f"Unsupported file type. Upload one of: {', '.join(sorted(VIDEO_EXTENSIONS))}.")
        if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES:
            raise LibraryError("Videos must be smaller than 20 GB.")
        temp = library.incoming_file()
        digest, size = hashlib.sha256(), 0
        try:
            with open(temp, "wb") as file:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise LibraryError("Videos must be smaller than 20 GB.")
                    digest.update(chunk)
                    file.write(chunk)
            if not size:
                raise LibraryError("The uploaded file is empty.")
            record, created = await run_in_threadpool(library.ingest, temp, digest.hexdigest(), name)
        except ClientDisconnect:
            temp.unlink(missing_ok=True)
            return JSONResponse({"error": "Upload cancelled."}, status_code=400)
        finally:
            temp.unlink(missing_ok=True)
        if created or record["status"] in ("uploaded", "failed", "cancelled", "interrupted"):
            service.start_index(record["id"])
        return JSONResponse(service.video(record["id"]), status_code=201 if created else 200)

    async def video(request):
        video_id = request.path_params["video_id"]
        if request.method == "DELETE":
            await run_in_threadpool(service.delete_video, video_id)
            return JSONResponse({"deleted": video_id})
        return JSONResponse(service.video(video_id))

    async def reindex(request):
        return JSONResponse(service.start_index(request.path_params["video_id"]).to_dict(), status_code=202)

    async def media(request):
        return FileResponse(library.media_path(request.path_params["video_id"]))

    async def thumbnail(request):
        return FileResponse(library.thumbnail_path(request.path_params["video_id"]), headers={"Cache-Control": "max-age=86400"})

    async def transcript(request):
        return JSONResponse({"segments": library.transcript(request.path_params["video_id"])})

    async def searches(request):
        video_id = request.path_params["video_id"]
        if request.method == "GET":
            return JSONResponse({"searches": library.list_searches(video_id)})
        data = await body_json(request)
        job = service.start_search(video_id, data.get("query"), data.get("mode", "verified"), data.get("options"))
        return JSONResponse(service.search(video_id, job.result["search_id"]), status_code=202)

    async def search(request):
        video_id, search_id = request.path_params["video_id"], request.path_params["search_id"]
        if request.method == "DELETE":
            record = library.get_search(video_id, search_id)
            if record.get("status") in ("queued", "running") and record.get("job_id"):
                try:
                    service.jobs.cancel(record["job_id"])
                except KeyError:
                    pass
            library.delete_search(video_id, search_id)
            return JSONResponse({"deleted": search_id})
        return JSONResponse(service.search(video_id, search_id))

    async def feedback(request):
        params = request.path_params
        data = await body_json(request)
        return JSONResponse(await run_in_threadpool(
            service.set_feedback, params["video_id"], params["search_id"], data.get("match_key"), data.get("label")))

    async def refine(request):
        params = request.path_params
        service.start_refine(params["video_id"], params["search_id"])
        return JSONResponse(service.search(params["video_id"], params["search_id"]), status_code=202)

    async def search_file(request):
        params = request.path_params
        return FileResponse(library.search_file(params["video_id"], params["search_id"], params["path"]))

    async def jobs(request):
        return JSONResponse({"jobs": [job.to_dict() for job in service.jobs.list()]})

    async def cancel_job(request):
        try:
            job = service.jobs.cancel(request.path_params["job_id"])
        except KeyError:
            raise NotFound("Job not found.")
        return JSONResponse(job.to_dict())

    async def library_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    routes = [
        Route("/", index),
        Route("/api/system", system),
        Route("/api/settings", settings, methods=["GET", "PUT"]),
        Route("/api/videos", videos, methods=["GET", "POST"]),
        Route("/api/videos/{video_id}", video, methods=["GET", "DELETE"]),
        Route("/api/videos/{video_id}/index", reindex, methods=["POST"]),
        Route("/api/videos/{video_id}/media", media),
        Route("/api/videos/{video_id}/thumbnail", thumbnail),
        Route("/api/videos/{video_id}/transcript", transcript),
        Route("/api/videos/{video_id}/searches", searches, methods=["GET", "POST"]),
        Route("/api/videos/{video_id}/searches/{search_id}", search, methods=["GET", "DELETE"]),
        Route("/api/videos/{video_id}/searches/{search_id}/feedback", feedback, methods=["POST"]),
        Route("/api/videos/{video_id}/searches/{search_id}/refine", refine, methods=["POST"]),
        Route("/api/videos/{video_id}/searches/{search_id}/files/{path:path}", search_file),
        Route("/api/jobs", jobs),
        Route("/api/jobs/{job_id}/cancel", cancel_job, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    return Starlette(
        routes=routes,
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOCAL_HOSTS),
            Middleware(RequireAppHeader),
        ],
        exception_handlers={LibraryError: library_error},
    )
