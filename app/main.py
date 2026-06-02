import asyncio
import base64
import contextlib
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import parse

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from mijiaAPI import mijiaAPI
except Exception:  # pragma: no cover - dependency is optional in local planning env
    mijiaAPI = None


MOCK_ENABLED = os.getenv("MIJIA_CONTROL_MOCK_ENABLED", "false").lower() in {"1", "true", "yes"}
AUTH_DIR = Path(os.getenv("MIJIA_AUTH_DIR", "/var/lib/mijia-control/auth"))
AUTH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="guiji-mijia-control")
login_sessions: dict[str, dict[str, Any]] = {}


class CredentialsPayload(BaseModel):
    credentials: dict[str, Any] = Field(default_factory=dict)


class DeviceListRequest(CredentialsPayload):
    pass


class ControlRequest(CredentialsPayload):
    did: str
    command: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


class StateRequest(CredentialsPayload):
    did: str
    capabilities: dict[str, Any] = Field(default_factory=dict)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _session_auth_path(session_id: str) -> Path:
    return AUTH_DIR / f"{session_id}.json"


def _credential_identity(auth_data: dict[str, Any]) -> tuple[str, str]:
    provider_user_id = str(auth_data.get("cUserId") or auth_data.get("userId") or "").strip()
    if not provider_user_id:
        provider_user_id = f"mijia-{secrets.token_hex(6)}"
    display_name = f"米家账号 {provider_user_id[-4:]}" if len(provider_user_id) > 4 else "米家账号"
    return provider_user_id, display_name


def _qr_image_base64(qr_image_url: str | None) -> str | None:
    if not qr_image_url:
        return None
    try:
        response = requests.get(qr_image_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return base64.b64encode(response.content).decode("ascii")


def _cleanup_runtime_auth(api) -> None:
    runtime_path: Path | None = getattr(api, "_runtime_auth_path", None)
    if runtime_path and runtime_path.exists():
        with contextlib.suppress(OSError):
            runtime_path.unlink()


def _build_api(credentials: dict[str, Any] | None = None, auth_path: Path | None = None):
    if mijiaAPI is None:
        raise HTTPException(status_code=501, detail="mijiaAPI 未安装，无法调用真实米家服务")

    runtime_path: Path | None = None
    if credentials and credentials.get("auth_json"):
        runtime_path = AUTH_DIR / f"runtime-{secrets.token_hex(8)}.json"
        runtime_path.write_text(credentials["auth_json"], encoding="utf-8")
        auth_path = runtime_path

    if auth_path:
        try:
            api = mijiaAPI(str(auth_path))
            api._runtime_auth_path = runtime_path
            return api
        except TypeError:
            pass
        candidates = [
            {"auth_data_path": str(auth_path)},
        ]
    else:
        candidates = [{}]

    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return mijiaAPI(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    raise HTTPException(status_code=500, detail=f"mijiaAPI 初始化失败：{last_error}")


def _mock_devices() -> list[dict[str, Any]]:
    return [
        {
            "external_device_id": "mock-plug-001",
            "name": "阳台主缸智能插座",
            "model": "chuangmi.plug.mock",
            "room_name": "阳台",
            "device_kind": "plug",
            "capabilities": {"power": {"siid": 2, "piid": 1}},
            "online_status": "online",
            "last_state": {"power": "off"},
            "last_seen_at": now_utc().isoformat(),
        },
        {
            "external_device_id": "mock-temp-001",
            "name": "主缸温湿度计",
            "model": "miaomiaoce.sensor_ht.mock",
            "room_name": "阳台",
            "device_kind": "temperature_sensor",
            "capabilities": {"temperature": {"siid": 2, "piid": 1}, "humidity": {"siid": 2, "piid": 2}},
            "online_status": "online",
            "last_state": {"temperature": 26.4, "humidity": 68},
            "last_seen_at": now_utc().isoformat(),
        },
    ]


def _classify_device(raw: dict[str, Any], spec: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    name = str(raw.get("name") or "").lower()
    model = str(raw.get("model") or "").lower()
    capabilities: dict[str, Any] = {}

    services = []
    if spec:
        services = spec.get("services") or spec.get("instances") or []
    for service in services if isinstance(services, list) else []:
        siid = service.get("iid") or service.get("siid")
        for prop in service.get("properties", []) or []:
            piid = prop.get("iid") or prop.get("piid")
            prop_type = str(prop.get("type") or prop.get("description") or prop.get("name") or "").lower()
            access = prop.get("access") or []
            if "on" in prop_type and "write" in access:
                capabilities["power"] = {"siid": siid, "piid": piid}
            if "temperature" in prop_type:
                capabilities["temperature"] = {"siid": siid, "piid": piid}
            if "humidity" in prop_type:
                capabilities["humidity"] = {"siid": siid, "piid": piid}

    if "power" in capabilities or "plug" in model or "插座" in name:
        capabilities.setdefault("power", {"siid": 2, "piid": 1})
        return "plug", capabilities
    if "temperature" in capabilities or "sensor_ht" in model or "温湿" in name or "温度" in name:
        return "temperature_sensor", capabilities
    if "humidity" in capabilities or "湿度" in name:
        return "humidity_sensor", capabilities
    return "unknown", capabilities


def _activate_session_from_auth(session_id: str, api, auth_path: Path, message: str) -> None:
    session = login_sessions[session_id]
    auth_json = auth_path.read_text(encoding="utf-8") if auth_path.exists() else ""
    if not auth_json:
        raise RuntimeError("米家登录成功但未找到凭证文件，请检查 mijiaAPI auth_path 适配")
    provider_user_id, display_name = _credential_identity(api.auth_data)
    session.update(
        {
            "status": "active",
            "provider_user_id": provider_user_id,
            "display_name": display_name,
            "credentials": {"auth_json": auth_json, "provider_user_id": provider_user_id},
            "message": message,
        }
    )


def _login_worker(session_id: str) -> None:
    session = login_sessions[session_id]
    auth_path = _session_auth_path(session_id)
    try:
        api = _build_api(auth_path=auth_path)
        location_data = api._get_location()
        if location_data.get("code", -1) == 0 and location_data.get("message") == "刷新Token成功":
            api._save_auth_data()
            api._init_session()
            _activate_session_from_auth(session_id, api, auth_path, "Token 有效，已自动登录")
            return

        location_data.update(
            {
                "theme": "",
                "bizDeviceType": "",
                "_hasLogo": "false",
                "_qrsize": "240",
                "_dc": str(int(time.time() * 1000)),
            }
        )
        headers = {
            "User-Agent": api.user_agent,
            "Accept-Encoding": "gzip",
            "Content-Type": "application/x-www-form-urlencoded",
            "Connection": "keep-alive",
        }
        url = api.login_url + "?" + parse.urlencode(location_data)
        login_ret = requests.get(url, headers=headers, timeout=20)
        login_data = api._handle_ret(login_ret)
        session.update(
            {
                "status": "waiting_for_scan",
                "qr_url": login_data.get("loginUrl"),
                "qr_image_base64": _qr_image_base64(login_data.get("qr")),
                "message": "请使用米家 App 扫码登录",
            }
        )

        poll_session = requests.Session()
        try:
            lp_ret = poll_session.get(login_data["lp"], headers=headers, timeout=120)
            lp_data = api._handle_ret(lp_ret)
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("二维码登录超时，请重新发起登录") from exc

        for key in ["psecurity", "nonce", "ssecurity", "passToken", "userId", "cUserId"]:
            api.auth_data[key] = lp_data[key]
        poll_session.get(lp_data["location"], headers=headers, timeout=20)
        api.auth_data.update(poll_session.cookies.get_dict())
        api.auth_data["expireTime"] = int((datetime.now() + timedelta(days=30)).timestamp() * 1000)
        api._save_auth_data()
        api._init_session()
        _activate_session_from_auth(session_id, api, auth_path, "登录成功")
    except Exception as exc:
        session["status"] = "failed"
        session["message"] = str(exc)


@app.post("/login-sessions")
async def create_login_session():
    expires_at = now_utc() + timedelta(minutes=10)
    if MOCK_ENABLED:
        session_id = f"mock-{secrets.token_hex(8)}"
        login_sessions[session_id] = {
            "session_id": session_id,
            "status": "waiting_for_scan",
            "qr_url": "https://mijia.mock/login/scan",
            "expires_at": expires_at,
            "credentials": {"mock": True, "provider_user_id": f"mock-{session_id}"},
            "provider_user_id": f"mock-{session_id}",
            "display_name": "米家测试账号",
        }
        return {k: v for k, v in login_sessions[session_id].items() if k != "credentials"}

    session_id = secrets.token_hex(16)
    login_sessions[session_id] = {
        "session_id": session_id,
        "status": "pending",
        "expires_at": expires_at,
        "qr_url": None,
        "qr_image_base64": None,
        "message": "正在初始化米家登录",
    }
    threading.Thread(target=_login_worker, args=(session_id,), daemon=True).start()
    await asyncio.sleep(0.5)
    return {k: v for k, v in login_sessions[session_id].items() if k != "credentials"}


@app.get("/login-sessions/{session_id}")
async def get_login_session(session_id: str):
    session = login_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="登录会话不存在")
    if session["expires_at"] < now_utc() and session["status"] != "active":
        session["status"] = "failed"
        session["message"] = "二维码已过期，请重新发起登录"
    if MOCK_ENABLED and session["status"] == "waiting_for_scan":
        session["status"] = "active"
        session["message"] = "Mock 登录成功"
    return session


@app.post("/devices")
async def list_devices(req: DeviceListRequest):
    if MOCK_ENABLED or req.credentials.get("mock"):
        return {"devices": _mock_devices()}

    api = _build_api(req.credentials)
    try:
        devices = api.get_devices_list() or []
        normalized = []
        for raw in devices:
            spec = None
            model = raw.get("model")
            if model and hasattr(api, "get_device_info"):
                with contextlib.suppress(Exception):
                    spec = api.get_device_info(model)
            kind, capabilities = _classify_device(raw, spec)
            normalized.append(
                {
                    "external_device_id": raw.get("did"),
                    "name": raw.get("name") or "米家设备",
                    "model": model,
                    "room_name": raw.get("room_name") or raw.get("room"),
                    "device_kind": kind,
                    "capabilities": capabilities,
                    "online_status": "online" if raw.get("isOnline", True) else "offline",
                    "last_state": {},
                    "last_seen_at": now_utc().isoformat(),
                }
            )
        return {"devices": normalized}
    finally:
        _cleanup_runtime_auth(api)


@app.post("/devices/control")
async def control_device(req: ControlRequest):
    if MOCK_ENABLED or req.credentials.get("mock"):
        return {"ok": True, "power": "on" if req.command == "turn_on" else "off", "mock": True}

    power = req.capabilities.get("power") or {"siid": 2, "piid": 1}
    api = _build_api(req.credentials)
    try:
        result = api.set_devices_prop(
            {
                "did": req.did,
                "siid": int(power.get("siid", 2)),
                "piid": int(power.get("piid", 1)),
                "value": req.command == "turn_on",
            }
        )
        return {"ok": True, "provider_result": result, "power": "on" if req.command == "turn_on" else "off"}
    finally:
        _cleanup_runtime_auth(api)


@app.post("/devices/state")
async def get_device_state(req: StateRequest):
    if MOCK_ENABLED or req.credentials.get("mock"):
        return {"power": "off", "online_status": "online", "mock": True}

    power = req.capabilities.get("power") or {"siid": 2, "piid": 1}
    api = _build_api(req.credentials)
    try:
        result = api.get_devices_prop(
            [
                {
                    "did": req.did,
                    "siid": int(power.get("siid", 2)),
                    "piid": int(power.get("piid", 1)),
                }
            ]
        )
        value = None
        if isinstance(result, list) and result:
            value = result[0].get("value")
        return {"power": "on" if value is True else "off" if value is False else "unknown", "provider_result": result}
    finally:
        _cleanup_runtime_auth(api)


@app.get("/health")
async def health():
    return {"status": "ok", "mock": MOCK_ENABLED, "mijia_api_installed": mijiaAPI is not None}
