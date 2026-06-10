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


class BatchStateRequest(CredentialsPayload):
    devices: list[dict[str, Any]] = Field(default_factory=list)


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
            "capabilities": {"power": {"siid": 2, "piid": 1}, "_meta": {"supports_wifi": True}},
            "supports_wifi": True,
            "online_status": "online",
            "last_state": {"power": "off"},
            "last_seen_at": now_utc().isoformat(),
        },
        {
            "external_device_id": "mock-temp-001",
            "name": "主缸 Wi-Fi 温度计",
            "model": "mijia.sensor_ht.wifi.mock",
            "room_name": "阳台",
            "device_kind": "temperature_sensor",
            "capabilities": {
                "temperature": {"siid": 2, "piid": 1},
                "humidity": {"siid": 2, "piid": 2},
                "_meta": {"supports_wifi": True},
            },
            "supports_wifi": True,
            "online_status": "online",
            "last_state": {"temperature": 26.4, "humidity": 68},
            "last_seen_at": now_utc().isoformat(),
        },
    ]


def _is_ble_device(raw: dict[str, Any]) -> bool:
    """Detect BLE/Zigbee devices (not Wi-Fi connected).

    BLE devices have DID starting with 'blt.' and use different property ID
    ranges (piid 1001+) compared to Wi-Fi devices (piid 1+).
    """
    did = str(raw.get("did") or raw.get("external_device_id") or "")
    if did.startswith("blt.") or did.startswith("ble."):
        return True
    model = str(raw.get("model") or "").lower()
    # Known BLE-only sensor models
    ble_models = ("miaomiaoce.sensor_ht", "cgllc.sensor_ht", "vchon.sensor_ht")
    return any(m in model for m in ble_models)


def _classify_device(raw: dict[str, Any], spec: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    name = str(raw.get("name") or "").lower()
    model = str(raw.get("model") or "").lower()
    is_ble = _is_ble_device(raw)
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
    # Fallback: use flattened spec.properties list (from patched get_device_info)
    if not capabilities and spec:
        for prop in (spec.get("properties") or []):
            method = prop.get("method")
            if not method:
                continue
            siid = method.get("siid")
            piid = method.get("piid")
            prop_type = str(prop.get("name") or "").lower()
            if "temperature" in prop_type:
                capabilities.setdefault("temperature", {"siid": siid, "piid": piid})
            if "humidity" in prop_type:
                capabilities.setdefault("humidity", {"siid": siid, "piid": piid})

    # BLE sensors use piid 1001+ range (e.g. vchon.sensor_ht.mbs17: temp=1001, hum=1002)
    ble_temp_piid = 1001
    ble_hum_piid = 1002

    if "power" in capabilities or "plug" in model or "插座" in name:
        capabilities.setdefault("power", {"siid": 2, "piid": 1})
        return "plug", capabilities
    if "temperature" in capabilities or "sensor_ht" in model or "温湿" in name or "温度" in name:
        capabilities.setdefault("temperature", {"siid": 2, "piid": ble_temp_piid if is_ble else 1})
        capabilities.setdefault("humidity", {"siid": 2, "piid": ble_hum_piid if is_ble else 2})
        return "temperature_sensor", capabilities
    if "humidity" in capabilities or "湿度" in name:
        capabilities.setdefault("humidity", {"siid": 2, "piid": 1})
        return "humidity_sensor", capabilities
    return "unknown", capabilities


def _retry_sensor_piid(
    req: StateRequest,
    prop_names: list[str],
    props: list[dict[str, Any]],
    response: dict[str, Any],
    api,
) -> None:
    """Retry temperature/humidity queries with BLE-common piid range.

    BLE sensors (e.g. vchon.sensor_ht.mbs17) use piid 1001+ instead of the
    Wi-Fi-device defaults (piid 1/2).  When the initial query returned no value
    for a sensor property, try the higher range.
    """
    missing: list[tuple[int, str, int]] = []  # (prop_index, name, siid)
    for i, name in enumerate(prop_names):
        if name in ("temperature", "humidity") and response.get(name) is None:
            orig_piid = props[i]["piid"]
            if orig_piid < 100:  # likely a Wi-Fi-default, try BLE range
                missing.append((i, name, props[i]["siid"]))

    if not missing:
        return

    retry_props: list[dict[str, Any]] = []
    retry_map: list[tuple[str, int]] = []  # (name, candidate_piid)
    for _orig_idx, name, siid in missing:
        # Temperature: try 1001, 1003, 1005.  Humidity: try 1002, 1004, 1006.
        base = 1001 if name == "temperature" else 1002
        for candidate_piid in (base, base + 2, base + 4):
            retry_props.append({"did": req.did, "siid": siid, "piid": candidate_piid})
            retry_map.append((name, candidate_piid))

    try:
        retry_result = api.get_devices_prop(retry_props)
    except Exception:
        return

    if not isinstance(retry_result, list):
        return

    for (name, _candidate_piid), item in zip(retry_map, retry_result):
        if response.get(name) is not None:
            continue  # already filled by earlier candidate
        value = item.get("value") if isinstance(item, dict) else None
        code = item.get("code") if isinstance(item, dict) else None
        if value is not None and code is None:
            with contextlib.suppress(TypeError, ValueError):
                response[name] = float(value)


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "wifi", "wi-fi"}:
            return True
        if text in {"0", "false", "no", "n", "ble", "bluetooth", "zigbee"}:
            return False
    return None


def _has_non_empty_field(raw: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip() not in {"", "0", "none", "null"}:
            return True
    return False


def _supports_wifi(raw: dict[str, Any], kind: str) -> bool | None:
    explicit_keys = (
        "supports_wifi",
        "support_wifi",
        "is_wifi",
        "isWifi",
        "wifi",
        "wlan",
        "supportWiFi",
        "support_wlan",
    )
    for key in explicit_keys:
        parsed = _bool_value(raw.get(key))
        if parsed is not None:
            return parsed

    text = " ".join(
        str(raw.get(key) or "").lower()
        for key in (
            "did",
            "name",
            "model",
            "conn_type",
            "connection_type",
            "connect_type",
            "net_type",
            "network_type",
            "protocol",
            "communication_protocol",
        )
    )
    if "wifi" in text or "wi-fi" in text or "wlan" in text:
        return True

    if _has_non_empty_field(raw, ("ssid", "bssid", "localip", "local_ip", "ip", "token")):
        return True

    if _has_non_empty_field(raw, ("parent_id", "parent_did", "parentDid", "gateway_id", "gateway_did", "gatewayDid")):
        return False

    non_wifi_markers = ("blt.", "ble", "bluetooth", "zigbee", "lumi.", "aqara.", "miaomiaoce.sensor_ht", "cgllc.sensor_ht")
    if any(marker in text for marker in non_wifi_markers):
        return False

    if kind == "plug":
        return True
    return None


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
            supports_wifi = _supports_wifi(raw, kind)
            capabilities["_meta"] = {"supports_wifi": supports_wifi is not False}
            normalized.append(
                {
                    "external_device_id": raw.get("did"),
                    "name": raw.get("name") or "米家设备",
                    "model": model,
                    "room_name": raw.get("room_name") or raw.get("room"),
                    "device_kind": kind,
                    "capabilities": capabilities,
                    "supports_wifi": supports_wifi is not False,
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
        return {"power": "off", "temperature": 26.4, "humidity": 68, "online_status": "online", "mock": True}

    props = []
    prop_names = []
    has_sensor_capability = "temperature" in req.capabilities or "humidity" in req.capabilities
    has_power_capability = "power" in req.capabilities
    for name in ("power", "temperature", "humidity"):
        capability = req.capabilities.get(name)
        if not capability:
            if name == "power" and not has_sensor_capability:
                capability = {"siid": 2, "piid": 1}
            elif name == "temperature" and not has_power_capability:
                capability = {"siid": 2, "piid": 1}
            elif name == "humidity" and not has_power_capability:
                capability = {"siid": 2, "piid": 2}
        if not capability:
            continue
        props.append(
            {
                "did": req.did,
                "siid": int(capability.get("siid", 2)),
                "piid": int(capability.get("piid", 1)),
            }
        )
        prop_names.append(name)
    if not props:
        props = [{"did": req.did, "siid": 2, "piid": 1}]
        prop_names = ["power"]

    api = _build_api(req.credentials)
    try:
        result = api.get_devices_prop(props)
        response = {"online_status": "online", "provider_result": result}
        if isinstance(result, list):
            for name, item in zip(prop_names, result):
                value = item.get("value") if isinstance(item, dict) else None
                if name == "power":
                    response["power"] = "on" if value is True else "off" if value is False else "unknown"
                elif name in {"temperature", "humidity"}:
                    with contextlib.suppress(TypeError, ValueError):
                        response[name] = float(value)

        # Auto-discover: if temperature/humidity returned no value with a low
        # piid (1-10), retry with BLE-common piid range (1001-1006).
        _retry_sensor_piid(req, prop_names, props, response, api)

        response.setdefault("power", "unknown")
        return response
    finally:
        _cleanup_runtime_auth(api)


@app.post("/devices/batch-state")
async def get_devices_batch_state(req: BatchStateRequest):
    if MOCK_ENABLED or req.credentials.get("mock"):
        results = []
        for d in req.devices:
            did = d.get("did", "")
            caps = d.get("capabilities", {})
            if "temperature" in caps or "humidity" in caps:
                results.append(
                    {
                        "did": did,
                        "temperature": 26.4,
                        "humidity": 68,
                        "online_status": "online",
                    }
                )
            else:
                results.append({"did": did, "power": "off", "online_status": "online"})
        return {"results": results}

    api = _build_api(req.credentials)
    try:
        results = []
        for d in req.devices:
            did = d.get("did")
            caps = d.get("capabilities", {})
            try:
                props = []
                prop_names = []
                has_sensor = "temperature" in caps or "humidity" in caps
                has_power = "power" in caps
                for name in ("power", "temperature", "humidity"):
                    capability = caps.get(name)
                    if not capability:
                        if name == "power" and not has_sensor:
                            capability = {"siid": 2, "piid": 1}
                        elif name == "temperature" and not has_power:
                            capability = {"siid": 2, "piid": 1}
                        elif name == "humidity" and not has_power:
                            capability = {"siid": 2, "piid": 2}
                    if not capability:
                        continue
                    props.append(
                        {
                            "did": did,
                            "siid": int(capability.get("siid", 2)),
                            "piid": int(capability.get("piid", 1)),
                        }
                    )
                    prop_names.append(name)
                if not props:
                    props = [{"did": did, "siid": 2, "piid": 1}]
                    prop_names = ["power"]

                result = api.get_devices_prop(props)
                entry: dict[str, Any] = {"did": did, "online_status": "online"}
                if isinstance(result, list):
                    for name, item in zip(prop_names, result):
                        value = item.get("value") if isinstance(item, dict) else None
                        if name == "power":
                            entry["power"] = (
                                "on"
                                if value is True
                                else "off" if value is False else "unknown"
                            )
                        elif name in {"temperature", "humidity"}:
                            with contextlib.suppress(TypeError, ValueError):
                                entry[name] = float(value)
                entry.setdefault("power", "unknown")
                results.append(entry)
            except Exception as exc:
                results.append(
                    {
                        "did": did,
                        "error": str(exc),
                        "online_status": "offline",
                    }
                )
        return {"results": results}
    finally:
        _cleanup_runtime_auth(api)


@app.get("/health")
async def health():
    return {"status": "ok", "mock": MOCK_ENABLED, "mijia_api_installed": mijiaAPI is not None}
