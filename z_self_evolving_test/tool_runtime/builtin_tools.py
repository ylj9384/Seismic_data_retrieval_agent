from typing import Dict
try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNNoDataException
except Exception:
    UTCDateTime = None
    Client = None
    class FDSNNoDataException(Exception):
        ...

def fetch_waveforms(network: str,
                    station: str,
                    start: str,
                    end: str,
                    channel: str,
                    location: str,
                    attach_response: bool = False,
                    limit_traces: int = 12) -> Dict:
    """
    仅一次请求：
      - 有数据 -> status=success
      - 无数据 -> status=no_data (不抛异常)
    """
    if Client is None:
        return {"status": "error", "reason": "缺少 obspy 依赖", "network": network, "station": station}

    alias = {
        "主传感器": "00", "主要位置": "00", "标准位置": "00",
        "备用传感器": "10", "次要位置": "10",
        "未指定位置": "", "默认位置": "", "空位置": ""
    }
    if location in alias:
        location = alias[location]

    t0 = UTCDateTime(start); t1 = UTCDateTime(end)
    client = Client("IRIS")
    try:
        st = client.get_waveforms(network=network, station=station,
                                  location=location, channel=channel,
                                  starttime=t0, endtime=t1,
                                  attach_response=attach_response)
    except FDSNNoDataException:
        return {
            "status": "no_data",
            "reason": "FDSN 服务器返回 no data",
            "network": network, "station": station,
            "location": location, "channel": channel,
            "start": start, "end": end
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"{type(e).__name__}: {e}",
            "network": network, "station": station
        }

    if not st or len(st) == 0:
        return {
            "status": "no_data",
            "reason": "请求成功但结果为空",
            "network": network, "station": station,
            "location": location, "channel": channel,
            "start": start, "end": end
        }

    traces = []
    for tr in st:
        s = tr.stats
        tid = getattr(tr, "id", None)
        if not tid:
            tid = ".".join(filter(None, [
                str(getattr(s, "network", "")),
                str(getattr(s, "station", "")),
                str(getattr(s, "location", "")),
                str(getattr(s, "channel", "")),
            ]))
        d = tr.data
        try:
            vmin = float(d.min()); vmax = float(d.max())
        except Exception:
            vmin = vmax = None
        traces.append({
            "id": tid,
            "network": getattr(s, "network", ""),
            "station": getattr(s, "station", ""),
            "location": getattr(s, "location", ""),
            "channel": getattr(s, "channel", ""),
            "start": str(getattr(s, "starttime", "")),
            "end": str(getattr(s, "endtime", "")),
            "npts": int(getattr(s, "npts", 0)),
            "sr": float(getattr(s, "sampling_rate", 0.0)),
            "delta": float(getattr(s, "delta", 0.0)),
            "min": vmin,
            "max": vmax
        })
        if len(traces) >= limit_traces:
            break

    return {
        "status": "success",
        "network": network,
        "station": station,
        "location": location,
        "channel": channel,
        "start": start,
        "end": end,
        "returned_traces": len(traces),
        "traces": traces
    }

