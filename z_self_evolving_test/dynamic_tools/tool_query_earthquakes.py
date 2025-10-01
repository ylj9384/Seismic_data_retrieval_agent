# dynamic tool: query_earthquakes
def query_earthquakes(start: str, end: str):
    """查询指定时间范围内震级大于5.0的地震事件
    参数: start=YYYY-MM-DDTHH:MM:SS, end=YYYY-MM-DDTHH:MM:SS
    返回: dict(status=...)"""
    import obspy
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNNoDataException
    if not (start and end):
        return {'status':'error','reason':'缺少必要参数'}
    try:
        tstart = obspy.UTCDateTime(start)
        tend = obspy.UTCDateTime(end)
    except Exception as e:
        return {'status':'error','reason':f'时间解析失败: {type(e).__name__}'}
    client = Client('IRIS')
    try:
        catalog = client.get_events(starttime=tstart, endtime=tend, minmagnitude=5.0)
    except FDSNNoDataException:
        return {'status':'no_data','reason':'FDSN no data','start':start,'end':end,'minmagnitude':5.0}
    except Exception as e:
        return {'status':'error','reason':f'{type(e).__name__}: {e}'}
    events = []
    for event in catalog:
        mag = None
        for magnitude in event.magnitudes:
            mag = magnitude.mag
            break
        events.append({
            'time': str(event.origins[0].time),
            'latitude': event.origins[0].latitude,
            'longitude': event.origins[0].longitude,
            'depth': event.origins[0].depth,
            'magnitude': mag,
            'region': event.event_descriptions[0].text if len(event.event_descriptions) > 0 else None
        })
    if not events:
        return {'status':'no_data','reason':'无匹配地震事件','start':start,'end':end,'minmagnitude':5.0}
    return {'status':'success','start':start,'end':end,'minmagnitude':5.0,'event_count':len(events),'events':events[:50]}
