# dynamic tool: query_station_metadata
def query_station_metadata(network: str, station: str, date: str):
    """查询台站元数据, 返回统一协议
    参数: network, station 通配(可含 *), date=YYYY-MM-DD 或 含时间
    返回: dict(status=...)"""
    import obspy
    from obspy.clients.fdsn import Client
    from obspy.clients.fdsn.header import FDSNNoDataException
    if not (network and station and date):
        return {'status':'error','reason':'缺少必要参数'}
    try:
        if 'T' in date:
            t0 = obspy.UTCDateTime(date)
            t1 = t0 + 1
        else:
            t0 = obspy.UTCDateTime(date + 'T00:00:00')
            t1 = obspy.UTCDateTime(date + 'T23:59:59')
    except Exception as e:
        return {'status':'error','reason':f'时间解析失败: {type(e).__name__}'}
    client = Client('IRIS')
    try:
        inv = client.get_stations(network=network, station=station, starttime=t0, endtime=t1, level='station')
    except FDSNNoDataException:
        return {'status':'no_data','reason':'FDSN no data','network':network,'station_pattern':station,'date':date}
    except Exception as e:
        return {'status':'error','reason':f'{type(e).__name__}: {e}'}
    stations = []
    for net in inv:
        for sta in net:
            stations.append({
                'network': net.code,
                'station': sta.code,
                'latitude': getattr(sta, 'latitude', None),
                'longitude': getattr(sta, 'longitude', None),
                'elevation': getattr(sta, 'elevation', None),
                'start_date': str(sta.start_date) if sta.start_date else None,
                'end_date': str(sta.end_date) if sta.end_date else None
            })
    if not stations:
        return {'status':'no_data','reason':'无匹配站点','network':network,'station_pattern':station,'date':date}
    return {
        'status': 'success',
        'network': network,
        'station_pattern': station,
        'date': date,
        'station_count': len(stations),
        'stations': stations[:50]
    }
