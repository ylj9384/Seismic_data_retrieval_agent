from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client("IRIS")
starttime = UTCDateTime("2020-01-01")
endtime = UTCDateTime("2020-01-02")
cat = client.get_events(starttime=starttime, endtime=endtime,
                        minmagnitude=5.0)
print(cat)
# cat.plot()