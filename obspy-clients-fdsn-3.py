from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client()
starttime = UTCDateTime("2002-01-01")
endtime = UTCDateTime("2002-01-02")
inventory = client.get_stations(network="IU", station="A*",
                                starttime=starttime,
                                endtime=endtime)
inventory.plot()