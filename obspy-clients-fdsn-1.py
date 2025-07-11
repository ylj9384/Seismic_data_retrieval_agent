from obspy import UTCDateTime
from obspy.clients.fdsn import Client
client = Client()
t = UTCDateTime("2010-02-27T06:45:00.000")
st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
for tr in st:
    print(tr.stats)
st.plot()