from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
client = Client()
t = UTCDateTime("2010-02-27T06:45:00.000")
st = client.get_waveforms("IU", "ANMO", "00", "LH*", t, t + 60 * 60)
for tr in st:
    print(tr.stats)


fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
for i in range(3):
    ax.plot(st[i].times(), st[i].data, label=st[i].stats.channel)
ax.legend()

# st.plot()