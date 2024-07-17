# perftools

## apkrepacker

```bash
$ python -m perftools.apkrepacker --help

usage: apkrepacker.py [-h] -i APK [-o OUTPUT] [--enable-debuggable] [--enable-profileable]

optional arguments:
  -h, --help            show this help message and exit
  -i APK, --apk APK     path of input apk
  -o OUTPUT, --output OUTPUT
                        output path of repacked apk file
  --enable-debuggable   add debuggable in AndroidManifest.xml
  --enable-profileable  add profileable in AndroidManifest.xml
```

## simpleperf

```bash
python -m perftools.simpleperf -h 

usage: simpleperf.py [-h] -s SERIAL -p APP [-o OUTPUT] [-f FREQ] [-d DURATION]

optional arguments:
  -h, --help            show this help message and exit
  -s SERIAL, --serial SERIAL
                        serial of device
  -p APP, --app APP     package name of app to be profiled
  -o OUTPUT, --output OUTPUT
                        output path of perf data collected in device
  -f FREQ, --freq FREQ  frequency of simpleperf record
  -d DURATION, --duration DURATION
                        duration of simpleperf record
```

## perfdata

```python
from perftools.perfdata import Perfdata
from pathlib import Path
import json

data = Perfdata("./perf.data")

threads = data.get_threads()

print(threads)

unitymain = threads["UnityMain"][0]

Path("a.json").write_text(json.dumps(unitymain.aggregate().json(), indent=1, ensure_ascii=False))
```
