from pathlib import Path
import json, re, hashlib
root=Path(__file__).resolve().parents[1]
case=root/'carsim'/'case'
manifest=json.loads((case/'manifest.json').read_text())
source=Path(manifest['source_parsfile']).read_text(errors='replace')
private=(case/'Run_all.par').read_text(errors='replace')
def table(text):
    return re.search(r'^LTARG_TABLE SPLINE_FLAT\s*\n(.*?)^ENDTABLE',text,re.M|re.S)[1].strip().splitlines()
assert table(source)==table(private),'Reference trajectory changed'
exports=re.findall(r'^EXPORT (\S+)',private,re.M)
assert exports==manifest['exports'] and len(exports)==20
units=[]
for line in (case/'LastRun_echo.par').read_text(errors='replace').splitlines():
    if line.startswith('EXPORT '): units.append(line)
(root/'results'/'interface_units.txt').write_text('\n'.join(units)+'\n')
checks={}
for name in ['mpc.cpp','mpc.h','vehicle.cpp','vehicle.h','reference.cpp','reference.h','rl.cpp','rl.h']:
    original=(root.parent/'src'/name).read_bytes()
    copied=(root/'controllers'/'native'/name).read_bytes()
    assert original==copied,name
    checks[name]=hashlib.sha256(original).hexdigest()
report=dict(reference_rows=len(table(private)),reference_equal=True,source_core_identical=True,
            source_core_sha256=checks,import_channels=re.findall(r'^IMPORT (\S+) (\S+)',private,re.M),
            source_pars_sha256=hashlib.sha256(Path(manifest['source_parsfile']).read_bytes()).hexdigest())
(root/'results'/'source_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
