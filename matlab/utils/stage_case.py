"""Freeze the selected CarSim run without changing the Browser database."""
from pathlib import Path
import hashlib, json, re

root = Path(__file__).resolve().parents[1]
source = Path(r'D:\Program Files (x86)\CarSim2024_Date')
case = root / 'carsim' / 'case'
case.mkdir(parents=True, exist_ok=True)
sim = (source / 'simfile.sim').read_text()
macros = dict(re.findall(r'^SET_MACRO (\$\([^)]+\)\$) (.*)$', sim, re.M))
input_path = re.search(r'^INPUT (.*)$', sim, re.M)[1]
for _ in range(8):
    for key, value in macros.items():
        input_path = input_path.replace(key, value)
original = Path(input_path)
data = original.read_bytes()
pars = data.decode('utf-8', errors='replace').replace('\r\n','\n')
assert pars.count('IMPORT IMP_STEER_L2 Replace') + pars.count('IMPORT IMP_STEER_R1 Replace') == 1
corrected_import = 'IMPORT IMP_STEER_L2 Replace' in pars
pars = pars.replace('IMPORT IMP_STEER_L2 Replace', 'IMPORT IMP_STEER_R1 Replace')
# Same private station margin as extendCarSimStationStop in the original C++.
# The native model ends at CG X=120, independent of the road-station margin.
pars = re.sub(r'^SSTOP 150\s*$', 'SSTOP 160', pars, flags=re.M)
native_model = root / 'models' / 'DRL_MPC_MATLAB.slx'
if native_model.exists():
    pars = re.sub(r'^SIMULINK_MODEL_FILE .+$',
                  lambda _: f'SIMULINK_MODEL_FILE {native_model}', pars, flags=re.M)
(case / 'Run_all.par').write_text(pars, encoding='utf-8')
(case / 'source_Run_all.par').write_bytes(data)
(case / 'source_simfile.sim').write_bytes((source / 'simfile.sim').read_bytes())
prefix = str(case / 'LastRun')
lines = ['SIMFILE', f'FILEBASE {prefix}', f'INPUT {case / "Run_all.par"}',
         f'INPUTARCHIVE {prefix}_all.par', f'ECHO {prefix}_echo.par',
         f'FINAL {prefix}_end.par', f'LOGFILE {prefix}_log.txt', f'ERDFILE {prefix}.vs',
         'PROGDIR D:\\Program Files (x86)\\CarSim2024.1_Prog\\',
         f'DATADIR {source}\\', 'PRODUCT_ID CarSim', 'PRODUCT_VER 2024.1',
         'VEHICLE_CODE i_i', 'EXT_MODEL_STEP 0.00050000', 'PORTS_IMP 1,2', 'PORTS_EXP 1,20', 'END']
(case / 'simfile.sim').write_text('\n'.join(lines) + '\n', encoding='utf-8')
manifest = dict(source_parsfile=str(original), source_sha256=hashlib.sha256(data).hexdigest(),
                changes=[('Second import corrected in private copy' if corrected_import else 'Source front-wheel imports already correct'),
                         'Private station margin SSTOP 160; native evaluation ends at CG X=120',
                         'Private Simulink model link points to the native model when available',
                         'All solver output paths redirected under matlab/carsim/case'],
                reference='DLC SYX; entire LTARG_TABLE preserved',
                exports=re.findall(r'^EXPORT (\S+)', pars, re.M))
(case / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
