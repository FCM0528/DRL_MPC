from pathlib import Path
import subprocess,json,configparser
import numpy as np
root=Path(__file__).resolve().parents[1]
state=np.loadtxt(root/'results'/'parity_states.csv',delimiter=',')
n=len(state);i=np.arange(n)
base=configparser.ConfigParser(inline_comment_prefixes=('#',';'));base.read(root/'configs'/'default.ini')
rl=configparser.ConfigParser(inline_comment_prefixes=('#',';'));rl.read(root/'configs'/'carsim_python_rl.ini')
horizons=[rl.getint('rl','min_np'),base.getint('mpc','np'),rl.getint('rl','max_np'),max(base.getint('mpc','nc'),22)]
tuning=np.column_stack((np.array(horizons)[i//8%4],
    np.array([4600,6000,7500,8250,9200,9850,10500,12000,13800])[i//2%9],
    np.array([1400,1800,2300,2550,2800,3150,3500,4400,5600])[i//4%9],
    np.array([6750,8500,11000,12250,13500,15250,17000,21500,27000])[i//6%9]))
input_file=root/'results'/'adaptive_replay_states.csv'
np.savetxt(input_file,np.column_stack((state,tuning)),delimiter=',',fmt='%.17g')
arrays=[]
for label in ['original','pipeline']:
    output=root/'results'/f'adaptive_replay_{label}.csv'
    subprocess.run([str(root/'build'/f'parity_{label}.exe'),str(root/'configs'/'carsim_python_rl.ini'),str(root/'carsim'/'case'),str(input_file),str(output)],check=True)
    arrays.append(np.loadtxt(output,delimiter=','))
error=np.max(np.abs(arrays[0]-arrays[1]),axis=0)
assert np.all(error==0),error
report=dict(samples=n,columns=len(error),max_difference=float(error.max()),
    prediction_horizons=sorted(set(tuning[:,0].astype(int).tolist())),all_weight_candidates_exercised=True,
    qp_failures=int(np.count_nonzero(arrays[0][:,1]==0)))
(root/'results'/'adaptive_pipeline_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
fault=root/'configs'/'forced_qp_failure.ini'
text=(root/'configs'/'carsim_python_rl.ini').read_text()
fault.write_text(text+'\n[solver]\nmax_iterations = 10\nprimal_tol = 1e-12\ndual_tol = 1e-12\n')
arrays=[]
for label in ['original','pipeline']:
    output=root/'results'/f'forced_qp_failure_{label}.csv'
    subprocess.run([str(root/'build'/f'parity_{label}.exe'),str(fault),str(root/'carsim'/'case'),str(input_file),str(output)],check=True)
    arrays.append(np.loadtxt(output,delimiter=','))
error=np.max(np.abs(arrays[0]-arrays[1]),axis=0)
assert np.all(error==0),error
failures=int(np.count_nonzero(arrays[0][:,1]==0))
assert failures>0 and np.all(np.isfinite(arrays[0])) and np.max(np.abs(arrays[0][:,0]))<np.pi/2
report=dict(samples=n,forced_qp_failures=failures,max_difference=float(error.max()),all_outputs_finite=True)
(root/'results'/'forced_qp_failure_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
