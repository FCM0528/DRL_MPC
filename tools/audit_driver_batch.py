"""Compare CarSim's internal driver on the ten archived road/speed fixtures."""
import argparse
import bisect
import csv
import math
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, default=Path('policies/tgt/b2/f'))
    parser.add_argument('--out', type=Path, default=Path('policies/audit/dr'))
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true', help='recompute from existing driver states')
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.reuse)
    result=[]
    for name in ('a60','a80','b60','b80','c60','c80','d60','d80','e60','e80'):
        src=args.fixtures/name
        out=args.out/name
        if not args.reuse:
            subprocess.run([sys.executable,'tools/audit_carsim_driver.py','--source',str(src),'--out',str(out),'--dll',str(args.dll)], check=True)
        lines=(src/'Run_all.par').read_text(encoding='utf-8',errors='replace').splitlines()
        i=next(i for i,line in enumerate(lines) if line.startswith('LTARG_TABLE'))
        table=[]
        for line in lines[i+1:]:
            if line.strip()=='ENDTABLE': break
            x,y=map(float,line.split(',')); table.append((x,y))
        xs=[x for x,_ in table]
        n=len(table); lower=[0.0]*n; diag=[1.0]*n; upper=[0.0]*n; rhs=[0.0]*n
        for j in range(1,n-1):
            h0=xs[j]-xs[j-1]; h1=xs[j+1]-xs[j]
            lower[j]=h0;diag[j]=2*(h0+h1);upper[j]=h1
            rhs[j]=6*((table[j+1][1]-table[j][1])/h1-(table[j][1]-table[j-1][1])/h0)
        for j in range(1,n):
            factor=lower[j]/diag[j-1];diag[j]-=factor*upper[j-1];rhs[j]-=factor*rhs[j-1]
        d2=[0.0]*n;d2[-1]=rhs[-1]/diag[-1]
        for j in range(n-2,-1,-1):d2[j]=(rhs[j]-upper[j]*d2[j+1])/diag[j]
        def reference(x):
            j=bisect.bisect_right(xs,x)
            if j==0: return table[0][1]
            if j==len(xs): return table[-1][1]
            x0,y0=table[j-1]; x1,y1=table[j]
            h=x1-x0;a=(x1-x)/h;b=(x-x0)/h
            return a*y0+b*y1+((a*a*a-a)*d2[j-1]+(b*b*b-b)*d2[j])*h*h/6
        with (out/'driver_states.csv').open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        errors=[float(r['YCG_TM'])-reference(float(r['XCG_TM'])) for r in rows]
        rmse=math.sqrt(sum(e*e for e in errors)/len(errors))
        result.append([name,len(rows),rows[-1]['XCG_TM'],rmse,max(map(abs,errors))])
        print(name,'rmse',rmse,'max',result[-1][-1],flush=True)
    with (args.out/'driver_metrics_cubic_reference.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['case','samples','last_x_m','rmse_y_m','max_abs_y_m']);writer.writerows(result)

if __name__=='__main__': main()
