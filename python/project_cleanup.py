"""Conservative workspace inventory and explicitly bounded, backed-up cleanup."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

CACHES={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
FAILURES={
    'strong_slew_response_20261116_v1':('strong_slew_response_20261116_v3','response_metrics.csv'),
    'td3_lateral_audit_20261116_v1':('td3_lateral_audit_20261116_v2','reward_reconstruction_residuals.csv'),
    'td3_reward_comparison_20261116_v1':('td3_reward_comparison_20261116_v2','reward_ablation_metrics.csv')}


def inventory(workspace,exclude):
    records=[]
    for current,dirs,files in os.walk(workspace,followlinks=False):
        dirs[:]=[d for d in dirs if d!='.git' and not (Path(current)/d).is_symlink() and (Path(current)/d).resolve()!=exclude]
        for name in files:
            path=Path(current)/name
            if path.is_symlink():continue
            relative=path.relative_to(workspace)
            try:size=path.stat().st_size
            except FileNotFoundError:continue
            parts=relative.parts
            category=('regenerable_cache' if set(parts)&CACHES or path.suffix=='.pyc' else
                'independent_matlab' if parts[0]=='matlab' else
                'core_source' if parts[0]=='src' else 'configuration' if parts[0]=='configs' else
                'tests' if 'tests' in parts else 'reproduction_data' if parts[0] in ('runs','policies','results') else
                'build_products' if parts[0] in ('out','.vs','slprj') else
                'experiment_tools' if parts[0]=='python' else 'documentation_or_other')
            records.append(dict(path=relative.as_posix(),bytes=size,category=category))
    return records


def csv_write(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summary(records):
    result={}
    for r in records:
        top=r['path'].split('/')[0];bucket=result.setdefault(top,dict(files=0,bytes=0))
        bucket['files']+=1;bucket['bytes']+=r['bytes']
    return dict(files=len(records),bytes=sum(r['bytes'] for r in records),directories=result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--backup',required=True);p.add_argument('--apply',action='store_true');args=p.parse_args()
    workspace=Path(__file__).resolve().parents[1];out=Path(args.out).resolve();backup=Path(args.backup).resolve()
    if not out.is_relative_to(workspace) or out==workspace:raise ValueError('Audit output must be a new workspace subdirectory')
    if backup.is_relative_to(workspace):raise ValueError('Backup must be outside the workspace')
    out.mkdir(parents=True,exist_ok=False)
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=workspace,text=True).splitlines())
    (out/'git_ls_files.txt').write_text('\n'.join(sorted(tracked)),encoding='utf-8')
    (out/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=workspace,text=True),encoding='utf-8')
    before=inventory(workspace,out);csv_write(out/'inventory_before.csv',before,['path','bytes','category'])
    imports=[];references=[]
    audit_sources=tracked | {r['path'] for r in before if r['path'].startswith(('python/','docs/')) and Path(r['path']).suffix in ('.py','.md')}
    for relative in sorted(audit_sources):
        path=workspace/relative
        if path.suffix=='.py' and path.is_file():
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    imports += [dict(source=relative,target=n.name) for n in node.names]
                elif isinstance(node,ast.ImportFrom):imports.append(dict(source=relative,target=node.module or ''))
        if path.suffix in ('.md','.py','.ps1','.txt') and path.is_file():
            for number,line in enumerate(path.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
                if any(key in line for key in ('python/','python\\','configs/','docs/','runs/','python ','.py"',".py'")):
                    references.append(dict(source=relative,line=number,text=line))
    csv_write(out/'python_imports.csv',imports,['source','target'])
    csv_write(out/'cli_and_document_references.csv',references,['source','line','text'])
    removed=[];archived=[]
    if args.apply:
        backup.mkdir(parents=True,exist_ok=False)
        # Remove only generated caches outside MATLAB; never a tracked file.
        candidates=[]
        for current,dirs,files in os.walk(workspace,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('.git','matlab') and not (Path(current)/d).is_symlink() and (Path(current)/d).resolve()!=out]
            for d in list(dirs):
                if d in CACHES:candidates.append(Path(current)/d);dirs.remove(d)
            candidates.extend(Path(current)/f for f in files if f.endswith('.pyc'))
        for target in candidates:
            resolved=target.resolve()
            if not resolved.is_relative_to(workspace) or resolved==workspace or target.is_symlink():raise ValueError('Unsafe cache target')
            payload=list(target.rglob('*')) if target.is_dir() else [target]
            payload=[v for v in payload if v.is_file()]
            if any(v.relative_to(workspace).as_posix() in tracked for v in payload):continue
            for v in payload:removed.append(dict(path=v.relative_to(workspace).as_posix(),bytes=v.stat().st_size,reason='regenerable Python/tool cache'))
            if target.is_dir():shutil.rmtree(target)
            else:target.unlink()
        for name,(fixed,evidence) in FAILURES.items():
            target=workspace/'runs'/name
            if not target.exists():continue
            resolved=target.resolve()
            if not resolved.is_relative_to(workspace/'runs') or resolved.name!=name or target.is_symlink():raise ValueError('Unsafe failed-run target')
            proof=workspace/'runs'/fixed/evidence
            if not proof.is_file():continue
            with proof.open(encoding='utf-8-sig',newline='') as f:proof_rows=list(csv.DictReader(f))
            if name.startswith('strong_slew'):
                verified=len(proof_rows)==3 and all(r['completed']=='True' and int(r['qp_failures'])==0 for r in proof_rows)
            elif name.startswith('td3_lateral'):
                verified=bool(proof_rows) and all(float(r['max_abs_residual'])<1e-9 for r in proof_rows)
            else:
                verified=len(proof_rows)==6 and all(r['terminated']=='True' and r['stable']=='True' for r in proof_rows)
            if not verified:raise ValueError('Repaired version has not passed its numerical checks')
            files=[v for v in target.rglob('*') if v.is_file()]
            if any(v.suffix.lower() in ('.zip','.pt','.pth','.slx','.mat','.dll','.exe') or 'checkpoint' in v.as_posix().lower() for v in files):
                raise ValueError('Unexpected unique model or binary in failed run; preserve and inspect')
            archive=backup/(name+'.zip')
            expected={v.relative_to(target).as_posix():hashlib.sha256(v.read_bytes()).hexdigest() for v in files}
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for v in files:z.write(v,v.relative_to(target).as_posix())
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None or {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}!=expected:
                    raise ValueError('External backup failed verification')
            archived.append(dict(directory=str(target),backup=str(archive),files=expected,
                repaired_evidence=str(proof),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
            for v in files:removed.append(dict(path=v.relative_to(workspace).as_posix(),bytes=v.stat().st_size,reason='failed run externally backed up and hash-verified; repaired version retained'))
            shutil.rmtree(target)
    after=inventory(workspace,out);csv_write(out/'inventory_after.csv',after,['path','bytes','category'])
    csv_write(out/'removed_files.csv',removed,['path','bytes','reason'])
    (out/'external_archives.json').write_text(json.dumps(archived,indent=2),encoding='utf-8')
    by_size={}
    for r in before:
        if r['path'].startswith(('matlab/results/','matlab/backups/')):
            by_size.setdefault(r['bytes'],[]).append(r['path'])
    duplicates=[]
    for size,paths in by_size.items():
        if len(paths)<2:continue
        hashes={}
        for relative in paths:
            digest=hashlib.sha256((workspace/relative).read_bytes()).hexdigest()
            hashes.setdefault(digest,[]).append(relative)
        for digest,members in hashes.items():
            if len(members)>1:duplicates.append(dict(bytes_each=size,sha256=digest,paths=members,action='retain; archival suggestion only'))
    (out/'matlab_duplicate_candidates.json').write_text(json.dumps(duplicates,indent=2),encoding='utf-8')
    result=dict(before=summary(before),after=summary(after),removed_files=len(removed),
        removed_bytes=sum(r['bytes'] for r in removed),backup=str(backup),
        measurement='logical file sizes, excludes .git and this audit directory; MATLAB inventoried but never cleaned')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if args.apply:
        inventory_archive=backup/'complete_inventory.zip'
        originals={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                   for name in ('inventory_before.csv','inventory_after.csv')}
        with zipfile.ZipFile(inventory_archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name in originals:z.write(out/name,name)
        with zipfile.ZipFile(inventory_archive) as z:
            if z.testzip() is not None or any(hashlib.sha256(z.read(name)).hexdigest()!=digest for name,digest in originals.items()):
                raise ValueError('Inventory backup verification failed')
        (out/'inventory_location.json').write_text(json.dumps(dict(archive=str(inventory_archive),files=originals),indent=2))
        for name in originals:(out/name).unlink()  # Only this tool's generated and verified audit CSVs.
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
