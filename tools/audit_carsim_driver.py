"""Run CarSim's own driver on a private copy of an existing run snapshot.

Requires the same licensed VS Solver as the C++ adapter. No Browser files are changed.
"""
import argparse
import csv
import ctypes
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path, help='directory containing simfile.sim and Run_all.par')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dll', required=True, type=Path)
    parser.add_argument('--control-csv', type=Path, help='replay front_steer_command_deg rather than run built-in driver')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve()
    source = args.source.resolve()
    pars = []
    removed_imports = []
    for line in (source / 'Run_all.par').read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if fields and fields[0] == 'IMPORT' and args.control_csv is None:
            removed_imports.append(line)
        elif not fields or fields[0] not in ('EXPORT', 'PORTS_EXP'):
            pars.append(line)
    if args.control_csv is None and (len(removed_imports) != 2 or not all('IMP_STEER_' in item for item in removed_imports)):
        raise RuntimeError('Expected exactly two road-wheel steering imports in source')
    (output / 'Run_all.par').write_text('\n'.join(pars) + '\n', encoding='utf-8')
    paths = {'FILEBASE': output/'LastRun', 'INPUT': output/'Run_all.par',
             'INPUTARCHIVE': output/'LastRun_all.par', 'ECHO': output/'LastRun_echo.par',
             'FINAL': output/'LastRun_end.par', 'LOGFILE': output/'LastRun_log.txt',
             'ERDFILE': output/'LastRun.vs'}
    sim = []
    for line in (source / 'simfile.sim').read_text(encoding='utf-8', errors='replace').splitlines():
        key = line.split(maxsplit=1)[0] if line.split() else ''
        if key in paths:
            sim.append(f'{key} {paths[key]}')
        elif key == 'PORTS_EXP':
            sim.append('PORTS_EXP 0')
        elif key == 'PORTS_IMP':
            sim.append('PORTS_IMP 1,2' if args.control_csv else 'PORTS_IMP 0')
        else:
            sim.append(line)
    (output/'simfile.sim').write_text('\n'.join(sim)+'\n', encoding='utf-8')
    (output/'removed_imports.txt').write_text('\n'.join(removed_imports)+'\n', encoding='utf-8')
    dll = ctypes.CDLL(str(args.dll.resolve()))
    dbl = ctypes.c_double
    dll.vs_read_configuration.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(dbl), ctypes.POINTER(dbl), ctypes.POINTER(dbl)]
    dll.vs_set_opt_error_dialog.argtypes = [ctypes.c_int]
    dll.vs_initialize.argtypes = [dbl, ctypes.c_void_p, ctypes.c_void_p]
    dll.vs_integrate_io.argtypes = [dbl, ctypes.POINTER(dbl), ctypes.POINTER(dbl)]
    dll.vs_integrate_io.restype = ctypes.c_int
    dll.vs_get_var_ptr.argtypes = [ctypes.c_char_p]
    dll.vs_get_var_ptr.restype = ctypes.POINTER(dbl)
    dll.vs_error_occurred.restype = ctypes.c_int
    dll.vs_get_error_message.restype = ctypes.c_char_p
    dll.vs_terminate_run.argtypes = [dbl]
    dll.vs_set_opt_error_dialog(0)
    ni, ne, time, stop, step = ctypes.c_int(), ctypes.c_int(), dbl(), dbl(), dbl()
    dll.vs_read_configuration(str(output/'simfile.sim').encode(), ctypes.byref(ni), ctypes.byref(ne),
        ctypes.byref(time), ctypes.byref(stop), ctypes.byref(step))
    def check():
        if dll.vs_error_occurred():
            raise RuntimeError((dll.vs_get_error_message() or b'CarSim error').decode(errors='replace'))
    check()
    dll.vs_initialize(time.value, None, None)
    check()
    try:
        if ni.value != (2 if args.control_csv else 0):
            raise RuntimeError(f'Unexpected number of steering imports: {ni.value}')
        keys = ['XCG_TM','YCG_TM','YAW','VX','VY','AVZ','STEER_L1','STEER_R1']
        pointers = [dll.vs_get_var_ptr(key.encode()) for key in keys]
        if any(not p for p in pointers):
            raise RuntimeError('Missing VS state variables')
        inputs, outputs = (dbl*max(2, ni.value))(), (dbl*1)()
        controls = []
        if args.control_csv:
            with args.control_csv.open(newline='', encoding='utf-8') as stream:
                controls = [float(row['front_steer_command_deg']) for row in csv.DictReader(stream)
                            if row['is_command'] == '1']
        with (output/'driver_states.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s']+keys)
            count = 0
            while time.value < stop.value:
                if count % 100 == 0:
                    writer.writerow([time.value]+[p[0] for p in pointers])
                    if pointers[0][0] >= 180.0:
                        break
                    if controls:
                        action = controls[min(count//100, len(controls)-1)]
                        inputs[0] = inputs[1] = action
                status = dll.vs_integrate_io(time.value, inputs, outputs)
                check()
                time.value += step.value
                count += 1
                if status:
                    break
        print(f'Internal-driver result: {output}; steps={count}; time={time.value}')
    finally:
        dll.vs_terminate_run(time.value)

if __name__ == '__main__':
    main()
