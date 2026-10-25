"""Check whether a generated Simfile can initialize the installed CarSim solver."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver_dll", type=Path)
    parser.add_argument("simfile", type=Path)
    args = parser.parse_args()
    solver = ctypes.CDLL(str(args.solver_dll.resolve()))
    solver.vs_read_configuration.argtypes = [ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)]
    solver.vs_read_configuration.restype = None
    solver.vs_error_occurred.argtypes = []
    solver.vs_error_occurred.restype = ctypes.c_int
    solver.vs_get_error_message.argtypes = []
    solver.vs_get_error_message.restype = ctypes.c_char_p
    solver.vs_terminate_run.argtypes = [ctypes.c_double]
    solver.vs_terminate_run.restype = None
    solver.vs_set_opt_error_dialog.argtypes = [ctypes.c_int]
    solver.vs_set_opt_error_dialog.restype = None
    solver.vs_set_opt_error_dialog(0)
    n_import, n_export = ctypes.c_int(), ctypes.c_int()
    t_start, t_stop, t_step = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    simfile = str(args.simfile.resolve()).encode("mbcs")
    solver.vs_read_configuration(simfile, ctypes.byref(n_import),
        ctypes.byref(n_export), ctypes.byref(t_start), ctypes.byref(t_stop),
        ctypes.byref(t_step))
    error = solver.vs_error_occurred()
    print(f"error={error}, imports={n_import.value}, exports={n_export.value}, "
          f"start={t_start.value}, stop={t_stop.value}, step={t_step.value}")
    if error:
        message = solver.vs_get_error_message()
        print(message.decode("mbcs", errors="replace") if message else "(no message)")
        raise SystemExit(1)
    solver.vs_terminate_run(t_start.value)


if __name__ == "__main__":
    main()
