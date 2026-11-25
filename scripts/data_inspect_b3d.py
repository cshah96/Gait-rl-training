"""Print everything stored in AddBiomechanics .b3d files (e.g. the Camargo 2021 dataset).

    .venv-data/bin/python scripts/data_inspect_b3d.py ../Data            # all .b3d under a folder
    .venv-data/bin/python scripts/data_inspect_b3d.py file.b3d --frames 2 # also dump 2 sample frames

Needs nimblephysics (installed in .venv-data, Python 3.11 - separate from the cp39 training env).
"""
import argparse
import collections
import glob
import os
import re
import sys

import numpy as np
import nimblephysics as nimble


KEEP = []


def fmt(a, n=6):
    a = np.asarray(a).ravel()
    s = np.array2string(a[:n], precision=3, suppress_small=True, max_line_width=200)
    return f"{s}{' ...' if a.size > n else ''} (len {a.size})"


def inspect(path, n_frames):
    s = nimble.biomechanics.SubjectOnDisk(path)
    print("=" * 100)
    print(f"FILE {os.path.basename(path)}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    print("=" * 100)
    print(f" subject      : height {s.getHeightM():.3f} m, mass {s.getMassKg():.2f} kg, "
          f"sex {s.getBiologicalSex()}, age {s.getAgeYears()}")
    print(f" tags / notes : {list(s.getSubjectTags())} / {s.getNotes()!r}  href={s.getHref()!r}")
    print(f" quality      : {s.getQuality()}")
    print(f" model        : {s.getNumDofs()} DoFs, {s.getNumJoints()} joints, "
          f"{s.getNumForcePlates(0)} force plates (trial 0), contact bodies {list(s.getGroundForceBodies())}")
    print(f" filtering    : lowpass {s.getLowpassCutoffFrequency(0, s.getNumProcessingPasses() - 1)} Hz "
          f"order {s.getLowpassFilterOrder(0, s.getNumProcessingPasses() - 1)}")
    passes = [str(s.getProcessingPassType(i)).split(".")[-1] for i in range(s.getNumProcessingPasses())]
    print(f" proc. passes : {passes}")

    skel = s.readSkel(s.getNumProcessingPasses() - 1, ignoreGeometry=True)
    names = [d.getName() for d in skel.getDofs()]
    print(f" DoF names    : {names}")
    print(f" bodies       : {[b.getName() for b in skel.getTreeBodyNodes(0)]}")
    osim = s.getOpensimFileText(s.getNumProcessingPasses() - 1)
    muscles = re.findall(r'<(?:Millard2012EquilibriumMuscle|Thelen2003Muscle|DeGrooteFregly2016Muscle)\s+name="([^"]+)"', osim)
    model_name = re.search(r'<Model\s+name="([^"]+)"', osim)
    print(f" OpenSim model: {model_name.group(1) if model_name else '?'}; {len(muscles)} muscles "
          f"{muscles[:6]}{' ...' if len(muscles) > 6 else ''}")

    # ---- trials -----------------------------------------------------------
    nT = s.getNumTrials()
    print(f"\n TRIALS: {nT}")
    print(f" {'#':>3} {'name':<34} {'original':<34} {'frames':>6} {'dt':>7} {'dur s':>6} "
          f"{'missGRF':>7} {'mRMS cm':>7}  tags")
    kinds = collections.Counter()
    for t in range(nT):
        L = s.getTrialLength(t)
        miss = np.mean([str(r).split(".")[-1] != "notMissingGRF" for r in s.getMissingGRF(t)]) if L else 1
        rms = s.getTrialMarkerRMSs(t, 0) if L else []
        orig = s.getTrialOriginalName(t)
        kinds[re.sub(r"[_\d]+$", "", orig.split("/")[0]) or orig] += 1
        print(f" {t:>3} {s.getTrialName(t)[:34]:<34} {orig[:34]:<34} {L:>6} {s.getTrialTimestep(t):>7.4f} "
              f"{L * s.getTrialTimestep(t):>6.1f} {miss:>7.0%} {np.mean(rms) * 100 if len(rms) else float('nan'):>7.2f}  "
              f"{list(s.getTrialTags(t))}")
    print(f" trial types (by original-name prefix): {dict(kinds)}")

    # ---- what a frame contains -------------------------------------------
    t0 = max(range(nT), key=lambda t: s.getTrialLength(t))
    mid = s.getTrialLength(t0) // 2           # frame 0 has zero vel/acc/tau (filter edge)
    fr = s.readFrames(t0, mid, max(n_frames, 1), includeSensorData=True, includeProcessingPasses=True)
    f = fr[0]
    print(f"\n FRAME CONTENTS (trial {t0}, frame {mid}, t={f.t:.3f}s)")
    print(f"  markers      : {len(f.markerObservations)} -> {[m[0] for m in f.markerObservations][:12]} ...")
    print(f"  IMU acc/gyro : {len(f.accObservations)} / {len(f.gyroObservations)} "
          f"{[a[0] for a in f.accObservations][:6]}")
    print(f"  EMG          : {len(f.emgSignals)} channels {[e[0] for e in f.emgSignals][:12]}")
    print(f"  exo torques  : {len(f.exoTorques)}")
    # (raw force-plate arrays are skipped: accessing them here segfaults in nimblephysics 0.10.x;
    #  the processed per-foot GRF / CoP below is what the tracking reward uses)
    print(f"  raw forceplt : {s.getNumForcePlates(t0)} plates in this trial (raw arrays not printed)")
    print(f"  custom values: {[c[0] for c in f.customValues]}")
    for p in f.processingPasses:
        print(f"  pass {str(p.type).split('.')[-1]}:")
        for k in ["pos", "vel", "acc", "tau", "groundContactForce", "groundContactCenterOfPressure",
                  "groundContactTorque", "contact", "comPos", "comVel", "residualWrenchInRootFrame",
                  "jointCenters", "markerRMS"]:
            print(f"     {k:<30} {fmt(getattr(p, k))}")
    for i, f in enumerate(fr[1:n_frames], 1):
        print(f"  -- frame {i}: pos {fmt(f.processingPasses[-1].pos, 12)}")
    KEEP.append((s, fr, skel))   # never free these: nimblephysics segfaults on destruction
    return dict(height=s.getHeightM(), mass=s.getMassKg(), trials=nT, dofs=names,
                emg=len(fr[0].emgSignals), passes=passes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--frames", type=int, default=1)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.b3d"), recursive=True)) if os.path.isdir(p) else [p]
    for f in files:
        inspect(f, a.frames)
    sys.stdout.flush()
    os._exit(0)   # nimblephysics 0.10.x segfaults while tearing down SubjectOnDisk/Frame objects
