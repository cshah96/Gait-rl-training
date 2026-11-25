"""Step 1a of the gait-imitation pipeline: AddBiomechanics .b3d -> plain numpy reference file.

Reads the treadmill trials of one subject (e.g. Camargo 2021, AB07_split5), keeps the
segments whose belt speed is close to a target speed, and writes everything the imitation
reward / model scaling needs into one .npz that the training env (Python 3.9, no
nimblephysics) can load with numpy only.

    .venv-data/bin/python scripts/mimic_extract_b3d.py ../Data/AB07_split5-001.b3d --subject AB07
    .venv-data/bin/python scripts/mimic_extract_b3d.py <file.b3d> --subject AB07 --speed 1.2 --tol 0.1

Output: data/mimic/<subject>/<subject>_treadmill_raw.npz  (world frame of the lab, OpenSim
coordinate conventions, radians, 200 Hz). Step 1b (scripts/mimic_build_subject.py) turns it
into a scaled SCONE model + reference motion.

Stored arrays (N = frames of all kept segments, concatenated; `seg_start` marks boundaries):
    dof_names (23,)            OpenSim coordinate names (incl. mtp_r/l, dropped later)
    q, dq, ddq, tau (N, 23)    positions / velocities / accelerations / inverse-dynamics torques
                               (from the AddBiomechanics DYNAMICS pass: dynamically consistent)
    grf, cop, grm (N, 2, 3)    per-foot ground reaction force [N], centre of pressure [m],
                               free torque, order (calcn_r, calcn_l)
    contact (N, 2)             foot contact flags
    joint_names (12,)          ground_pelvis, hip_r, walker_knee_r, ankle_r, subtalar_r, mtp_r, ...
    joint_centers (N, 12, 3)   joint centre positions in the world
    com (N, 3)                 whole-body centre of mass
    belt_speed (N,)            estimated treadmill belt speed [m/s] (stance-foot speed, 1 s median)
    seg_start, seg_len, seg_name, seg_speed   per kept segment
    height, mass, sex, age, dt
    segment_lengths            dict-like arrays: femur, tibia, foot, pelvis_width (median, m)
"""
import argparse
import os
import sys

import numpy as np
import nimblephysics as nimble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = []   # nimblephysics 0.10.x segfaults when SubjectOnDisk / Frame objects are freed


def belt_speed_estimate(heel_xz, contact, dt, win_s=1.0):
    """Treadmill belt speed = horizontal speed of a foot while it is on the belt."""
    v = np.r_[np.linalg.norm(np.diff(heel_xz, axis=0), axis=1) / dt, np.nan]
    v[contact == 0] = np.nan
    w = int(win_s / dt)
    out = np.empty_like(v)
    for i in range(len(v)):
        seg = v[max(0, i - w // 2): i + w // 2]
        out[i] = np.nanmedian(seg) if np.isfinite(seg).any() else np.nan
    # fill gaps (flight / both-feet-unloaded frames) by interpolation
    good = np.isfinite(out)
    out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("b3d")
    ap.add_argument("--subject", required=True, help="subject id used for the output folder, e.g. AB07")
    ap.add_argument("--speed", type=float, default=1.2, help="target belt speed [m/s]")
    ap.add_argument("--tol", type=float, default=0.1, help="keep segments with |median speed - target| <= tol")
    ap.add_argument("--max-std", type=float, default=0.06,
                    help="drop segments whose belt speed varies more than this (std, m/s): speed ramps")
    ap.add_argument("--out", help="output .npz (default data/mimic/<subject>/<subject>_treadmill_raw.npz)")
    a = ap.parse_args()

    s = nimble.biomechanics.SubjectOnDisk(a.b3d)
    KEEP.append(s)
    last = s.getNumProcessingPasses() - 1                    # DYNAMICS pass
    skel = s.readSkel(last, ignoreGeometry=True)
    KEEP.append(skel)
    dof_names = [d.getName() for d in skel.getDofs()]
    joint_names = [skel.getJoint(i).getName() for i in range(skel.getNumJoints())]
    J = {n: i for i, n in enumerate(joint_names)}
    print(f"subject {a.subject}: {s.getHeightM()} m, {s.getMassKg()} kg, {s.getBiologicalSex()}, "
          f"{s.getNumTrials()} trials; target {a.speed} +- {a.tol} m/s")

    keep = {k: [] for k in ["q", "dq", "ddq", "tau", "grf", "cop", "grm", "contact", "jc", "com", "belt"]}
    seg_name, seg_start, seg_len, seg_speed = [], [], [], []
    n_total = 0
    for t in range(s.getNumTrials()):
        name = s.getTrialName(t)
        if not name.startswith("treadmill"):
            continue
        L = s.getTrialLength(t)
        if L < 400:
            continue
        missing = [str(r).split(".")[-1] != "notMissingGRF" for r in s.getMissingGRF(t)]
        if np.mean(missing) > 0:
            print(f"  skip {name}: missing GRF on {np.mean(missing):.0%} of frames")
            continue
        fr = s.readFrames(t, 0, L, includeSensorData=False, includeProcessingPasses=True)
        KEEP.append(fr)
        P = [f.processingPasses[last] for f in fr]
        dt = s.getTrialTimestep(t)
        jc = np.array([p.jointCenters for p in P]).reshape(L, -1, 3)
        contact = np.array([p.contact for p in P])
        belt = belt_speed_estimate(jc[:, J["subtalar_r"], [0, 2]], contact[:, 0], dt)
        med, sd = float(np.median(belt)), float(np.std(belt))
        ok = abs(med - a.speed) <= a.tol and sd <= a.max_std
        print(f"  {'KEEP' if ok else 'skip'} {name:<32} belt {med:.2f} +- {sd:.2f} m/s, {L * dt:.1f} s")
        if not ok:
            continue
        keep["q"].append([p.pos for p in P])
        keep["dq"].append([p.vel for p in P])
        keep["ddq"].append([p.acc for p in P])
        keep["tau"].append([p.tau for p in P])
        keep["grf"].append(np.array([p.groundContactForce for p in P]).reshape(L, 2, 3))
        keep["cop"].append(np.array([p.groundContactCenterOfPressure for p in P]).reshape(L, 2, 3))
        keep["grm"].append(np.array([p.groundContactTorque for p in P]).reshape(L, 2, 3))
        keep["contact"].append(contact)
        keep["jc"].append(jc)
        keep["com"].append([p.comPos for p in P])
        keep["belt"].append(belt)
        seg_name.append(name)
        seg_start.append(n_total)
        seg_len.append(L)
        seg_speed.append(med)
        n_total += L

    if not seg_name:
        sys.exit("no treadmill segment matched - try a different --speed / --tol / --max-std")
    cat = {k: np.concatenate([np.asarray(x, dtype=np.float64) for x in v]) for k, v in keep.items()}
    jc = cat["jc"]

    def seglen(a_, b_):
        return float(np.median(np.linalg.norm(jc[:, J[a_]] - jc[:, J[b_]], axis=1)))

    lengths = {
        "femur": np.mean([seglen("hip_r", "walker_knee_r"), seglen("hip_l", "walker_knee_l")]),
        "tibia": np.mean([seglen("walker_knee_r", "ankle_r"), seglen("walker_knee_l", "ankle_l")]),
        "foot": np.mean([seglen("ankle_r", "mtp_r"), seglen("ankle_l", "mtp_l")]),
        "pelvis_width": seglen("hip_r", "hip_l"),
        "pelvis_to_back": seglen("ground_pelvis", "back"),
    }
    print("segment lengths [m]: " + ", ".join(f"{k} {v:.3f}" for k, v in lengths.items()))

    out = a.out or os.path.join(ROOT, "data", "mimic", a.subject, f"{a.subject}_treadmill_raw.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        subject=a.subject, source_file=os.path.abspath(a.b3d),
        height=s.getHeightM(), mass=s.getMassKg(), sex=str(s.getBiologicalSex()), age=s.getAgeYears(),
        dt=dt, dof_names=np.array(dof_names), joint_names=np.array(joint_names),
        q=cat["q"], dq=cat["dq"], ddq=cat["ddq"], tau=cat["tau"],
        grf=cat["grf"], cop=cat["cop"], grm=cat["grm"], contact=cat["contact"].astype(np.int8),
        joint_centers=jc, com=cat["com"], belt_speed=cat["belt"],
        seg_name=np.array(seg_name), seg_start=np.array(seg_start), seg_len=np.array(seg_len),
        seg_speed=np.array(seg_speed), target_speed=a.speed,
        **{f"len_{k}": v for k, v in lengths.items()},
    )
    print(f"wrote {out}: {len(seg_name)} segments, {n_total} frames ({n_total * dt:.0f} s), "
          f"mean belt speed {np.mean(seg_speed):.2f} m/s")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # skip nimblephysics teardown (segfault)
