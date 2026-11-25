"""Step 1b of the gait-imitation pipeline: scaled SCONE model + reference motion for one subject.

    python scripts/mimic_build_subject.py --subject AB07
    python scripts/mimic_build_subject.py --subject AB07 --scaling height   # uniform, by body height

Input : data/mimic/<subject>/<subject>_treadmill_raw.npz   (from scripts/mimic_extract_b3d.py)
Output:
  data/mimic/<subject>/model/H2190_<subject>.scone   scaled H2190 (BodyModifier + ModelModifier), with the
                                                    .hfd / .zml files it needs next to it
  data/mimic/<subject>/<subject>_reference.npz      reference motion in H2190 coordinates (21 DoFs,
                                                    model DoF order, walking forward along +x at the
                                                    belt speed, feet on the ground): what the
                                                    imitation reward will track
  data/mimic/<subject>/scaling.txt                  scale factors + validation report
  results/mimic/<subject>/                          SCONE Studio playback: config.scone + one .sto per
                                                    10 s treadmill segment, and the same motion on
                                                    the unscaled model (default_model/) to compare
                                                    (= ~/SCONE/results/scone-rl/mimic/<subject>)

What it does
  1. scale factors: segment lengths measured in the data (hip-knee, knee-ankle, ankle-toe, hip
     width) divided by the same lengths in H2190; torso by body height (no head markers);
     total mass set to the subject's mass (SCONE redistributes it over the bodies).
  2. motion: drops mtp_r/l, rotates the lab frame so the subject faces +x, turns treadmill walking
     into overground walking (pelvis_tx += integral of belt speed), and shifts pelvis_ty so the
     lowest foot contact sphere touches the ground during stance.
  3. validation: sets every reference frame on the scaled model in Hyfydy and compares the model's
     hip / knee / ankle joint centres with the measured ones (reported in scaling.txt).
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SCONEGYM_DATA = os.path.join(os.path.dirname(ROOT), "sconegym", "sconegym", "data-v1")
BASE_SCONE = "H2190.scone"
MODEL_FILES = ["H2190v2k.hfd", "InitStateGait10.zml", "neural_delays_FEA_v4.zml"]
REFERENCE_HEIGHT = 1.80     # H2190 is based on the 1.80 m / 75 kg OpenSim gait models

# H2190 geometry (from H2190v2k.hfd): joint positions in the body frames
H2190 = {
    "femur": 0.17 + 0.226,              # hip (pos_in_child of hip) -> knee (pos_in_parent of knee)
    "tibia": 0.1867 + 0.2433,           # knee -> ankle
    "foot": np.hypot(0.085 + 0.05123, 0.03 + 0.01195),  # ankle -> front (medial toe) contact sphere
    "pelvis_width": 2 * 0.0835,         # hip_r -> hip_l
}
# local points used to compare the model with the measured joint centres
JOINT_POINTS = {   # data joint name -> (H2190 body, local point)
    "hip_r": ("femur_r", (0, 0.17, 0)), "hip_l": ("femur_l", (0, 0.17, 0)),
    "walker_knee_r": ("tibia_r", (0, 0.1867, 0)), "walker_knee_l": ("tibia_l", (0, 0.1867, 0)),
    "ankle_r": ("calcn_r", (-0.05123, 0.01195, -0.00792)), "ankle_l": ("calcn_l", (-0.05123, 0.01195, 0.00792)),
}
CONTACT_SPHERES = {  # body -> [(local centre, radius)]  (H2190v2k.hfd)
    "calcn_r": [((-0.085, -0.015, -0.005), 0.03), ((0.0425, -0.03, 0.041), 0.02), ((0.085, -0.03, -0.0275), 0.02)],
    "calcn_l": [((-0.085, -0.015, 0.005), 0.03), ((0.0425, -0.03, -0.041), 0.02), ((0.085, -0.03, 0.0275), 0.02)],
}
PELVIS_ROT = ["pelvis_tilt", "pelvis_list", "pelvis_rotation"]   # OpenSim: intrinsic Z, X, Y
# sign differences between the AddBiomechanics (Rajagopal 2016, walker_knee) model and H2190
# (gait2392 conventions): H2190 knee flexion is NEGATIVE (range -120..10 deg)
SIGN = {"knee_angle_r": -1.0, "knee_angle_l": -1.0}
PELVIS_POS = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]


def scale_factors(raw, mode):
    s_h = float(raw["height"]) / REFERENCE_HEIGHT
    if mode == "height":
        f = dict(femur=s_h, tibia=s_h, foot=s_h, pelvis_width=s_h)
    else:
        f = {k: float(raw[f"len_{k}"]) / H2190[k] for k in H2190}
    return {
        "pelvis": (s_h, s_h, f["pelvis_width"]),
        "femur_r;femur_l": (f["femur"],) * 3,
        "tibia_r;tibia_l": (f["tibia"],) * 3,
        "calcn_r;calcn_l": (f["foot"],) * 3,
        "torso": (s_h,) * 3,
    }, f, s_h


def write_scaled_scone(path, factors, mass):
    base = open(os.path.join(SCONEGYM_DATA, BASE_SCONE)).read().rstrip()
    assert base.endswith("}")
    mods = "\n\t# --- subject scaling (scripts/mimic_build_subject.py) ---\n"
    for inc, (x, y, z) in factors.items():
        mods += f"\tBodyModifier {{ include = \"{inc}\" scale = [ {x:.4f} {y:.4f} {z:.4f} ] }}\n"
    mods += f"\tModelModifier {{ mass = {mass:.2f} }}\n"
    open(path, "w").write(base[:-1].rstrip() + "\n" + mods + "}\n")


def yaw_of(R):
    fwd = R.apply([1.0, 0, 0])                    # pelvis forward axis in the world
    return np.arctan2(-fwd[:, 2], fwd[:, 0])      # rotation about +y that maps +x onto it


def to_model_frame(raw, model_dofs, hip_mid_local):
    """Lab treadmill frame -> walking forward along +x; returns dict of arrays in model DoF order.

    hip_mid_local: midpoint of the two hip joint centres in the H2190 pelvis frame. The pelvis
    origin of H2190 is not the OpenSim pelvis origin (H2190 hips sit straight below it), so the
    model pelvis is placed such that its hip midpoint coincides with the measured one."""
    names = list(raw["dof_names"])
    ix = {n: i for i, n in enumerate(names)}
    missing = [d for d in model_dofs if d not in ix]
    if missing:
        raise SystemExit(f"model DoFs not in data: {missing}")
    dt = float(raw["dt"])
    q, dq, tau = raw["q"].copy(), raw["dq"].copy(), raw["tau"].copy()
    for n, sgn in SIGN.items():
        q[:, ix[n]] *= sgn
        dq[:, ix[n]] *= sgn
        tau[:, ix[n]] *= sgn
    jc, grf, cop, com = raw["joint_centers"].copy(), raw["grf"].copy(), raw["cop"].copy(), raw["com"].copy()
    N = len(q)
    starts, lens = raw["seg_start"], raw["seg_len"]

    R = Rotation.from_euler("ZXY", q[:, [ix[n] for n in PELVIS_ROT]])
    yaw = np.median(np.unwrap(yaw_of(R)))          # treadmill: constant heading
    Ry = Rotation.from_euler("y", -yaw)
    R = Ry * R
    q[:, [ix[n] for n in PELVIS_ROT]] = R.as_euler("ZXY")

    def rot(v):                                    # rotate (..., 3) world vectors
        sh = v.shape
        return Ry.apply(v.reshape(-1, 3)).reshape(sh)

    jc, grf, cop, com = rot(jc), rot(grf), rot(cop), rot(com)
    jn = list(raw["joint_names"])
    hip_mid = 0.5 * (jc[:, jn.index("hip_r")] + jc[:, jn.index("hip_l")])
    p = hip_mid - R.apply(np.tile(hip_mid_local, (N, 1)))     # H2190 pelvis origin
    tau[:, [ix[n] for n in PELVIS_POS]] = rot(tau[:, [ix[n] for n in PELVIS_POS]])   # residual force

    # treadmill -> overground: add the belt travel along +x, per segment starting at x = 0
    for s0, L in zip(starts, lens):
        sl = slice(s0, s0 + L)
        travel = np.cumsum(raw["belt_speed"][sl]) * dt
        x0 = p[s0, 0]
        for arr in (p[sl], com[sl]):
            arr[:, 0] += travel - x0
        jc[sl, :, 0] += (travel - x0)[:, None]
        cop[sl, :, 0] += (travel - x0)[:, None]
        z0 = p[s0:s0 + L, 2].mean()
        for arr in (p[sl], com[sl]):
            arr[:, 2] -= z0
        jc[sl, :, 2] -= z0
        cop[sl, :, 2] -= z0
    q[:, [ix[n] for n in PELVIS_POS]] = p
    # pelvis velocities changed (rotation + belt): recompute those 6 by finite differences per segment
    for s0, L in zip(starts, lens):
        sl = slice(s0, s0 + L)
        for n in PELVIS_ROT + PELVIS_POS:
            dq[sl, ix[n]] = np.gradient(np.unwrap(q[sl, ix[n]]) if n in PELVIS_ROT else q[sl, ix[n]], dt)
    sel = [ix[d] for d in model_dofs]
    return dict(q=q[:, sel], dq=dq[:, sel], tau=tau[:, sel], joint_centers=jc, grf=grf, cop=cop, com=com,
                contact=raw["contact"], belt_speed=raw["belt_speed"], seg_start=starts, seg_len=lens, dt=dt)


def lowest_contact_points(model, bodies, sconepy):
    out = []
    for name in ("calcn_r", "calcn_l"):
        b = bodies[name]
        out.append(min(b.point_pos(sconepy.Vec3(*c)).y - r for c, r in CONTACT_SPHERES[name]))
    return out


def write_sto(path, names, rows):
    with open(path, "w") as f:
        f.write(f"{os.path.splitext(os.path.basename(path))[0]}\nversion=1\nnRows={len(rows)}\n"
                f"nColumns={len(names) + 1}\ninDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(names) + "\n")
        for t, r in rows:
            f.write(f"{t:.4f}\t" + "\t".join(f"{v:.6g}" for v in r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--scaling", choices=["segments", "height"], default="segments",
                    help="segments: per-segment lengths from the data (default); height: uniform by height")
    ap.add_argument("--playback-segments", type=int, default=3, help="how many 10 s segments to write as .sto")
    a = ap.parse_args()

    from sconetools import sconepy
    sconepy.set_log_level(3)

    sub_dir = os.path.join(ROOT, "data", "mimic", a.subject)
    raw = dict(np.load(os.path.join(sub_dir, f"{a.subject}_treadmill_raw.npz"), allow_pickle=False))
    mass = float(raw["mass"])

    # ---- 1. scaled model ---------------------------------------------------
    factors, seg_f, s_h = scale_factors(raw, a.scaling)
    model_dir = os.path.join(sub_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    for f in MODEL_FILES:
        shutil.copy(os.path.join(SCONEGYM_DATA, f), model_dir)
    scone_file = os.path.join(model_dir, f"H2190_{a.subject}.scone")
    write_scaled_scone(scone_file, factors, mass)
    model = sconepy.load_model(scone_file)
    model_dofs = [d.name() for d in model.dofs()]
    bodies = {b.name(): b for b in model.bodies()}

    def hip_mid_local_of(model_):
        bod = {b.name(): b for b in model_.bodies()}
        model_.set_dof_positions(np.zeros(len(model_dofs)))
        model_.set_dof_velocities(np.zeros(len(model_dofs)))
        model_.init_state_from_dofs()
        h = [np.array(bod[b].point_pos(sconepy.Vec3(*JOINT_POINTS[k][1])).array())
             for k, b in (("hip_r", "femur_r"), ("hip_l", "femur_l"))]
        return 0.5 * (h[0] + h[1]) - np.array(bod["pelvis"].pos().array())   # zero pose: frame = world
    print(f"scaled model {scone_file}: mass {model.mass():.2f} kg, {len(model_dofs)} DoFs")

    # ---- 2. reference motion in model coordinates ----------------------------
    ref = to_model_frame(raw, model_dofs, hip_mid_local_of(model))
    # check the pelvis rotation convention against Hyfydy (quaternion of the pelvis body)
    set_q = ref["q"][len(ref["q"]) // 2]
    model.set_dof_positions(set_q); model.init_state_from_dofs()
    qt = bodies["pelvis"].orientation()
    rs = Rotation.from_euler("ZXY", [set_q[model_dofs.index(n)] for n in PELVIS_ROT]).as_quat()  # x y z w
    ang = np.degrees((Rotation.from_quat([qt.x, qt.y, qt.z, qt.w]).inv() * Rotation.from_quat(rs)).magnitude())
    print(f"pelvis orientation convention check: {ang:.3f} deg difference (should be ~0)")
    q, dq = ref["q"], ref["dq"]
    ity = model_dofs.index("pelvis_ty")

    def set_pose(i, model_):
        model_.set_dof_positions(q[i])
        model_.set_dof_velocities(dq[i])
        model_.init_state_from_dofs()

    # ground alignment: lowest contact sphere of the stance foot touches y = 0
    idx = np.arange(0, len(q), 10)
    gaps = []
    for i in idx:
        set_pose(i, model)
        lo = lowest_contact_points(model, bodies, sconepy)
        c = ref["contact"][i]
        if c.any():
            gaps.append(min(l for l, ci in zip(lo, c) if ci))
    dy = -float(np.median(gaps))
    q[:, ity] += dy
    for k in ("joint_centers", "cop", "com"):
        ref[k][..., 1] += dy        # keep data points consistent with the shifted pelvis
    print(f"ground alignment: pelvis_ty shifted by {dy * 100:+.1f} cm")

    # ---- 3. validation: model joint centres vs measured -----------------------
    jn = list(raw["joint_names"])
    err = {k: [] for k in JOINT_POINTS}
    pen = []
    for i in idx:
        set_pose(i, model)
        for k, (body, loc) in JOINT_POINTS.items():
            pm = np.array(bodies[body].point_pos(sconepy.Vec3(*loc)).array())
            err[k].append(np.linalg.norm(pm - ref["joint_centers"][i, jn.index(k)]))
        lo = lowest_contact_points(model, bodies, sconepy)
        pen.append(min(lo))
    pen = np.array(pen)

    report = [f"subject {a.subject}: height {float(raw['height']):.3f} m, mass {mass:.2f} kg, scaling = {a.scaling}",
              "segment lengths  data / H2190 = factor"]
    for k in H2190:
        report.append(f"  {k:<13} {float(raw[f'len_{k}']):.3f} / {H2190[k]:.3f} = {float(raw[f'len_{k}']) / H2190[k]:.3f}"
                      f"   (used {seg_f[k]:.3f})")
    report.append(f"  torso / pelvis height: body height ratio {s_h:.3f}")
    report.append("BodyModifier scale factors written to the .scone:")
    report += [f"  {k:<16} {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}" for k, v in factors.items()]
    report.append(f"ModelModifier mass = {mass:.2f} kg (model reports {model.mass():.2f} kg)")
    report.append(f"reference: {len(q)} frames ({len(q) * ref['dt']:.0f} s) in {len(ref['seg_start'])} segments, "
                  f"belt {np.mean(ref['belt_speed']):.2f} m/s, pelvis_ty shift {dy * 100:+.1f} cm")
    report.append("validation (model joint centre vs measured, cm): median / 95th percentile")
    for k, v in err.items():
        report.append(f"  {k:<14} {np.median(v) * 100:5.1f} / {np.percentile(v, 95) * 100:5.1f}")
    report.append(f"lowest foot point over all frames: median {np.median(pen) * 100:+.1f} cm, "
                  f"min {pen.min() * 100:+.1f} cm (negative = into the ground)")
    txt = "\n".join(report)
    print(txt)
    open(os.path.join(sub_dir, "scaling.txt"), "w").write(txt + "\n")

    # ---- 4. model-consistent end-effector positions + GRF sanity ---------------
    # (the imitation reward compares the simulated feet / torso with these, so they are
    #  computed with the same scaled model rather than taken from the marker-based data)
    EE_BODIES = ["calcn_r", "calcn_l", "torso"]
    ee = np.zeros((len(q), len(EE_BODIES), 3))
    for i in range(len(q)):
        set_pose(i, model)
        ee[i] = [bodies[b].pos().array() for b in EE_BODIES]
    ref["ee_pos"] = ee
    ref["ee_names"] = np.array(EE_BODIES)
    # inverse-dynamics torque spikes -> clipped copy used by the torque reward
    tau = ref["tau"]
    lim = np.percentile(np.abs(tau), 99.5, axis=0)
    ref["tau_clipped"] = np.clip(tau, -lim, lim)
    report_tau = {d: (round(float(np.abs(tau[:, j]).max() / mass), 2), round(float(lim[j] / mass), 2))
                  for j, d in enumerate(model_dofs) if d.split("_")[0] in ("hip", "knee", "ankle")}
    print("ID torque peak -> clip level [N.m/kg]:", report_tau)
    open(os.path.join(sub_dir, "scaling.txt"), "a").write(f"ID torque peak -> 99.5th pct clip [N.m/kg]: {report_tau}\n")

    # ---- 5. save reference for training --------------------------------------
    out = os.path.join(sub_dir, f"{a.subject}_reference.npz")
    np.savez_compressed(out, dof_names=np.array(model_dofs), model_file=os.path.relpath(scone_file, ROOT),
                        mass=mass, height=float(raw["height"]), **{k: v for k, v in ref.items()})
    print(f"wrote {out}")

    # ---- 6. SCONE Studio playback ---------------------------------------------
    def playback(model_, scone_src, folder, label):
        os.makedirs(folder, exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy(os.path.join(SCONEGYM_DATA, f), folder)
        shutil.copy(scone_src, os.path.join(folder, "config.scone"))
        dofs = [d.name() for d in model_.dofs()]
        names = [x for d in dofs for x in (d, d + "_u")] + list(model_.state().keys())
        for k, (s0, L) in enumerate(zip(ref["seg_start"][:a.playback_segments], ref["seg_len"][:a.playback_segments])):
            rows = []
            for i in range(s0, s0 + L, 2):            # 100 Hz is plenty for viewing
                set_pose(i, model_)
                dof_cols = [x for j in range(len(dofs)) for x in (q[i, j], dq[i, j])]
                rows.append(((i - s0) * ref["dt"], dof_cols + list(model_.state().values())))
            p = os.path.join(folder, f"{a.subject}_{k:02d}_{raw['seg_name'][k]}.sto")
            write_sto(p, names, rows)
        print(f"{label}: {folder}")

    play = os.path.join(ROOT, "results", "mimic", a.subject)
    playback(model, scone_file, play, "playback (scaled model)")
    default = sconepy.load_model(os.path.join(SCONEGYM_DATA, BASE_SCONE))
    playback(default, os.path.join(SCONEGYM_DATA, BASE_SCONE), os.path.join(play, "default_model"),
             "playback (unscaled H2190, for comparison)")


if __name__ == "__main__":
    main()
