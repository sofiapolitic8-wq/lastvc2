"""Runs on GitHub BEFORE the exe is built: proves the pedalboard chain works with every slider extreme."""
import itertools
import numpy as np
import engine

rng = np.random.default_rng(0)
x = (rng.standard_normal(960) * 0.1).astype(np.float32)

LOW = dict(volume=0, pitch=-12, timbre=0, pers_on=True, thick=-100, persona=-100, mix=0, gate=-20, lowcut=300,
           bass=-12, mid=-12, treble=-12, comp_thr=-40, comp_ratio=1, reverb=0, low_hz=80, high_hz=1500,
           presence_hz=1000, clarity=0, chorus_mix=0, chorus_depth=0, chorus_rate=0.1, vib_depth=0, vib_rate=1,
           comp_attack=1, comp_release=20, room=0, ceiling=-12)
HIGH = dict(volume=200, pitch=12, timbre=100, pers_on=True, thick=100, persona=100, mix=100, gate=-80, lowcut=20,
            bass=12, mid=12, treble=12, comp_thr=0, comp_ratio=10, reverb=50, low_hz=300, high_hz=8000,
            presence_hz=5000, clarity=100, chorus_mix=100, chorus_depth=100, chorus_rate=5, vib_depth=100,
            vib_rate=10, comp_attack=100, comp_release=500, room=100, ceiling=0)

for mode in ("fast", "balanced", "safe"):
    e = engine.Engine(dict(LOW), mode)
    for name, params in (("low", LOW), ("high", HIGH), ("low", LOW), ("high", HIGH)):
        e.params = dict(params)
        for _ in range(40):
            y = e.process(rng.standard_normal(e.hop).astype(np.float32) * 0.1)
            assert y.shape == (e.hop,) and y.dtype == np.float32 and np.all(np.isfinite(y)), (mode, name)
            assert np.max(np.abs(y)) <= 1.0001, "limiter failed"
    print(mode, "ok | latency ~", round(e.latency_samples / engine.SR * 1000), "ms")
# the engine must really let sound through (a silent output would pass every check above)
for pitch in (0, -6, 6):
    e = engine.Engine(dict(volume=100, pitch=pitch, pers_on=False), "balanced")
    sig = engine._tone(e.hop * 80)
    ys = [e.process(sig[i * e.hop:(i + 1) * e.hop]) for i in range(80)]
    rms = float(np.sqrt(np.mean(np.concatenate(ys[-30:]) ** 2)))
    assert rms > 0.05, f"engine output is silent (pitch {pitch}, backend {e.backend})"
print("backend:", e.backend, "| tone rms", round(rms, 3))
print("SELFTEST OK")
