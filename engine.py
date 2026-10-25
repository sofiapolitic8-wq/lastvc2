"""engine.py - CuteVoice v4 voice DSP, built on Spotify's `pedalboard`.

Signal chain (mono, SR = 48 kHz, processed in whole hops):

  NoiseGate -> HighpassFilter(Low Cut) -> Compressor -> PitchShift -> Chorus(thickness)
            -> Chorus(vibrato, 100 % wet) -> LowShelfFilter -> PeakFilter(presence)
            -> HighShelfFilter -> Reverb -> [smooth volume ramp] -> Limiter

Every number in the chain is driven by a GUI slider (see app.py).  The GUI only writes numbers into
`engine.params` (a dict).  At the start of every audio block `process()` converts them into target
values for the plugins and glides the real plugin values towards those targets (~60 ms time
constant), so dragging a slider while talking never clicks or zippers.  Plugins are only touched
from the worker thread (the thread that runs them), so no locks are needed.

Interface expected by app.py / audio_io.py:
  SR, Engine(params, mode), .params, .mode, .set_mode(), .hop, .latency_samples,
  .peak (mic level), .out_peak (output level), .process(block) -> block
"""
import math

import numpy as np
try:
    from pedalboard import (Chorus, Compressor, HighpassFilter, HighShelfFilter, Limiter,
                            LowShelfFilter, NoiseGate, Pedalboard, PeakFilter, PitchShift, Reverb)
    PEDAL_IMPORT_ERROR = None
except Exception as _e:                      # missing DLL etc.: the numpy-only SafeEngine takes over
    PEDAL_IMPORT_ERROR = _e

SR = 48000
HOPS = {"fast": 480, "balanced": 960, "safe": 1920}      # 10 / 20 / 40 ms per block
GLIDE_SECONDS = 0.06                                      # time constant of the slider smoothing
VIB_CENTRE_MS = 8.0                                       # fixed delay of the vibrato stage
CHORUS_CENTRE_MS = 7.0

_latency_cache = {}


def _measure_latency():
    """Delay of PitchShift in samples, measured once with noise (so the delay readout is honest)."""
    if "v" in _latency_cache:
        return _latency_cache["v"]
    try:
        if PEDAL_IMPORT_ERROR is not None:
            raise RuntimeError("no pedalboard")
        rng = np.random.default_rng(1)
        sig = (rng.standard_normal(SR // 2) * 0.1).astype(np.float32)
        board = Pedalboard([PitchShift(semitones=0.0)])
        outs = []
        for i in range(0, len(sig) - 479, 480):
            y = board(sig[i:i + 480][None, :], SR, reset=(i == 0))
            outs.append(np.asarray(y, np.float32).reshape(-1))
        out = np.concatenate(outs)
        sig = sig[:len(out)]
        n = 1 << int(np.ceil(np.log2(2 * len(out))))
        corr = np.fft.irfft(np.fft.rfft(out, n) * np.conj(np.fft.rfft(sig, n)), n)
        lat = int(np.argmax(corr[:16384]))
    except Exception:
        lat = 2688
    lat += int((CHORUS_CENTRE_MS + VIB_CENTRE_MS) / 1000.0 * SR)
    _latency_cache["v"] = lat
    return lat


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _targets(p):
    def g(k, d=0.0):
        try:
            return float(p.get(k, d))
        except (TypeError, ValueError):
            return d

    on = bool(p.get("pers_on", False))
    mix = g("mix", 100.0) / 100.0 if on else 0.0
    thick = g("thick") / 100.0 * mix                   # -1 thin ... +1 thick
    persona = g("persona") / 100.0 * mix               # -1 ... +1
    semis = _clamp(g("pitch") + persona * 6.0, -24.0, 24.0)
    tilt = (g("timbre", 50.0) - 50.0) / 50.0           # -1 dark/heavy ... +1 light/bright
    pos_thick = max(thick, 0.0)

    low_db = g("bass") - tilt * 4.0 + thick * 4.0      # heavy bass / body
    clarity = g("clarity", 100.0) / 100.0              # auto-restore of clarity when pitched down
    bright = min(max(-semis, 0.0) * 0.4, 4.0) * clarity
    high_db = g("treble") + tilt * 3.0 - thick * 2.0 + bright

    gate = g("gate", -80.0)
    return dict(
        gate_db=-100.0 if gate <= -80.0 else _clamp(gate, -100.0, -20.0),
        lowcut_hz=_clamp(g("lowcut", 20.0), 20.0, 300.0),
        comp_thr=_clamp(g("comp_thr", 0.0), -60.0, 0.0),
        comp_ratio=_clamp(g("comp_ratio", 1.0), 1.0, 20.0),
        comp_attack=_clamp(g("comp_attack", 5.0), 0.5, 200.0),
        comp_release=_clamp(g("comp_release", 120.0), 5.0, 1000.0),
        semitones=semis,
        chorus_mix=_clamp(g("chorus_mix", 0.0) / 100.0 + pos_thick * 0.45, 0.0, 1.0),
        chorus_depth=_clamp(g("chorus_depth", 25.0) / 100.0 + 0.15 * pos_thick, 0.0, 1.0),
        chorus_rate=_clamp(g("chorus_rate", 0.8), 0.05, 10.0),
        vib_depth=_clamp(g("vib_depth", 0.0) / 100.0 * 0.6, 0.0, 1.0),
        vib_rate=_clamp(g("vib_rate", 5.0), 0.5, 15.0),
        low_hz=_clamp(g("low_hz", 150.0), 40.0, 600.0),
        low_db=_clamp(low_db, -18.0, 18.0),
        presence_hz=_clamp(g("presence_hz", 2500.0), 500.0, 8000.0),
        mid_db=_clamp(g("mid"), -12.0, 12.0),
        high_hz=_clamp(g("high_hz", 3000.0), 1000.0, 12000.0),
        high_db=_clamp(high_db, -18.0, 18.0),
        room=_clamp(g("room", 35.0) / 100.0, 0.0, 1.0),
        reverb_wet=_clamp(g("reverb") / 100.0 * 0.8, 0.0, 0.5),
        ceiling=_clamp(g("ceiling", -1.0), -12.0, 0.0),
    ), _clamp(g("volume", 100.0) / 100.0, 0.0, 2.0)



class PedalEngine:
    def __init__(self, params, mode="balanced"):
        self.params = params
        self.peak = 0.0
        self.out_peak = 0.0
        self._cur = {}                       # last value written to each plugin attribute
        self._sm = None                      # smoothed (current) control values
        self._gain = None                    # current linear volume (for the click-free ramp)
        self._reset = True

        self.gate = NoiseGate(threshold_db=-100.0, ratio=10.0, attack_ms=1.0, release_ms=120.0)
        self.lowcut = HighpassFilter(cutoff_frequency_hz=20.0)
        self.comp = Compressor(threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=120.0)
        self.pitch = PitchShift(semitones=0.0)
        self.chorus = Chorus(rate_hz=0.8, depth=0.25, centre_delay_ms=CHORUS_CENTRE_MS, feedback=0.0, mix=0.0)
        self.vibrato = Chorus(rate_hz=5.0, depth=0.0, centre_delay_ms=VIB_CENTRE_MS, feedback=0.0, mix=1.0)
        self.low = LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=0.0, q=0.7071)
        self.presence = PeakFilter(cutoff_frequency_hz=2500.0, gain_db=0.0, q=0.9)
        self.high = HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=0.0, q=0.7071)
        self.reverb = Reverb(room_size=0.35, damping=0.6, wet_level=0.0, dry_level=1.0, width=1.0)
        self.limiter = Limiter(threshold_db=-1.0, release_ms=80.0)
        self.board = Pedalboard([self.gate, self.lowcut, self.comp, self.pitch, self.chorus, self.vibrato,
                                 self.low, self.presence, self.high, self.reverb])
        self.limiter_board = Pedalboard([self.limiter])

        # (plugin, attribute, control key): one line per adjustable plugin value
        self._map = [
            (self.gate, "threshold_db", "gate_db"),
            (self.lowcut, "cutoff_frequency_hz", "lowcut_hz"),
            (self.comp, "threshold_db", "comp_thr"),
            (self.comp, "ratio", "comp_ratio"),
            (self.comp, "attack_ms", "comp_attack"),
            (self.comp, "release_ms", "comp_release"),
            (self.pitch, "semitones", "semitones"),
            (self.chorus, "mix", "chorus_mix"),
            (self.chorus, "depth", "chorus_depth"),
            (self.chorus, "rate_hz", "chorus_rate"),
            (self.vibrato, "depth", "vib_depth"),
            (self.vibrato, "rate_hz", "vib_rate"),
            (self.low, "cutoff_frequency_hz", "low_hz"),
            (self.low, "gain_db", "low_db"),
            (self.presence, "cutoff_frequency_hz", "presence_hz"),
            (self.presence, "gain_db", "mid_db"),
            (self.high, "cutoff_frequency_hz", "high_hz"),
            (self.high, "gain_db", "high_db"),
            (self.reverb, "room_size", "room"),
            (self.reverb, "wet_level", "reverb_wet"),
            (self.limiter, "threshold_db", "ceiling"),
        ]

        self.mode = None
        self.hop = HOPS["balanced"]
        self._alpha = 0.3
        self.set_mode(mode)
        self.latency_samples = _measure_latency()

    # ------------------------------------------------------------------ mode
    def set_mode(self, mode):
        self.mode = mode if mode in HOPS else "balanced"
        self.hop = HOPS[self.mode]
        self._alpha = 1.0 - math.exp(-(self.hop / float(SR)) / GLIDE_SECONDS)
        self._reset = True                    # flush plugin state on the next block

    # ------------------------------------------------------------------ slider -> target values
    def _write(self, plugin, attr, value):
        value = round(float(value), 3)
        key = (id(plugin), attr)
        if self._cur.get(key) != value:
            try:
                setattr(plugin, attr, value)
            except Exception:                              # out-of-range / unsupported value: keep the old one
                pass
            self._cur[key] = value

    def _sync(self):
        target, volume = _targets(self.params)
        if self._sm is None:                               # first block: jump straight there
            self._sm = dict(target)
        else:
            a = self._alpha
            for k, t in target.items():
                c = self._sm[k]
                d = t - c
                self._sm[k] = t if abs(d) < max(1e-3, 1e-3 * abs(t)) else c + d * a
        for plugin, attr, key in self._map:
            self._write(plugin, attr, self._sm[key])
        return volume

    # ------------------------------------------------------------------ audio
    def process(self, x):
        x = np.asarray(x, np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return x
        volume = self._sync()
        self.peak = float(np.max(np.abs(x)))

        y = self.board(x[None, :], SR, reset=self._reset)
        self._reset = False
        y = np.asarray(y, np.float32).reshape(-1)
        if len(y) != n:                                    # defensive: keep block size exact
            y = np.pad(y, (0, max(0, n - len(y))))[:n]

        # volume: ramp across the block, so dragging the slider never clicks
        if self._gain is None:
            self._gain = volume
        y *= np.linspace(self._gain, volume, n, endpoint=False, dtype=np.float32)
        self._gain = volume

        y = np.asarray(self.limiter_board(y[None, :], SR, reset=False), np.float32).reshape(-1)
        np.nan_to_num(y, copy=False)
        np.clip(y, -1.0, 1.0, out=y)                       # hard safety net: the limiter alone can overshoot
        self.out_peak = float(np.max(np.abs(y)))
        return y


# =====================================================================================================
#  SafeEngine - numpy only (no pedalboard).  Used automatically when the pedalboard chain fails its
#  start-up check or goes silent / raises while running, so the person always gets sound.
#  Pitch (granular two-tap shifter), Low Cut, Bass / Presence / Treble EQ, Noise gate, Compressor,
#  Volume and Limiter.  (No chorus / vibrato / reverb in this engine.)
# =====================================================================================================
class _Biquad:
    def __init__(self):
        self.b0, self.b1, self.b2, self.a1, self.a2 = 1.0, 0.0, 0.0, 0.0, 0.0
        self.z1 = self.z2 = 0.0
        self.on = False

    def set(self, b0, b1, b2, a0, a1, a2):
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0

    def enable(self, flag):
        if flag and not self.on:
            self.z1 = self.z2 = 0.0
        self.on = flag

    def run(self, x):
        b0, b1, b2, a1, a2, z1, z2 = self.b0, self.b1, self.b2, self.a1, self.a2, self.z1, self.z2
        out = [0.0] * len(x)
        i = 0
        for v in x.tolist():
            y = b0 * v + z1
            z1 = b1 * v - a1 * y + z2
            z2 = b2 * v - a2 * y
            out[i] = y
            i += 1
        self.z1, self.z2 = z1, z2
        return np.asarray(out, np.float32)


def _rbj(kind, f0, gain_db=0.0, q=0.7071):
    f0 = _clamp(f0, 20.0, SR * 0.45)
    w0 = 2.0 * math.pi * f0 / SR
    c, s = math.cos(w0), math.sin(w0)
    if kind == "hp":
        al = s / (2.0 * q)
        return ((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al)
    A = 10.0 ** (gain_db / 40.0)
    if kind == "peak":
        al = s / (2.0 * q)
        return (1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A)
    sq = 2.0 * math.sqrt(A) * (s / 2.0 * math.sqrt(2.0))
    if kind == "low":
        return (A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c),
                A * ((A + 1) - (A - 1) * c - sq), (A + 1) + (A - 1) * c + sq,
                -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq)
    return (A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c),   # "high"
            A * ((A + 1) + (A - 1) * c - sq), (A + 1) - (A - 1) * c + sq,
            2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq)


class _Granular:
    """Two-tap delay-line pitch shifter (Hann cross-fade, 50 % overlap)."""

    def __init__(self, win):
        self.D = float(win)
        self.H = int(win) + 8
        self.hist = np.zeros(self.H, np.float32)
        self.ph = 0.0

    def run(self, x, ratio):
        n = len(x)
        buf = np.concatenate((self.hist, x))
        i = np.arange(n, dtype=np.float64)
        p1 = (self.ph + i * (1.0 - ratio) / self.D) % 1.0
        p2 = (p1 + 0.5) % 1.0
        out = np.zeros(n, np.float64)
        for p in (p1, p2):
            pos = self.H + i - (2.0 + p * self.D)
            i0 = np.floor(pos).astype(np.int64)
            fr = pos - i0
            v = buf[i0] * (1.0 - fr) + buf[np.minimum(i0 + 1, len(buf) - 1)] * fr
            out += v * (np.sin(np.pi * p) ** 2)
        self.ph = float((self.ph + n * (1.0 - ratio) / self.D) % 1.0)
        self.hist = buf[-self.H:].copy()
        return out.astype(np.float32)


class SafeEngine:
    def __init__(self, params, mode="balanced"):
        self.params = params
        self.peak = 0.0
        self.out_peak = 0.0
        self.mode = None
        self.hop = HOPS["balanced"]
        self._alpha = 0.3
        self._sm = None
        self._gain = None
        self._wet = 0.0
        self._comp_env = -90.0
        self._comp_g = 1.0
        self._gate_g = 1.0
        self._lim_g = 1.0
        self.hp, self.lo, self.pk, self.hi = _Biquad(), _Biquad(), _Biquad(), _Biquad()
        self.set_mode(mode)
        self.latency_samples = int(self.gr.D / 2) + 2

    def set_mode(self, mode):
        self.mode = mode if mode in HOPS else "balanced"
        self.hop = HOPS[self.mode]
        self._alpha = 1.0 - math.exp(-(self.hop / float(SR)) / GLIDE_SECONDS)
        win = {"fast": 960, "balanced": 1440, "safe": 1920}[self.mode]
        if not hasattr(self, "gr") or int(self.gr.D) != win:
            self.gr = _Granular(win)
            self.latency_samples = int(win / 2) + 2

    @staticmethod
    def _ramp(a, b, n):
        return np.linspace(a, b, n, endpoint=False, dtype=np.float32)

    def process(self, x):
        x = np.asarray(x, np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return x
        target, volume = _targets(self.params)
        if self._sm is None:
            self._sm = dict(target)
        else:
            a = self._alpha
            for k, t in target.items():
                c = self._sm[k]
                d = t - c
                self._sm[k] = t if abs(d) < max(1e-3, 1e-3 * abs(t)) else c + d * a
        s = self._sm
        self.peak = float(np.max(np.abs(x)))
        y = x.copy()

        # noise gate (block level, smooth gain)
        if s["gate_db"] > -99.0:
            lvl = 20.0 * math.log10(max(self.peak, 1e-9))
            tg = 1.0 if lvl > s["gate_db"] else 0.0
            ng = tg if tg > self._gate_g else self._gate_g + (tg - self._gate_g) * 0.25
            y *= self._ramp(self._gate_g, ng, n)
            self._gate_g = ng
        else:
            self._gate_g = 1.0

        # low cut
        if s["lowcut_hz"] > 25.0:
            self.hp.enable(True)
            self.hp.set(*_rbj("hp", s["lowcut_hz"]))
            y = self.hp.run(y)
        else:
            self.hp.enable(False)

        # compressor (block peak, smooth gain)
        if s["comp_ratio"] > 1.05 and s["comp_thr"] < -0.5:
            lvl = 20.0 * math.log10(max(float(np.max(np.abs(y))), 1e-9))
            blk = n / float(SR) * 1000.0
            att = 1.0 - math.exp(-blk / max(s["comp_attack"], 1.0))
            rel = 1.0 - math.exp(-blk / max(s["comp_release"], 10.0))
            k = att if lvl > self._comp_env else rel
            self._comp_env += (lvl - self._comp_env) * k
            over = max(0.0, self._comp_env - s["comp_thr"])
            g = 10.0 ** (-(over * (1.0 - 1.0 / s["comp_ratio"])) / 20.0)
            y *= self._ramp(self._comp_g, g, n)
            self._comp_g = g
        else:
            self._comp_g, self._comp_env = 1.0, -90.0

        # pitch (cross-fade to dry when the shift is ~0 to avoid the comb effect of two equal taps)
        semis = s["semitones"]
        want = 1.0 if abs(semis) > 0.02 else 0.0
        wet = self.gr.run(y, 2.0 ** (semis / 12.0))
        w = self._ramp(self._wet, want, n)
        y = wet * w + y * (1.0 - w)
        self._wet = want

        # EQ
        for bq, kind, f, g in ((self.lo, "low", s["low_hz"], s["low_db"]),
                               (self.pk, "peak", s["presence_hz"], s["mid_db"]),
                               (self.hi, "high", s["high_hz"], s["high_db"])):
            if abs(g) > 0.15:
                bq.enable(True)
                bq.set(*_rbj(kind, f, g, 0.9 if kind == "peak" else 0.7071))
                y = bq.run(y)
            else:
                bq.enable(False)

        # volume (ramped) + limiter (block gain, ramped) + hard safety clip
        if self._gain is None:
            self._gain = volume
        y *= self._ramp(self._gain, volume, n)
        self._gain = volume
        ceil = 10.0 ** (s["ceiling"] / 20.0)
        pk = float(np.max(np.abs(y)))
        tg = min(1.0, ceil / pk) if pk > 1e-9 else 1.0
        ng = tg if tg < self._lim_g else self._lim_g + (1.0 - self._lim_g) * 0.05
        y *= self._ramp(self._lim_g, ng, n)
        self._lim_g = ng
        np.nan_to_num(y, copy=False)
        np.clip(y, -1.0, 1.0, out=y)
        self.out_peak = float(np.max(np.abs(y)))
        return y


# =====================================================================================================
#  Engine - what app.py / audio_io.py use.  Picks the pedalboard chain when it really produces sound
#  and switches to SafeEngine (for good) if it fails its check, raises, or stays silent while the mic
#  is loud.  Everything is written to cutevoice.log through `log_fn`.
# =====================================================================================================
log_fn = lambda msg: None          # noqa: E731  (app.py replaces this with its logger)
_pedal_state = {"checked": False, "ok": False, "why": ""}


def _tone(n, f=1000.0, amp=0.5):
    return (amp * np.sin(2.0 * math.pi * f * np.arange(n) / SR)).astype(np.float32)


def _check_pedalboard():
    """True only if the pedalboard chain turns a loud tone into a loud tone (not silence, no exception)."""
    if PEDAL_IMPORT_ERROR is not None:
        return False, f"pedalboard import failed: {PEDAL_IMPORT_ERROR}"
    try:
        for pitch in (0.0, -4.0, 5.0):
            e = PedalEngine(dict(volume=100, pitch=pitch, pers_on=False), "balanced")
            sig = _tone(e.hop * 80)
            outs = []
            for i in range(80):
                outs.append(e.process(sig[i * e.hop:(i + 1) * e.hop]))
            tail = np.concatenate(outs[-30:])
            rms = float(np.sqrt(np.mean(tail ** 2)))
            if not np.isfinite(rms) or rms < 0.05:
                return False, f"pedalboard chain output is silent (pitch {pitch:+.0f}: rms {rms:.5f})"
        return True, ""
    except Exception as ex:
        return False, f"pedalboard chain raised: {ex!r}"


class Engine:
    def __init__(self, params, mode="balanced"):
        self._params = params
        self.notice = ""
        self.backend = "pedalboard"
        self._silent_blocks = 0
        self._impl = None
        if not _pedal_state["checked"]:
            ok, why = _check_pedalboard()
            _pedal_state.update(checked=True, ok=ok, why=why)
            log_fn(f"engine check: {'pedalboard OK' if ok else 'pedalboard FAILED -> SafeEngine (' + why + ')'}")
        if _pedal_state["ok"]:
            try:
                self._impl = PedalEngine(params, mode)
            except Exception as ex:
                self._switch(mode, f"pedalboard init failed: {ex!r}")
        else:
            self._switch(mode, _pedal_state["why"])

    # -- backend switching (never goes back to pedalboard once it misbehaved)
    def _switch(self, mode, why):
        _pedal_state.update(ok=False, why=why)
        self._impl = SafeEngine(self._params, mode)
        self.backend = "safe"
        self.notice = "Safe engine (pedalboard band)"
        self._silent_blocks = 0
        log_fn(f"SafeEngine active: {why}")

    # -- attributes the rest of the app reads / writes
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, v):
        self._params = v
        self._impl.params = v

    @property
    def mode(self):
        return self._impl.mode

    @property
    def hop(self):
        return self._impl.hop

    @property
    def latency_samples(self):
        return self._impl.latency_samples

    @property
    def peak(self):
        return self._impl.peak

    @peak.setter
    def peak(self, v):
        self._impl.peak = v

    @property
    def out_peak(self):
        return self._impl.out_peak

    @out_peak.setter
    def out_peak(self, v):
        self._impl.out_peak = v

    def set_mode(self, mode):
        self._impl.set_mode(mode)

    # -- audio
    def process(self, x):
        impl = self._impl
        if isinstance(impl, PedalEngine):
            try:
                y = impl.process(x)
            except Exception as ex:
                self._switch(impl.mode, f"pedalboard raised while running: {ex!r}")
                return self._impl.process(x)
            try:
                vol = float(self._params.get("volume", 100.0))
            except (TypeError, ValueError):
                vol = 100.0
            if impl.peak > 0.15 and impl.out_peak < 1e-5 and vol >= 5.0:
                self._silent_blocks += 1
                if self._silent_blocks >= 60:                      # ~1.2 s of loud mic and a dead output
                    self._switch(impl.mode, "pedalboard chain stayed silent while the mic was loud")
                    return self._impl.process(x)
            else:
                self._silent_blocks = 0
            return y
        return impl.process(x)
