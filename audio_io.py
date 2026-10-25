"""Audio plumbing for CuteVoice v3:  mic -> engine -> CABLE Input (+ optional headphone monitor).

Why v2 stuttered ("lag in places"):
  * fixed 10-40 ms blocks that did not match the shifter hop, and a whole block was re-primed
    after every hiccup.
  * mic and cable run on different clocks -> the buffer slowly drained / filled -> periodic drop-outs.
v3:
  * device callbacks can be any size, the engine always works on whole shifter hops
  * the output ring watches its (smoothed) fill level and stretches / squeezes the audio by
    1 sample per callback to stay centred, and grows its own cushion if the PC ever stalls
  * worker thread: MMCSS "Pro Audio", high priority, GIL released after 0.5 ms, no power throttling
"""
import copy
import queue
import threading
import time
import traceback

import numpy as np

try:
    import sounddevice as sd
except Exception as _e:          # PortAudio missing -> GUI shows the message
    sd = None
    SD_ERROR = _e
else:
    SD_ERROR = None

import winperf
from engine import SR, Engine

# name -> (engine mode, chunk = this many hops, safety margin of the output cushion in ms)
BUFFERS = {
    "Fast (lowest delay)": ("fast", 1, 4),
    "Balanced (recommended)": ("balanced", 1, 6),
    "Safe (no glitches)": ("safe", 1, 12),
}
DEFAULT_BUFFER = "Balanced (recommended)"


# ------------------------------------------------------------------ ring buffer
class Ring:
    """Thread-safe float32 FIFO between the worker thread and an output callback.

    cushion  = output block + input block + half a hop + `margin`  (sizes measured at run time)
    * playback starts once `cushion` samples are queued
    * drift compensation: the smoothed fill is kept at the cushion by playing 1 sample
      faster / slower per callback (linear-interpolated, inaudible)
    * underrun: plays what is left (short fade), waits for the cushion again and adds to it
      (`extra`); after 45 s of clean audio the extra shrinks again
    """

    def __init__(self, chunk, margin, cap):
        self.buf = np.zeros(cap, np.float32)
        self.cap = cap
        self.chunk = chunk
        self.margin = int(margin)
        self.max_extra = int(chunk * 6)
        self.extra = 0
        self.out_seen = 0
        self.in_seen = 0
        self.r = self.w = self.n = 0
        self.primed = False
        self.fade_in = 0
        self.quiet = 0
        self.underruns = 0
        self.trims = 0
        self.fill_avg = 0.0
        self.active = True                 # False: callback gets silence, writes are ignored
        self.lock = threading.Lock()

    @property
    def target(self):
        return (self.out_seen or 480) + max(self.in_seen, self.chunk) + self.chunk // 2 + self.margin + self.extra

    def clear(self):
        with self.lock:
            self.r = self.w = self.n = 0
            self.primed = False
            self.extra = 0

    def set_active(self, flag):
        """Switch the ring on/off.  Off = the audio callback instantly gets silence (even if it is
        running right now) and the producer's writes are dropped; leftovers are always discarded."""
        with self.lock:
            self.active = bool(flag)
            self.r = self.w = self.n = 0
            self.primed = False
            self.extra = 0
            self.fill_avg = 0.0

    # ---- producer
    def write(self, x):
        x = np.asarray(x, np.float32)
        L = len(x)
        with self.lock:
            if not self.active:
                return
            if L > self.cap:
                x, L = x[-self.cap:], self.cap
            over = self.n + L - self.cap
            if over > 0:                                  # consumer is far behind: drop the oldest audio
                self.r = (self.r + over) % self.cap
                self.n -= over
                self.trims += 1
            end = self.w + L
            if end <= self.cap:
                self.buf[self.w:end] = x
            else:
                k = self.cap - self.w
                self.buf[self.w:] = x[:k]
                self.buf[:end - self.cap] = x[k:]
            self.w = end % self.cap
            self.n += L

    # ---- consumer (audio callback)
    def _take(self, m):
        end = self.r + m
        if end <= self.cap:
            out = self.buf[self.r:end].copy()
        else:
            out = np.concatenate((self.buf[self.r:], self.buf[:end - self.cap]))
        self.r = end % self.cap
        self.n -= m
        return out

    def read(self, k):
        with self.lock:
            if not self.active:                           # monitor switched OFF: never replay buffered audio
                return np.zeros(k, np.float32)
            if k > self.out_seen:
                self.out_seen = k
            target = self.target
            if not self.primed:
                if self.n < target:
                    return np.zeros(k, np.float32)
                self.primed = True
                self.fade_in = 64
                self.fill_avg = float(self.n)

            # smoothed fill (~0.5 s) drives the drift control; instantaneous fill only the safety trim
            self.fill_avg += (self.n - self.fill_avg) * 0.02

            if self.n > target + 3 * self.chunk + k:      # a stall delivered a burst: skip back to the cushion
                drop = self.n - target
                self.r = (self.r + drop) % self.cap
                self.n -= drop
                self.fill_avg = float(self.n)
                self.trims += 1

            band = self.chunk / 2.0
            if self.fill_avg > target + band:
                d = 1                                       # too full  -> play 1 sample faster
            elif self.fill_avg < target - band:
                d = -1                                      # too empty -> play 1 sample slower
            else:
                d = 0
            need = k + d

            if self.n < need:                              # underrun
                out = np.zeros(k, np.float32)
                m = min(self.n, k)
                if m > 0:
                    seg = self._take(m)
                    f = min(m, 48)
                    seg[m - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
                    out[:m] = seg
                self.primed = False
                self.underruns += 1
                self.quiet = 0
                self.extra = min(self.max_extra, self.extra + max(1, self.chunk // 2))
                return out

            seg = self._take(need)
            if d:
                seg = np.interp(np.linspace(0.0, need - 1.0, k), np.arange(need), seg).astype(np.float32)
            if self.fade_in > 0:
                f = min(self.fade_in, k)
                seg[:f] *= np.linspace(1.0 - self.fade_in / 64.0, 1.0, f, dtype=np.float32)
                self.fade_in -= f
            self.quiet += k
            if self.quiet > 45 * SR and self.extra > 0:
                self.extra = max(0, self.extra - max(1, self.chunk // 2))
                self.quiet = 0
            return seg


def warm_up(engine, chunk):
    """Run a throw-away engine for a moment so numpy / FFT caches are hot before real audio arrives
    (the very first block of a cold engine can take 10x longer than normal)."""
    try:
        e = Engine(copy.deepcopy(engine.params), engine.mode)
        z = (np.random.default_rng(0).standard_normal(chunk) * 1e-3).astype(np.float32)
        for _ in range(10):
            e.process(z)
    except Exception:
        pass


# ------------------------------------------------------------------ router
class Router:
    """mic stream -> queue -> worker thread (engine, whole hops) -> cable ring (+ monitor ring)."""

    def __init__(self, engine, buffer_name=DEFAULT_BUFFER):
        mode, mult, extra_ms = BUFFERS.get(buffer_name, BUFFERS[DEFAULT_BUFFER])
        if engine.mode != mode:
            engine.set_mode(mode)
        self.engine = engine
        self.chunk = engine.hop * mult
        margin = int(extra_ms / 1000.0 * SR)
        cap = self.chunk * 32
        self.cable_ring = Ring(self.chunk, margin, cap)
        self.mon_ring = Ring(self.chunk, margin, cap)
        self.mon_ring.active = False                   # headphone ring is silent until a monitor stream is really open
        self.q = queue.Queue(maxsize=128)
        self.monitor_enabled = False
        self.running = False
        self.error = None
        self.streams = []
        self.thread = None
        self.in_over = 0
        self.out_under = 0
        self.proc_ms = 0.0
        self.dev_latency = 0.0
        self.silent_since = None
        self.heard_signal = False
        # one lock serialises start / stop / monitor on/off (all called from the GUI thread, but a
        # double-click or auto-fix must never interleave them)
        self._lock = threading.RLock()
        self._seq_lock = threading.Lock()
        self._mon_seq = 0                # every Hear request / stop() bumps this; an older request that was
                                         # still waiting for the lock sees the change and does nothing
        self._mon_stream = None
        self._kw_base = dict(samplerate=SR, blocksize=0, dtype="float32", latency="low")
        self.cable_dev = None
        self.monitor_error = None

    # ---- callbacks (audio threads: keep tiny)
    def _in_cb(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self.in_over += 1
        if frames > self.cable_ring.in_seen:
            self.cable_ring.in_seen = self.mon_ring.in_seen = frames
        mono = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
        blk = mono.astype(np.float32, copy=True)
        try:
            self.q.put_nowait(blk)
        except queue.Full:                       # worker is far behind: drop the oldest block
            try:
                self.q.get_nowait()
                self.q.put_nowait(blk)
            except (queue.Empty, queue.Full):
                pass

    def _out_cb_factory(self, ring, ch):
        def cb(outdata, frames, time_info, status):
            if status and status.output_underflow:
                self.out_under += 1
            mono = ring.read(frames)
            if ch == 1:
                outdata[:, 0] = mono
            else:
                outdata[:] = mono[:, None]
        return cb

    # ---- worker
    def _worker(self):
        winperf.boost_thread()
        chunk = self.chunk
        acc = np.zeros(0, np.float32)
        while self.running:
            try:
                blk = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            blks = [blk]
            while True:                              # catch up on anything that piled up
                try:
                    blks.append(self.q.get_nowait())
                except queue.Empty:
                    break
            acc = np.concatenate([acc] + blks)
            while len(acc) >= chunk:
                c, acc = acc[:chunk], acc[chunk:]
                t0 = time.perf_counter()
                try:
                    y = self.engine.process(c)
                    self.error = None
                except Exception:
                    if self.error is None:
                        try:
                            import engine as _em
                            _em.log_fn("worker engine error:\n" + traceback.format_exc())
                        except Exception:
                            pass
                    self.error = "Engine error (cutevoice.log dekho)"
                    y = np.zeros(len(c), np.float32)          # never leak the raw voice
                dt = (time.perf_counter() - t0) * 1000.0
                self.proc_ms += (dt - self.proc_ms) * 0.05
                self.cable_ring.write(y)
                if self.monitor_enabled:
                    self.mon_ring.write(y)               # ring ignores it unless the monitor stream is open
                if self.engine.peak > 0.0:
                    self.heard_signal = True

    # ---- control
    @staticmethod
    def _api_name(dev):
        try:
            return sd.query_hostapis(sd.query_devices(dev)["hostapi"])["name"]
        except Exception:
            return ""

    def _kw_for(self, dev):
        """Stream settings for one device: WASAPI shared-mode auto-convert only on WASAPI devices."""
        kw = dict(self._kw_base)
        if "WASAPI" in self._api_name(dev) and hasattr(sd, "WasapiSettings"):
            try:
                kw["extra_settings"] = sd.WasapiSettings(auto_convert=True)
            except TypeError:
                pass
        return kw

    @staticmethod
    def _out_channels(dev):
        return 2 if sd.query_devices(dev)["max_output_channels"] >= 2 else 1

    def start(self, api_name, in_dev, cable_dev, mon_dev, monitor=False):
        """`monitor` = the person wants Hear-myself ON.  monitor_enabled only becomes True after the
        headphone stream has really been opened (see set_monitor) - never before."""
        if sd is None:
            raise RuntimeError(f"PortAudio / sounddevice load nahi hua: {SD_ERROR}")
        with self._lock:
            if self.running:
                return                                   # never open a second set of streams
            self.monitor_enabled = False
            self.mon_ring.set_active(False)
            # Voice OUT must be a virtual cable.  A real speaker / headphone here would play the
            # processed voice straight into the room, so refuse to start at all.
            try:
                cable_name = sd.query_devices(cable_dev)["name"]
            except Exception as ex:
                raise RuntimeError(f"Voice OUT device nahi khula: {ex}")
            if not is_virtual_output(cable_name):
                raise RuntimeError(
                    f"Voice OUT galat hai: '{cable_name}' asli speaker/headphone hai.\n"
                    "Voice OUT -> Cable mein 'CABLE Input (VB-Audio Virtual Cable)' chuno. "
                    "Headphone mein khud sunne ke liye 'Hear myself' use karo.")
            for d, what in ((in_dev, "Mic"), (cable_dev, "Voice OUT")):
                if "wdm" in self._api_name(d).lower():
                    raise RuntimeError(f"{what} WDM-KS audio system par hai (exclusive, aawaaz nahi aati). "
                                       "Audio system mein WASAPI / DirectSound / MME chuno.")
            self.cable_dev = cable_dev

            warm_up(self.engine, self.chunk)
            self.streams = []
            self.heard_signal = False
            self.silent_since = time.time()
            self.in_over = self.out_under = 0
            self.monitor_error = None
            try:
                try:
                    s_in = sd.InputStream(device=in_dev, channels=1, callback=self._in_cb, **self._kw_for(in_dev))
                except Exception:
                    ch = min(2, max(1, sd.query_devices(in_dev)["max_input_channels"]))
                    s_in = sd.InputStream(device=in_dev, channels=ch, callback=self._in_cb, **self._kw_for(in_dev))
                self.streams.append(s_in)

                c = self._out_channels(cable_dev)
                s_cab = sd.OutputStream(device=cable_dev, channels=c,
                                        callback=self._out_cb_factory(self.cable_ring, c), **self._kw_for(cable_dev))
                self.streams.append(s_cab)
                self.dev_latency = float(s_in.latency) + float(s_cab.latency)
                self.running = True
                self.thread = threading.Thread(target=self._worker, name="cv-worker", daemon=True)
                self.thread.start()
                for s in self.streams:
                    s.start()
            except Exception:
                self.stop()
                raise
            # headphone monitor is its own stream, opened only while "Hear myself" is ON
            if monitor:
                ok, msg = self.set_monitor(True, mon_dev)
                if not ok:
                    self.monitor_error = msg

    def set_monitor(self, enabled, mon_dev=None):
        """Hear myself ON/OFF.  Returns (ok, message).  ON opens one headphone stream, OFF closes it.

        Race safety:
          * every call takes a sequence number; if a newer call (or stop()) arrived while this one was
            waiting for the lock, this one does nothing - an old request can never re-activate the monitor
          * OFF first silences the ring (a callback that is running right now gets zeros, buffered audio is
            thrown away), then the stream is closed
          * monitor_enabled turns True only after the stream really started
        """
        with self._seq_lock:
            self._mon_seq += 1
            seq = self._mon_seq
        with self._lock:
            if seq != self._mon_seq:                     # superseded while waiting
                return True, ""
            if not enabled:
                self._monitor_off()
                return True, ""
            if not self.running:
                self._monitor_off()
                return False, "Pehle START dabao."
            if mon_dev is None:
                self._monitor_off()
                return False, "Monitor (Headphones) device select karo."
            if mon_dev == self.cable_dev:
                self._monitor_off()
                return False, "Monitor device cable wala nahi ho sakta - headphones select karo."
            try:
                if is_virtual_output(sd.query_devices(mon_dev)["name"]):
                    self._monitor_off()
                    return False, "Monitor ke liye cable device nahi chalega - asli headphones select karo."
            except Exception as ex:
                self._monitor_off()
                return False, f"Monitor device nahi khula: {ex}"
            if self._mon_stream is not None and self.monitor_enabled:      # already on
                return True, ""
            self._monitor_off()                          # clean slate: no old stream, silent empty ring
            stream, last_err = None, None
            for dev in self._monitor_candidates(mon_dev):
                s = None
                try:
                    m = self._out_channels(dev)
                    s = sd.OutputStream(device=dev, channels=m,
                                        callback=self._out_cb_factory(self.mon_ring, m), **self._kw_for(dev))
                    s.start()
                    stream = s
                    break
                except Exception as ex:
                    last_err = f"{self._api_name(dev) or '?'} [{dev}]: {ex}"
                    if s is not None:
                        try:
                            s.close()
                        except Exception:
                            pass
            if stream is None:
                self._monitor_off()
                return False, f"Monitor start nahi hua (kisi bhi audio system par nahi khula).\n{last_err}"
            self._mon_stream = stream
            self.mon_ring.set_active(True)               # only now may audio flow to the headphones
            self.monitor_enabled = True
            return True, ""

    def _monitor_candidates(self, mon_dev):
        """The chosen headphone device first, then the same physical device on the other host APIs
        (WASAPI -> DirectSound -> MME).  WDM-KS is never used: it is exclusive and often fails to open."""
        cands = []
        try:
            name = sd.query_devices(mon_dev)["name"]
            if "wdm" not in self._api_name(mon_dev).lower():
                cands.append(mon_dev)
            apis = hostapis()
            for key in API_ORDER:
                for idx, nm in enumerate(apis):
                    if key.lower() in nm.lower():
                        d = find_equivalent(name, idx, False)
                        if d is not None and d not in cands and not is_virtual_output(sd.query_devices(d)["name"]):
                            cands.append(d)
        except Exception:
            pass
        return cands or [mon_dev]

    def _monitor_off(self):
        """Caller holds self._lock.  Order matters: stop feeding -> silence ring -> close stream."""
        self.monitor_enabled = False
        self.mon_ring.set_active(False)
        self._close_monitor()

    def _close_monitor(self):
        s, self._mon_stream = self._mon_stream, None
        if s is not None:
            try:
                s.abort()
            except Exception:
                pass
            try:
                s.close()
            except Exception:
                pass

    def stop(self):
        with self._seq_lock:
            self._mon_seq += 1                           # cancels any Hear request still waiting
        with self._lock:
            self.running = False
            self.monitor_enabled = False
            self.mon_ring.set_active(False)
            # 1) worker first (nothing writes into the rings any more) ...
            t, self.thread = self.thread, None
            if t is not None and t is not threading.current_thread():
                t.join(timeout=2.0)
            # 2) ... then every stream ...
            self._close_monitor()
            for s in self.streams:
                try:
                    s.abort()
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
            self.streams = []
            # 3) ... then reset the buffers
            self.cable_ring.clear()
            self.mon_ring.set_active(False)
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

    # ---- stats for the GUI
    def delay_ms(self):
        """Rough mic -> cable delay: devices + hop wait + shifter + cushion."""
        e = self.engine
        samples = e.latency_samples + self.chunk / 2.0 + self.cable_ring.fill_avg
        return self.dev_latency * 1000.0 + samples / SR * 1000.0

    def load_pct(self):
        return self.proc_ms / (self.chunk / SR * 1000.0) * 100.0

    @property
    def dropouts(self):
        return self.cable_ring.underruns + self.in_over


# ------------------------------------------------------------------ devices
SKIP_NAMES = ("Sound Mapper", "Primary Sound")
API_ORDER = ("WASAPI", "DirectSound", "MME")


def is_cable_output(name):
    n = name.lower()
    return "cable" in n and ("output" in n or "cable out" in n)


def is_cable_input(name):
    n = name.lower()
    return "cable" in n and ("input" in n or "cable in" in n)


def is_virtual_output(name):
    """True only for virtual-cable style playback devices (VB-CABLE, VoiceMeeter ...).
    A real speaker / headphone must NEVER be used as 'Voice OUT'."""
    n = name.lower()
    return is_cable_input(name) or "vb-audio" in n or "voicemeeter" in n


def hostapis():
    return [a["name"] for a in sd.query_hostapis()]


def usable_hostapis():
    """Host APIs the app may use (WDM-KS is exclusive-mode and breaks routing -> never offered)."""
    return [n for n in hostapis() if "wdm" not in n.lower()]


def collect_devices(api_idx):
    ins, outs = [], []
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != api_idx or any(s in d["name"] for s in SKIP_NAMES):
            continue
        label = f"{i}: {d['name']}"
        if d["max_input_channels"] > 0 and not is_cable_output(d["name"]):
            ins.append((i, label))                      # never offer the cable's own mic side as our mic
        if d["max_output_channels"] > 0:
            outs.append((i, label))
    return ins, outs


def label_name(label):
    return label.split(": ", 1)[1] if ": " in label else label


def find_equivalent(name, api_idx, want_input):
    """Same physical device on another host API (names differ slightly / are truncated in MME)."""
    key = "max_input_channels" if want_input else "max_output_channels"
    devs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["hostapi"] == api_idx and d[key] > 0]
    lname = name.lower()
    for i, d in devs:
        if d["name"].lower() == lname:
            return i
    p = lname[:24]
    for i, d in devs:
        if d["name"].lower().startswith(p) or lname.startswith(d["name"].lower()[:24]):
            return i
    return None


def reinit_portaudio():
    """Re-scan audio devices (needed after VB-Cable was installed while the app was open)."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass
