"""
CuteVoice Studio v3 - real-time voice changer for Windows

    Mic  ->  [engine thread]  ->  CABLE Input (always)  ->  WhatsApp / OBS / Discord (mic = CABLE Output)
                              \\->  Headphones (optional "Hear myself")

Files:  app.py (GUI)  engine.py (voice DSP)  audio_io.py (devices / buffers)  winperf.py (Windows tweaks)
Log:    cutevoice.log  (next to this file)
"""
import copy
import json
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
LOG_FILE = os.path.join(BASE_DIR, "cutevoice.log")

# pythonw.exe has no console: send prints / tracebacks to a log file instead of losing them
try:
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 500000:
        os.remove(LOG_FILE)
    if sys.stderr is None or sys.stdout is None:
        _lf = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or _lf
        sys.stderr = sys.stderr or _lf
except Exception:
    pass


def log(msg):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg) + "\n")
    except Exception:
        pass


sys.path.insert(0, BASE_DIR)
import numpy as np                                   # noqa: E402,F401
import audio_io                                      # noqa: E402
import winperf                                       # noqa: E402
from audio_io import (Router, BUFFERS, DEFAULT_BUFFER, API_ORDER, collect_devices, find_equivalent,  # noqa: E402
                      hostapis, is_cable_input, is_virtual_output, label_name, reinit_portaudio,
                      usable_hostapis)
import engine as engine_mod                          # noqa: E402
from engine import Engine                            # noqa: E402

engine_mod.log_fn = log                              # engine check / fallback messages go to cutevoice.log

AUTO_API = "Auto (best)"

# ------------------------------------------------------------------ voices
BASE = dict(volume=100, pitch=0, timbre=50,
            pers_on=False, thick=0, persona=0, mix=100,
            gate=-80, lowcut=20, bass=0, mid=0, treble=0,
            comp_thr=0, comp_ratio=1, reverb=0,
            # --- pro controls (v4)
            low_hz=150, high_hz=3000, presence_hz=2500, clarity=100,
            chorus_mix=0, chorus_depth=25, chorus_rate=0.8, vib_depth=0, vib_rate=5,
            comp_attack=5, comp_release=120, room=35, ceiling=-1)

FACTORY = {
    "Original": dict(BASE),
    # Natural presets: moderate pitch shift + matching formant shift = most believable result.
    "Natural Female": dict(BASE, volume=125, pitch=5, timbre=70, gate=-60, lowcut=110, bass=-2, mid=1.5, treble=1.5,
                           comp_thr=-24, comp_ratio=2),
    "Natural Male": dict(BASE, volume=125, pitch=-4, timbre=36, gate=-60, lowcut=70, bass=3, mid=0.5, treble=-1,
                         comp_thr=-24, comp_ratio=2),
    "Deep Voice": dict(BASE, pitch=-4, timbre=38, pers_on=True, thick=35, persona=-22, mix=68,
                       gate=-65, lowcut=75, bass=4.5, mid=1, treble=-1.5,
                       comp_thr=-24, comp_ratio=2.5, reverb=2),
    "Soft Girl": dict(BASE, pitch=4.5, timbre=64, pers_on=True, thick=-22, persona=28, mix=58,
                      gate=-65, lowcut=100, bass=0, mid=1, treble=1.5,
                      comp_thr=-24, comp_ratio=2, reverb=1),
    "Cute Girl": dict(BASE, pitch=6, timbre=72, pers_on=True, thick=-40, persona=40, mix=100,
                      gate=-55, lowcut=140, bass=-3, treble=3,
                      comp_thr=-20, comp_ratio=3, reverb=6),
}

# (label, key, min, max, step, display-format)
ROWS_ORIGINAL = [
    ("Volume", "volume", 0, 200, 1, "{:.0f}%"),
    ("Pitch", "pitch", -12, 12, 0.5, "{:+.1f}"),
    ("Timbre", "timbre", 0, 100, 1, "{:.0f}"),
]
ROWS_PERSONALITY = [
    ("Vocal thickness", "thick", -100, 100, 1, "{:.0f}"),
    ("Persona", "persona", -100, 100, 1, "{:.0f}"),
    ("Strength (Mix)", "mix", 0, 100, 1, "{:.0f}"),
]
ROWS_ADV_LEFT = [
    ("Noise Gate (min = off)", "gate", -80, -20, 1, "{:.0f} dB"),
    ("Low Cut", "lowcut", 20, 300, 5, "{:.0f} Hz"),
    ("Bass", "bass", -12, 12, 0.5, "{:+.1f} dB"),
    ("Mid / Presence", "mid", -12, 12, 0.5, "{:+.1f} dB"),
]
ROWS_ADV_RIGHT = [
    ("Treble / Air", "treble", -12, 12, 0.5, "{:+.1f} dB"),
    ("Compressor Threshold", "comp_thr", -40, 0, 1, "{:.0f} dB"),
    ("Compressor Ratio", "comp_ratio", 1, 10, 0.5, "{:.1f}:1"),
    ("Reverb", "reverb", 0, 50, 1, "{:.0f}%"),
]

ROWS_PRO_TONE = [
    ("Bass Frequency (low shelf)", "low_hz", 80, 300, 5, "{:.0f} Hz"),
    ("Treble Frequency (high shelf)", "high_hz", 1500, 8000, 100, "{:.0f} Hz"),
    ("Presence Frequency", "presence_hz", 1000, 5000, 100, "{:.0f} Hz"),
    ("Auto Clarity (deep voice)", "clarity", 0, 100, 1, "{:.0f}%"),
]
ROWS_PRO_FX = [
    ("Chorus Mix", "chorus_mix", 0, 100, 1, "{:.0f}%"),
    ("Chorus Depth", "chorus_depth", 0, 100, 1, "{:.0f}%"),
    ("Chorus Speed", "chorus_rate", 0.1, 5, 0.1, "{:.1f} Hz"),
    ("Vibrato Depth", "vib_depth", 0, 100, 1, "{:.0f}%"),
    ("Vibrato Speed", "vib_rate", 1, 10, 0.5, "{:.1f} Hz"),
]
ROWS_PRO_DYN = [
    ("Compressor Attack", "comp_attack", 1, 100, 1, "{:.0f} ms"),
    ("Compressor Release", "comp_release", 20, 500, 10, "{:.0f} ms"),
    ("Reverb Size", "room", 0, 100, 1, "{:.0f}%"),
    ("Limiter Ceiling", "ceiling", -12, 0, 0.5, "{:.1f} dB"),
]

# ------------------------------------------------------------------ theme
BG, PANEL, FG, MUTED = "#141414", "#1e1e1e", "#ffffff", "#8c8c8c"
ACC, TRACK, THUMB = "#6c5ce7", "#3a3a3a", "#d4d0ff"
OK, BAD = "#2ecc71", "#ff6b6b"


PRO_CLOSED = ">  Pro Controls (frequencies / chorus / vibrato / compressor)  - right side"
PRO_OPEN = "v  Pro Controls (frequencies / chorus / vibrato / compressor)  - right side"


class Slider(tk.Canvas):
    """Flat dark slider with purple fill. Double-click = reset to neutral."""
    W, H, PAD = 360, 46, 10

    def __init__(self, parent, row, on_change):
        super().__init__(parent, width=self.W, height=self.H, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.label, self.key, self.lo, self.hi, self.step, self.fmt = row
        self.value = BASE[self.key]
        self.on_change = on_change
        self.bind("<Button-1>", self._mouse)
        self.bind("<B1-Motion>", self._mouse)
        self.bind("<Double-Button-1>", self._reset)
        self.draw()

    def _frac(self):
        return (self.value - self.lo) / float(self.hi - self.lo)

    def draw(self):
        self.delete("all")
        x0, x1, y = self.PAD, self.W - self.PAD, 32
        xv = x0 + self._frac() * (x1 - x0)
        self.create_text(self.PAD, 11, text=self.label, anchor="w", fill=FG, font=("Segoe UI", 10))
        self.create_text(self.W - self.PAD, 11, text=self.fmt.format(self.value), anchor="e", fill=FG,
                         font=("Segoe UI", 10))
        self.create_line(x0, y, x1, y, fill=TRACK, width=4, capstyle="round")
        self.create_line(x0, y, xv, y, fill=ACC, width=4, capstyle="round")
        self.create_oval(xv - 7, y - 7, xv + 7, y + 7, fill=THUMB, outline=THUMB)

    def set(self, v, notify=False):
        v = min(self.hi, max(self.lo, v))
        v = round(v / self.step) * self.step
        v = min(self.hi, max(self.lo, v))
        self.value = v
        self.draw()
        if notify:
            self.on_change(self.key, v)

    def _mouse(self, e):
        frac = (e.x - self.PAD) / float(self.W - 2 * self.PAD)
        frac = min(1.0, max(0.0, frac))
        self.set(self.lo + frac * (self.hi - self.lo), notify=True)

    def _reset(self, _e=None):
        self.set(BASE[self.key], notify=True)


class Toggle(tk.Canvas):
    def __init__(self, parent, command):
        super().__init__(parent, width=46, height=24, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.value, self.command = False, command
        self.bind("<Button-1>", self._click)
        self.draw()

    def draw(self):
        self.delete("all")
        col = ACC if self.value else TRACK
        self.create_oval(1, 2, 23, 22, fill=col, outline=col)
        self.create_oval(23, 2, 45, 22, fill=col, outline=col)
        self.create_rectangle(12, 2, 34, 22, fill=col, outline=col)
        cx = 34 if self.value else 12
        self.create_oval(cx - 8, 4, cx + 8, 20, fill="white", outline="white")

    def set(self, v):
        self.value = bool(v)
        self.draw()

    def _click(self, _e=None):
        self.value = not self.value
        self.draw()
        self.command(self.value)



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CuteVoice Studio v4 - Real-time Voice Changer")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.build_scroll_area()

        self.voices = copy.deepcopy(FACTORY)
        self.voice = "Original"
        self.engine = Engine(self.voices[self.voice])
        self.router = None
        self.active_api = ""
        self.listed_api = ""
        self.ins, self.outs, self.cable_outs, self.mon_outs = [], [], [], []
        self.shown_in = self.shown_out = 0.0
        self.sliders = {}
        self.last_drop_count = 0
        self.last_drop_time = 0.0
        self._mon_busy = False                  # a Hear on/off request is being handled
        self._mon_gen = 0                       # bumped on START / STOP: results of older Hear requests are ignored
        self._mon_q = queue.Queue()             # results from the monitor worker thread -> Tk thread
        self.saved = self.load_settings()

        self.build_style()
        self.build_devices()
        self.build_voice_bar()
        self.build_main()
        self.build_advanced()
        self.build_pro()
        self.build_footer()
        first_voice = self.saved.get("voice") if self.saved.get("voice") in FACTORY else "Natural Female"
        self.select_voice(first_voice)
        self.fit_window()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.report_callback_exception = self.on_tk_error
        self.after(60, self.tick)
        self.after(80, self._poll_monitor)
        if self.autostart_var.get():
            self.after(700, self.auto_start)

    def on_tk_error(self, exc, val, tb):
        log("".join(traceback.format_exception(exc, val, tb)))

    # ---------------------------------------------------------------- settings
    def load_settings(self):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        try:
            for name, vals in data.get("voices", {}).items():
                if name in self.voices:
                    self.voices[name].update({k: v for k, v in vals.items() if k in BASE})
        except Exception:
            pass
        return data

    def save_settings(self):
        try:
            data = dict(
                voice=self.voice, voices=self.voices,
                api=self.api_box.get(), mic=label_name(self.in_box.get()),
                cable=label_name(self.cable_box.get()), monitor_dev=label_name(self.mon_box.get()),
                monitor_on=bool(self.router and self.router.monitor_enabled) or self.monitor_wanted,
                buffer=self.buf_box.get(), autostart=bool(self.autostart_var.get()))
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
        except Exception:
            pass

    # ---------------------------------------------------------------- UI
    def build_style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=FG,
                     arrowcolor=FG, bordercolor=TRACK, lightcolor=PANEL, darkcolor=PANEL)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", FG)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACC)

    def panel(self, parent, side=None):
        f = tk.Frame(parent, bg=PANEL, padx=10, pady=8)
        if side:
            f.pack(side=side, fill="both", expand=True, padx=4)
        return f

    def build_devices(self):
        f = self.panel(self.left)
        f.pack(fill="x", padx=10, pady=(10, 4))
        self.monitor_wanted = bool(self.saved.get("monitor_on", False))

        def lbl(text, r):
            tk.Label(f, text=text, bg=PANEL, fg=FG).grid(row=r, column=0, sticky="w", pady=3)

        apis = [AUTO_API] + usable_hostapis()
        lbl("Audio system", 0)
        self.api_box = ttk.Combobox(f, state="readonly", width=64, values=apis)
        self.api_box.grid(row=0, column=1, padx=8)
        want = self.saved.get("api")
        self.api_box.set(want if want in apis else AUTO_API)
        self.api_box.bind("<<ComboboxSelected>>", lambda _e: self.refresh_devices())

        lbl("Mic (Input)", 1)
        self.in_box = ttk.Combobox(f, state="readonly", width=64)
        self.in_box.grid(row=1, column=1, padx=8)
        lbl("Voice OUT -> Cable", 2)
        self.cable_box = ttk.Combobox(f, state="readonly", width=64)
        self.cable_box.grid(row=2, column=1, padx=8)
        lbl("Monitor (Headphones only)", 3)
        self.mon_box = ttk.Combobox(f, state="readonly", width=64)
        self.mon_box.grid(row=3, column=1, padx=8)
        lbl("Delay / Quality", 4)
        self.buf_box = ttk.Combobox(f, state="readonly", width=28, values=list(BUFFERS))
        self.buf_box.grid(row=4, column=1, padx=8, sticky="w")
        self.buf_box.set(self.saved.get("buffer") if self.saved.get("buffer") in BUFFERS else DEFAULT_BUFFER)

        self.autostart_var = tk.BooleanVar(value=bool(self.saved.get("autostart", True)))
        tk.Checkbutton(f, text="App khulte hi START (auto)", variable=self.autostart_var, bg=PANEL, fg=FG,
                       selectcolor=TRACK, activebackground=PANEL, activeforeground=FG,
                       highlightthickness=0, bd=0).grid(row=4, column=1, sticky="e", padx=8)

        self.cable_status = tk.Label(f, bg=PANEL, fg=MUTED, justify="left", wraplength=700, anchor="w")
        self.cable_status.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.warn = tk.Label(f, bg=PANEL, fg="#f1c40f", justify="left", wraplength=700, anchor="w", cursor="hand2")
        self.warn.grid(row=6, column=0, columnspan=2, sticky="w")
        self.warn.bind("<Button-1>", lambda _e: self.open_mic_privacy())

        self.start_btn = tk.Button(f, text="START", width=12, bg=ACC, fg="white", relief="flat",
                                   activebackground="#8579f0", activeforeground="white",
                                   font=("Segoe UI", 10, "bold"), command=self.toggle_stream)
        self.start_btn.grid(row=0, column=2, rowspan=3, padx=10, ipady=6)
        self.monitor_btn = tk.Button(f, text="", width=16, relief="flat", fg=FG, command=self.toggle_monitor)
        self.monitor_btn.grid(row=3, column=2, padx=10)
        self.fix_btn = tk.Button(f, text="Auto-fix / Rescan", width=16, relief="flat", bg=TRACK, fg=FG,
                                 activebackground=ACC, activeforeground="white", command=self.autofix)
        self.fix_btn.grid(row=4, column=2, padx=10)
        self.update_monitor_btn()
        self.refresh_devices(first=True)

    def _fill(self, box, items, want_name, prefer):
        box["values"] = [l for _, l in items]
        if want_name:
            w = want_name.lower()
            for n, (_, l) in enumerate(items):          # exact, then same-start (MME truncates names)
                if label_name(l).lower() == w:
                    box.current(n)
                    return
            for n, (_, l) in enumerate(items):
                if label_name(l).lower()[:24] == w[:24] and len(w) >= 8:
                    box.current(n)
                    return
        for n, (_, l) in enumerate(items):
            if prefer(l):
                box.current(n)
                return
        if items:
            box.current(0)

    def effective_api(self):
        names = hostapis()
        sel = self.api_box.get()
        if sel in names and "wdm" not in sel.lower():
            return sel
        for key in API_ORDER:
            for nm in names:
                if key.lower() in nm.lower():
                    return nm
        return names[0]

    def refresh_devices(self, first=False):
        names = hostapis()
        self.listed_api = self.effective_api()
        api_idx = names.index(self.listed_api)
        apis = audio_io.sd.query_hostapis()
        keep = self.saved if first else dict(mic=label_name(self.in_box.get()) if self.in_box.get() else "",
                                             cable=label_name(self.cable_box.get()) if self.cable_box.get() else "",
                                             monitor_dev=label_name(self.mon_box.get()) if self.mon_box.get() else "")
        self.ins, self.outs = collect_devices(api_idx)
        virt = [(i, l) for i, l in self.outs if is_virtual_output(label_name(l))]
        self.cable_outs = virt if virt else list(self.outs)      # real speakers are never offered as Voice OUT
        self.mon_outs = [(None, "(none)")] + [(i, l) for i, l in self.outs if not is_virtual_output(label_name(l))]
        def_in = apis[api_idx]["default_input_device"]
        def_out = apis[api_idx]["default_output_device"]
        self._fill(self.in_box, self.ins, keep.get("mic"), lambda l: l.startswith(f"{def_in}:"))
        if any("cable input" in l.lower() for _, l in self.cable_outs):
            cable_pref = lambda l: "cable input" in l.lower()                    # noqa: E731
        else:
            cable_pref = lambda l: is_cable_input(label_name(l))                 # noqa: E731
        self._fill(self.cable_box, self.cable_outs, keep.get("cable"), cable_pref)
        self._fill(self.mon_box, self.mon_outs, keep.get("monitor_dev"), lambda l: l.startswith(f"{def_out}:"))
        self.cable_found = bool(virt)
        if self.cable_found:
            self.cable_status.config(fg=OK, text="OK: VB-Cable mila. WhatsApp / OBS mein mic = 'CABLE Output (VB-Audio Virtual Cable)'.")
        else:
            self.cable_status.config(fg=BAD, text="VB-Cable nahi mila. 'Auto-fix / Rescan' dabao (ya SETUP.bat chalao), "
                                                  "phir agar zaroorat ho to PC restart karo.")

    def build_voice_bar(self):
        bar = tk.Frame(self.left, bg=BG)
        bar.pack(fill="x", padx=10, pady=4)
        tk.Label(bar, text="Voice:", bg=BG, fg=MUTED).pack(side="left", padx=(4, 8))
        self.voice_btns = {}
        for name in FACTORY:
            b = tk.Button(bar, text=name, width=13, relief="flat", bg=TRACK, fg=FG,
                          activeforeground=FG, font=("Segoe UI", 9, "bold"),
                          command=lambda n=name: self.select_voice(n))
            b.pack(side="left", padx=3, ipady=3)
            self.voice_btns[name] = b
        tk.Button(bar, text="Reset this voice", relief="flat", bg=TRACK, fg=FG,
                  command=self.reset_voice).pack(side="right", padx=4, ipady=3)

    def build_main(self):
        row = tk.Frame(self.left, bg=BG)
        row.pack(fill="x", padx=6, pady=4)
        left, right = self.panel(row, "left"), self.panel(row, "left")

        head = tk.Frame(left, bg=PANEL)
        head.pack(fill="x")
        self.pers_toggle = Toggle(head, self.on_pers_toggle)
        self.pers_toggle.pack(side="left")
        tk.Label(head, text="Personality", bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold")).pack(side="left", padx=10)
        tk.Label(left, bg=PANEL, fg=MUTED, justify="left", wraplength=350,
                 text="Extra pitch/formant/body offset on the same voice (no second copy mixed in, so it stays clean).").pack(
            anchor="w", pady=(6, 6))
        for r in ROWS_PERSONALITY:
            self.add_slider(left, r)

        self.title_lbl = tk.Label(right, text="Original", bg=PANEL, fg=FG, font=("Segoe UI", 13, "bold"))
        self.title_lbl.pack(pady=(0, 14))
        for r in ROWS_ORIGINAL:
            self.add_slider(right, r)

    def build_advanced(self):
        self.adv_btn = tk.Button(self.left, text="v  Advanced Voice Edit", relief="flat", bg=BG, fg=FG,
                                 anchor="w", font=("Segoe UI", 10, "bold"), command=self.toggle_adv)
        self.adv_btn.pack(fill="x", padx=10, pady=(6, 0))
        self.adv = tk.Frame(self.left, bg=BG)
        self.adv.pack(fill="x", padx=6, pady=4)
        for rows in (ROWS_ADV_LEFT, ROWS_ADV_RIGHT):
            p = self.panel(self.adv, "left")
            for r in rows:
                self.add_slider(p, r)
        self.adv_open = True

    def build_pro(self):
        self.pro_btn = tk.Button(self.left, text=PRO_CLOSED, relief="flat", bg=BG, fg=FG, anchor="w",
                                 font=("Segoe UI", 10, "bold"), command=self.toggle_pro)
        self.pro_btn.pack(fill="x", padx=10, pady=(6, 0))
        self.pro = tk.Frame(self.right, bg=BG)               # right-side column
        for rows in (ROWS_PRO_TONE, ROWS_PRO_FX, ROWS_PRO_DYN):
            p = self.panel(self.pro)
            p.pack(side="top", fill="x", pady=3)
            for r in rows:
                self.add_slider(p, r)
        self.pro_open = False
        # wide screen -> open Pro Controls on the right from the start
        if self.winfo_screenwidth() >= 1280:
            self.toggle_pro(fit=False)

    def build_footer(self):
        f = tk.Frame(self.left, bg=BG)
        f.pack(fill="x", padx=10, pady=(2, 10))
        tk.Label(f, text="Mic", bg=BG, fg=MUTED).pack(side="left")
        self.meter_in = tk.Canvas(f, width=170, height=10, bg=TRACK, highlightthickness=0)
        self.meter_in.pack(side="left", padx=(6, 14))
        tk.Label(f, text="To cable", bg=BG, fg=MUTED).pack(side="left")
        self.meter_out = tk.Canvas(f, width=170, height=10, bg=TRACK, highlightthickness=0)
        self.meter_out.pack(side="left", padx=(6, 14))
        self.status = tk.Label(f, text="Stopped", bg=BG, fg=BAD, font=("Segoe UI", 10, "bold"))
        self.status.pack(side="right")

    def add_slider(self, parent, row):
        s = Slider(parent, row, self.on_slider)
        s.pack(pady=1)
        self.sliders[row[1]] = s

    # ---------------------------------------------------------------- actions
    def on_slider(self, key, value):
        self.voices[self.voice][key] = value

    def on_pers_toggle(self, on):
        self.voices[self.voice]["pers_on"] = bool(on)

    def select_voice(self, name):
        self.voice = name
        params = self.voices[name]
        self.engine.params = params
        self.title_lbl.config(text=name)
        for n, b in self.voice_btns.items():
            b.config(bg=ACC if n == name else TRACK)
        self.refresh_controls()

    def refresh_controls(self):
        params = self.voices[self.voice]
        for key, s in self.sliders.items():
            s.set(params[key])
        self.pers_toggle.set(params["pers_on"])

    def reset_voice(self):
        self.voices[self.voice].update(copy.deepcopy(FACTORY[self.voice]))
        self.refresh_controls()

    def toggle_adv(self):
        if self.adv_open:
            self.adv.pack_forget()
            self.adv_btn.config(text=">  Advanced Voice Edit")
        else:
            self.adv.pack(fill="x", padx=6, pady=4, after=self.adv_btn)
            self.adv_btn.config(text="v  Advanced Voice Edit")
        self.adv_open = not self.adv_open

    def toggle_pro(self, fit=True):
        if self.pro_open:
            self.pro.pack_forget()
            self.pro_btn.config(text=PRO_CLOSED)
        else:
            self.pro.pack(side="top", fill="x")
            self.pro_btn.config(text=PRO_OPEN)
        self.pro_open = not self.pro_open
        if fit:
            self.after(30, lambda: self.fit_window(grow_only=True))

    # ---------------------------------------------------------------- scrolling / window fit
    def build_scroll_area(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True)
        self.vbar = ttk.Scrollbar(outer, orient="vertical")
        self.hbar = ttk.Scrollbar(outer, orient="horizontal")
        self.canvas = tk.Canvas(outer, bg=BG, highlightthickness=0,
                                yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.vbar.config(command=self.canvas.yview)
        self.hbar.config(command=self.canvas.xview)
        self.vbar.pack(side="right", fill="y")
        self.hbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=BG)
        self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.left = tk.Frame(self.body, bg=BG)               # main controls
        self.right = tk.Frame(self.body, bg=BG)              # Pro Controls
        self.left.pack(side="left", fill="y", anchor="n")
        self.right.pack(side="left", fill="y", anchor="n", padx=(0, 6))
        self.bind_all("<MouseWheel>", self._wheel)
        self.bind_all("<Shift-MouseWheel>", self._wheel_x)

    def _wheel(self, e):
        if e.widget.winfo_class() in ("Listbox", "TCombobox"):
            return
        self.canvas.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")

    def _wheel_x(self, e):
        self.canvas.xview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1), "units")

    def fit_window(self, grow_only=False):
        """Size the window to its content, but never bigger than the screen (the rest scrolls)."""
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = self.body.winfo_reqwidth() + self.vbar.winfo_reqwidth() + 4
        h = self.body.winfo_reqheight() + self.hbar.winfo_reqheight() + 4
        w, h = min(w, sw - 30), min(h, sh - 110)           # leave room for title bar + taskbar
        if grow_only:
            w, h = max(w, self.winfo_width()), max(h, self.winfo_height())
            w, h = min(w, sw - 30), min(h, sh - 110)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h - 60) // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(520, 360)

    # ---------------------------------------------------------------- audio
    def candidate_apis(self):
        names = hostapis()
        sel = self.api_box.get()
        if sel in names and "wdm" not in sel.lower():
            return [sel]                         # user forced one system: do not silently switch
        order = []
        for key in API_ORDER:
            for nm in names:
                if key.lower() in nm.lower() and nm not in order:
                    order.append(nm)
        return order or names[:1]

    def devices_for(self, api_name):
        """(mic, cable, monitor) device indices for `api_name`."""
        names = hostapis()
        mon_name = label_name(self.mon_box.get()) if self.mon_box.get() else "(none)"
        if api_name == self.listed_api:
            try:
                in_dev = self.ins[self.in_box.current()][0]
                cab_dev = self.cable_outs[self.cable_box.current()][0]
                mon_dev = self.mon_outs[self.mon_box.current()][0]
            except IndexError:
                raise RuntimeError("Mic aur Cable output select karo.")
            return in_dev, cab_dev, mon_dev
        idx = names.index(api_name)
        in_dev = find_equivalent(label_name(self.in_box.get()), idx, True)
        cab_dev = find_equivalent(label_name(self.cable_box.get()), idx, False)
        mon_dev = None if mon_name == "(none)" else find_equivalent(mon_name, idx, False)
        if in_dev is None or cab_dev is None:
            raise RuntimeError("is audio system mein mic / cable device nahi mila")
        return in_dev, cab_dev, mon_dev

    def update_monitor_btn(self):
        on = self.monitor_wanted
        self.monitor_btn.config(text="Hear myself: ON" if on else "Hear myself: OFF", bg=ACC if on else TRACK)

    def toggle_monitor(self):
        """Hear myself ON/OFF. Opening / closing the headphone stream can take a few hundred ms (WASAPI),
        so it runs in a short worker thread; the Tk window never freezes and the result is picked up by
        _poll_monitor() on the Tk thread."""
        if self._mon_busy:
            return                                          # previous request still running
        want = not self.monitor_wanted
        if self.router is None:                             # not started yet: just remember the choice
            self.monitor_wanted = want
            self.update_monitor_btn()
            self.save_settings()
            return
        try:
            mon_dev = self.devices_for(self.active_api)[2]
        except Exception as ex:
            log(f"monitor toggle failed: {ex}\n{traceback.format_exc()}")
            self._monitor_done(want, False, str(ex))
            return
        self._mon_busy = True
        router, gen = self.router, self._mon_gen

        def work():
            try:
                ok, msg = router.set_monitor(want, mon_dev)
            except Exception as ex:                         # never let the worker die silently
                log(f"monitor worker failed: {ex}\n{traceback.format_exc()}")
                ok, msg = False, str(ex)
            self._mon_q.put((gen, router, want, ok, msg))

        threading.Thread(target=work, name="cv-monitor", daemon=True).start()

    def _poll_monitor(self):
        try:
            while True:
                gen, router, want, ok, msg = self._mon_q.get_nowait()
                if gen != self._mon_gen or router is not self.router:
                    continue                                # old request (STOP / new START happened): ignore
                self._monitor_done(want, ok, msg)
        except queue.Empty:
            pass
        self.after(80, self._poll_monitor)

    def _monitor_done(self, want, ok, msg):
        self._mon_busy = False
        if not ok:
            log(f"monitor {'on' if want else 'off'} failed: {msg}")
            if want:
                messagebox.showwarning("Hear myself", msg)
            want = False                                    # on failed -> OFF; off failed -> treat as OFF
        if self.router is not None:                         # the router's real state is the truth
            want = bool(self.router.monitor_enabled)
        self.monitor_wanted = want
        self.update_monitor_btn()
        # Persist the toggle immediately so the next launch matches the visible OFF/ON state.
        self.save_settings()

    def toggle_stream(self):
        if self.router is not None:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self, silent=False):
        if self.router is not None:
            return True
        if audio_io.sd is None:
            messagebox.showerror("Audio", f"PortAudio / sounddevice load nahi hua:\n{audio_io.SD_ERROR}\n\nSETUP.bat dobara chalao.")
            return False
        # Voice OUT must be a virtual cable.  Starting with a real speaker / headphone selected would play
        # the processed voice out loud, so START is refused (also for auto-start).
        sel_cable = label_name(self.cable_box.get()) if self.cable_box.get() else ""
        if not self.cable_found or not is_virtual_output(sel_cable):
            why = ("VB-Cable nahi mila." if not self.cable_found else
                   f"Voice OUT -> Cable mein '{sel_cable}' select hai - ye asli speaker/headphone hai.")
            if not silent:
                messagebox.showerror("Cable", why + "\n\nVoice OUT mein 'CABLE Input (VB-Audio Virtual Cable)' chuno "
                                     "(ya 'Auto-fix / Rescan' dabao). Warna aapki processed awaaz seedha "
                                     "speaker/headphone par bajegi, isliye START nahi hoga.")
            else:
                self.status.config(text="Voice OUT galat / cable nahi - START nahi hua", fg=BAD)
            return False
        errors = []
        for api in self.candidate_apis():
            router = None
            try:
                in_dev, cab_dev, mon_dev = self.devices_for(api)
                router = Router(self.engine, self.buf_box.get())
                router.start(api, in_dev, cab_dev, mon_dev, monitor=self.monitor_wanted)
            except Exception as ex:
                log(f"start failed on {api}: {ex}\n{traceback.format_exc()}")
                errors.append(f"{api}: {ex}")
                if router is not None:
                    try:
                        router.stop()
                    except Exception:
                        pass
                continue
            self.router, self.active_api = router, api
            self._mon_gen += 1                             # results of requests for an older router are ignored
            self._mon_busy = False
            self.monitor_wanted = bool(router.monitor_enabled)      # show what really happened, not what was wished
            self.update_monitor_btn()
            if router.monitor_error:                       # headphones could not be opened: stay OFF
                log(f"monitor: {router.monitor_error}")
                if not silent:
                    messagebox.showwarning("Hear myself", router.monitor_error)
            self.last_drop_count, self.last_drop_time = 0, 0.0
            self.start_btn.config(text="STOP", bg="#e74c3c")
            self.status.config(text="Running", fg=OK)
            self.save_settings()
            log(f"started on {api} | buffer={self.buf_box.get()} | perf={winperf.boost_process()}")
            return True
        if not silent:
            messagebox.showerror("Audio error", "Audio start nahi hua:\n\n" + "\n".join(errors) +
                                 "\n\nMic ko koi doosri app exclusive mode mein use kar rahi ho to usse band karo,"
                                 " phir 'Auto-fix / Rescan' dabao.")
        else:
            self.status.config(text="Start nahi hua - Auto-fix dabao", fg=BAD)
        return False

    def stop_stream(self):
        self._mon_gen += 1                                  # any Hear request still running is now stale
        self._mon_busy = False
        if self.router is not None:
            self.router.stop()
            self.router = None
        self.start_btn.config(text="START", bg=ACC)
        self.status.config(text="Stopped", fg=BAD)
        self.warn.config(text="")
        self.engine.peak = self.engine.out_peak = 0.0

    def auto_start(self):
        if self.router is None and self.cable_found:
            self.start_stream(silent=True)

    def autofix(self):
        """Re-scan audio devices, go back to safe defaults, and restart the stream."""
        self.stop_stream()
        reinit_portaudio()
        apis = [AUTO_API] + usable_hostapis()
        self.api_box["values"] = apis
        self.api_box.set(AUTO_API)
        self.buf_box.set(DEFAULT_BUFFER)
        self.refresh_devices()
        if not self.cable_found:
            if messagebox.askyesno("VB-Cable nahi mila",
                                   "VB-Cable abhi Windows mein dikh nahi raha.\n\n"
                                   "SETUP chalaun? (Windows admin permission poochega, baaki sab automatic)"):
                self.run_setup()
            return
        self.start_stream()

    def run_setup(self):
        bat = os.path.join(BASE_DIR, "SETUP.bat")
        if not os.path.exists(bat):
            messagebox.showinfo("Setup", "SETUP.bat nahi mila (CuteVoice folder mein hona chahiye).")
            return
        try:
            import ctypes
            ctypes.windll.shell32.ShellExecuteW(None, "runas", bat, None, BASE_DIR, 1)
        except Exception:
            try:
                os.startfile(bat)
            except Exception:
                pass

    def open_mic_privacy(self):
        try:
            os.startfile("ms-settings:privacy-microphone")
        except Exception:
            pass

    def tick(self):
        self.shown_in = max(self.engine.peak, self.shown_in * 0.85)
        self.shown_out = max(self.engine.out_peak, self.shown_out * 0.85)
        for cv, v in ((self.meter_in, self.shown_in), (self.meter_out, self.shown_out)):
            cv.delete("all")
            cv.create_rectangle(0, 0, int(min(1.0, v * 1.4) * 170), 10, fill=ACC, outline=ACC)
        r = self.router
        if r is not None:
            now = time.time()
            drops = r.dropouts
            if drops != self.last_drop_count:
                self.last_drop_count, self.last_drop_time = drops, now
            if r.error:
                txt, col = r.error, BAD
            else:
                txt = f"Running | delay ~{r.delay_ms():.0f} ms | CPU {r.load_pct():.0f}% | dropouts {drops}"
                if self.engine.notice:
                    txt += " | " + self.engine.notice
                col = "#f1c40f" if now - self.last_drop_time < 4.0 else OK
            if self.status.cget("text") != txt or self.status.cget("fg") != col:
                self.status.config(text=txt, fg=col)
            silent = (not r.heard_signal) and r.silent_since is not None and now - r.silent_since > 4.0
            msg = ("Mic se signal nahi aa raha (0). Mic galat select hai / mute hai, ya Windows Settings > Privacy > "
                   "Microphone mein 'desktop apps' ko allow nahi. (Yahan click = Settings khulegi)") if silent else ""
            if self.warn.cget("text") != msg:
                self.warn.config(text=msg)
        self.after(60, self.tick)

    def on_close(self):
        self.save_settings()
        self.stop_stream()
        self.destroy()


def main():
    if not winperf.single_instance():
        r = tk.Tk()
        r.withdraw()
        messagebox.showinfo("CuteVoice", "CuteVoice Studio pehle se khula hai.")
        return
    if audio_io.sd is None:
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("CuteVoice", f"Audio library load nahi hui:\n{audio_io.SD_ERROR}\n\nSETUP.bat chalao.")
        return
    winperf.boost_process()
    log("---- CuteVoice v3 start")
    App().mainloop()


if __name__ == "__main__":
    main()
