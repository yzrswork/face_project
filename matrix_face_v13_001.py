"""
╔══════════════════════════════════════════════════════════════════╗
║   E N T I T Y  //  v13.0                                        ║
║                                                                  ║
║   FACE    → 顔として存在（心臓鼓動）                            ║
║   SWARM   → ボイドAIとして群れ飛ぶ                              ║
║   HELIX   → DNA二重螺旋に変形                                   ║
║   SPHERE  → フィボナッチ球面に変形                              ║
║   VORTEX  → 竜巻状渦巻きに変形                                  ║
║   SINGULARITY → 1点に収束 → ビッグバン爆発 → 顔に戻る          ║
║                                                                  ║
║   v13 NEW                                                        ║
║   + 無音検出を実測RMS値ベースに最適化（RMS閾値=150）             ║
║   + PCMゼロカット強化（閾値50）                                  ║
║   + ノイズフロア減算値を実測ベースに調整                         ║
║   + 無音3秒でFACEにピタッと戻って静止                           ║
║   + FACEモードで心臓鼓動（ドックン）                            ║
║   + SINGULARITYクールダウン（連発防止）                         ║
╚══════════════════════════════════════════════════════════════════╝
【起動】  py -3.11 matrix_face_v13.py
【キー】  ESC=終了  D=デバッグ  +/-=感度  SPACE=SINGULARITY強制
         1=FACE  2=SWARM  3=HELIX  4=SPHERE  5=VORTEX
"""

import os, sys, json, math, random, colorsys
import pygame
import numpy as np

# ═══════════════════════════════════════════════════════════
#  ■ CONFIG
# ═══════════════════════════════════════════════════════════
WIDTH, HEIGHT  = 1280, 720
FPS            = 60
FIT_RATIO      = 0.65

FORM_FACE        = "FACE"
FORM_SWARM       = "SWARM"
FORM_HELIX       = "HELIX"
FORM_SPHERE      = "SPHERE"
FORM_VORTEX      = "VORTEX"
FORM_SINGULARITY = "SINGULARITY"
MORPH_FORMS = [FORM_HELIX, FORM_SPHERE, FORM_VORTEX]

FORM_COLOR = {
    FORM_FACE:        (  0, 220,  70),
    FORM_SWARM:       ( 80, 255, 180),
    FORM_HELIX:       (180, 100, 255),
    FORM_SPHERE:      (100, 200, 255),
    FORM_VORTEX:      (255, 160,  40),
    FORM_SINGULARITY: (255, 255, 200),
}
C_BG        = (  0,   0,   0)
C_WIRE_DIM  = (  0,  45,  14)
C_TRI_CORE  = (200, 255, 220)
C_TRI_GLOW1 = ( 80, 255, 160,  30)
C_TRI_GLOW2 = ( 20, 200, 100,  12)
C_ARC       = ( 80, 255, 180)
C_ARC_CORE  = (220, 255, 240)
C_SHOCK_OUT = (  0, 255, 100)
C_SHOCK_IN  = (200, 255, 230)
C_DOT_NEAR  = (  0, 150,  45)
C_DOT_FAR   = (  0,  22,   6)
C_HUD       = (  0, 130,  40)
C_HUD_HI    = (  0, 255,  75)
C_CRT_SCAN  = (  0,  18,   5,  12)
C_BORDER    = (  0, 190,  56)
C_VNOISE    = (  0, 255,  80,  42)

# ── Audio ─────────────────────────────────────────────────
CHUNK        = 2048
SAMPLE_RATE  = 48000
DEVICE_INDEX = 31
CHANNELS     = 2
NORM_LOW     = 2800000.0
NORM_MID     =  900000.0
NORM_HI      =  850000.0
KICK_GATE    = 0.03
VALUE_CAP    = 0.88
SENS_MASTER  = 1.0
SMOOTH_LOW   = 0.68
SMOOTH_MID   = 0.88
SMOOTH_HI    = 0.85

# ── キック検出 ────────────────────────────────────────────
KICK_THRESH  = 0.08
KICK_MIN     = 0.18
SING_THRESH  = 0.75   # SINGULARITYトリガー（高めに維持）

# ── 無音検出 ──────────────────────────────────────────────
SILENCE_THRESH  = 0.04   # s_low/s_mid閾値（ノイズフロア引き後）
SILENCE_DELAY   = 180    # 約3秒（60fps×3）
PCM_ZERO_CUT    = 50.0   # 実測ノイズmax=2700に対してPCM段階でカット
NOISE_FLOOR_LOW = 0.008  # 実測ベース調整済み
NOISE_FLOOR_MID = 0.006  # 実測ベース調整済み
NOISE_FLOOR_HI  = 0.006  # 実測ベース調整済み
RMS_SILENCE     = 1500.0 # 実測avg=570 推奨1427×1.05
RMS_WINDOW      = 20     # 中央値フィルター（スパイク耐性）

# ── SINGULARITYクールダウン ───────────────────────────────
SING_COOLDOWN   = 600    # SINGULARITY後のロック時間（約10秒）

# ── 形態遷移クールダウン ──────────────────────────────────
FORM_COOLDOWN   = 90     # 形態変化後のロック時間（約1.5秒）

# ── 物理 ──────────────────────────────────────────────────
MORPH_LERP   = 0.06    # ゆっくり変形
DAMPING      = 0.975
SPEED_MAX    = 7.0

# ── 心臓鼓動（FACEモード） ────────────────────────────────
PULSE_PEAK   = 0.22    # kick時の最大膨張率
PULSE_DECAY  = 0.82    # フレームごとの減衰
PULSE_THRESH = 0.10    # 鼓動検出しきい値

# ── ボイドAI ──────────────────────────────────────────────
BOID_SEP_R   = 28.0
BOID_ALI_R   = 65.0
BOID_COH_R   = 90.0
BOID_SEP_F   = 1.8
BOID_ALI_F   = 0.9
BOID_COH_F   = 0.5
BOID_ATTRACT_F = 0.08

# ── SINGULARITY ───────────────────────────────────────────
SING_CONTRACT = 0.18
SING_EXPLODE_V = 22.0
SING_REFORM_L  = 0.14

# ── パーティクル・エコー ──────────────────────────────────
TRAIL_LEN      = 5
TRAIL_INTERVAL = 2
ECHO_COUNT     = 3

# ── 衝撃波 ────────────────────────────────────────────────
SHOCK_SPEED  = 12
SHOCK_MAX_R  = 480

# ── 3Dドット ──────────────────────────────────────────────
DOT_COUNT    = 160
DOT_Z_RANGE  = (200, 1400)
DOT_ROT_BASE = 0.0025

# ── 電弧 ──────────────────────────────────────────────────
ARC_DIST_MAX = 90
ARC_P_BASE   = 0.015
ARC_SEGS     = 7
ARC_JITTER   = 14

# ── 三角形 ────────────────────────────────────────────────
TRIANGLE_EDGE_IDX = {150, 151, 152, 153, 154, 155}
LEFT_TRI_VERTS    = [152, 153, 154]
RIGHT_TRI_VERTS   = [155, 156, 157]

# ═══════════════════════════════════════════════════════════
#  ■ INIT
# ═══════════════════════════════════════════════════════════
pygame.init()
screen    = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("E N T I T Y  //  v13.0")
clock     = pygame.time.Clock()

layer_bg  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
layer_wire= pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
layer_fx  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
layer_crt = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
layer_abr = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
phosphor  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
phosphor.fill((0,0,0,0))
_ph_fade  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
_ph_fade.fill((0,0,0,32))

font_hud  = pygame.font.SysFont("consolas", 10, bold=True)
font_debug= pygame.font.SysFont("consolas", 11)
font_form = pygame.font.SysFont("consolas", 36, bold=True)
font_sub  = pygame.font.SysFont("consolas", 14, bold=True)

# ─── JSON ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for fname in ("face_wireframe.json", "20260218_extracted.json"):
    p = os.path.join(SCRIPT_DIR, fname)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        break
else:
    sys.exit("[ERROR] face_wireframe.json が見つかりません")

verts_orig = np.array([v[:2] for v in raw["vertices"]], dtype=float)
edges_all  = [(i,j) for i,j in raw["edges"]
              if np.linalg.norm(verts_orig[i]-verts_orig[j]) > 0.5]
span       = verts_orig.max(axis=0) - verts_orig.min(axis=0)
offset     = (verts_orig.min(axis=0) + verts_orig.max(axis=0)) / 2
base_scale = min(WIDTH*FIT_RATIO/span[0], HEIGHT*FIT_RATIO/span[1])
cx, cy     = WIDTH//2, HEIGHT//2
N          = len(verts_orig)
print(f"[JSON] 頂点={N} エッジ={len(edges_all)}")

FACE_TX = cx + (verts_orig[:,0] - offset[0]) * base_scale
FACE_TY = cy - (verts_orig[:,1] - offset[1]) * base_scale
LEFT_TRI_C  = np.mean(verts_orig[LEFT_TRI_VERTS],  axis=0)
RIGHT_TRI_C = np.mean(verts_orig[RIGHT_TRI_VERTS], axis=0)

# ── 形態ターゲット座標 ────────────────────────────────────
def make_helix():
    tx, ty = np.zeros(N), np.zeros(N)
    R, H = 160, 380
    for i in range(N):
        t  = i / N * 4 * math.pi
        st = 1 if i % 2 == 0 else -1
        tx[i] = cx + math.cos(t * st) * R
        ty[i] = cy - H/2 + (i/N)*H + math.sin(t)*12
    return tx, ty

def make_sphere():
    tx, ty = np.zeros(N), np.zeros(N)
    golden = math.pi * (3 - math.sqrt(5))
    R = 240
    for i in range(N):
        yn = 1 - (i/(N-1))*2
        r  = math.sqrt(max(0, 1-yn**2))
        th = golden * i
        tx[i] = cx + math.cos(th)*r*R
        ty[i] = cy + yn*R*0.85
    return tx, ty

def make_vortex():
    tx, ty = np.zeros(N), np.zeros(N)
    for i in range(N):
        t  = i / N
        ag = t*6*math.pi + t*2
        r  = (1-t)*220 + 20
        tx[i] = cx + math.cos(ag)*r
        ty[i] = cy - 220 + t*440
    return tx, ty

HELIX_TX,  HELIX_TY  = make_helix()
SPHERE_TX, SPHERE_TY = make_sphere()
VORTEX_TX, VORTEX_TY = make_vortex()
FORM_TARGETS = {
    FORM_FACE:   (FACE_TX,   FACE_TY),
    FORM_HELIX:  (HELIX_TX,  HELIX_TY),
    FORM_SPHERE: (SPHERE_TX, SPHERE_TY),
    FORM_VORTEX: (VORTEX_TX, VORTEX_TY),
}

# ─── Audio ────────────────────────────────────────────────
import pyaudio
pa = pyaudio.PyAudio()
try:
    pa.get_device_info_by_index(DEVICE_INDEX)
    astream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                      rate=SAMPLE_RATE, input=True,
                      input_device_index=DEVICE_INDEX,
                      frames_per_buffer=CHUNK)
    print(f"[AUDIO] デバイス {DEVICE_INDEX} 接続完了")
except Exception as e:
    print(f"[AUDIO] エラー: {e}")
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            print(f"  [{i}] {d['name']}")
    DEVICE_INDEX = int(input("インデックス > "))
    astream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                      rate=SAMPLE_RATE, input=True,
                      input_device_index=DEVICE_INDEX,
                      frames_per_buffer=CHUNK)

freqs   = np.fft.rfftfreq(CHUNK, 1/SAMPLE_RATE)
idx_low = np.where((freqs >=  20) & (freqs <=  150))[0]
idx_mid = np.where((freqs >= 250) & (freqs <= 2000))[0]
idx_hi  = np.where((freqs >= 2000) & (freqs <= 8000))[0]

# ═══════════════════════════════════════════════════════════
#  ■ 状態変数（すべてリスト or スカラー、globalで管理）
# ═══════════════════════════════════════════════════════════
s_low = s_mid = s_hi = 0.0
prev_s_low   = 0.0
prev_s_low_pulse = 0.0   # 鼓動用
sens         = SENS_MASTER
tick         = 0
debug_mode   = False
raw_lo = raw_mi = raw_hi = 0.0
cur_rms = 0.0
rms_history = []  # RMS中央値フィルター用履歴
rms_median  = 0.0

vert_x = FACE_TX.copy()
vert_y = FACE_TY.copy()
vel_x  = np.zeros(N)
vel_y  = np.zeros(N)

current_form     = FORM_FACE
morph_progress   = 1.0
morph_from_x     = FACE_TX.copy()
morph_from_y     = FACE_TY.copy()
swarm_beat_count = 0
sing_phase       = "none"
sing_timer       = 0
sing_cooldown    = 0     # SINGULARITYクールダウンカウンタ
form_cooldown    = 0     # 形態遷移クールダウン

# 無音管理
silence_count    = 0     # 無音フレーム数
is_silent        = False # 無音状態フラグ

# 心臓鼓動
cur_pulse        = 0.0
pulse_scale      = 1.0   # 鼓動による顔のスケール倍率

form_flash_timer = 0
form_flash_text  = ""
form_flash_color = (0, 255, 80)

trail_buf = np.zeros((TRAIL_LEN, N, 2))
trail_ptr = 0
echo_bufs = []
shockwaves= []
lens_str  = 0.0
aberr_str = 0.0
vnoise_list = []

dot_x  = np.random.uniform(-WIDTH*0.7,  WIDTH*0.7,  DOT_COUNT)
dot_y  = np.random.uniform(-HEIGHT*0.7, HEIGHT*0.7, DOT_COUNT)
dot_z  = np.random.uniform(DOT_Z_RANGE[0], DOT_Z_RANGE[1], DOT_COUNT)
dot_vz = np.random.uniform(-1.8, -0.3, DOT_COUNT)
dot_rot= 0.0

# ═══════════════════════════════════════════════════════════
#  ■ UTILS
# ═══════════════════════════════════════════════════════════
def lerp(a, b, t): return a + (b-a)*t
def clamp(v, lo, hi): return max(lo, min(hi, v))
def blend(c1, c2, t): return tuple(int(a+(b-a)*t) for a,b in zip(c1,c2))
def hsv_col(h, s=1.0, v=1.0):
    r,g,b = colorsys.hsv_to_rgb(h%1.0, s, v)
    return (int(r*255), int(g*255), int(b*255))

def read_fft():
    buf = astream.read(CHUNK, exception_on_overflow=False)
    pcm = np.frombuffer(buf, dtype=np.int16).astype(float)
    if CHANNELS == 2 and len(pcm) == CHUNK*2:
        pcm = (pcm[0::2]+pcm[1::2])/2.0
    pcm = pcm[:CHUNK]
    if len(pcm) < CHUNK:
        pcm = np.pad(pcm, (0, CHUNK-len(pcm)))
    # ゼロカット：ドライバノイズを除去
    pcm[np.abs(pcm) < PCM_ZERO_CUT] = 0.0
    rms = float(np.sqrt(np.mean(pcm**2)))
    return np.abs(np.fft.rfft(pcm)), rms

def trigger_form_flash(name, color):
    global form_flash_timer, form_flash_text, form_flash_color
    form_flash_timer = 55
    form_flash_text  = name
    form_flash_color = color

def start_morph_to(form):
    global current_form, morph_progress, morph_from_x, morph_from_y
    global swarm_beat_count, form_cooldown
    if form == current_form and form != FORM_SWARM:
        return
    current_form     = form
    morph_progress   = 0.0
    morph_from_x     = vert_x.copy()
    morph_from_y     = vert_y.copy()
    swarm_beat_count = 0
    form_cooldown    = FORM_COOLDOWN
    trigger_form_flash(form, FORM_COLOR.get(form, (0,255,80)))

# ═══════════════════════════════════════════════════════════
#  ■ ボイドAI
# ═══════════════════════════════════════════════════════════
def boid_update(attract_face=False):
    global vel_x, vel_y
    new_vx = vel_x.copy()
    new_vy = vel_y.copy()
    for i in range(N):
        dx = vert_x - vert_x[i]
        dy = vert_y - vert_y[i]
        d  = np.sqrt(dx**2 + dy**2) + 0.001
        sep_mask = d < BOID_SEP_R; sep_mask[i] = False
        if sep_mask.any():
            new_vx[i] += -(dx[sep_mask]/d[sep_mask]).mean() * BOID_SEP_F
            new_vy[i] += -(dy[sep_mask]/d[sep_mask]).mean() * BOID_SEP_F
        ali_mask = (d < BOID_ALI_R) & (d > BOID_SEP_R); ali_mask[i] = False
        if ali_mask.any():
            new_vx[i] += vel_x[ali_mask].mean() * BOID_ALI_F * 0.1
            new_vy[i] += vel_y[ali_mask].mean() * BOID_ALI_F * 0.1
        coh_mask = (d < BOID_COH_R) & (d > BOID_ALI_R); coh_mask[i] = False
        if coh_mask.any():
            new_vx[i] += (vert_x[coh_mask].mean()-vert_x[i]) * BOID_COH_F * 0.02
            new_vy[i] += (vert_y[coh_mask].mean()-vert_y[i]) * BOID_COH_F * 0.02
        if attract_face:
            new_vx[i] += (FACE_TX[i]-vert_x[i]) * BOID_ATTRACT_F
            new_vy[i] += (FACE_TY[i]-vert_y[i]) * BOID_ATTRACT_F
    vel_x, vel_y = new_vx, new_vy

# ═══════════════════════════════════════════════════════════
#  ■ 無音処理
# ═══════════════════════════════════════════════════════════
def update_silence():
    global silence_count, is_silent, current_form, morph_progress
    global vert_x, vert_y, vel_x, vel_y, cur_pulse, pulse_scale

    if s_low < SILENCE_THRESH and s_mid < SILENCE_THRESH:
        silence_count += 1
        # 無音3秒でFACEにピタッと戻る
        if silence_count == SILENCE_DELAY and sing_phase == "none":
            is_silent = True
            current_form   = FORM_FACE
            morph_progress = 1.0
            vert_x = FACE_TX.copy()
            vert_y = FACE_TY.copy()
            vel_x[:] = 0.0
            vel_y[:] = 0.0
            cur_pulse   = 0.0
            pulse_scale = 1.0
            trigger_form_flash("- SILENCE -", (0, 80, 30))
    else:
        if is_silent:
            is_silent = False
            trigger_form_flash("FACE", FORM_COLOR[FORM_FACE])
        silence_count = 0

# ═══════════════════════════════════════════════════════════
#  ■ 形態更新メインロジック
# ═══════════════════════════════════════════════════════════
def update_forms(kick_detected):
    global vert_x, vert_y, vel_x, vel_y
    global current_form, morph_progress, swarm_beat_count
    global sing_phase, sing_timer, sing_cooldown, form_cooldown
    global lens_str, aberr_str, shockwaves, echo_bufs
    global morph_from_x, morph_from_y
    global cur_pulse, pulse_scale, prev_s_low_pulse

    # クールダウン更新
    if sing_cooldown  > 0: sing_cooldown  -= 1
    if form_cooldown  > 0: form_cooldown  -= 1

    # ── 無音中は何もしない ──────────────────────────────
    if is_silent:
        return

    # ── SINGULARITY チェック ─────────────────────────────
    if (kick_detected and s_low > SING_THRESH
            and sing_phase == "none"
            and sing_cooldown == 0
            and current_form != FORM_SINGULARITY):
        sing_phase   = "contract"
        sing_timer   = 0
        sing_cooldown = SING_COOLDOWN
        current_form = FORM_SINGULARITY
        if len(echo_bufs) >= ECHO_COUNT: echo_bufs.pop(0)
        echo_bufs.append((vert_x.copy(), vert_y.copy(), 180))
        trigger_form_flash("SINGULARITY", FORM_COLOR[FORM_SINGULARITY])

    # ── SINGULARITY 処理 ────────────────────────────────
    if sing_phase == "contract":
        lens_str  = min(1.0, lens_str + 0.06)
        aberr_str = min(1.0, aberr_str + 0.04)
        vert_x += (cx - vert_x) * SING_CONTRACT
        vert_y += (cy - vert_y) * SING_CONTRACT
        vel_x *= 0.5; vel_y *= 0.5
        sing_timer += 1
        if np.sqrt((vert_x-cx)**2+(vert_y-cy)**2).mean() < 6:
            sing_phase = "explode"; sing_timer = 0
            ang = np.random.uniform(0, math.tau, N)
            spd = np.random.uniform(SING_EXPLODE_V*0.5, SING_EXPLODE_V, N)
            vel_x = np.cos(ang)*spd; vel_y = np.sin(ang)*spd
            for dr in [0, 8, 16]:
                shockwaves.append({"r": float(dr), "max_r": SHOCK_MAX_R, "intensity": 1.2})

    elif sing_phase == "explode":
        lens_str  = max(0.0, lens_str  - 0.04)
        aberr_str = max(0.0, aberr_str - 0.03)
        vert_x += vel_x; vert_y += vel_y
        vel_x *= 0.92;   vel_y *= 0.92
        sing_timer += 1
        if sing_timer > 35:
            sing_phase = "reform"; sing_timer = 0

    elif sing_phase == "reform":
        lens_str  = max(0.0, lens_str  - 0.02)
        aberr_str = max(0.0, aberr_str - 0.02)
        vert_x += (FACE_TX - vert_x) * SING_REFORM_L
        vert_y += (FACE_TY - vert_y) * SING_REFORM_L
        vel_x *= 0.7; vel_y *= 0.7
        sing_timer += 1
        d = np.sqrt((vert_x-FACE_TX)**2+(vert_y-FACE_TY)**2)
        if d.mean() < 5 or sing_timer > 80:
            sing_phase     = "none"
            current_form   = FORM_FACE
            morph_progress = 1.0
            vert_x = FACE_TX.copy(); vert_y = FACE_TY.copy()
            vel_x[:] = 0; vel_y[:] = 0
            trigger_form_flash("FACE", FORM_COLOR[FORM_FACE])
        return

    if sing_phase != "none":
        return

    # ── 心臓鼓動（FACEモード） ───────────────────────────
    if current_form == FORM_FACE and morph_progress >= 1.0:
        kick_delta = s_low - prev_s_low_pulse
        if kick_delta > PULSE_THRESH:
            strength = clamp(kick_delta / (1.0-PULSE_THRESH+0.01), 0.0, 1.0)
            cur_pulse = PULSE_PEAK * strength
        else:
            cur_pulse *= PULSE_DECAY
        prev_s_low_pulse = s_low
        pulse_scale = 1.0 + cur_pulse
    else:
        cur_pulse   *= PULSE_DECAY
        pulse_scale  = 1.0 + cur_pulse

    # ── SWARM ────────────────────────────────────────────
    if current_form == FORM_SWARM:
        boid_update(attract_face=(kick_detected and s_low > 0.25))
        spd = np.sqrt(vel_x**2+vel_y**2)
        ov  = spd > SPEED_MAX
        vel_x[ov] = vel_x[ov]/spd[ov]*SPEED_MAX
        vel_y[ov] = vel_y[ov]/spd[ov]*SPEED_MAX
        vert_x += vel_x; vert_y += vel_y
        vel_x *= DAMPING; vel_y *= DAMPING
        margin = 40
        flip_x = (vert_x<margin)|(vert_x>WIDTH-margin)
        flip_y = (vert_y<margin)|(vert_y>HEIGHT-margin)
        vel_x[flip_x] *= -0.6; vel_y[flip_y] *= -0.6
        vert_x = np.clip(vert_x, margin, WIDTH-margin)
        vert_y = np.clip(vert_y, margin, HEIGHT-margin)
        if kick_detected:
            swarm_beat_count += 1
            if swarm_beat_count >= 16:
                start_morph_to(random.choice(MORPH_FORMS))
        return

    # ── 幾何形態補間 ─────────────────────────────────────
    if current_form in FORM_TARGETS:
        tx, ty = FORM_TARGETS[current_form]
        if morph_progress < 1.0:
            morph_progress = min(1.0, morph_progress + MORPH_LERP)
            t = morph_progress; t = t*t*(3-2*t)
            vert_x = morph_from_x + (tx-morph_from_x)*t
            vert_y = morph_from_y + (ty-morph_from_y)*t
        else:
            # 鼓動スケールをFACE形態に適用
            if current_form == FORM_FACE and pulse_scale != 1.0:
                vert_x = cx + (FACE_TX - cx) * pulse_scale
                vert_y = cy + (FACE_TY - cy) * pulse_scale
            else:
                vert_x = tx.copy(); vert_y = ty.copy()

            if form_cooldown == 0:
                # FACE → SWARM（強いkickのみ、確率30%）
                if current_form == FORM_FACE and kick_detected and s_low > 0.55:
                    if random.random() < 0.30:
                        if len(echo_bufs) >= ECHO_COUNT: echo_bufs.pop(0)
                        echo_bufs.append((vert_x.copy(), vert_y.copy(), 140))
                        start_morph_to(FORM_SWARM)
                # 幾何形態 → 次（強いkickかつ確率15%）
                elif current_form != FORM_FACE and kick_detected and s_low > 0.52:
                    if random.random() < 0.15:
                        nf = [f for f in MORPH_FORMS if f != current_form]
                        if len(echo_bufs) >= ECHO_COUNT: echo_bufs.pop(0)
                        echo_bufs.append((vert_x.copy(), vert_y.copy(), 140))
                        start_morph_to(FORM_FACE if random.random()<0.35 else random.choice(nf))

        noise_str = s_hi*2.5 + s_mid*1.0
        vert_x += np.random.uniform(-noise_str, noise_str, N)
        vert_y += np.random.uniform(-noise_str, noise_str, N)

    echo_bufs[:] = [(ex,ey,max(0,ea-2)) for ex,ey,ea in echo_bufs if ea > 0]

# ═══════════════════════════════════════════════════════════
#  ■ DRAW — 3Dドット
# ═══════════════════════════════════════════════════════════
def draw_3d_dots(surf):
    global dot_z, dot_rot
    dot_rot += DOT_ROT_BASE*(1.0+s_mid*3.5)
    dot_z   += dot_vz*(1.0+s_low*3.5)
    reset    = dot_z < 50
    dot_z[reset] = np.random.uniform(*DOT_Z_RANGE, reset.sum())
    dot_x[reset] = np.random.uniform(-WIDTH*0.7,  WIDTH*0.7,  reset.sum())
    dot_y[reset] = np.random.uniform(-HEIGHT*0.7, HEIGHT*0.7, reset.sum())
    for i in range(DOT_COUNT):
        z  = dot_z[i]
        rx = dot_x[i]*math.cos(dot_rot) - z*math.sin(dot_rot)
        rz = max(dot_x[i]*math.sin(dot_rot) + z*math.cos(dot_rot), 10)
        proj = 600.0/rz
        sx = cx + rx*proj; sy = cy + dot_y[i]*proj
        if lens_str > 0.01:
            ddx = sx-cx; ddy = sy-cy
            dd  = math.sqrt(ddx**2+ddy**2)+0.1
            warp = lens_str*3000/(dd+100)
            sx -= ddx/dd*warp; sy -= ddy/dd*warp
        sx, sy = int(sx), int(sy)
        if not (0 <= sx < WIDTH and 0 <= sy < HEIGHT): continue
        t   = clamp(1.0-(rz-50)/1400, 0.0, 1.0)
        col = blend(C_DOT_FAR, C_DOT_NEAR, t)
        # 無音時は暗くする
        alpha_mul = 0.3 if is_silent else 1.0
        pygame.draw.circle(surf, (*col, int((t*150+20)*alpha_mul)), (sx,sy), max(1,int(t*2.5)))

# ═══════════════════════════════════════════════════════════
#  ■ DRAW — エコー・トレイル
# ═══════════════════════════════════════════════════════════
def draw_echoes(surf):
    for ei, (ex, ey, ea) in enumerate(echo_bufs):
        if ea < 4: continue
        col = hsv_col((tick*0.003+ei*0.15)%1.0, 0.6, 0.8)
        px  = ex.astype(int); py = ey.astype(int)
        for i,j in edges_all:
            pygame.draw.line(surf, (*col, int(ea*0.45)), (px[i],py[i]), (px[j],py[j]), 1)

def update_trail():
    global trail_ptr
    if tick % TRAIL_INTERVAL == 0:
        trail_buf[trail_ptr%TRAIL_LEN,:,0] = vert_x
        trail_buf[trail_ptr%TRAIL_LEN,:,1] = vert_y
        trail_ptr += 1

def draw_trails(surf):
    hue_base = (tick*0.004)%1.0
    for age in range(1, min(TRAIL_LEN, trail_ptr)):
        buf_idx = (trail_ptr-age-1)%TRAIL_LEN
        t = 1.0 - age/TRAIL_LEN
        col   = hsv_col((hue_base+age*0.07)%1.0, 0.8, 1.0)
        alpha = int(t**2*120)
        if alpha < 4: continue
        px = trail_buf[buf_idx,:,0].astype(int)
        py = trail_buf[buf_idx,:,1].astype(int)
        r  = max(1, int(t*2.5))
        for i in range(N):
            pygame.draw.circle(surf, (*col, alpha), (px[i],py[i]), r)

# ═══════════════════════════════════════════════════════════
#  ■ DRAW — ワイヤーフレーム
# ═══════════════════════════════════════════════════════════
def draw_face(surf):
    px = vert_x.astype(int)
    py = vert_y.astype(int)
    form_col = FORM_COLOR.get(current_form, (0,200,70))

    # 無音時は暗く
    dim = 0.25 if is_silent else 1.0

    if current_form == FORM_SINGULARITY: base_b = 0.9
    elif current_form == FORM_SWARM:     base_b = 0.3 + s_mid*0.7
    else:                                base_b = 0.35 + s_mid*0.65 + morph_progress*0.15
    base_b *= dim

    for idx, (i,j) in enumerate(edges_all):
        p1 = (px[i],py[i]); p2 = (px[j],py[j])
        if current_form in (FORM_SWARM, FORM_SINGULARITY):
            di = math.sqrt((vert_x[i]-FACE_TX[i])**2+(vert_y[i]-FACE_TY[i])**2)
            dj = math.sqrt((vert_x[j]-FACE_TX[j])**2+(vert_y[j]-FACE_TY[j])**2)
            cls = clamp(1.0-(di+dj)/2/200, 0.1, 1.0)
        else:
            cls = 1.0
        is_tri = idx in TRIANGLE_EDGE_IDX
        ev     = 0.78 + random.random()*0.44
        bright = clamp(base_b*ev*cls, 0.0, 1.0)
        if is_tri:
            glow = clamp(bright*(0.6+s_low*0.8), 0.0, 1.0)
            pygame.draw.line(surf, (*C_TRI_GLOW2[:3], int(clamp(glow*28,0,28))), p1, p2, 8)
            pygame.draw.line(surf, (*C_TRI_GLOW1[:3], int(clamp(glow*65,0,65))), p1, p2, 4)
            pygame.draw.line(surf, (*blend(form_col,C_TRI_CORE,glow), int(255*bright)), p1, p2, 2)
        else:
            if   bright > 0.70: col = blend(form_col, (255,255,255), (bright-0.70)/0.30)
            elif bright > 0.30: col = blend(C_WIRE_DIM, form_col,    (bright-0.30)/0.40)
            else:               col = blend((0,0,0), C_WIRE_DIM,     bright/0.30)
            lw = 2 if bright > 0.65 else 1
            if bright > 0.38:
                pygame.draw.line(surf, (*form_col, int(bright*32)), p1, p2, lw+3)
            pygame.draw.line(surf, (*col, int(255*bright)), p1, p2, lw)

    if current_form == FORM_SWARM:
        spds = np.sqrt(vel_x**2+vel_y**2)
        for i in range(N):
            col = hsv_col((i/N+tick*0.005)%1.0, 0.9, 1.0)
            r   = max(1, int(1.5+spds[i]/SPEED_MAX*4))
            pygame.draw.circle(surf, (*col, int(80+spds[i]/SPEED_MAX*175)), (px[i],py[i]), r)
    else:
        for i in range(N):
            pygame.draw.circle(surf, (*form_col, int(120*dim)), (px[i],py[i]), 1)

    glow_str = clamp(s_low*0.75+s_mid*0.5, 0.0, 1.0) * dim
    if glow_str > 0.2:
        for tri_v in (LEFT_TRI_VERTS, RIGHT_TRI_VERTS):
            tx_s = int(np.mean(vert_x[tri_v]))
            ty_s = int(np.mean(vert_y[tri_v]))
            r_g  = int(11*glow_str); a_g = int(45*glow_str)
            pygame.draw.circle(surf, (*C_TRI_GLOW1[:3], a_g),    (tx_s,ty_s), r_g+3)
            pygame.draw.circle(surf, (*C_TRI_GLOW2[:3], a_g//2), (tx_s,ty_s), r_g)
            pygame.draw.circle(surf, (*C_TRI_CORE, min(a_g+18,85)), (tx_s,ty_s), max(2,r_g//3))

# ═══════════════════════════════════════════════════════════
#  ■ DRAW — 電弧・衝撃波・色収差・CRT
# ═══════════════════════════════════════════════════════════
def draw_arcs(surf):
    arc_p = ARC_P_BASE*(0.5+s_hi*2.5) * (0.0 if is_silent else 1.0)
    if sing_phase == "explode": arc_p *= 4.0
    px = vert_x.astype(int); py = vert_y.astype(int)
    for i in range(N):
        for j in range(i+1, N):
            dx = vert_x[i]-vert_x[j]; dy = vert_y[i]-vert_y[j]
            dist = math.sqrt(dx*dx+dy*dy)
            if dist > ARC_DIST_MAX or random.random() > arc_p: continue
            pts = [(px[i],py[i])]
            for s in range(1, ARC_SEGS):
                t   = s/ARC_SEGS
                mx  = int(lerp(px[i],px[j],t)); my = int(lerp(py[i],py[j],t))
                jit = ARC_JITTER*(1.0-abs(t-0.5)*2)
                pts.append((mx+random.randint(-int(jit),int(jit)),
                             my+random.randint(-int(jit),int(jit))))
            pts.append((px[j],py[j]))
            fade = clamp(1.0-dist/ARC_DIST_MAX, 0.0, 1.0)
            arc_col = hsv_col((tick*0.01+i*0.03)%1.0, 0.7, 1.0)
            if len(pts) >= 2:
                pygame.draw.lines(surf, (*arc_col,   int(fade*65)),  False, pts, 3)
                pygame.draw.lines(surf, (*C_ARC_CORE, int(fade*170)), False, pts, 1)

def update_draw_shockwaves(surf):
    dead = []
    for sw in shockwaves:
        sw["r"] += SHOCK_SPEED*(1.0+s_mid*1.5)
        if sw["r"] > sw["max_r"]: dead.append(sw); continue
        t     = sw["r"]/sw["max_r"]
        alpha = int(clamp((1.0-t)*200*sw["intensity"],0,200))
        r     = int(sw["r"])
        pygame.draw.circle(surf, (*C_SHOCK_OUT, alpha//3), (cx,cy), r+5, 5)
        pygame.draw.circle(surf, (*C_SHOCK_IN,  alpha),    (cx,cy), r,   2)
        if r > 20:
            pygame.draw.circle(surf, (*C_SHOCK_OUT, alpha//2), (cx,cy), max(1,r-14), 1)
    for d in dead: shockwaves.remove(d)

def draw_aberration(surf, wire_surf):
    if aberr_str < 0.02: return
    shift = int(aberr_str*12)
    if shift < 1: return
    r_surf = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
    r_surf.blit(wire_surf, (0,0))
    r_map  = pygame.surfarray.array3d(r_surf)
    for ch, off in [(0, (-shift,0)), (2, (shift,0))]:
        only = np.zeros_like(r_map); only[:,:,ch] = r_map[:,:,ch]
        s = pygame.surfarray.make_surface(only)
        s.set_alpha(int(aberr_str*120))
        surf.blit(s, off)

def draw_crt(surf):
    for y in range(0, HEIGHT, 3):
        pygame.draw.line(surf, C_CRT_SCAN, (0,y), (WIDTH,y), 1)
    if random.random() < 0.038*(0.4+s_low*1.2):
        vnoise_list.append({"y":random.randint(0,HEIGHT),
                            "shift":random.randint(-14,14),
                            "h":random.randint(1,3),"life":1.0})
    dead = []
    for vn in vnoise_list:
        a = int(clamp(vn["life"]*43,0,43)); sx=vn["shift"]; y0,h=vn["y"],vn["h"]
        if sx>0: pygame.draw.rect(surf,(*C_VNOISE[:3],a),(0,y0,sx,h))
        else:    pygame.draw.rect(surf,(*C_VNOISE[:3],a),(WIDTH+sx,y0,-sx,h))
        vn["life"] -= 0.22
        if vn["life"] <= 0: dead.append(vn)
    for d in dead: vnoise_list.remove(d)
    pygame.draw.rect(surf, (*C_BORDER,98), (0,0,WIDTH,HEIGHT), 3)
    for (bx,by),(dx,dy) in zip(
        [(3,3),(WIDTH-3,3),(3,HEIGHT-3),(WIDTH-3,HEIGHT-3)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        pygame.draw.line(surf,C_BORDER,(bx,by),(bx+dx*40,by),2)
        pygame.draw.line(surf,C_BORDER,(bx,by),(bx,by+dy*40),2)

# ═══════════════════════════════════════════════════════════
#  ■ HUD
# ═══════════════════════════════════════════════════════════
FORM_LABELS = {
    FORM_FACE:        "FACE        // 顔",
    FORM_SWARM:       "SWARM       // 群れ",
    FORM_HELIX:       "HELIX       // 螺旋",
    FORM_SPHERE:      "SPHERE      // 球面",
    FORM_VORTEX:      "VORTEX      // 渦",
    FORM_SINGULARITY: "SINGULARITY // 特異点",
}

def draw_hud(surf):
    a   = int(clamp(65+s_mid*145, 50, 210))
    col = FORM_COLOR.get(current_form, C_HUD_HI)
    global form_flash_timer
    if form_flash_timer > 0:
        fa = int(clamp(form_flash_timer/55*220, 0, 220))
        ft = font_form.render(form_flash_text, True, form_flash_color)
        ft.set_alpha(fa); surf.blit(ft, (cx-ft.get_width()//2, 30))
        form_flash_timer -= 1
    if form_flash_timer <= 0:
        sm = font_sub.render(FORM_LABELS.get(current_form,""), True, col)
        sm.set_alpha(a); surf.blit(sm, (cx-sm.get_width()//2, 34))
    for row, line in enumerate([
        "E N T I T Y  v13",
        f"TICK :{tick:06d}",
        f"LOW  :{s_low:.3f}",
        f"MID  :{s_mid:.3f}",
        f"HI   :{s_hi:.3f}",
        f"SENS :{sens:.1f}",
    ]):
        c = C_HUD_HI if row==0 else C_HUD
        t = font_hud.render(line,True,c); t.set_alpha(a); surf.blit(t,(24,24+row*13))
    for bi,(label,val,hi_col) in enumerate([
        ("KICK",  s_low,       (0,255,78)),
        ("PULSE", cur_pulse/PULSE_PEAK, (0,200,100)),
        ("MORPH", morph_progress,       (80,200,255)),
        ("LENS",  lens_str,    (200,150,255)),
    ]):
        bx,by,bw,bh = WIDTH-160, HEIGHT-78+bi*18, 130, 6
        pygame.draw.rect(surf,(*C_HUD,34),(bx,by,bw,bh))
        fill = int(clamp(val,0,1)*bw)
        if fill > 0:
            pygame.draw.rect(surf,(*hi_col,a),(bx,by,fill,bh))
        t = font_hud.render(label,True,C_HUD); t.set_alpha(a); surf.blit(t,(bx-55,by))
    # 無音インジケータ
    if silence_count > 0 and not is_silent:
        prog = silence_count / SILENCE_DELAY
        bx2,by2,bw2,bh2 = cx-100, HEIGHT-20, 200, 4
        pygame.draw.rect(surf,(*C_HUD,30),(bx2,by2,bw2,bh2))
        pygame.draw.rect(surf,(*C_HUD,120),(bx2,by2,int(prog*bw2),bh2))

def draw_debug(surf):
    for row, line in enumerate([
        f"[DEBUG] form={current_form}  sing={sing_phase}  silent={is_silent}",
        f"raw_LOW:{raw_lo:8.0f}  s_low={s_low:.3f}",
        f"raw_MID:{raw_mi:8.0f}  s_mid={s_mid:.3f}",
        f"raw_HI :{raw_hi:8.0f}  s_hi ={s_hi:.3f}",
        f"RMS={cur_rms:.1f}  med={rms_median:.1f}  pulse={cur_pulse:.3f}  sing_cd={sing_cooldown}",
        f"silence={silence_count}/{SILENCE_DELAY}  echoes={len(echo_bufs)}  FPS:{clock.get_fps():.1f}",
    ]):
        t = font_debug.render(line,True,(185,255,100))
        t.set_alpha(190); surf.blit(t,(WIDTH-440,18+row*14))

# ═══════════════════════════════════════════════════════════
#  ■ MAIN LOOP
# ═══════════════════════════════════════════════════════════
print("[SYS] E N T I T Y v13 起動  ESC=終了  D=デバッグ  +/-=感度")
print("[SYS] SPACE=SINGULARITY  1=FACE  2=SWARM  3=HELIX  4=SPHERE  5=VORTEX")
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN:
            if   event.key == pygame.K_ESCAPE: running = False
            elif event.key == pygame.K_d:
                debug_mode = not debug_mode
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                sens = round(clamp(sens+0.1,0.1,5.0),1)
            elif event.key == pygame.K_MINUS:
                sens = round(clamp(sens-0.1,0.1,5.0),1)
            elif event.key == pygame.K_SPACE:
                if sing_phase == "none" and sing_cooldown == 0:
                    sing_phase = "contract"
                    current_form = FORM_SINGULARITY
                    sing_cooldown = SING_COOLDOWN
                    trigger_form_flash("SINGULARITY", FORM_COLOR[FORM_SINGULARITY])
            elif event.key == pygame.K_1: start_morph_to(FORM_FACE)
            elif event.key == pygame.K_2: start_morph_to(FORM_SWARM)
            elif event.key == pygame.K_3: start_morph_to(FORM_HELIX)
            elif event.key == pygame.K_4: start_morph_to(FORM_SPHERE)
            elif event.key == pygame.K_5: start_morph_to(FORM_VORTEX)

    # ── 音声 ────────────────────────────────────────────
    cur_rms = 0.0
    try:
        fft, cur_rms = read_fft()
        # RMS中央値フィルター（間欠スパイク耐性）
        rms_history.append(cur_rms)
        if len(rms_history) > RMS_WINDOW:
            rms_history.pop(0)
        rms_median = float(np.median(rms_history))
        if rms_median < RMS_SILENCE:
            r_lo = r_mi = r_hi = 0.0
        else:
            raw_lo = float(np.mean(fft[idx_low]))
            raw_mi = float(np.mean(fft[idx_mid]))
            raw_hi = float(np.mean(fft[idx_hi]))
            r_lo = clamp(max(0.0,raw_lo/NORM_LOW-KICK_GATE)*sens, 0.0, VALUE_CAP)
            r_mi = clamp(raw_mi/NORM_MID*sens, 0.0, VALUE_CAP)
            r_hi = clamp(raw_hi/NORM_HI *sens, 0.0, VALUE_CAP)
    except Exception:
        r_lo = r_mi = r_hi = 0.0

    s_low = lerp(s_low, r_lo, 1.0-SMOOTH_LOW)
    s_mid = lerp(s_mid, r_mi, 1.0-SMOOTH_MID)
    s_hi  = lerp(s_hi,  r_hi, 1.0-SMOOTH_HI)

    # ノイズフロア減算（残留スムージング分をカット）
    s_low = max(0.0, s_low - NOISE_FLOOR_LOW)
    s_mid = max(0.0, s_mid - NOISE_FLOOR_MID)
    s_hi  = max(0.0, s_hi  - NOISE_FLOOR_HI)

    kick = ((s_low - prev_s_low) > KICK_THRESH and s_low > KICK_MIN)
    prev_s_low = s_low

    # ── 無音検出 ────────────────────────────────────────
    update_silence()

    # ── 物理・形態更新 ──────────────────────────────────
    update_trail()
    update_forms(kick)

    # ── 描画 ────────────────────────────────────────────
    screen.fill(C_BG)
    layer_bg.fill((0,0,0,0))
    layer_wire.fill((0,0,0,0))
    layer_fx.fill((0,0,0,0))
    layer_crt.fill((0,0,0,0))
    layer_abr.fill((0,0,0,0))

    draw_3d_dots(layer_bg)
    screen.blit(layer_bg, (0,0))
    screen.blit(phosphor, (0,0))
    draw_echoes(layer_fx)
    draw_trails(layer_fx)
    draw_arcs(layer_fx)
    update_draw_shockwaves(layer_fx)
    screen.blit(layer_fx, (0,0))
    draw_face(layer_wire)
    phosphor.blit(_ph_fade, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
    phosphor.blit(layer_wire, (0,0), special_flags=pygame.BLEND_RGBA_ADD)
    screen.blit(layer_wire, (0,0))
    draw_aberration(screen, layer_wire)
    draw_crt(layer_crt)
    screen.blit(layer_crt, (0,0))
    draw_hud(screen)
    if debug_mode: draw_debug(screen)

    pygame.display.flip()
    clock.tick(FPS)
    tick += 1

astream.stop_stream()
astream.close()
pa.terminate()
pygame.quit()
print("[SYS] 終了")
