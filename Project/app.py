import pickle
import numpy as np
import torch
import torch.nn as nn
import gradio as gr


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — MODEL ARCHITECTURE (must match training)
# ══════════════════════════════════════════════════════════════════════════════
class MultiTaskNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3,
                 dropout: float = 0.25):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.clf_head = nn.Linear(32, n_classes)
        self.reg_head = nn.Linear(32, 1)

    def forward(self, x):
        shared = self.backbone(x)
        return self.clf_head(shared), self.reg_head(shared)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SKLEARN-COMPATIBLE WRAPPERS
# These mimic the exact API the UI expects:
#   apnea_model.predict(row)
#   apnea_model.predict_proba(row)
#   stress_model.predict(row)
# ══════════════════════════════════════════════════════════════════════════════
class _ApneaWrapper:
    """
    Wraps the classification head of MultiTaskNet.
    Input row has 12 columns (with Stress Level placeholder at index 6).
    We drop that column before inference since the DL model uses 11 features.
    """
    def __init__(self, net, scaler):
        self.net    = net
        self.scaler = scaler

    def _prepare(self, X):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Drop Stress Level column (index 6) — the UI always passes it
        if X.shape[1] == 12:
            X = np.delete(X, 6, axis=1)      # now 11 features
        X = self.scaler.transform(X).astype(np.float32)
        return torch.tensor(X)

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(self._prepare(X))
            return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(self._prepare(X))
            return torch.softmax(logits, dim=1).numpy()


class _StressWrapper:
    """
    Wraps the regression head of MultiTaskNet.
    Same column-drop logic as _ApneaWrapper.
    """
    def __init__(self, net, scaler):
        self.net    = net
        self.scaler = scaler

    def _prepare(self, X):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == 12:
            X = np.delete(X, 6, axis=1)
        X = self.scaler.transform(X).astype(np.float32)
        return torch.tensor(X)

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            _, stress = self.net(self._prepare(X))
            return stress.squeeze().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — LOAD MODEL  (replaces pickle.load)
# ══════════════════════════════════════════════════════════════════════════════
with open('encoders.pkl', 'rb') as f:
    enc = pickle.load(f)

occ_le     = enc['occ_le']
scaler     = enc['scaler']
n_features = enc['n_features']       # 11

_net = MultiTaskNet(n_features=n_features)
_net.load_state_dict(torch.load('model.pth', map_location='cpu'))
_net.eval()

# Drop-in replacements — same names the UI uses
apnea_model  = _ApneaWrapper(_net, scaler)
stress_model = _StressWrapper(_net, scaler)

print('✅ Deep learning model loaded successfully')


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — METADATA (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ['None', 'Insomnia', 'Sleep Apnea']

OCCUPATION_LIST = [
    'Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager',
    'Nurse', 'Sales Representative', 'Salesperson',
    'Scientist', 'Software Engineer', 'Teacher'
]

BMI_MAP  = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
BMI_LIST = ['Normal', 'Overweight', 'Obese']

BENCHMARKS = {
    '18-30': {'sleep_duration': 7.4, 'quality': 7.1, 'steps': 8200, 'hr': 68},
    '31-45': {'sleep_duration': 7.1, 'quality': 6.8, 'steps': 7100, 'hr': 71},
    '46-65': {'sleep_duration': 6.8, 'quality': 6.3, 'steps': 6000, 'hr': 73},
}

def get_benchmark(age):
    if age <= 30: return BENCHMARKS['18-30']
    elif age <= 45: return BENCHMARKS['31-45']
    else: return BENCHMARKS['46-65']


def run_bias_audit():
    groups = {
        'Young Male':   [25, 1, 2, 7, 7, 40, 0, 120, 80, 70, 6000],
        'Young Female': [25, 0, 2, 7, 7, 40, 0, 120, 80, 70, 6000],
        'Older Male':   [55, 1, 2, 7, 7, 40, 0, 130, 85, 75, 5000],
        'Older Female': [55, 0, 2, 7, 7, 40, 0, 130, 85, 75, 5000],
        'Overweight':   [35, 1, 2, 7, 7, 40, 1, 125, 82, 72, 5500],
        'Obese':        [35, 1, 2, 7, 7, 40, 2, 135, 88, 78, 4500],
    }
    results = {}
    for name, features in groups.items():
        # 11-feature row (no stress placeholder needed here)
        row = np.array([features])
        pred  = apnea_model.predict(row)[0]
        proba = apnea_model.predict_proba(row)[0]
        results[name] = {
            'prediction': CLASS_NAMES[pred],
            'confidence': round(max(proba) * 100, 1)
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — PREDICT FUNCTION  (signature 100% unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def predict(
    age, gender, occupation,
    sleep_duration, quality_of_sleep,
    physical_activity, bmi_category,
    bp_systolic, bp_diastolic,
    heart_rate, daily_steps
):
    gender_enc = 1 if gender == 'Male' else 0
    occ_enc    = OCCUPATION_LIST.index(occupation)
    bmi_enc    = BMI_MAP[bmi_category]

    bm          = get_benchmark(age)
    sleep_delta = sleep_duration - bm['sleep_duration']

    # stress_est is still computed for the HTML display heuristic;
    # the model no longer uses it as input (no leakage)
    stress_est = 8 - (quality_of_sleep * 0.6) - (physical_activity / 30) + (1 if bmi_enc >= 1 else 0)
    stress_est = int(np.clip(round(stress_est), 3, 8))

    # 11-feature row — Stress Level column NOT included
    row = np.array([[
        age, gender_enc, occ_enc,
        sleep_duration, quality_of_sleep,
        physical_activity,
        bmi_enc, bp_systolic, bp_diastolic,
        heart_rate, daily_steps
    ]])

    # ── Sleep Apnea Prediction ────────────────────────────────────────────────
    apnea_idx   = apnea_model.predict(row)[0]
    apnea_proba = apnea_model.predict_proba(row)[0] if hasattr(apnea_model, 'predict_proba') else [0, 0, 0]
    apnea_label = CLASS_NAMES[apnea_idx]

    risk_map = {
        'None':        ('No Risk',          'LOW'),
        'Insomnia':    ('Insomnia Risk',    'MEDIUM'),
        'Sleep Apnea': ('Sleep Apnea Risk', 'HIGH'),
    }
    result_text, risk_level = risk_map[apnea_label]

    # ── Stress Prediction ─────────────────────────────────────────────────────
    stress_raw   = stress_model.predict(row)
    stress_score = int(np.clip(round(float(stress_raw)), 3, 8))
    stress_label = 'Low' if stress_score <= 4 else ('Medium' if stress_score <= 6 else 'High')
    stress_pct   = int(stress_score / 8 * 100)
    stress_col   = '#1a9e75' if stress_score <= 4 else ('#e5a020' if stress_score <= 6 else '#e05252')
    stress_bg    = '#f0fdf4' if stress_score <= 4 else ('#fffbeb' if stress_score <= 6 else '#fef2f2')

    # ── Overall Sleep Score ───────────────────────────────────────────────────
    sleep_score = int(np.clip(
        (quality_of_sleep / 10 * 100) * 0.35 +
        (min(sleep_duration, 9) / 9 * 100) * 0.30 +
        ((8 - stress_score) / 8 * 100) * 0.20 +
        (physical_activity / 90 * 100) * 0.15,
        0, 100
    ))

    if sleep_score >= 80:
        score_msg = "That's a great score!"
        score_sub = "You're well on your way to improving your health."
    elif sleep_score >= 60:
        score_msg = 'Moderate score.'
        score_sub = 'Some areas need attention for better sleep health.'
    else:
        score_msg = 'Low score.'
        score_sub = 'Please follow the recommendations below carefully.'

    dur_score    = int(min(sleep_duration / 9 * 70, 70))
    qual_score   = int(quality_of_sleep * 2)
    activity_pts = int(min(physical_activity / 90 * 5, 5))
    stress_pts   = int(max(0, (8 - stress_score) / 5 * 5))

    none_pct  = round(apnea_proba[0] * 100, 1)
    ins_pct   = round(apnea_proba[1] * 100, 1)
    apnea_pct = round(apnea_proba[2] * 100, 1)

    rcolour  = {'LOW': '#1a9e75', 'MEDIUM': '#e5a020', 'HIGH': '#e05252'}[risk_level]
    ring_col = '#1a9e75' if sleep_score >= 80 else ('#e5a020' if sleep_score >= 60 else '#e05252')

    R    = 54
    circ = 2 * 3.14159 * R
    dash = circ * sleep_score / 100
    gap  = circ - dash

    # ════════════════════════════════════════
    # Sleep Apnea Score Card HTML
    # ════════════════════════════════════════
    apnea_result_html = f"""
<div style="font-family:'DM Sans',system-ui,sans-serif; background:#ffffff;
     border-radius:18px; border:1px solid #dde9f0; overflow:hidden;
     box-shadow:0 2px 16px rgba(13,33,55,0.09);">

  <div style="background:linear-gradient(160deg,#0d2137 0%,#123d5e 55%,#0f5c42 100%);
       padding:30px 24px 26px; text-align:center;">
    <div style="font-size:0.68rem; letter-spacing:2px; color:rgba(255,255,255,0.45);
         text-transform:uppercase; margin-bottom:14px;">NeuroRest Score</div>

    <svg width="148" height="148" viewBox="0 0 148 148" style="display:block;margin:0 auto 12px;">
      <circle cx="74" cy="74" r="{R}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="11"/>
      <circle cx="74" cy="74" r="{R}" fill="none" stroke="{ring_col}" stroke-width="11"
              stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {gap:.1f}"
              transform="rotate(-90 74 74)"/>
      <text x="74" y="68" text-anchor="middle" font-size="38" font-weight="700"
            fill="white" font-family="DM Sans,sans-serif">{sleep_score}</text>
      <text x="74" y="86" text-anchor="middle" font-size="11"
            fill="rgba(255,255,255,0.5)" font-family="DM Sans,sans-serif">100 points</text>
    </svg>

    <div style="color:white;font-size:1rem;font-weight:600;">{score_msg}</div>
    <div style="color:rgba(255,255,255,0.55);font-size:0.8rem;margin-top:4px;">{score_sub}</div>
  </div>

  <div style="padding:20px 22px;">

    <!-- Duration row -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:#e8f7f3;
           display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" stroke="#1a9e75" stroke-width="2"/>
          <path d="M12 7v5l3 3" stroke="#1a9e75" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div style="flex:1;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span style="font-size:0.85rem;color:#374151;">{sleep_duration:.1f}h &nbsp;sleep duration</span>
          <span style="font-size:0.85rem;font-weight:700;color:#1a9e75;">{dur_score}&nbsp;/&nbsp;70</span>
        </div>
        <div style="height:5px;background:#f0f5f8;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{min(dur_score/70*100,100):.0f}%;background:#1a9e75;border-radius:3px;"></div>
        </div>
      </div>
    </div>

    <!-- Quality row -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:#eef3ff;
           display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" stroke="#3b82f6" stroke-width="2"/>
          <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="#3b82f6" stroke-width="2" stroke-linecap="round"/>
          <circle cx="9" cy="9" r="1" fill="#3b82f6"/><circle cx="15" cy="9" r="1" fill="#3b82f6"/>
        </svg>
      </div>
      <div style="flex:1;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span style="font-size:0.85rem;color:#374151;">Sleep quality rating</span>
          <span style="font-size:0.85rem;font-weight:700;color:#3b82f6;">{qual_score}&nbsp;/&nbsp;20</span>
        </div>
        <div style="height:5px;background:#f0f5f8;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{min(qual_score/20*100,100):.0f}%;background:#3b82f6;border-radius:3px;"></div>
        </div>
      </div>
    </div>

    <!-- Apnea row -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:50%;
           background:{'#fef2f2' if risk_level=='HIGH' else ('#fffbeb' if risk_level=='MEDIUM' else '#f0fdf4')};
           display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M3 12c0 0 4-8 9-8s9 8 9 8-4 8-9 8-9-8-9-8z" stroke="{rcolour}" stroke-width="2"/>
          <circle cx="12" cy="12" r="3" stroke="{rcolour}" stroke-width="2"/>
        </svg>
      </div>
      <div style="flex:1;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span style="font-size:0.85rem;color:#374151;">
            <span style="color:{rcolour};font-weight:600;">{apnea_label}</span> detected
          </span>
          <span style="font-size:0.85rem;font-weight:700;color:{rcolour};">{activity_pts}&nbsp;/&nbsp;5</span>
        </div>
        <div style="height:5px;background:#f0f5f8;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{min(activity_pts/5*100,100):.0f}%;background:{rcolour};border-radius:3px;"></div>
        </div>
      </div>
    </div>

    <!-- Stress row -->
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:#fff7ed;
           display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4"
                stroke="#e5a020" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </div>
      <div style="flex:1;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
          <span style="font-size:0.85rem;color:#374151;">Stress — {stress_label}</span>
          <span style="font-size:0.85rem;font-weight:700;color:#e5a020;">{stress_pts}&nbsp;/&nbsp;5</span>
        </div>
        <div style="height:5px;background:#f0f5f8;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{min(stress_pts/5*100,100):.0f}%;background:#e5a020;border-radius:3px;"></div>
        </div>
      </div>
    </div>

    <!-- Probability bar -->
    <div style="background:#f8fafc;border-radius:10px;padding:14px 16px;margin-top:4px;">
      <div style="font-size:0.75rem;font-weight:600;color:#64748b;margin-bottom:10px;
           letter-spacing:0.5px;text-transform:uppercase;">Disorder Probability</div>
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">
        <span style="width:70px;font-size:0.75rem;color:#374151;">None</span>
        <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
          <div style="height:100%;width:{none_pct}%;background:#1a9e75;border-radius:4px;"></div></div>
        <span style="width:38px;font-size:0.75rem;font-weight:600;color:#1a9e75;text-align:right;">{none_pct}%</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">
        <span style="width:70px;font-size:0.75rem;color:#374151;">Insomnia</span>
        <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
          <div style="height:100%;width:{ins_pct}%;background:#e5a020;border-radius:4px;"></div></div>
        <span style="width:38px;font-size:0.75rem;font-weight:600;color:#e5a020;text-align:right;">{ins_pct}%</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center;">
        <span style="width:70px;font-size:0.75rem;color:#374151;">Sleep Apnea</span>
        <div style="flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">
          <div style="height:100%;width:{apnea_pct}%;background:#e05252;border-radius:4px;"></div></div>
        <span style="width:38px;font-size:0.75rem;font-weight:600;color:#e05252;text-align:right;">{apnea_pct}%</span>
      </div>
    </div>

  </div>
</div>"""

    # ════════════════════════════════════════
    # Stress Output HTML
    # ════════════════════════════════════════
    stress_output = f"""
<div style="font-family:'DM Sans',system-ui,sans-serif;background:{stress_bg};
     border-radius:14px;border:1px solid #dde9f0;padding:20px 22px;
     box-shadow:0 2px 10px rgba(13,33,55,0.07);">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <span style="font-size:0.85rem;font-weight:600;color:#1e293b;">Stress Level</span>
    <span style="font-size:1.4rem;font-weight:700;color:{stress_col};">{stress_score}/8</span>
  </div>
  <div style="height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;margin-bottom:10px;">
    <div style="height:100%;width:{stress_pct}%;background:{stress_col};
         border-radius:5px;transition:width 0.4s ease;"></div>
  </div>
  <div style="font-size:0.82rem;color:{stress_col};font-weight:600;">Category: {stress_label}</div>
</div>"""

    # ════════════════════════════════════════
    # Recommendations HTML
    # ════════════════════════════════════════
    recs = []
    if sleep_duration < bm['sleep_duration']:
        recs.append(('🛌', 'Increase Sleep Duration',
                     f"You sleep {sleep_duration:.1f}h vs the {bm['sleep_duration']}h benchmark for your age. "
                     f"Aim for 7–9 hours nightly."))
    if quality_of_sleep < bm['quality']:
        recs.append(('🌙', 'Improve Sleep Quality',
                     'Maintain a consistent sleep schedule, keep your room cool and dark.'))
    if stress_score >= 6:
        recs.append(('🧘', 'Manage Stress',
                     'High stress detected. Consider meditation, journaling, or speaking with a professional.'))
    if physical_activity < 30:
        recs.append(('🏃', 'Increase Physical Activity',
                     'At least 30 minutes of moderate exercise daily improves sleep quality significantly.'))
    if bmi_enc >= 1:
        recs.append(('⚖️', 'Monitor BMI',
                     'Elevated BMI increases sleep apnea risk. Consider dietary adjustments.'))
    if daily_steps < bm['steps']:
        recs.append(('👟', 'Increase Daily Steps',
                     f"You average {int(daily_steps):,} steps vs benchmark {bm['steps']:,}. "
                     f"Aim for 7,000–10,000 steps."))
    if bp_systolic > 130 or bp_diastolic > 85:
        recs.append(('❤️', 'Monitor Blood Pressure',
                     'Elevated BP detected. Reduce sodium, increase potassium, and consult your doctor.'))
    if not recs:
        recs.append(('✅', 'Great Health Indicators',
                     'Your metrics are within healthy ranges. Keep maintaining your current lifestyle!'))

    rec_items = ''.join(f"""
      <div style="display:flex;gap:12px;padding:12px;background:#f8fafc;border-radius:10px;
           border-left:3px solid #1a6b8a;margin-bottom:9px;">
        <div style="font-size:1.2rem;flex-shrink:0;">{icon}</div>
        <div>
          <div style="font-size:0.87rem;font-weight:600;color:#1e293b;margin-bottom:3px;">{title}</div>
          <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">{body}</div>
        </div>
      </div>""" for icon, title, body in recs)

    rec_html = f"""
<div style="font-family:'DM Sans',system-ui,sans-serif;background:white;border-radius:18px;
     border:1px solid #dde9f0;overflow:hidden;box-shadow:0 2px 16px rgba(13,33,55,0.09);">
  <div style="background:linear-gradient(135deg,#0d2137 0%,#123d5e 100%);
       padding:16px 22px;">
    <div style="color:white;font-weight:600;font-size:0.92rem;">💡 Recommendations</div>
  </div>
  <div style="padding:16px 18px;">{rec_items}</div>
</div>"""

    # ════════════════════════════════════════
    # AI Coaching HTML
    # ════════════════════════════════════════
    coaching_tips = []
    if apnea_label == 'Sleep Apnea':
        coaching_tips.append(('🩺', 'Consult a Sleep Specialist',
                               'Sleep Apnea detected. A polysomnography test can confirm and guide CPAP therapy.'))
    if apnea_label == 'Insomnia':
        coaching_tips.append(('🧠', 'Try CBT-I',
                               'Cognitive Behavioural Therapy for Insomnia is the most effective non-drug treatment.'))
    if stress_score >= 5:
        coaching_tips.append(('🌿', 'Wind-down Routine',
                               'Start a 30-minute pre-sleep routine: dim lights, no screens, light stretching.'))
    coaching_tips.append(('📱', 'Track Your Sleep',
                           'Use a wearable or sleep-tracking app to monitor trends over 2–4 weeks.'))
    coaching_tips.append(('☀️', 'Morning Light Exposure',
                           'Get 10–15 min of natural light within an hour of waking to regulate your circadian rhythm.'))

    tips_html = ''.join(f"""
      <div style="display:flex;gap:12px;padding:13px;background:#f8fafc;border-radius:10px;
           border-left:3px solid #1a6b8a;margin-bottom:9px;">
        <div style="font-size:1.2rem;flex-shrink:0;line-height:1.4;">{icon}</div>
        <div>
          <div style="font-size:0.87rem;font-weight:600;color:#1e293b;margin-bottom:3px;">{title}</div>
          <div style="font-size:0.8rem;color:#64748b;line-height:1.6;">{body}</div>
        </div>
      </div>""" for icon, title, body in coaching_tips)

    coaching_html = f"""
<div style="font-family:'DM Sans',system-ui,sans-serif; background:white; border-radius:18px;
     border:1px solid #dde9f0; overflow:hidden; box-shadow:0 2px 16px rgba(13,33,55,0.09);">
  <div style="background:linear-gradient(135deg,#0d2137 0%,#1a6b8a 100%);
       padding:16px 22px;display:flex;align-items:center;gap:12px;">
    <div style="width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,0.13);
         display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🤖</div>
    <div>
      <div style="color:white;font-weight:600;font-size:0.92rem;">AI Sleep Coach</div>
      <div style="color:rgba(255,255,255,0.5);font-size:0.73rem;">Personalized insights based on your vitals</div>
    </div>
  </div>
  <div style="padding:16px 18px;">
    {tips_html}
    <div style="margin-top:4px;padding:10px 14px;background:#e8f7f3;border-radius:8px;
         font-size:0.77rem;color:#0f6e56;">
      💬 Re-analyze weekly to get updated coaching as your health data changes.
    </div>
  </div>
</div>"""

    return apnea_result_html, stress_output, rec_html, coaching_html


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — GRADIO UI  (100% original — zero changes)
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

body, .gradio-container {
    background: #eef3f8 !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.gradio-container { max-width: 1340px !important; margin: 0 auto !important; }

.gap, .form, .block, .contain,
.gradio-group, .gr-group, .gr-box,
div.svelte-vt1mxs, div.svelte-1gfkn6j,
.wrap.svelte-z7cif2, .padded {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#nr-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #123d5e 55%, #0f5c42 100%);
    border-radius: 18px; padding: 26px 36px; margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(13,27,42,0.2);
}
#nr-header h1 {
    font-size: 1.8rem !important; font-weight: 700 !important;
    color: white !important; margin: 0 0 4px !important; letter-spacing: -0.4px;
}
#nr-header p { color: rgba(255,255,255,0.58) !important; font-size: 0.87rem !important; margin: 0 !important; }

#nr-dot {
    display: inline-block; width: 9px; height: 9px;
    background: #2ecc8f; border-radius: 50%; margin-right: 9px;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.nr-lbl {
    font-size: 0.67rem !important; font-weight: 700 !important;
    letter-spacing: 1.8px !important; text-transform: uppercase !important;
    color: #1a6b8a !important; margin-bottom: 8px !important; padding-left: 2px !important;
}

.nr-card {
    background: white !important;
    border-radius: 16px !important;
    border: 1px solid #d1e2ed !important;
    padding: 20px 22px !important;
    box-shadow: 0 2px 12px rgba(15,82,110,0.07) !important;
    margin-bottom: 14px !important;
}

input[type=range] {
    accent-color: #1a6b8a !important;
    height: 4px !important;
    cursor: pointer !important;
}

.nr-card label > span,
.nr-card .block > label > span,
label > span {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #1a4b6a !important;
    letter-spacing: 0.1px !important;
}

.nr-card input[type=number] {
    background: #f0f7fb !important;
    border: 1px solid #c4d9e8 !important;
    border-radius: 8px !important;
    color: #1a6b8a !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-align: center !important;
}

.nr-card .wrap label {
    background: #f4f8fb !important;
    border: 1.5px solid #c4d9e8 !important;
    border-radius: 10px !important;
    padding: 7px 14px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #2d4a5e !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
}
.nr-card .wrap label:hover {
    border-color: #1a6b8a !important;
    background: #e8f4fb !important;
}
.nr-card .wrap input[type=radio]:checked + label,
.nr-card .wrap label.selected {
    background: linear-gradient(135deg, #1a6b8a, #0f5c42) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 2px 8px rgba(26,107,138,0.25) !important;
}

.nr-card select, select {
    border-radius: 10px !important;
    border: 1.5px solid #c4d9e8 !important;
    background: #f4f8fb !important;
    font-size: 0.85rem !important;
    color: #1e293b !important;
    padding: 8px 12px !important;
    font-weight: 500 !important;
    transition: border-color 0.2s !important;
}
.nr-card select:focus, select:focus {
    border-color: #1a6b8a !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(26,107,138,0.1) !important;
}

#nr-btn button {
    background: linear-gradient(135deg, #1a6b8a 0%, #0f5c42 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(26,107,138,0.35) !important;
    transition: opacity 0.2s, transform 0.1s !important;
    letter-spacing: 0.3px !important;
}
#nr-btn button:hover  { opacity: 0.88 !important; transform: translateY(-2px) !important; }
#nr-btn button:active { transform: translateY(0) !important; }

.nr-html-out > div { border-radius: 14px !important; }

#nr-footer {
    text-align: center; padding: 18px; font-size: 0.75rem;
    color: #94a3b8; margin-top: 4px;
}
"""

with gr.Blocks(css=CSS, title='NeuroRest AI') as demo:

    gr.HTML("""
    <div id="nr-header">
      <h1><span id="nr-dot"></span>NeuroRest AI</h1>
      <p>Sleep Apnea &amp; Stress Prediction &nbsp;·&nbsp; AI Coaching &nbsp;·&nbsp; Detailed Health Insights
         &nbsp;|&nbsp; Enter your vitals and click Analyze</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=5):

            gr.HTML('<div class="nr-lbl">👤 Personal Information</div>')
            with gr.Group(elem_classes='nr-card'):
                with gr.Row():
                    age    = gr.Slider(18, 65, value=30, step=1,  label='Age')
                    gender = gr.Radio(['Male', 'Female'], value='Male', label='Gender')
                occupation = gr.Dropdown(OCCUPATION_LIST, value='Engineer', label='Occupation')

            gr.HTML('<div class="nr-lbl">😴 Sleep Metrics</div>')
            with gr.Group(elem_classes='nr-card'):
                with gr.Row():
                    sleep_duration   = gr.Slider(4, 10, value=7, step=0.5, label='Sleep Duration (hrs)')
                    quality_of_sleep = gr.Slider(1, 10, value=7, step=1,   label='Sleep Quality (1–10)')

            gr.HTML('<div class="nr-lbl">🏃 Physical Health</div>')
            with gr.Group(elem_classes='nr-card'):
                with gr.Row():
                    physical_activity = gr.Slider(0, 90, value=40, step=5, label='Physical Activity (min/day)')
                    bmi_category      = gr.Radio(BMI_LIST, value='Normal', label='BMI Category')

            gr.HTML('<div class="nr-lbl">❤️ Vitals</div>')
            with gr.Group(elem_classes='nr-card'):
                with gr.Row():
                    bp_systolic  = gr.Slider(90, 180, value=120, step=1, label='Systolic BP (mmHg)')
                    bp_diastolic = gr.Slider(60, 110, value=80,  step=1, label='Diastolic BP (mmHg)')
                with gr.Row():
                    heart_rate  = gr.Slider(55, 100,    value=70,   step=1,   label='Heart Rate (bpm)')
                    daily_steps = gr.Slider(1000, 12000, value=6000, step=500, label='Daily Steps')

            with gr.Row(elem_id='nr-btn'):
                analyze_btn = gr.Button('🔍  Analyze My Sleep & Stress', variant='primary')

        with gr.Column(scale=5):

            gr.HTML('<div class="nr-lbl">📊 Sleep Apnea Score</div>')
            apnea_out = gr.HTML(
                value="<div style='padding:44px;text-align:center;color:#94a3b8;"
                      "font-family:DM Sans,sans-serif;background:white;"
                      "border-radius:18px;border:1px solid #dde9f0;'>"
                      "Run analysis to see your sleep score</div>"
            )

            gr.HTML('<div class="nr-lbl" style="margin-top:16px;">😓 Stress Level</div>')
            stress_out = gr.HTML(
                value="<div style='padding:28px;text-align:center;color:#94a3b8;"
                      "font-family:DM Sans,sans-serif;background:white;"
                      "border-radius:14px;border:1px solid #dde9f0;'>"
                      "Stress prediction will appear here...</div>",
                elem_classes='nr-html-out'
            )

            gr.HTML('<div class="nr-lbl" style="margin-top:16px;">💡 Health Recommendations</div>')
            rec_out = gr.HTML(
                value="<div style='padding:28px;text-align:center;color:#94a3b8;"
                      "font-family:DM Sans,sans-serif;background:white;"
                      "border-radius:18px;border:1px solid #dde9f0;'>"
                      "Personalized recommendations will appear here...</div>",
                elem_classes='nr-html-out'
            )

            gr.HTML('<div class="nr-lbl" style="margin-top:16px;">🤖 AI Sleep Coaching</div>')
            coaching_out = gr.HTML(
                value="<div style='padding:32px;text-align:center;color:#94a3b8;"
                      "font-family:DM Sans,sans-serif;background:white;"
                      "border-radius:18px;border:1px solid #dde9f0;'>"
                      "AI coaching will appear after analysis</div>"
            )

    gr.HTML("""
    <div id="nr-footer">
      NeuroRest AI &nbsp;·&nbsp; For informational purposes only
      &nbsp;·&nbsp; Always consult a qualified medical professional for diagnosis
    </div>
    """)

    analyze_btn.click(
        fn=predict,
        inputs=[
            age, gender, occupation,
            sleep_duration, quality_of_sleep,
            physical_activity, bmi_category,
            bp_systolic, bp_diastolic,
            heart_rate, daily_steps
        ],
        outputs=[apnea_out, stress_out, rec_out, coaching_out]
    )

demo.launch()