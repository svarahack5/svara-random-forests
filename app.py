import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ==============================================================================
# 1. MODEL & DATASET MANAGEMENT (RELATIVE PATH FOR GITHUB DEPLOYMENT)
# ==============================================================================
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'dementia_real_model.joblib')

# Uses relative path relative to the app file for seamless running on GitHub / servers
DATASET_FILE = os.path.join(os.path.dirname(__file__), 'dementia_dataset.csv')

# Non-clinical feature keys corresponding to daily life FAQ inputs
FEATURE_KEYS = [
    'age',
    'daily_faq_memory_conversations',     # Maps to MMSE recall
    'daily_faq_orientation_time_date',    # Maps to MMSE time orientation
    'daily_faq_navigating_places',        # Maps to MMSE spatial orientation
    'daily_faq_managing_finances',        # Maps to EDUC / MMSE calculation
    'daily_faq_household_chores',         # Maps to MMSE execution
    'daily_faq_learning_new_tasks',       # Maps to SES / EDUC adaptation
    'daily_faq_language_word_finding',    # Maps to MMSE language
    'daily_faq_object_misplacement',      # Maps to brain atrophy / nWBV surrogate
    'daily_faq_decision_making'           # Maps to eTIV / CDR impairment degree
]


def prepare_real_dataset():
    """Loads dementia_dataset.csv and maps clinical columns to non-clinical daily life FAQ proxies."""
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset file '{DATASET_FILE}' not found. Please ensure 'dementia_dataset.csv' is in the root directory."
        )

    raw_df = pd.read_csv(DATASET_FILE)

    # Clean target column CDR (Clinical Dementia Rating)
    # Mapping: 0.0 -> Stage 0 (Normal), 0.5 -> Stage 1 (Mild), 1.0 -> Stage 2 (Moderate), 2.0+ -> Stage 3 (Advanced)
    raw_df = raw_df.dropna(subset=['CDR']).copy()

    def map_cdr_to_stage(cdr):
        if cdr == 0.0:
            return 0
        elif cdr == 0.5:
            return 1
        elif cdr == 1.0:
            return 2
        else:
            return 3

    mapped_df = pd.DataFrame()
    mapped_df['stage'] = raw_df['CDR'].apply(map_cdr_to_stage)

    # Base Age
    mapped_df['age'] = raw_df['Age'].values

    # MMSE score inverted scale (Lower MMSE = higher difficulty severity)
    mmse = raw_df['MMSE'].fillna(raw_df['MMSE'].median())
    mmse_severity = np.clip((30.0 - mmse) / 15.0, 0.0, 1.0)

    # Brain Atrophy / Volume metrics (nWBV, eTIV) mapped as daily fatigue / misplacement surrogates
    nwbv = raw_df['nWBV'].fillna(raw_df['nWBV'].median())
    nwbv_severity = np.clip((0.85 - nwbv) / 0.20, 0.0, 1.0)

    # Mapping non-clinical daily life FAQ proxy scores (0.0 = Normal, 0.5 = Mild, 1.0 = Frequent)
    mapped_df['daily_faq_memory_conversations'] = np.round(mmse_severity * 2) / 2.0
    mapped_df['daily_faq_orientation_time_date'] = np.round(mmse_severity * 2) / 2.0
    mapped_df['daily_faq_navigating_places'] = np.round(mmse_severity * 2) / 2.0
    mapped_df['daily_faq_managing_finances'] = np.round(((30.0 - mmse) + (18 - raw_df['EDUC'].fillna(12))) / 30.0 * 2) / 2.0
    mapped_df['daily_faq_managing_finances'] = np.clip(mapped_df['daily_faq_managing_finances'], 0.0, 1.0)

    mapped_df['daily_faq_household_chores'] = np.round(mmse_severity * 2) / 2.0
    mapped_df['daily_faq_learning_new_tasks'] = np.round((raw_df['SES'].fillna(3) / 5.0) * mmse_severity * 2) / 2.0
    mapped_df['daily_faq_learning_new_tasks'] = np.clip(mapped_df['daily_faq_learning_new_tasks'], 0.0, 1.0)

    mapped_df['daily_faq_language_word_finding'] = np.round(mmse_severity * 2) / 2.0
    mapped_df['daily_faq_object_misplacement'] = np.round(nwbv_severity * 2) / 2.0
    mapped_df['daily_faq_object_misplacement'] = np.clip(mapped_df['daily_faq_object_misplacement'], 0.0, 1.0)

    mapped_df['daily_faq_decision_making'] = np.round(mmse_severity * 2) / 2.0

    return mapped_df


def load_or_train_model():
    """Trains a pipeline model on dementia_dataset.csv with leak-prevention train-test splitting."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading existing model pipeline from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)

    print("Loading real dementia dataset and building training pipeline...")
    df = prepare_real_dataset()

    X = df[FEATURE_KEYS]
    y = df['stage']

    # 1. Stratified Train-Test Split to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Pipeline combining imputer and random forest classifier (n_estimators=500, max_depth=25)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42))
    ])

    # Train on training split only
    pipeline.fit(X_train, y_train)

    # Validate model on separate test split
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Real Dataset Pipeline trained successfully! Test Accuracy: {acc * 100:.2f}%")

    # Fit full pipeline for deployment
    pipeline.fit(X, y)

    # Save trained pipeline locally
    try:
        joblib.dump(pipeline, MODEL_FILE)
        print(f"Model pipeline saved to {MODEL_FILE}")
    except Exception as e:
        print(f"Notice: Could not save model file locally ({e}). Continuing with in-memory model.")

    return pipeline


model = load_or_train_model()

# ==============================================================================
# 2. NON-CLINICAL DAILY LIFE INTERFACE (HTML, CSS, JS)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="theme-color" content="#0e0a17">
    <title>Daily Living & Cognitive Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">

    <style>
        :root {
            color-scheme: dark;
            --bg: #0e0a17;
            --text: #efeaf8;
            --muted: #b3a7c9;
            --p1: #b79bf2;
            --p2: #7d55d6;
            --line: #352b4d;
            --glass: rgba(255, 255, 255, .06);
            --card-bg: rgba(22, 16, 38, 0.65);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            position: relative;
            background: var(--bg);
            color: var(--text);
            font: 17px/1.6 'Atkinson Hyperlegible', Verdana, system-ui, sans-serif;
            overflow-x: hidden;
        }

        body::before {
            content: "";
            position: fixed;
            inset: -25%;
            z-index: 0;
            pointer-events: none;
            background:
                radial-gradient(40% 45% at 20% 25%, rgba(125, 85, 214, .5), transparent),
                radial-gradient(35% 40% at 80% 70%, rgba(154, 111, 224, .32), transparent),
                radial-gradient(28% 35% at 55% 5%, rgba(226, 120, 200, .16), transparent);
            animation: drift 22s ease-in-out infinite alternate;
        }

        header {
            position: sticky;
            top: 0; left: 0; right: 0;
            z-index: 5;
            padding: calc(14px + env(safe-area-inset-top, 0px)) 22px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            background: rgba(16, 10, 28, .55);
            -webkit-backdrop-filter: blur(18px) saturate(1.7);
            backdrop-filter: blur(18px) saturate(1.7);
            border-bottom: 1px solid rgba(183, 155, 242, .14);
        }

        .brand { display: flex; align-items: center; gap: 12px; }
        .mark {
            width: 40px; height: 40px;
            border-radius: 50%;
            display: grid; place-items: center;
            font-size: 20px;
            background: radial-gradient(circle at 30% 25%, rgba(255,255,255,.4), rgba(154,111,224,.35) 45%, rgba(75,44,154,.65));
            box-shadow: 0 0 22px rgba(154, 111, 224, .55);
            animation: breathe 6s ease-in-out infinite;
        }
        h1.logo-title { margin: 0; font-size: 22px; letter-spacing: .04em; line-height: 1.1; font-weight: 700; color: var(--text); }
        .tag { display: block; color: var(--muted); font-size: 12px; letter-spacing: 0; margin-top: 2px; font-weight: 400; }
        .right { display: flex; align-items: center; gap: 10px; }

        .home {
            display: inline-flex; align-items: center;
            color: var(--text); text-decoration: none; font-weight: 700;
            padding: 8px 18px; font-size: 14px;
            border: 1px solid rgba(183, 155, 242, .25); border-radius: 999px;
            background: var(--glass); transition: background .25s, transform .25s;
        }
        .home:hover { background: rgba(183, 155, 242, .18); transform: translateY(-1px); }

        .main-container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 820px;
            margin: 0 auto;
            padding: 40px 20px 60px;
            flex: 1;
        }

        .header-box {
            text-align: center;
            margin-bottom: 36px;
            animation: rise .8s both;
        }

        .header-box h2 {
            font-size: clamp(26px, 5vw, 34px);
            margin: 0 0 10px 0;
            color: var(--text);
            line-height: 1.2;
        }

        p.subtitle {
            font-size: 18px;
            color: var(--muted);
            margin: 0;
        }

        .section-header {
            color: var(--p1);
            padding: 8px 0;
            border-bottom: 1px solid var(--line);
            font-size: 20px;
            font-weight: bold;
            margin: 36px 0 20px 0;
            letter-spacing: .02em;
        }

        .form-card {
            background: var(--card-bg);
            border: 1px solid var(--line);
            border-left: 4px solid var(--p2);
            border-radius: 16px;
            padding: 22px 20px;
            margin-bottom: 20px;
            -webkit-backdrop-filter: blur(12px);
            backdrop-filter: blur(12px);
            transition: border-color .25s, box-shadow .25s, transform .25s;
            animation: rise .45s cubic-bezier(.2, .8, .2, 1) both;
        }

        .form-card:focus-within, .form-card:hover {
            border-color: var(--p1);
            border-left-color: var(--p1);
            box-shadow: 0 8px 24px rgba(125, 85, 214, .15);
        }

        label {
            display: block;
            font-size: 18px;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 12px;
            line-height: 1.4;
        }

        input[type="number"], select {
            width: 100%;
            font: inherit;
            font-size: 16px;
            padding: 14px 18px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--glass);
            color: var(--text);
            outline: none;
            transition: border-color .25s, box-shadow .25s;
            appearance: none;
            -webkit-appearance: none;
        }

        select {
            background-image: url("data:image/svg+xml;utf8,<svg fill='%23b3a7c9' height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'><path d='M7 10l5 5 5-5z'/></svg>");
            background-repeat: no-repeat;
            background-position: right 14px center;
            padding-right: 42px;
        }

        select option {
            background-color: #161026;
            color: var(--text);
        }

        input[type="number"]:focus, select:focus {
            border-color: var(--p2);
            box-shadow: 0 0 0 4px rgba(125, 85, 214, .25);
        }

        .btn-submit {
            display: block;
            width: 100%;
            min-height: 54px;
            background: linear-gradient(135deg, var(--p2), #a47cf0);
            color: #FFFFFF;
            font: inherit;
            font-size: 20px;
            font-weight: bold;
            padding: 16px;
            border: none;
            border-radius: 999px;
            cursor: pointer;
            box-shadow: 0 8px 24px rgba(125, 85, 214, .35);
            text-align: center;
            margin-top: 36px;
            transition: transform .25s, box-shadow .25s, opacity .25s;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(125, 85, 214, .5);
        }

        .result-card {
            display: none;
            margin-top: 40px;
            padding: 28px;
            border-radius: 20px;
            text-align: center;
            border: 1px solid var(--line);
            -webkit-backdrop-filter: blur(16px);
            backdrop-filter: blur(16px);
            animation: rise .5s cubic-bezier(.2, .8, .2, 1) both;
        }

        .result-card.stage0 { background: rgba(39, 171, 131, 0.12); border-color: rgba(39, 171, 131, 0.4); color: #7ee0a4; }
        .result-card.stage1 { background: rgba(214, 158, 46, 0.12); border-color: rgba(214, 158, 46, 0.4); color: #f6ad55; }
        .result-card.stage2 { background: rgba(221, 107, 32, 0.12); border-color: rgba(221, 107, 32, 0.4); color: #ed8936; }
        .result-card.stage3 { background: rgba(229, 62, 62, 0.12); border-color: rgba(229, 62, 62, 0.4); color: #fc8181; }

        .result-title { font-size: 28px; font-weight: bold; margin-bottom: 10px; }
        .result-severity { font-size: 20px; font-weight: bold; margin-bottom: 16px; opacity: 0.9; }
        .result-text { font-size: 17px; text-align: left; line-height: 1.7; color: var(--text); }

        .error-banner {
            display: none;
            background: rgba(229, 62, 62, 0.15);
            color: #fc8181;
            border: 1px solid rgba(229, 62, 62, 0.4);
            padding: 14px 20px;
            border-radius: 12px;
            margin-bottom: 24px;
            font-weight: bold;
            text-align: center;
            animation: rise .3s ease;
        }

        footer {
            position: relative;
            z-index: 2;
            padding: 20px;
            text-align: center;
            color: var(--muted);
            font-size: 14px;
            background: rgba(16, 10, 28, .5);
            -webkit-backdrop-filter: blur(18px) saturate(1.7);
            backdrop-filter: blur(18px) saturate(1.7);
            border-top: 1px solid rgba(183, 155, 242, .14);
        }

        @keyframes drift { to { transform: translate(6%, -4%) rotate(14deg) scale(1.12); } }
        @keyframes breathe { 50% { transform: scale(1.07); } }
        @keyframes rise { from { opacity: 0; transform: translateY(18px); } }

        @media (max-width: 600px) {
            header { padding-left: 14px; padding-right: 14px; }
            .tag { display: none; }
            .main-container { padding: 24px 14px 40px; }
            .form-card { padding: 18px 14px; }
            label { font-size: 16px; }
            input[type="number"], select { font-size: 15px; }
            .btn-submit { font-size: 18px; }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="mark" aria-hidden="true">🧠</div>
            <h1 class="logo-title">SVARA<span class="tag">Daily Life & Memory Assessment</span></h1>
        </div>
        <div class="right">
            <a class="home" href="https://svara-homepage.vercel.app">Home</a>
        </div>
    </header>

    <main class="main-container">
        <div class="header-box">
            <h2>Daily Life & Cognitive Assessment</h2>
            <p class="subtitle">Please answer these 10 simple questions about daily routines and activities.</p>
        </div>

        <div id="errorBanner" class="error-banner"></div>

        <form id="assessmentForm">
            <div class="section-header">1. General Information</div>
            <div class="form-card">
                <label for="age">1. What is your current age?</label>
                <input type="number" id="age" min="50" max="105" value="68" required>
            </div>

            <div class="section-header">2. Daily Routines & FAQs</div>

            <div class="form-card">
                <label for="daily_faq_memory_conversations">2. How often do you forget recent conversations or events that happened earlier in the day?</label>
                <select id="daily_faq_memory_conversations" required>
                    <option value="0">Rarely or Never (Normal memory)</option>
                    <option value="0.5">Occasionally (Sometimes forget small details)</option>
                    <option value="1">Frequently (Often forget whole conversations)</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_orientation_time_date">3. How often do you lose track of the date, month, or time of day?</label>
                <select id="daily_faq_orientation_time_date" required>
                    <option value="0">Never (Always aware of date/time)</option>
                    <option value="0.5">Occasionally (Mix up dates briefly)</option>
                    <option value="1">Frequently (Confused about time or day)</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_navigating_places">4. Do you experience hesitation or confusion when walking or driving in familiar neighborhoods?</label>
                <select id="daily_faq_navigating_places" required>
                    <option value="0">No, navigate easily</option>
                    <option value="0.5">Mild hesitation in less frequent routes</option>
                    <option value="1">Frequent confusion in familiar places</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_managing_finances">5. Do you find it challenging to manage daily bills, checkbook, or bank balance?</label>
                <select id="daily_faq_managing_finances" required>
                    <option value="0">No difficulty</option>
                    <option value="0.5">Mild difficulty (Needs occasional review)</option>
                    <option value="1">Significant trouble with routine finances</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_household_chores">6. How difficult is it to complete daily household tasks (e.g. using TV remote, cooking, appliance setup)?</label>
                <select id="daily_faq_household_chores" required>
                    <option value="0">Easy, no help needed</option>
                    <option value="0.5">Occasional pauses or minor mistakes</option>
                    <option value="1">Struggles with routine appliances or steps</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_learning_new_tasks">7. Do you find it harder than usual to adapt to new tools, routines, or instructions?</label>
                <select id="daily_faq_learning_new_tasks" required>
                    <option value="0">Adapts easily</option>
                    <option value="0.5">Takes extra time to learn new things</option>
                    <option value="1">Avoids or struggles with new tasks</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_language_word_finding">8. How often do you struggle to find common words during regular conversations?</label>
                <select id="daily_faq_language_word_finding" required>
                    <option value="0">Rarely (Normal speech)</option>
                    <option value="0.5">Occasional word pauses ("tip of the tongue")</option>
                    <option value="1">Frequent pauses or losing train of thought</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_object_misplacement">9. How often do you misplace everyday objects in unusual locations (e.g., keys in the freezer)?</label>
                <select id="daily_faq_object_misplacement" required>
                    <option value="0">Rarely or never</option>
                    <option value="0.5">Occasionally misplace items</option>
                    <option value="1">Frequently leave items in odd places</option>
                </select>
            </div>

            <div class="form-card">
                <label for="daily_faq_decision_making">10. Do family members notice changes in your daily decision-making or planning ability?</label>
                <select id="daily_faq_decision_making" required>
                    <option value="0">No changes noticed</option>
                    <option value="0.5">Mild occasional lapses</option>
                    <option value="1">Noticeable changes in judgment</option>
                </select>
            </div>

            <button type="submit" class="btn-submit"> Calculate Stage & Result</button>
        </form>

        <div id="resultCard" class="result-card">
            <div id="resultTitle" class="result-title"></div>
            <div id="resultSeverity" class="result-severity"></div>
            <div id="resultText" class="result-text"></div>
        </div>
    </main>

    <footer>
        © 2026 Svara AI. All rights reserved.
    </footer>

<script>
    document.getElementById('assessmentForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const errorBanner = document.getElementById('errorBanner');
        errorBanner.style.display = 'none';

        const payload = {
            age: parseFloat(document.getElementById('age').value),
            daily_faq_memory_conversations: parseFloat(document.getElementById('daily_faq_memory_conversations').value),
            daily_faq_orientation_time_date: parseFloat(document.getElementById('daily_faq_orientation_time_date').value),
            daily_faq_navigating_places: parseFloat(document.getElementById('daily_faq_navigating_places').value),
            daily_faq_managing_finances: parseFloat(document.getElementById('daily_faq_managing_finances').value),
            daily_faq_household_chores: parseFloat(document.getElementById('daily_faq_household_chores').value),
            daily_faq_learning_new_tasks: parseFloat(document.getElementById('daily_faq_learning_new_tasks').value),
            daily_faq_language_word_finding: parseFloat(document.getElementById('daily_faq_language_word_finding').value),
            daily_faq_object_misplacement: parseFloat(document.getElementById('daily_faq_object_misplacement').value),
            daily_faq_decision_making: parseFloat(document.getElementById('daily_faq_decision_making').value)
        };

        if (isNaN(payload.age) || payload.age < 50 || payload.age > 105) {
            errorBanner.innerText = "Please enter a valid age between 50 and 105.";
            errorBanner.style.display = 'block';
            window.scrollTo({ top: 0, behavior: 'smooth' });
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            const resultCard = document.getElementById('resultCard');
            const resultTitle = document.getElementById('resultTitle');
            const resultSeverity = document.getElementById('resultSeverity');
            const resultText = document.getElementById('resultText');

            resultCard.style.display = 'block';
            resultCard.className = "result-card stage" + data.stage;

            resultTitle.innerHTML = data.stage_title;
            resultSeverity.innerHTML = "Severity Level: " + data.severity;
            resultText.innerHTML = data.message;

            resultCard.scrollIntoView({ behavior: 'smooth' });

        } catch (err) {
            errorBanner.innerText = "Unable to connect to backend server. Please verify connection.";
            errorBanner.style.display = 'block';
        }
    });
</script>

</body>
</html>
"""


# ==============================================================================
# 3. FLASK API ENDPOINTS WITH ZERO-SYMPTOM OVERRIDE
# ==============================================================================
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}

        input_vector = [
            float(data.get(feat, 0.0)) for feat in FEATURE_KEYS
        ]

        # Zero-symptom override rule: If all daily FAQ scores are 0
        symptom_values = input_vector[1:]  # Exclude age
        all_zeros = all(val == 0.0 for val in symptom_values)

        if all_zeros:
            return jsonify({
                "stage": 0,
                "stage_title": "🟢 You do not have dementia",
                "severity": "None (All indicators are normal)",
                "message": "All your reported responses are normal (0). You do not have dementia. Keep maintaining your healthy daily lifestyle!"
            })

        # Predict Dementia Stage using the pipeline model trained on real data
        predicted_stage = int(model.predict([input_vector])[0])

        stage_info = {
            0: {
                "stage_title": "🟢 Stage 0: Healthy / Normal",
                "severity": "None to Very Minimal",
                "message": "Your answers indicate healthy cognitive and daily living functions. Keep enjoying your active lifestyle and daily routines!"
            },
            1: {
                "stage_title": "🟡 Stage 1: Very Mild / Early Cognitive Impairment",
                "severity": "Mild Impairment",
                "message": "Your answers suggest mild daily memory or recall changes. It is recommended to discuss these observations during your next routine family doctor check-up."
            },
            2: {
                "stage_title": "🟠 Stage 2: Moderate Stage",
                "severity": "Moderate Impairment",
                "message": "Your answers reflect moderate challenges with daily chores, orientation, or memory. Scheduling a friendly visit with a physician for a thorough review is advised."
            },
            3: {
                "stage_title": "🔴 Stage 3: Advanced / Severe Stage",
                "severity": "Higher Care & Attention Required",
                "message": "Your responses show noticeable daily care needs and safety considerations. We strongly recommend speaking with a healthcare professional and family caregivers for supportive care guidance."
            }
        }

        response_data = stage_info.get(predicted_stage, stage_info[0])
        response_data["stage"] = predicted_stage

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "stage": 0,
            "stage_title": "⚠️ Processing Error",
            "severity": "Invalid Input",
            "message": "An error occurred while processing your inputs. Please ensure all fields are selected correctly."
        }), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Svara Assessment Application on port {port}...")
    app.run(host='0.0.0.0', port=port)
