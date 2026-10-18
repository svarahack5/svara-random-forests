import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==============================================================================
# 1. MODEL & DATASET MANAGEMENT
# ==============================================================================
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model.joblib')
DATASET_FILE = 'dementia_multi_stage_dataset.csv'

FEATURE_KEYS = [
    'age', 'speech_find_words', 'memory_recent_events', 'memory_repeat_questions',
    'orientation_date_time', 'orientation_familiar_places', 'tasks_managing_money',
    'tasks_using_appliances', 'tasks_cooking_meals', 'mood_unmotivated_apathy',
    'mood_anxiety_agitation', 'mood_personality_change', 'motor_misplacing_items',
    'motor_trouble_dressing', 'sleep_restlessness', 'social_withdrawal',
    'safety_wandering', 'judgment_poor_decisions'
]


def generate_multi_stage_dataset():
    """Generates a synthetic dataset with 18 self-assessment parameters and 4 dementia stages."""
    np.random.seed(42)
    n_samples = 3000

    data = {'age': np.random.randint(50, 96, n_samples)}

    for feat in FEATURE_KEYS[1:]:
        data[feat] = np.random.choice([0.0, 0.5, 1.0], size=n_samples, p=[0.55, 0.30, 0.15])

    df = pd.DataFrame(data)

    weighted_score = (
            (df['age'] > 75).astype(int) * 1.5 +
            df['speech_find_words'] * 1.5 +
            df['memory_recent_events'] * 2.0 +
            df['memory_repeat_questions'] * 1.5 +
            df['orientation_date_time'] * 2.0 +
            df['orientation_familiar_places'] * 2.5 +
            df['tasks_managing_money'] * 2.0 +
            df['tasks_using_appliances'] * 1.5 +
            df['tasks_cooking_meals'] * 1.5 +
            df['mood_unmotivated_apathy'] * 1.0 +
            df['mood_anxiety_agitation'] * 1.0 +
            df['mood_personality_change'] * 1.5 +
            df['motor_misplacing_items'] * 1.0 +
            df['motor_trouble_dressing'] * 2.5 +
            df['sleep_restlessness'] * 1.0 +
            df['social_withdrawal'] * 1.0 +
            df['safety_wandering'] * 3.0 +
            df['judgment_poor_decisions'] * 1.5
    )

    def calculate_stage(score):
        if score < 5.0:
            return 0
        elif score < 12.0:
            return 1
        elif score < 20.0:
            return 2
        else:
            return 3

    df['stage'] = weighted_score.apply(calculate_stage)
    df.to_csv(DATASET_FILE, index=False)
    return df


def load_or_train_model():
    """Loads a pre-trained joblib model if present; otherwise generates synthetic data and trains it."""
    if os.path.exists(MODEL_FILE):
        print(f"Loading existing model from {MODEL_FILE}...")
        return joblib.load(MODEL_FILE)

    print("No joblib model found. Generating synthetic dataset and training model...")
    if not os.path.exists(DATASET_FILE):
        df = generate_multi_stage_dataset()
    else:
        df = pd.read_csv(DATASET_FILE)

    X = df[FEATURE_KEYS]
    y = df['stage']

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X, y)

    # Save model locally for quick reuse
    try:
        joblib.dump(clf, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")
    except Exception as e:
        print(f"Notice: Could not save model file locally ({e}). Continuing with in-memory model.")

    return clf


model = load_or_train_model()

# ==============================================================================
# 2. SENIOR-FRIENDLY EMBEDDED HTML, CSS, AND JS INTERFACE
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta name="theme-color" content="#0e0a17">
    <title>Dementia Stage Analysis</title>
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

        /* drifting aurora behind everything */
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

        /* translucent header */
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

        .btn-submit:active {
            transform: translateY(0);
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

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation: none !important; transition-duration: .01s !important; }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="mark" aria-hidden="true">🧠</div>
            <h1 class="logo-title">SVARA<span class="tag">Comprehensive Mind & Memory Assessment</span></h1>
        </div>
        <div class="right">
            <a class="home" href="https://svara-homepage.vercel.app">Home</a>
        </div>
    </header>

    <main class="main-container">
        <div class="header-box">
            <h2>Dementia Stage Analysis</h2>
            <p class="subtitle">Please complete these 18 easy daily living questions.</p>
        </div>

        <div id="errorBanner" class="error-banner"></div>

        <form id="assessmentForm">
            <div class="section-header">1. General Information</div>
            <div class="form-card">
                <label for="age">1. What is your current age?</label>
                <input type="number" id="age" min="50" max="105" value="68" required>
            </div>

            <div class="section-header">2. Memory & Word Recall</div>
            <div class="form-card">
                <label for="speech_find_words">2. Difficulty finding the right words or speaking smoothly?</label>
                <select id="speech_find_words" required>
                    <option value="0">No, speaking is normal</option>
                    <option value="0.5">Mild difficulty (Occasional word pausing)</option>
                    <option value="1">Frequent difficulty completing sentences</option>
                </select>
            </div>

            <div class="form-card">
                <label for="memory_recent_events">3. Forgetting recent events, conversations, or messages?</label>
                <select id="memory_recent_events" required>
                    <option value="0">No problem</option>
                    <option value="0.5">Mild problem (Sometimes forget details)</option>
                    <option value="1">Frequent problem forgetting recent events</option>
                </select>
            </div>

            <div class="form-card">
                <label for="memory_repeat_questions">4. Repeating the same questions or stories in a single conversation?</label>
                <select id="memory_repeat_questions" required>
                    <option value="0">No repetition</option>
                    <option value="0.5">Mild (Occasional repetition)</option>
                    <option value="1">Frequent repetition</option>
                </select>
            </div>

            <div class="section-header">3. Time & Orientation</div>
            <div class="form-card">
                <label for="orientation_date_time">5. Feeling confused about the day of the week, month, or time of day?</label>
                <select id="orientation_date_time" required>
                    <option value="0">No confusion</option>
                    <option value="0.5">Mild (Occasionally mix up dates)</option>
                    <option value="1">Frequent time/date confusion</option>
                </select>
            </div>

            <div class="form-card">
                <label for="orientation_familiar_places">6. Getting lost or confused in familiar places or neighborhoods?</label>
                <select id="orientation_familiar_places" required>
                    <option value="0">Never lost</option>
                    <option value="0.5">Mild hesitation in places</option>
                    <option value="1">Frequent confusion in familiar places</option>
                </select>
            </div>

            <div class="section-header">4. Managing Chores & Tasks</div>
            <div class="form-card">
                <label for="tasks_managing_money">7. Trouble managing bank accounts, cash, or paying bills?</label>
                <select id="tasks_managing_money" required>
                    <option value="0">No trouble</option>
                    <option value="0.5">Mild trouble (Needs occasional help)</option>
                    <option value="1">Significant difficulty managing money</option>
                </select>
            </div>

            <div class="form-card">
                <label for="tasks_using_appliances">8. Difficulty operating household appliances (TV remote, stove, microwave)?</label>
                <select id="tasks_using_appliances" required>
                    <option value="0">Operates easily</option>
                    <option value="0.5">Mild hesitation</option>
                    <option value="1">Frequent trouble using appliances</option>
                </select>
            </div>

            <div class="form-card">
                <label for="tasks_cooking_meals">9. Trouble preparing or cooking regular meals safely?</label>
                <select id="tasks_cooking_meals" required>
                    <option value="0">Cooks normally</option>
                    <option value="0.5">Mild difficulty with steps</option>
                    <option value="1">Struggles to prepare meals</option>
                </select>
            </div>

            <div class="section-header">5. Mood & Personality</div>
            <div class="form-card">
                <label for="mood_unmotivated_apathy">10. Loss of interest in hobbies, family activities, or daily routine?</label>
                <select id="mood_unmotivated_apathy" required>
                    <option value="0">Stays engaged</option>
                    <option value="0.5">Mild lack of motivation</option>
                    <option value="1">Significant loss of interest</option>
                </select>
            </div>

            <div class="form-card">
                <label for="mood_anxiety_agitation">11. Uncharacteristic feelings of worry, restlessness, or irritability?</label>
                <select id="mood_anxiety_agitation" required>
                    <option value="0">No noticeable anxiety</option>
                    <option value="0.5">Mild occasional worry</option>
                    <option value="1">Frequent agitation or anxiety</option>
                </select>
            </div>

            <div class="form-card">
                <label for="mood_personality_change">12. Noticeable changes in personality or social behavior?</label>
                <select id="mood_personality_change" required>
                    <option value="0">No change</option>
                    <option value="0.5">Mild subtle changes</option>
                    <option value="1">Clear personality changes</option>
                </select>
            </div>

            <div class="section-header">6. Daily Routines & Safety</div>
            <div class="form-card">
                <label for="motor_misplacing_items">13. Misplacing items in unusual places (e.g. keys in the fridge)?</label>
                <select id="motor_misplacing_items" required>
                    <option value="0">Rarely or never</option>
                    <option value="0.5">Occasional misplacement</option>
                    <option value="1">Frequent misplacement in odd spots</option>
                </select>
            </div>

            <div class="form-card">
                <label for="motor_trouble_dressing">14. Trouble picking out clothing or dressing in proper order?</label>
                <select id="motor_trouble_dressing" required>
                    <option value="0">Dresses without help</option>
                    <option value="0.5">Mild struggle with buttons/shoes</option>
                    <option value="1">Needs assistance dressing</option>
                </select>
            </div>

            <div class="form-card">
                <label for="sleep_restlessness">15. Restless sleep or walking around during the night?</label>
                <select id="sleep_restlessness" required>
                    <option value="0">Sleeps well</option>
                    <option value="0.5">Mild restlessness</option>
                    <option value="1">Frequent nighttime awakening</option>
                </select>
            </div>

            <div class="form-card">
                <label for="social_withdrawal">16. Withdrawing from conversations or group social activities?</label>
                <select id="social_withdrawal" required>
                    <option value="0">Active socially</option>
                    <option value="0.5">Mild quietness</option>
                    <option value="1">Withdraws from conversations</option>
                </select>
            </div>

            <div class="form-card">
                <label for="safety_wandering">17. Wandering away or walking outside without knowing the destination?</label>
                <select id="safety_wandering" required>
                    <option value="0">Never happens</option>
                    <option value="0.5">Occasional wandering tendency</option>
                    <option value="1">Frequent wandering concern</option>
                </select>
            </div>

            <div class="form-card">
                <label for="judgment_poor_decisions">18. Making risky or unusually poor decisions (e.g. safety or scams)?</label>
                <select id="judgment_poor_decisions" required>
                    <option value="0">Normal judgment</option>
                    <option value="0.5">Occasional minor lapses</option>
                    <option value="1">Frequent poor decision making</option>
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
            speech_find_words: parseFloat(document.getElementById('speech_find_words').value),
            memory_recent_events: parseFloat(document.getElementById('memory_recent_events').value),
            memory_repeat_questions: parseFloat(document.getElementById('memory_repeat_questions').value),
            orientation_date_time: parseFloat(document.getElementById('orientation_date_time').value),
            orientation_familiar_places: parseFloat(document.getElementById('orientation_familiar_places').value),
            tasks_managing_money: parseFloat(document.getElementById('tasks_managing_money').value),
            tasks_using_appliances: parseFloat(document.getElementById('tasks_using_appliances').value),
            tasks_cooking_meals: parseFloat(document.getElementById('tasks_cooking_meals').value),
            mood_unmotivated_apathy: parseFloat(document.getElementById('mood_unmotivated_apathy').value),
            mood_anxiety_agitation: parseFloat(document.getElementById('mood_anxiety_agitation').value),
            mood_personality_change: parseFloat(document.getElementById('mood_personality_change').value),
            motor_misplacing_items: parseFloat(document.getElementById('motor_misplacing_items').value),
            motor_trouble_dressing: parseFloat(document.getElementById('motor_trouble_dressing').value),
            sleep_restlessness: parseFloat(document.getElementById('sleep_restlessness').value),
            social_withdrawal: parseFloat(document.getElementById('social_withdrawal').value),
            safety_wandering: parseFloat(document.getElementById('safety_wandering').value),
            judgment_poor_decisions: parseFloat(document.getElementById('judgment_poor_decisions').value)
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
# 3. FLASK API ENDPOINTS WITH ERROR HANDLING & ZERO-SYMPTOM OVERRIDE
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

        # ERROR HANDLING / OVERRIDE RULE: If all symptom fields are 0
        symptom_values = input_vector[1:]  # Exclude age
        all_zeros = all(val == 0.0 for val in symptom_values)

        if all_zeros:
            return jsonify({
                "stage": 0,
                "stage_title": "🟢 You do not have dementia",
                "severity": "None (All indicators are normal)",
                "message": "All your reported symptom responses are normal (0). You do not have dementia. Keep maintaining your healthy daily lifestyle!"
            })

        # Predict Dementia Stage using Random Forest Model
        predicted_stage = int(model.predict([input_vector])[0])

        stage_info = {
            0: {
                "stage_title": "🟢 Stage 0: Healthy / Normal",
                "severity": "None to Very Minimal",
                "message": "Your answers indicate healthy cognitive and daily living functions. Keep enjoying your active lifestyle and daily routines!"
            },
            1: {
                "stage_title": "🟡 Stage 1: Early Stage / Mild Cognitive Impairment",
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
    print(f"Starting Multi-Stage Dementia Assessment Application on port {port}...")
    app.run(host='0.0.0.0', port=port)
