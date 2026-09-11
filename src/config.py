"""Constants: intents, thresholds, model names. No other module hardcodes these."""

# Phase 0 decision — brand slice for all filtering / retrieval.
BRAND_HANDLE = "AppleSupport"

# Phase 2 — labels from notebook keyword clustering (+ other for residual).
INTENTS: list[str] = [
    "ios_update_bug",
    "battery_performance",
    "connectivity",
    "app_crash",
    "icloud_account",
    "hardware",
    "purchase_billing",
    "other",
]

# Keyword cues for the Phase 2 rule classifier (scored in intents.py).
INTENT_KEYWORDS: dict[str, list[str]] = {
    "ios_update_bug": ["ios", "update", "beta", "upgraded", "software update"],
    "battery_performance": ["battery", "drain", "drains", "charging", "power bank"],
    "connectivity": ["wifi", "wi-fi", "bluetooth", "signal", "cellular", "lte", "network"],
    "app_crash": ["crash", "crashes", "freeze", "frozen", "restart", "reboot", "not working"],
    "icloud_account": ["icloud", "apple id", "appleid", "password", "login", "sign in", "2fa"],
    "hardware": ["iphone", "ipad", "macbook", "watch", "airpods", "screen", "camera", "speaker"],
    "purchase_billing": [
        "bill",
        "billing",
        "refund",
        "purchase",
        "subscription",
        "app store",
        "payment",
        "charged",
    ],
    "other": [],
}

# Phase 3 — escalation labeling cues (also used later by escalation.py).
ESCALATION_SAFETY_KEYWORDS: list[str] = [
    "hacked",
    "stolen",
    "lawyer",
    "lawsuit",
    "attorney",
    "police",
    "threat",
    "suicide",
    "kill myself",
]
ESCALATION_LOW_CONFIDENCE = 0.4
ESCALATION_LOW_RETRIEVAL = 0.35
ESCALATION_SAFETY_INTENTS: list[str] = [
    "icloud_account",
    "purchase_billing",
]
GOLDEN_PER_INTENT_TARGET = 25
GOLDEN_PER_INTENT_CAP = 40
GOLDEN_MIN_MESSAGE_CHARS = 12
GOLDEN_SAMPLE_SEED = 42

# Phase 5 — retrieval
RETRIEVAL_K = 3

# Phase 6 — LLM reply drafting (Groq)
LLM_MODEL = "openai/gpt-oss-20b"
LLM_MAX_TOKENS = 300
LLM_MIN_COMPLETION_CHARS = 20  # retry/fallback below this (empty or truncated stubs)
LLM_MAX_RETRIES = 3            # fail fast on exhausted TPM (not 6+)
LLM_BASE_BACKOFF_SECONDS = 2.0
LLM_MAX_BACKOFF_SECONDS = 15.0  # cap — never climb into multi-minute sleeps
LLM_MIN_REQUEST_INTERVAL = 0.5  # shared limiter across worker threads

# Phase 9/12 — batch eval efficiency (real LLM path)
EVAL_LLM_WORKERS = 3          # keep under Groq free-tier TPM
EVAL_REPLY_SAMPLE = 40        # LLM reply+ablation sample; 0 = full golden set
EVAL_SAMPLE_SEED = 42
