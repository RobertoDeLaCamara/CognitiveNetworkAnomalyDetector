# Triple-Engine Network Anomaly Detection: Building a Weighted Ensemble with Isolation Forest, LSTM Autoencoder, and Rule-Based Heuristics

Network anomaly detection is a solved problem only if you ignore the edge cases. A single ML model trained on synthetic baseline traffic will catch the obvious stuff — port scans, bandwidth floods — but it will miss the slow, subtle deviations that represent the most dangerous intrusions. This article walks through the architecture of a triple-engine ensemble detector built around Scapy packet capture, explaining the trade-offs behind each design choice and the lessons learned from running it on a live interface.

---

## The Core Problem: No Single Model Generalizes

Isolation Forest is excellent at finding statistical outliers in tabular feature space. It can detect an IP behaving unusually compared to its own historical baseline, but it requires at least 10 packets before it has enough context to run, and it operates on aggregate features — not raw packet timing. It will miss a slow-scanning attacker who carefully stays within normal rate bounds.

LSTM Autoencoders capture temporal patterns. By maintaining a per-IP sequence buffer of the last 50 feature vectors, the LSTM learns what normal *transitions* look like for each source. An IP that alternates between HTTP and DNS traffic will have a very different reconstruction pattern than one that suddenly pivots to probing high-numbered ports. But the LSTM requires 50 packets before its buffer fills, so early-stage attacks are invisible to it.

Rule-based engines are deterministic and instantaneous. They fire on the first packet. They catch ICMP floods, payload fragments containing `UNION SELECT`, directory traversal strings like `../../../etc/passwd`, and traffic spikes beyond 2× the rolling average — all without any training data. But they produce false positives on legitimate traffic that happens to match a pattern.

None of these alone is sufficient. Together, with appropriate weighting, they cover each other's blind spots.

---

## Architecture Overview

```
Scapy sniff() → bounded queue (10,000 pkts) → 2 async workers
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
         analyze_packet(pkt)
              │
              ├─ Feature Extractor → 18-dim vector per source IP
              │   (min 3 packets; sliding window last 1000, 60s)
              │
              ├─ Rule Engine (runs on every packet)
              │   Traffic spike / ICMP flood / uncommon port /
              │   large payload / 30+ malicious patterns
              │
              ├─ Isolation Forest (requires ≥10 packets/IP)
              │   IF.predict(vector) → score [0,1]
              │
              └─ LSTM Autoencoder (requires 50-packet buffer)
                  update per-IP deque → predict if full
                  reconstruction error → score [0,1]
                          │
              Ensemble Scorer (IF×0.4 + LSTM×0.4 + Rules×0.2)
                          │
              combined_score ≥ 0.6 + cooldown check
                          │
              Alert → SQLite + log file + console
                          │
              Streamlit dashboard :8501
```

The queue is bounded at 10,000 packets. If the workers fall behind during a traffic burst, new packets are dropped and the drop count is tracked. This is a deliberate trade-off: the alternative — an unbounded queue — would let memory grow without limit under sustained high-throughput attacks, which is itself a denial-of-service vector.

---

## Feature Engineering: 18 Numbers That Describe Host Behavior

Every inference pipeline lives or dies on its features. For a network anomaly detector operating on raw packets (not flows), the challenge is extracting meaningful signal from incomplete information. We don't have full flow records — we have per-packet metadata and must reconstruct behavioral profiles incrementally.

The 18 features fall into five categories:

**Statistical (6)**: packets/sec, bytes/sec, average packet size, packet size variance, total packets, total bytes. These capture the volume and shape of traffic from a source.

**Temporal (4)**: inter-arrival time mean, inter-arrival time standard deviation, burst rate (last 5 seconds), session duration. IAT std is particularly discriminative — normal user traffic has high variance (web browsing produces irregular bursts), while automated scanners tend to send packets at fixed intervals.

**Protocol (3)**: TCP ratio, UDP ratio, ICMP ratio. An IP that suddenly shifts from 95% TCP to 60% ICMP is exhibiting behavior worth investigating.

**Port (2)**: unique destination port count, uncommon port ratio. Scanning behavior shows up clearly in these features — legitimate clients connect to a small set of known ports; scanners visit hundreds.

**Payload (3)**: average Shannon entropy, average payload size, payload size variance. Encrypted traffic and compressed data have high entropy; many exploit payloads contain structured low-entropy data.

All 18 features are extracted from a sliding window of the last 1,000 packets (or 60 seconds, whichever is shorter) per source IP. The window approach means the detector adapts to changing behavior — an IP that was normal for an hour but suddenly starts scanning will trigger detection without being permanently contaminated by its earlier good behavior.

### Why Shannon Entropy Matters

Payload entropy is underused in most network IDS implementations. The Shannon entropy formula:

```
H = -Σ p(x) × log₂(p(x))
```

applied to individual packet payload bytes discriminates between:

- **High entropy (≥7.0 bits/byte)**: encrypted or compressed data — normal HTTPS, SSH, video streams
- **Medium entropy (4.0–7.0)**: typical text, JSON, XML
- **Low entropy (<4.0)**: structured binary formats, certain exploit payloads, command-and-control beacon traffic with templated payloads

A sudden drop in entropy from a source that was previously sending encrypted traffic is a meaningful signal — it might indicate a TLS stripping attack or a compromised endpoint that switched from encrypted to plaintext C2 communication.

---

## The Three Engines in Detail

### Isolation Forest: Statistical Baseline Deviation

```python
# src/config/ml_config.py
contamination = 0.01   # assume 1% of traffic is anomalous
n_estimators  = 100    # 100 trees per forest
```

The model is trained on synthetic baseline data generated by `scripts/generate_synthetic_data.py`, which simulates normal traffic patterns across HTTP, DNS, SSH, database, and streaming protocols. The contamination parameter tells sklearn to expect 1% of training samples to be outliers — this affects the decision boundary placement.

Score normalization converts sklearn's `decision_function` output (typically `[-0.5, 0.5]`) to a `[0,1]` anomaly score:

```
normalized_score = (0.5 - decision_score).clip(0, 1)
```

A `decision_score` of `0.5` (deep inside normal territory) maps to `0.0`. A `decision_score` of `-0.5` (clear outlier) maps to `1.0`. The clipping handles the small fraction of cases where scores fall outside the typical range.

The model uses a fallback loading chain: MLflow registry → local joblib file → rules-only mode. In production without MLflow, the local file path is `models/isolation_forest_v{MODEL_VERSION}.joblib`.

### LSTM Autoencoder: Temporal Pattern Reconstruction

The LSTM Autoencoder is fundamentally different from the Isolation Forest: instead of asking "does this IP's current feature vector look unusual?", it asks "does this IP's recent *sequence* of behavior match what I'd expect?"

```
Architecture:
  Input: (50, 18) — 50 timesteps, 18 features per step
  Encoder: LSTM(hidden=64, layers=2) → Dense(32)  # latent
  Decoder: LSTM(hidden=64, layers=2, dropout=0.2) → (50, 18)
  Loss:    MSE(input_sequence, reconstructed_sequence)
```

During training on normal traffic, the autoencoder learns to compress and reconstruct typical behavioral sequences. At inference time, high reconstruction error (the model can't "explain" what it's seeing) indicates an anomaly.

The key design choice is per-IP sequence buffers using Python `deque(maxlen=50)`. Each source IP maintains its own LSTM input buffer, updated with every new feature vector. Inference only runs when the buffer is exactly full — partial buffers produce no score and the LSTM weight is redistributed to the other engines.

This matters for detecting slow attacks. An IP that over the course of 50 packets gradually increases its port scan rate will show up in the LSTM reconstruction error even if each individual packet, considered alone, looks benign.

**Hot reload** is worth highlighting: a background thread checks the model file's mtime every 60 seconds. When it changes, the model is reloaded in-place without interrupting detection. This enables live model updates — you can retrain overnight on accumulated traffic and deploy without a service restart.

### Rule-Based Engine: Deterministic, Instantaneous, Zero Training Required

The rule engine fires on the first packet and doesn't require any accumulated history. It catches:

- **Traffic spikes**: rate > average × 2.0
- **ICMP floods**: count > 50 in the current window
- **Uncommon ports**: TCP/UDP to any port not in the 14-port whitelist (80, 443, 53, 22, 21, 25, 587, 993, 995, 3306, 5432, 6379, 27017, 8080)
- **Large payloads**: payload size > 100 bytes
- **30+ malicious patterns**: SQL injection, command injection, web shells, XSS, directory traversal, file inclusion

The malicious pattern matching runs each regex in a daemon thread with a 1-second timeout:

```python
def _match_with_timeout(pattern, payload, timeout=1.0):
    result = []
    def _run():
        if re.search(pattern, payload):
            result.append(True)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return bool(result)
```

Without this guard, a crafted packet could trigger catastrophic backtracking in a vulnerable regex, blocking all packet processing indefinitely. The timeout ensures the worker thread moves on regardless.

Patterns are sorted longest-first before matching. This is a false-positive reduction technique: matching `UNION SELECT * FROM` before `UNION` means the specific pattern wins, preventing generic matches from firing on legitimate traffic.

---

## Ensemble Scoring: Making Three Numbers Into One Decision

```python
@dataclass
class EnsembleResult:
    confidence_score: float   # weighted combination [0,1]
    is_anomaly: bool          # score >= ENSEMBLE_ANOMALY_THRESHOLD (0.6)
    engines: dict             # per-engine scores and metadata
```

The default weights are IF: 40%, LSTM: 40%, Rules: 20%. The logic behind this allocation:

- IF and LSTM get equal weight because both are ML-based and cover different aspects of anomaly detection (spatial vs. temporal). Giving one priority over the other without evidence would be arbitrary.
- Rules get 20% because they're deterministic, fast, and high-precision for known patterns — but low-recall for novel attacks.

**Dynamic redistribution** handles the case where an engine can't score yet:

```
if LSTM buffer not full (< 50 packets):
    LSTM weight redistributed proportionally to IF and Rules
    → effective weights: IF = 0.4 / 0.6 × 0.8 = 53%, Rules = 0.2 / 0.6 × 0.8 = 27%
    (the exact math: each engine's weight / sum_of_available_weights)
```

This means the detector doesn't just stop working for new IPs — it gracefully degrades to the available engines while the LSTM buffer fills.

The threshold of 0.6 was chosen to balance false positive rate against detection coverage. At 0.6, a single engine scoring 1.0 is not sufficient to trigger an alert (IF alone at max score would contribute 0.4 with dynamic redistribution contributing the remainder). Two engines need to agree, or one needs to score high while another scores moderately.

---

## Security Considerations That Aren't Obvious

### The Pickle Problem

Loading ML models from disk using Python's pickle (which joblib uses internally) is a code execution vulnerability. A maliciously crafted `.joblib` file can execute arbitrary Python when loaded. The detector addresses this with a custom safe unpickler:

```python
ALLOWED_MODULES = {
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'numpy',
    'numpy.core.multiarray',
}
```

Any attempt to deserialize a class from a module outside this allowlist raises an exception. This prevents the attack vector where an adversary replaces the model file with a weaponized one.

### Memory Exhaustion Defense

A network anomaly detector that tracks per-IP state is vulnerable to a resource exhaustion attack: flood the detector with traffic from millions of unique source IPs, consuming all available memory.

The hard cap of 10,000 tracked IPs addresses this. When the limit is reached, the least-active 20% of IPs (by packet count) are evicted — not the oldest. This preserves behavioral history for high-traffic IPs (which are more likely to be legitimate and more likely to exhibit interesting anomalies) while making room for new sources.

### Log Injection

Before any string is written to the log file, IP addresses are replaced with `[REDACTED-IP]` and absolute paths with `[REDACTED-PATH]`. Without this, an attacker who can control their source IP (via spoofing or by naming their reverse DNS record maliciously) could inject fake log entries or log-monitoring alerts.

---

## Operational Lessons

**Start with the synthetic baseline, but augment with real traffic quickly.** The synthetic data generator produces reasonable normal traffic, but real networks have quirks — internal monitoring systems that ping every host, backup jobs that generate large transfers at 3 AM, DNS resolvers with unusual query patterns. Retraining the Isolation Forest on 24 hours of real traffic dramatically reduces false positives.

**The LSTM hot reload is more valuable than it sounds.** Being able to update the model without restarting the detector means you can iterate on the model while it's running in production. Retrain on accumulated data, drop the new `.pt` file in the `models/` directory, and the detector picks it up within 60 seconds with no service interruption and no packets dropped.

**Alert rate limiting is essential.** Without it, a single attacking IP can generate thousands of alerts per minute, overwhelming any downstream system. The 3-alerts-per-60-seconds-per-(IP, type) limit prevents this while preserving detection coverage — ML inference still runs on every packet, so you're not blind during the cooldown period.

**The dashboard is for humans, not automation.** The 8-page Streamlit dashboard reads directly from SQLite with no caching layer. For automated alert processing, write a consumer that tails the SQLite database or subscribes to the log file. The dashboard is optimized for manual investigation, not for feeding pipelines.

---

## What's Missing

This detector lacks several capabilities that a production deployment would need:

**No flow-level analysis.** The feature extractor operates on individual packets and maintains per-IP history, but it doesn't reconstruct TCP flows or correlate packets into sessions. CICFlowMeter-style flow features (TCP flags, packet IAT within a flow, handshake timing) are more discriminative for many attack types.

**No certificate/TLS inspection.** JA3 fingerprinting is absent. Two HTTPS connections with identical source/destination IPs can behave identically at the packet level but differ entirely at the TLS layer — one legitimate browser, one malware using a custom TLS implementation.

**No lateral movement detection.** The detector treats each IP independently. Coordinated attacks that use multiple source IPs to probe a target in sequence (each individual IP staying below the anomaly threshold) will not be detected.

**Single-host only.** There's no correlation across multiple sensors. A distributed attack where each node sends sub-threshold traffic to the same target won't trigger any single detector.

These limitations are well-understood and intentional scope decisions — the project targets a single-host deployment scenario where simplicity and reliability take priority over comprehensive coverage.

---

## Conclusion

The triple-engine architecture exists because no single approach to anomaly detection is sufficient. Isolation Forest provides statistical outlier detection with no latency requirement. LSTM Autoencoder adds temporal context that catches slow behavioral changes. Rule-based matching provides deterministic, zero-latency coverage for known attack patterns.

The ensemble isn't magic — it's a principled combination of three different views of the same traffic, with weights that reflect confidence in each view and dynamic redistribution that handles the reality that not all views are always available.

The full implementation is at `cognitive-anomaly-detector/`. Start with `scripts/generate_synthetic_data.py` and `scripts/train_model.py` before running the detector.
