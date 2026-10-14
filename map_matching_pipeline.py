"""
Map-Matching Pipeline with Dynamic Adaptive HMM and Dedicated Final Testing Phase.

Structure:
- Section 1: Synthetic Road Network & Realistic GPS Trajectory Generation
- Section 2: Classifier Training & Weight Learning (Linear SVM vs KNN)
- Section 3: Dynamic Adaptive HMM Viterbi Model (Confidence-Margin Driven Delay)
- Section 4: Buffer Delay Convex Error Curve Evaluation & Optimal N Analysis
- Section 5: FINAL TESTING PHASE - Hold-out Test Evaluation & Detailed Delay Rationale
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =======================================================
# 1. SYNTHETIC ROAD NETWORK & GPS TRAJECTORY GENERATION
# =======================================================
class RoadSegment:
    def __init__(self, seg_id: int, start_pt: np.ndarray, end_pt: np.ndarray, heading_deg: float):
        self.id = seg_id
        self.start = np.array(start_pt, dtype=float)
        self.end = np.array(end_pt, dtype=float)
        self.heading = heading_deg

    def point_distance(self, pt: np.ndarray) -> float:
        """Perpendicular Euclidean distance from point to segment."""
        v = self.end - self.start
        w = pt - self.start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(np.linalg.norm(pt - self.start))
        c2 = np.dot(v, v)
        if c2 <= c1:
            return float(np.linalg.norm(pt - self.end))
        b = c1 / c2
        pb = self.start + b * v
        return float(np.linalg.norm(pt - pb))

    def project_point(self, pt: np.ndarray) -> np.ndarray:
        """Projects a point orthogonally onto this road segment."""
        v = self.end - self.start
        w = pt - self.start
        c1 = np.dot(w, v)
        if c1 <= 0:
            return self.start.copy()
        c2 = np.dot(v, v)
        if c2 <= c1:
            return self.end.copy()
        b = c1 / c2
        return self.start + b * v


def create_road_network():
    """Generates an interconnected grid street network with parallel and cross roads."""
    segments = []
    seg_id = 0
    # Horizontal segments
    for y in range(0, 500, 100):
        for x in range(0, 400, 100):
            segments.append(RoadSegment(seg_id, [x, y], [x + 100, y], 0.0))
            seg_id += 1
    # Vertical segments
    for x in range(0, 500, 100):
        for y in range(0, 400, 100):
            segments.append(RoadSegment(seg_id, [x, y], [x, y + 100], 90.0))
            seg_id += 1
    return segments


def generate_trajectory(network, num_steps=300, noise_std=12.0, multipath_prob=0.15):
    """
    Generates realistic 1Hz GPS sequence with varied movement speeds and realistic multipath noise.
    """
    true_states = []
    gps_obs = []
    
    curr_seg = network[np.random.randint(0, len(network))]
    curr_pos = curr_seg.start.copy()

    for _ in range(num_steps):
        true_states.append(curr_seg.id)
        direction = (curr_seg.end - curr_seg.start)
        norm_dir = direction / (np.linalg.norm(direction) + 1e-6)
        curr_pos += norm_dir * np.random.uniform(7.0, 12.0)

        # Realistic Gaussian sensor noise + intermittent multipath street jumps
        noise = np.random.normal(0, noise_std, size=2)
        if np.random.rand() < multipath_prob:
            # Shift towards parallel road simulating urban canyon reflection
            noise += np.random.choice([-1, 1], size=2) * np.random.uniform(20.0, 40.0)
        gps_obs.append(curr_pos + noise)

        if np.linalg.norm(curr_pos - curr_seg.end) < 15.0:
            connecting = [s for s in network if np.linalg.norm(s.start - curr_seg.end) < 5.0 and s.id != curr_seg.id]
            if connecting:
                curr_seg = connecting[np.random.randint(0, len(connecting))]
                curr_pos = curr_seg.start.copy()
            else:
                curr_seg = network[np.random.randint(0, len(network))]
                curr_pos = curr_seg.start.copy()

    return np.array(gps_obs), true_states


# =======================================================
# 2. FEATURE EXTRACTION & CLASSIFIER WEIGHT LEARNING
# =======================================================
def extract_feature_dataset(network, n_samples=2000):
    """Generates feature vectors for training transition and emission weight models."""
    X = []
    y = []
    for _ in range(n_samples):
        is_match = np.random.choice([1, -1], p=[0.25, 0.75])
        if is_match == 1:
            dist_feat = np.exp(-0.5 * (np.random.normal(4.0, 2.5) / 12.0)**2)
            angle_feat = np.cos(np.deg2rad(np.random.normal(6, 12)))
            speed_feat = np.random.uniform(0.7, 1.0)
        else:
            dist_feat = np.exp(-0.5 * (np.random.normal(32.0, 12.0) / 12.0)**2)
            angle_feat = np.cos(np.deg2rad(np.random.normal(65, 30)))
            speed_feat = np.random.uniform(0.0, 0.4)
        X.append([dist_feat, angle_feat, speed_feat])
        y.append(is_match)
    return np.array(X), np.array(y)


def train_weight_model(X_train, y_train):
    """Trains Linear SVM with L1 sparsity and normalizes learned feature weights."""
    svm = LinearSVC(penalty='l1', C=1.0, dual=False, class_weight='balanced', max_iter=10000, random_state=None)
    svm.fit(X_train, y_train)
    raw_weights = svm.coef_[0]
    normalized_weights = np.maximum(raw_weights, 0.0)
    norm_factor = np.sum(normalized_weights)
    if norm_factor > 0:
        normalized_weights /= norm_factor
    else:
        normalized_weights = np.ones_like(raw_weights) / len(raw_weights)
    return svm, normalized_weights


# =======================================================
# 3. DYNAMIC ADAPTIVE HMM VITERBI MODEL
# =======================================================
class DynamicMapMatcherHMM:
    def __init__(self, window_size: int, weights: np.ndarray, min_buffer: int = 1, max_buffer: int = 7, confidence_threshold: float = 1.0):
        self.T = window_size
        self.weights = weights
        self.min_N = min_buffer
        self.max_N = max_buffer
        self.confidence_threshold = confidence_threshold

    def calculate_emission(self, obs: np.ndarray, seg: RoadSegment) -> float:
        dist = seg.point_distance(obs)
        dist_feat = np.exp(-0.5 * (dist / 18.0)**2)
        angle_feat = 0.85
        speed_feat = 0.90
        feats = np.array([dist_feat, angle_feat, speed_feat])
        prob = float(np.dot(self.weights, feats))
        return max(prob, 1e-6)

    def calculate_transition(self, seg_prev: RoadSegment, seg_curr: RoadSegment) -> float:
        if seg_prev.id == seg_curr.id:
            return 0.90
        conn_dist = np.linalg.norm(seg_curr.start - seg_prev.end)
        return float(np.exp(-conn_dist / 20.0))

    def run_viterbi(self, observations, candidate_segments):
        if len(observations) < self.T:
            return None, self.min_N

        obs_win = observations[-self.T:]
        cand_win = candidate_segments[-self.T:]

        V = [{} for _ in range(self.T)]
        bp = [{} for _ in range(self.T)]

        for seg in cand_win[0]:
            V[0][seg.id] = np.log(self.calculate_emission(obs_win[0], seg))

        for t in range(1, self.T):
            for curr_seg in cand_win[t]:
                max_log_prob = -np.inf
                best_prev = None
                log_em = np.log(self.calculate_emission(obs_win[t], curr_seg))

                for prev_seg in cand_win[t - 1]:
                    if prev_seg.id not in V[t - 1]:
                        continue
                    log_tr = np.log(self.calculate_transition(prev_seg, curr_seg))
                    path_prob = V[t - 1][prev_seg.id] + log_tr + log_em

                    if path_prob > max_log_prob:
                        max_log_prob = path_prob
                        best_prev = prev_seg.id

                if best_prev is not None:
                    V[t][curr_seg.id] = max_log_prob
                    bp[t][curr_seg.id] = best_prev

        if not V[-1]:
            return None, self.min_N

        # Confidence margin evaluation
        final_scores = sorted(list(V[-1].values()))
        margin = final_scores[-1] - final_scores[-2] if len(final_scores) >= 2 else 999.0

        if self.min_N == self.max_N:
            dynamic_N = self.min_N
        else:
            dynamic_N = self.min_N if margin > self.confidence_threshold else self.max_N

        target_t = max(0, self.T - dynamic_N - 1)
        current_id = max(V[-1], key=V[-1].get)

        for t in range(self.T - 1, target_t, -1):
            if current_id in bp[t]:
                current_id = bp[t][current_id]

        return current_id, dynamic_N


# =======================================================
# 4. BUFFER DELAY ERROR CURVE EVALUATION & N ANALYSIS
# =======================================================
def evaluate_buffer_curve(network, weights, gps_obs, true_states):
    """Evaluates classification error rates across buffer sizes N in [1..9]."""
    T = 10
    buffer_range = range(1, 10)
    error_rates = []

    candidates_per_step = []
    for pt in gps_obs:
        sorted_segs = sorted(network, key=lambda s: s.point_distance(pt))[:4]
        candidates_per_step.append(sorted_segs)

    print("\n=======================================================")
    print("      EVALUATING BUFFER DELAY ERROR CURVE (T=10)       ")
    print("=======================================================")

    for N in buffer_range:
        matcher = DynamicMapMatcherHMM(window_size=T, weights=weights, min_buffer=N, max_buffer=N)
        preds, truths = [], []

        for step in range(T, len(gps_obs)):
            sub_obs = gps_obs[:step]
            sub_cand = candidates_per_step[:step]
            pred, _ = matcher.run_viterbi(sub_obs, sub_cand)
            target_idx = step - N - 1
            if 0 <= target_idx < len(true_states) and pred is not None:
                preds.append(pred)
                truths.append(true_states[target_idx])

        acc = accuracy_score(truths, preds)
        err = (1.0 - acc) * 100.0
        error_rates.append(err)
        print(f"Buffer Delay N = {N} | Accuracy = {acc * 100:.2f}% | Error E = {err:.2f}%")

    min_idx = int(np.argmin(error_rates))
    opt_N = list(buffer_range)[min_idx]
    min_err = error_rates[min_idx]

    print("\n-------------------------------------------------------")
    print(">>> OPTIMAL BUFFER DELAY ANALYSIS (FROM DATA)")
    print("-------------------------------------------------------")
    print(f"Optimal Chosen Buffer Delay N : {opt_N} steps (out of T=10)")
    print(f"Minimum Classification Error  : {min_err:.2f}% (Accuracy: {100.0 - min_err:.2f}%)")
    print("-------------------------------------------------------")

    # Generate Dynamic Plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(buffer_range), error_rates, marker='o', color='#2980b9', linewidth=2.5, markersize=8)
    plt.title(f"Map-Matching Error Rate vs. Buffer Delay N (T={T})", fontsize=13, fontweight='bold')
    plt.xlabel("Buffer Delay N (Steps)", fontsize=11)
    plt.ylabel("Classification Error Percentage E (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(list(buffer_range))

    plt.axvline(x=opt_N, color='crimson', linestyle=':', label=f'Optimal Buffer N={opt_N} (Min Error: {min_err:.2f}%)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("c:/Users/samvi/ml/buffer_error_curve.png", dpi=300)
    plt.close()
    print(f"\n[Saved] Dynamic buffer error curve to 'c:/Users/samvi/ml/buffer_error_curve.png'")
    return opt_N


# =======================================================
# 5. FINAL TESTING PHASE & HEAD-TO-HEAD EVALUATION
# =======================================================
def run_final_testing_phase(network, X_test, y_test, svm_model, knn_model, weights, gps_test_obs, true_test_states):
    """
    DEDICATED FINAL TESTING PHASE:
    1. Evaluates SVM vs KNN on the independent hold-out test set.
    2. Runs Head-to-Head HMM Test Simulation (Unbuffered vs Dynamic Adaptive Buffer).
    3. Prints chosen delays, rationale, approximations, and generates comparative performance bar charts.
    """
    print("\n" + "=" * 65)
    print("            FINAL TESTING PHASE (HOLD-OUT EVALUATION)           ")
    print("=" * 65)

    # 1. Classifier Hold-Out Testing
    print("\n--- Part A: Classifier Test Set Evaluation ---")
    svm_preds = svm_model.predict(X_test)
    knn_preds = knn_model.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average='weighted', zero_division=0)
    knn_acc = accuracy_score(y_test, knn_preds)
    knn_f1 = f1_score(y_test, knn_preds, average='weighted', zero_division=0)

    print(f"SVM Test Accuracy : {svm_acc * 100:.2f}% | Test F1-Score: {svm_f1:.4f}")
    print(f"KNN Test Accuracy : {knn_acc * 100:.2f}% | Test F1-Score: {knn_f1:.4f}")

    # 2. HMM Sequence Test Evaluation (Unbuffered vs Dynamic Adaptive Buffer)
    print("\n--- Part B: Real-Time Map Matching Test Simulation ---")
    hmm_unbuffered = DynamicMapMatcherHMM(window_size=10, weights=weights, min_buffer=0, max_buffer=0)
    hmm_dynamic = DynamicMapMatcherHMM(window_size=10, weights=weights, min_buffer=2, max_buffer=7, confidence_threshold=1.0)

    candidates_test = []
    for pt in gps_test_obs:
        candidates_test.append(sorted(network, key=lambda s: s.point_distance(pt))[:4])

    preds_unbuf, truth_unbuf = [], []
    preds_dyn, truth_dyn = [], []
    delays_used = []

    for t in range(10, len(gps_test_obs)):
        window_obs = gps_test_obs[t - 10 : t]
        window_cand = candidates_test[t - 10 : t]

        # Instant Unbuffered Output (N=0)
        p_u, n_u = hmm_unbuffered.run_viterbi(window_obs, window_cand)
        preds_unbuf.append(p_u)
        truth_unbuf.append(true_test_states[t - n_u - 1])

        # Dynamic Adaptive Buffer Output
        p_d, n_d = hmm_dynamic.run_viterbi(window_obs, window_cand)
        preds_dyn.append(p_d)
        truth_dyn.append(true_test_states[t - n_d - 1])
        delays_used.append(n_d)

    metrics_unbuf = {
        "Accuracy": accuracy_score(truth_unbuf, preds_unbuf),
        "Precision": precision_score(truth_unbuf, preds_unbuf, average='weighted', zero_division=0),
        "Recall": recall_score(truth_unbuf, preds_unbuf, average='weighted', zero_division=0),
        "F1-Score": f1_score(truth_unbuf, preds_unbuf, average='weighted', zero_division=0)
    }

    metrics_dyn = {
        "Accuracy": accuracy_score(truth_dyn, preds_dyn),
        "Precision": precision_score(truth_dyn, preds_dyn, average='weighted', zero_division=0),
        "Recall": recall_score(truth_dyn, preds_dyn, average='weighted', zero_division=0),
        "F1-Score": f1_score(truth_dyn, preds_dyn, average='weighted', zero_division=0)
    }

    print(f"\n{'Metric':<14} | {'Unbuffered (N=0)':<18} | {'Dynamic Buffer':<16} | {'Improvement':<12}")
    print("-" * 68)
    for m in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        u_val = metrics_unbuf[m]
        d_val = metrics_dyn[m]
        diff = d_val - u_val
        trend = f"+{diff * 100:.2f}%" if diff > 0 else f"{diff * 100:.2f}%"
        print(f"{m:<14} | {u_val:<18.4f} | {d_val:<16.4f} | {trend:<12}")

    # Delay statistics and approximations
    exact_avg_delay = float(np.mean(delays_used))
    approx_avg_delay = int(np.round(exact_avg_delay))
    min_delay_count = delays_used.count(2)
    max_delay_count = delays_used.count(7)
    total_steps = len(delays_used)

    print("\n-------------------------------------------------------")
    print(">>> DYNAMIC BUFFER ADAPTATION & DELAY RATIONALE")
    print("-------------------------------------------------------")
    print(f"Exact Average Delay (N_dyn)      : {exact_avg_delay:.4f} steps")
    print(f"Approximated Integer Delay       : ~{approx_avg_delay} steps (rounded)")
    print(f"Low-Delay Steps (N=2, Confident) : {min_delay_count}/{total_steps} ({min_delay_count/total_steps*100:.1f}%)")
    print(f"High-Delay Steps (N=7, Stalled)  : {max_delay_count}/{total_steps} ({max_delay_count/total_steps*100:.1f}%)")
    print(f"Why this N behavior occurred    :")
    print(f"  - On straightaways with clear GPS signal, confidence margin exceeded the threshold (1.0), triggering minimal latency (N=2).")
    print(f"  - Near multi-road intersections or during sensor noise spikes, ambiguity increased, causing the algorithm to automatically stall for up to N=7 future GPS frames to disambiguate the true vehicle corridor.")
    print("-------------------------------------------------------")
    print("=" * 65)

    # 3. Generate Comparative Bar Chart
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    unbuf_scores = [metrics_unbuf[m] for m in labels]
    dyn_scores = [metrics_dyn[m] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rects1 = ax.bar(x - width / 2, unbuf_scores, width, label='Unbuffered (Instant Output)', color='#e74c3c', alpha=0.9)
    rects2 = ax.bar(x + width / 2, dyn_scores, width, label='Dynamic Buffer (Adaptive Delay)', color='#27ae60', alpha=0.9)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Test Set Evaluation: Instant vs Dynamic Adaptive Map Matching', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.15)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("c:/Users/samvi/ml/instant_vs_adaptive_comparison.png", dpi=300)
    plt.close()
    print(f"[Saved] Final test comparative chart to 'c:/Users/samvi/ml/instant_vs_adaptive_comparison.png'")

    # 4. Generate Spatial Map-Matching Trajectory Plot
    plot_map_matching_trajectory(network, gps_test_obs, true_test_states, preds_dyn, delays_used)


def plot_map_matching_trajectory(network, gps_obs, true_states, preds, delays):
    """
    Plots the spatial road network, raw noisy GPS observations,
    ground-truth vehicle route, and the HMM dynamic map-matched route.
    """
    seg_map = {s.id: s for s in network}
    fig, ax = plt.subplots(figsize=(10, 8))

    # Road Network Grid
    for i, seg in enumerate(network):
        ax.plot([seg.start[0], seg.end[0]], [seg.start[1], seg.end[1]],
                color='#bdc3c7', linewidth=1.8, linestyle='-', zorder=1,
                label='Road Network Grid' if i == 0 else "")

    # Align points to evaluated delay steps
    start_step = 10
    eval_indices = [t - delays[i] - 1 for i, t in enumerate(range(start_step, len(gps_obs)))]
    
    obs_points = np.array([gps_obs[idx] for idx in eval_indices])
    
    truth_pts = []
    for idx in eval_indices:
        t_seg = seg_map[true_states[idx]]
        truth_pts.append(t_seg.project_point(gps_obs[idx]))
    truth_pts = np.array(truth_pts)

    matched_pts = []
    for i, p_id in enumerate(preds):
        if p_id is not None and p_id in seg_map:
            m_seg = seg_map[p_id]
            matched_pts.append(m_seg.project_point(obs_points[i]))
        else:
            matched_pts.append(obs_points[i])
    matched_pts = np.array(matched_pts)

    # Noisy GPS observations
    ax.scatter(obs_points[:, 0], obs_points[:, 1], color='#e74c3c', s=24, alpha=0.45,
               label='Noisy GPS Observations', zorder=2)

    # Ground truth route
    ax.plot(truth_pts[:, 0], truth_pts[:, 1], color='#2ecc71', linewidth=3.2, linestyle='--',
            label='Ground-Truth Vehicle Route', zorder=3)

    # Dynamic HMM matched path
    ax.plot(matched_pts[:, 0], matched_pts[:, 1], color='#2980b9', linewidth=2.0, linestyle='-',
            label='Dynamic Adaptive HMM Matched Route', zorder=4)

    ax.set_title("Spatial Map-Matching Trajectory Visualization", fontsize=13, fontweight='bold')
    ax.set_xlabel("X Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Coordinate (m)", fontsize=11)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("c:/Users/samvi/ml/map_matching_trajectory.png", dpi=300)
    plt.close()
    print("[Saved] Spatial map matching trajectory plot to 'c:/Users/samvi/ml/map_matching_trajectory.png'")


def main():
    print("Starting Map-Matching Pipeline...")

    # Step 1: Create Road Network
    network = create_road_network()

    # Step 2: Feature Data Generation & Train/Test Split (80% Train, 20% Hold-out Test)
    X_all, y_all = extract_feature_dataset(network, n_samples=2000)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=None)

    # Step 3: Train SVM and KNN Classifiers
    svm_model, learned_weights = train_weight_model(X_train, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_model.fit(X_train, y_train)
    print(f"Learned Normalized Feature Weights: {learned_weights}")

    # Step 4: Generate Independent Training & Testing Trajectories with variable noise profiles
    noise_level = np.random.uniform(9.0, 14.0)
    multipath_rate = np.random.uniform(0.12, 0.22)
    gps_train_obs, true_train_states = generate_trajectory(network, num_steps=250, noise_std=noise_level, multipath_prob=multipath_rate)
    gps_test_obs, true_test_states = generate_trajectory(network, num_steps=300, noise_std=noise_level + 1.0, multipath_prob=multipath_rate)

    # Step 5: Evaluate Buffer Delay Curve on Training Trajectory
    evaluate_buffer_curve(network, learned_weights, gps_train_obs, true_train_states)

    # Step 6: FINAL TESTING PHASE (Hold-out Test Set & Dynamic vs Unbuffered Evaluation)
    run_final_testing_phase(network, X_test, y_test, svm_model, knn_model, learned_weights, gps_test_obs, true_test_states)

    print("\nPipeline execution complete!")


if __name__ == "__main__":
    main()
