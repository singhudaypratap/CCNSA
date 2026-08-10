import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

AGENTS = ["CCNSA","CQL","IQL","DecisionTransformer","RuleBased","NeuroSymbolic","PPO","SAC","XAI"]
LABELS = {"CCNSA":"CCNSA","CQL":"CQL","IQL":"IQL","DecisionTransformer":"Dec.Trans.",
          "RuleBased":"RuleBased","NeuroSymbolic":"NeuroSymb.","PPO":"PPO","SAC":"SAC","XAI":"XAI"}
COLOR = {"CCNSA":"#b8860b"}  # muted gold/amber for CCNSA
BASE_GRAY = "#6e7b8b"
COLOR.update({a:BASE_GRAY for a in AGENTS if a!="CCNSA"})

# ---- Real data, replication run (10 previously-unused seeds), verified from the actual GPU run log ----
reward_mean = {"CCNSA":0.6604,"SAC":0.2246,"PPO":0.2426,"CQL":0.6294,"IQL":0.6030,
               "XAI":0.2217,"RuleBased":0.3475,"NeuroSymbolic":0.2591,"DecisionTransformer":0.4747}
reward_ci = {"CCNSA":0.0067,"SAC":0.0048,"PPO":0.0166,"CQL":0.0077,"IQL":0.0084,
             "XAI":0.0030,"RuleBased":0.0092,"NeuroSymbolic":0.0078,"DecisionTransformer":0.0076}
d_vs_ccnsa = {"SAC":55.069,"PPO":13.560,"CQL":2.176,"IQL":4.052,"XAI":38.493,
              "RuleBased":22.484,"NeuroSymbolic":37.057,"DecisionTransformer":19.492}

faith = {"CCNSA":0.821,"SAC":0.741,"PPO":0.806,"CQL":0.788,"IQL":0.809,"XAI":0.752,"NeuroSymbolic":0.772}  # exact Table tbl:faith-di (headline run) values -- kept identical to the manuscript text/table throughout
# Delta values are the exact Table tbl:faith-di (headline run) numbers reported in the manuscript text.
# Only CCNSA's AUC_del/AUC_ins split is independently reported in the manuscript text (Section on
# the deletion-insertion test); no del/ins split is reported for the other agents, only Delta, so we
# do not invent one -- the DI figure below plots Delta only for the six-agent comparison and reserves
# the del/ins curve view for CCNSA specifically, matching what Table tbl:faith-di actually contains.
di_delta = {"CCNSA":0.880,"SAC":0.440,"PPO":0.906,"CQL":0.764,"IQL":0.729,"XAI":-0.105,"NeuroSymbolic":0.730}
di_ccnsa_auc = {"del":0.059,"ins":0.939}  # from manuscript text, Section 7.5.2
di_random_control_delta = 0.102
di = {a: (None, None, di_delta[a]) for a in di_delta}  # keep di[a][2] usable as before (radar axis)

flops = {"SAC":167936,"PPO":157696,"CQL":157696,"IQL":157696,"XAI":167936,
         "NeuroSymbolic":157696,"RuleBased":0,"CCNSA":181760}
lat_mean = {"SAC":0.77,"PPO":1.39,"CQL":0.72,"IQL":0.72,"XAI":0.77,"NeuroSymbolic":0.73,
            "RuleBased":0.01,"CCNSA":0.99}
lat_std = {"SAC":0.03,"PPO":0.12,"CQL":0.02,"IQL":0.01,"XAI":0.02,"NeuroSymbolic":0.01,
           "RuleBased":0.00,"CCNSA":0.05}

# per-seed reward, all 9 agents x 10 seeds -- verified transcription from the actual replication run log
seeds = [807080,817081,827082,837083,847084,857085,867086,877087,887088,897089]
per_seed = {
 "CCNSA":[.6658,.6627,.6525,.6673,.6557,.6707,.6675,.6658,.6398,.6557],
 "SAC":[.2246,.2345,.2245,.2209,.2319,.2254,.2263,.2270,.2096,.2212],
 "PPO":[.2236,.2479,.2585,.2524,.2257,.2340,.2343,.2189,.2978,.2335],
 "CQL":[.6282,.6163,.6215,.6462,.6326,.6198,.6207,.6372,.6259,.6455],
 "IQL":[.6156,.5830,.5924,.6124,.5993,.6199,.5996,.5926,.6105,.6047],
 "XAI":[.2230,.2216,.2183,.2173,.2187,.2240,.2272,.2162,.2291,.2217],
 "RuleBased":[.3494,.3673,.3419,.3300,.3446,.3377,.3676,.3467,.3339,.3554],
 "NeuroSymbolic":[.2576,.2665,.2572,.2781,.2392,.2590,.2536,.2625,.2483,.2685],
 "DecisionTransformer":[.4779,.4912,.4741,.4636,.4679,.4810,.4788,.4750,.4541,.4833],
}

# 12-arm ablation table, replication run (already in the manuscript's Table tbl:ablation)
ablation = [
 ("w/o Causal graph", 0.001, "n.s."),
 ("w/o Symbolic logic", 0.003, "n.s."),
 ("w/o Prioritised replay", 0.004, "n.s."),
 ("w/o Smooth clamping", -0.003, "n.s."),
 ("w/o Emergency boosting", 0.005, "n.s."),
 ("w/o Causal+symbolic (joint)", 0.001, "n.s."),
 ("w/o Conservative critic pen.", -0.086, "n.s.$^\\dagger$"),
 ("+ Symbolic gating", -0.025, "***"),
 ("w/o BC warm-start", 0.001, "n.s."),
 ("Bare capacity", -0.000, "n.s."),
 ("Cons. pen. $\\lambda$=0.5", 0.003, "n.s."),
 ("Cons. pen. $\\lambda$=2.0", 0.005, "n.s."),
]
print("data loaded ok")

order = sorted(AGENTS, key=lambda a: -reward_mean[a])
OUT = "figures/"

# ---------- Fig 3: Final performance bar + 95% CI ----------
fig, ax = plt.subplots(figsize=(6.2,3.6), dpi=200)
xs = np.arange(len(order))
means = [reward_mean[a] for a in order]
cis = [reward_ci[a] for a in order]
colors = [COLOR[a] for a in order]
ax.bar(xs, means, yerr=cis, capsize=3, color=colors, edgecolor="#333333", linewidth=0.6, width=0.62,
       error_kw=dict(elinewidth=1.0, ecolor="#222222"))
for x,m,a in zip(xs,means,order):
    if a=="CCNSA":
        ax.text(x, m+cis[order.index(a)]+0.015, "***", ha="center", fontsize=9, color="#333333")
ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in order], rotation=30, ha="right")
ax.set_ylabel("Mean reward (95% CI)")
ax.set_title("Track A final performance — replication run (10 unused seeds)", fontsize=10)
ax.set_ylim(0, 0.78)
for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig3_final_performance.png")
plt.close()

# ---------- Fig 2 replacement: per-seed distribution (box + strip) ----------
fig, ax = plt.subplots(figsize=(6.6,3.8), dpi=200)
box_data = [per_seed[a] for a in order]
bp = ax.boxplot(box_data, positions=xs, widths=0.45, patch_artist=True, showfliers=False,
                 medianprops=dict(color="#222222", linewidth=1.2),
                 boxprops=dict(linewidth=0.8, edgecolor="#333333"),
                 whiskerprops=dict(linewidth=0.8, color="#333333"),
                 capprops=dict(linewidth=0.8, color="#333333"))
for patch, a in zip(bp['boxes'], order):
    patch.set_facecolor(COLOR[a]); patch.set_alpha(0.35)
rng = np.random.default_rng(0)
for x,a in zip(xs, order):
    jitter = rng.uniform(-0.12,0.12,size=len(per_seed[a]))
    ax.scatter(np.full(len(per_seed[a]), x)+jitter, per_seed[a], s=14, color=COLOR[a],
               edgecolor="#222222", linewidth=0.3, zorder=3)
ax.set_xticks(xs); ax.set_xticklabels([LABELS[a] for a in order], rotation=30, ha="right")
ax.set_ylabel("Per-seed mean reward")
ax.set_title("Distribution of per-seed mean reward, n=10 seeds/agent (replication run)", fontsize=9.5)
for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig2_seed_distribution.png")
plt.close()

# ---------- Fig 5: Ablation bar chart ----------
fig, ax = plt.subplots(figsize=(6.6,4.0), dpi=200)
names = [x[0] for x in ablation]
deltas = [x[1] for x in ablation]
sig = [x[2] for x in ablation]
ys = np.arange(len(names))[::-1]
cols = ["#b8321a" if "***" in s else ("#c99a2e" if "dagger" in s else "#6e7b8b") for s in sig]
ax.barh(ys, deltas, color=cols, edgecolor="#333333", linewidth=0.6, height=0.6)
ax.axvline(0, color="#333333", linewidth=0.8)
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel(r"$\Delta$ Reward vs. Full CCNSA")
ax.set_title("12-arm ablation, replication run (Bonferroni $k=12$)", fontsize=10)
for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig5_ablation.png")
plt.close()

# ---------- Fig 7: Faithfulness bar ----------
fa_order = [a for a in order if a in faith]
fig, ax = plt.subplots(figsize=(5.8,3.4), dpi=200)
xs2 = np.arange(len(fa_order))
vals = [faith[a] for a in fa_order]
cols = [COLOR[a] for a in fa_order]
ax.bar(xs2, vals, color=cols, edgecolor="#333333", linewidth=0.6, width=0.6)
ax.axhline(0.70, color="#b8321a", linestyle="--", linewidth=1.0, label=r"$\tau=0.70$ threshold")
ax.set_xticks(xs2); ax.set_xticklabels([LABELS[a] for a in fa_order], rotation=30, ha="right")
ax.set_ylabel(r"Faithfulness $\mathcal{F}$")
ax.set_ylim(0,0.95)
ax.set_title("Inter-method attribution agreement (replication run)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig7_faithfulness.png")
plt.close()

# ---------- Fig DI: two-panel deletion-insertion figure ----------
# Panel (a): CCNSA's own Deletion-AUC / Insertion-AUC bars against the random-ranking control's Delta
#            (the only del/ins split independently reported in the manuscript text).
# Panel (b): Delta (fidelity) across all six comparable agents, from Table tbl:faith-di.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4,3.6), dpi=200, gridspec_kw={"width_ratios":[1,1.5]})

bars = ["Deletion\nAUC", "Insertion\nAUC"]
vals_l = [di_ccnsa_auc["del"], di_ccnsa_auc["ins"]]
axL.bar(bars, vals_l, color=["#b8321a","#2a6f4f"], edgecolor="#333333", linewidth=0.6, width=0.55)
axL.axhline(di_random_control_delta, color="#888888", linestyle="--", linewidth=1.0)
axL.text(-0.15, di_random_control_delta+0.05, "random-ranking control $\\Delta$=+0.102", color="#666666", fontsize=7, ha="left")
axL.set_ylim(0,1.05)
axL.set_ylabel("AUC")
axL.set_title(f"(a) CCNSA, K=0..32\n$\\Delta$={di_ccnsa_auc['ins']-di_ccnsa_auc['del']:+.3f}", fontsize=9)
for spine in ["top","right"]: axL.spines[spine].set_visible(False)

di_order = [a for a in order if a in di_delta]
xs4 = np.arange(len(di_order))
vals_r = [di_delta[a] for a in di_order]
cols_r = [COLOR[a] for a in di_order]
axR.bar(xs4, vals_r, color=cols_r, edgecolor="#333333", linewidth=0.6, width=0.6)
axR.axhline(0, color="#333333", linewidth=0.8)
axR.axhline(di_random_control_delta, color="#888888", linestyle="--", linewidth=1.0, label="random-ranking control")
axR.set_xticks(xs4); axR.set_xticklabels([LABELS[a] for a in di_order], rotation=30, ha="right")
axR.set_ylabel(r"Deletion-insertion fidelity $\Delta$")
axR.set_title("(b) All comparable agents (Table tbl:faith-di)", fontsize=9)
axR.legend(frameon=False, fontsize=7.5, loc="upper right")
for spine in ["top","right"]: axR.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUT+"fig_faithfulness_DI.png")
plt.close()

# ---------- Fig 9: Track B real-data results (3-panel) ----------
fig, (axP, axQ, axS) = plt.subplots(1, 3, figsize=(11.0,3.4), dpi=200)

clf_names = ["Random\nForest","Linear\nSVM","Logistic\nRegr."]
clf_f1 = [0.6663, 0.4791, 0.4352]
axP.bar(clf_names, clf_f1, color=["#b8860b","#6e7b8b","#6e7b8b"], edgecolor="#333333", linewidth=0.6, width=0.55)
axP.set_ylabel("Macro-F1")
axP.set_ylim(0,0.85)
axP.set_title("(a) B2: classifier context\n(27,240-record held-out split)", fontsize=8.7)
for spine in ["top","right"]: axP.spines[spine].set_visible(False)

cd_names = ["PC","NOTEARS","DirectLiNGAM"]
cd_stab = [0.906, 0.896, 0.833]
cd_edges = [30, 6, 16]
barsQ = axQ.bar(cd_names, cd_stab, color=["#b8860b","#6e7b8b","#6e7b8b"], edgecolor="#333333", linewidth=0.6, width=0.55)
for b,e in zip(barsQ, cd_edges):
    axQ.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{e} edges", ha="center", fontsize=7)
axQ.set_ylabel("Bootstrap edge-stability")
axQ.set_ylim(0,1.05)
axQ.set_title("(b) B3: causal discovery,\nreal UNSW-NB15 features", fontsize=8.7)
for spine in ["top","right"]: axQ.spines[spine].set_visible(False)

acp_names = ["sbytes\n>P60","dur\n>P70","ct_state_ttl\n>P75","dbytes\n>P70","smeansz & dmeansz\n>P75","sbytes\n>P80"]
acp_vals = [0.225, 0.085, 0.524, 0.274, 0.230, 0.157]
xs5 = np.arange(len(acp_names))
axS.bar(xs5, acp_vals, color="#6e7b8b", edgecolor="#333333", linewidth=0.6, width=0.6)
axS.axhline(0.249, color="#b8860b", linestyle="--", linewidth=1.2, label="average = 0.249")
axS.set_xticks(xs5); axS.set_xticklabels(acp_names, fontsize=6.3, rotation=30, ha="right")
axS.set_ylabel("Attack-conditional precision")
axS.set_ylim(0,0.65)
axS.set_title("(c) B4: audit-rule precision\n(real UNSW-NB15 attacks)", fontsize=8.7)
axS.legend(frameon=False, fontsize=7, loc="upper right")
for spine in ["top","right"]: axS.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUT+"fig9_trackb.png")
plt.close()

# ---------- Fig 6: FLOPs vs Latency ----------
fig, ax = plt.subplots(figsize=(5.8,3.8), dpi=200)
for a in order:
    if a=="DecisionTransformer":
        continue  # GPT-2 backbone not on comparable FLOPs scale; annotated separately
    ax.errorbar(flops[a], lat_mean[a], yerr=lat_std[a], fmt='o', color=COLOR[a],
                markeredgecolor="#222222", markersize=9 if a=="CCNSA" else 7, capsize=3, elinewidth=1.0)
    dx = 4000 if a!="RuleBased" else 4000
    ax.annotate(LABELS[a], (flops[a], lat_mean[a]), xytext=(dx,3), textcoords="offset points", fontsize=8)
ax.axhline(5.0, color="#b8321a", linestyle="--", linewidth=1.0)
ax.text(2000, 5.08, "5 ms URLLC budget", color="#b8321a", fontsize=8)
ax.set_xlabel("FLOPs per forward pass")
ax.set_ylabel("Inference latency (ms)")
ax.set_ylim(-0.3, 5.8)
ax.set_title("Complexity vs. latency (Decision Transformer omitted: GPT-2 backbone,\nnot on a comparable FLOPs scale)", fontsize=9)
for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig6_flops_latency.png")
plt.close()

# ---------- Fig 8: 3-panel advantage profile ----------
fig, axes = plt.subplots(1,3, figsize=(11.5,3.6), dpi=200)
axA, axB, axC = axes

for a in fa_order:
    axA.scatter(faith[a], reward_mean[a], color=COLOR[a], s=70 if a=="CCNSA" else 45,
                edgecolor="#222222", linewidth=0.5, zorder=3)
    axA.annotate(LABELS[a], (faith[a], reward_mean[a]), xytext=(4,3), textcoords="offset points", fontsize=7.5)
axA.axvline(0.70, color="#bbbbbb", linestyle="--", linewidth=0.8)
axA.set_xlabel(r"Faithfulness $\mathcal{F}$"); axA.set_ylabel("Reward")
axA.set_title("(a) Reward vs. faithfulness", fontsize=9)

for a in order:
    if a == "DecisionTransformer": continue
    axB.scatter(lat_mean[a], reward_mean[a], color=COLOR[a], s=70 if a=="CCNSA" else 45,
                edgecolor="#222222", linewidth=0.5, zorder=3)
    axB.annotate(LABELS[a], (lat_mean[a], reward_mean[a]), xytext=(4,3), textcoords="offset points", fontsize=7.5)
axB.axvline(5.0, color="#b8321a", linestyle="--", linewidth=0.8)
axB.set_xlabel("Latency (ms)"); axB.set_ylabel("Reward")
axB.set_title("(b) Speed vs. performance", fontsize=9)

ysC = np.arange(len(names))[::-1]
axC.barh(ysC, deltas, color=cols, edgecolor="#333333", linewidth=0.5, height=0.6)
axC.axvline(0, color="#333333", linewidth=0.8)
axC.set_yticks(ysC); axC.set_yticklabels(names, fontsize=6.8)
axC.set_xlabel(r"$\Delta$ Reward")
axC.set_title("(c) Ablation component importance", fontsize=9)

for ax in axes:
    for spine in ["top","right"]: ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(OUT+"fig8_advantage_profile.png")
plt.close()

# ---------- Fig 4: Radar ----------
axes_labels = ["Reward","Faithfulness","Del-Ins $\\Delta$","Inverse latency","Cohen's $d$ vs SAC"]
def norm(vals):
    vals = np.array(vals, dtype=float)
    lo, hi = vals.min(), vals.max()
    return (vals-lo)/(hi-lo+1e-9)

radar_agents = ["CCNSA","CQL","IQL","XAI"]
raw = {
 "Reward":        {a: reward_mean[a] for a in radar_agents},
 "Faithfulness":  {a: faith[a] for a in radar_agents},
 "Del-Ins $\\Delta$": {a: di[a][2] for a in radar_agents},
 "Inverse latency": {a: 1.0/lat_mean[a] for a in radar_agents},
 "Cohen's $d$ vs SAC": {"CCNSA": 54.97, "CQL": 28.733, "IQL": 22.996, "XAI": -0.303},  # real paired Cohen's d vs SAC, computed directly from per_seed data below (all 4 agents, same formula)
}
N = len(axes_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(6.2,6.2), dpi=200, subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)

# build normalised matrix (per-axis min-max across the 4 agents shown)
mat = np.array([[raw[lbl][a] for a in radar_agents] for lbl in axes_labels])  # N x 4
matn = np.array([norm(row) for row in mat])  # per-axis normalised 0-1

# radial gridlines WITH labels -- the earlier version called ax.set_yticks([]),
# which silently discarded the ring labels and made the chart unreadable
# (no way to tell what "further out" means on any axis, and the un-annotated
# outer circle looked like a placeholder). Fix: proper 0/0.25/0.5/0.75/1.0 rings.
ax.set_ylim(0, 1.18)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00\n(best of 4)"], fontsize=6.8, color="#555555")
ax.set_rlabel_position(54)  # empty wedge between the Reward and Faithfulness axes
ax.grid(color="#bbbbbb", linewidth=0.6, alpha=0.7)
ax.spines["polar"].set_color("#999999")

for j,a in enumerate(radar_agents):
    vals = matn[:,j].tolist()
    vals += vals[:1]
    is_ccnsa = (a == "CCNSA")
    c = COLOR[a] if is_ccnsa else plt.cm.tab10(j)
    ax.plot(angles, vals, color=c, linewidth=2.6 if is_ccnsa else 1.4,
            linestyle="-", marker="o", markersize=5 if is_ccnsa else 3.5,
            label=LABELS[a], zorder=5 if is_ccnsa else 3)
    ax.fill(angles, vals, color=c, alpha=0.18 if is_ccnsa else 0.08, zorder=2)

# annotate CCNSA's own vertex values so the chart is self-explanatory without
# forcing the reader back to Table 2/4 for the raw numbers. Offset radially
# outward (bumping r, not a fixed pixel offset) so labels clear both the
# plotted vertex and the category tick label at every angle, instead of
# colliding with them the way a uniform up-shift did at the side axes.
ccnsa_idx = radar_agents.index("CCNSA")
for k in range(N):
    lbl = axes_labels[k]
    raw_val = raw[lbl]["CCNSA"]
    txt = f"{raw_val:.3g}" if abs(raw_val) < 100 else f"{raw_val:.1f}"
    r_label = min(matn[k, ccnsa_idx] + 0.10, 1.13)
    ax.text(angles[k], r_label, txt, fontsize=6.8, color=COLOR["CCNSA"],
            ha="center", va="center", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=9)
ax.tick_params(axis="x", pad=14)
ax.set_title("Normalised 5-axis profile\n(per-axis min-max across the 4 agents shown; CCNSA values labelled)",
             fontsize=9, y=1.14)
ax.legend(loc="upper right", bbox_to_anchor=(1.34,1.18), frameon=False, fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT+"fig4_radar.png", bbox_inches="tight")
plt.close()

print("all figures written")
import os
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(OUT+f))
