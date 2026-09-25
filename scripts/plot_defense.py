import pandas as pd, numpy as np, matplotlib.pyplot as plt

CSV = "outputs/plots/physical_banana_defense.csv"
OUT = "outputs/plots/defense_summary.png"
df = pd.read_csv(CSV)

DEF_ORDER = ["none", "jpeg_q30", "blur_s2", "median_3", "seg_blind", "seg_oracle"]
DEF_LABEL = {
    "none": "no defense",
    "jpeg_q30": "JPEG q30",
    "blur_s2": "blur σ2",
    "median_3": "median 3×3",
    "seg_blind": "seg (blind)",
    "seg_oracle": "seg (oracle)",
}
AREA = 0.25
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

emb = df[df.metric == "embed"]
adv = emb[(emb.condition == "adversarial") & (emb.area_frac == AREA)]
adv_mean = adv.groupby("defense").banana_rate.mean().reindex(DEF_ORDER)
adv_std = adv.groupby("defense").banana_rate.std().reindex(DEF_ORDER)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
c_attack = "#c23b3b"

# Panel 1: attack success per defense (embedding ensemble mean)
ax = axes[0]
x = np.arange(len(DEF_ORDER))
ax.bar(x, adv_mean.values * 100, yerr=adv_std.values * 100, capsize=3, color=c_attack)
ax.set_xticks(x)
ax.set_xticklabels([DEF_LABEL[d] for d in DEF_ORDER], rotation=25, ha="right")
ax.set_ylabel('attack success — "says banana" (%)')
ax.set_ylim(0, 105)
ax.set_title(
    f"Embedding ensemble · patch @{int(AREA * 100)}% area\n(lower = defense works)"
)
for i, v in enumerate(adv_mean.values):
    ax.text(i, v * 100 + 2, f"{v * 100:.0f}", ha="center", fontsize=10)

# Panel 2: per-model heatmap (model × defense) attack success
piv = adv.pivot_table(index="model", columns="defense", values="banana_rate").reindex(
    columns=DEF_ORDER
)
ax = axes[1]
im = ax.imshow(piv.values * 100, cmap="Reds", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(DEF_ORDER)))
ax.set_xticklabels([DEF_LABEL[d] for d in DEF_ORDER], rotation=25, ha="right")
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(
            j,
            i,
            f"{v * 100:.0f}",
            ha="center",
            va="center",
            color="white" if v > 0.5 else "black",
            fontsize=9,
        )
ax.set_title("Attack success by model × defense (%)")
fig.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: VLM headline (SmolVLM)
vlm = df[df.metric == "vlm"]
v_adv = (
    vlm[vlm.condition == "adversarial"]
    .groupby("defense")
    .banana_rate.mean()
    .reindex(DEF_ORDER)
)
v_clean = vlm[vlm.condition == "clean"].banana_rate.mean()
ax = axes[2]
ax.bar(x, v_adv.values * 100, color="#e4572e")
ax.axhline(
    v_clean * 100, ls="--", color="#555", label=f"clean baseline ({v_clean * 100:.0f}%)"
)
ax.set_xticks(x)
ax.set_xticklabels([DEF_LABEL[d] for d in DEF_ORDER], rotation=25, ha="right")
ax.set_ylabel('SmolVLM "says banana" (%)')
ax.set_ylim(0, 105)
ax.legend()
ax.set_title(f"Reasoning VLM (SmolVLM-500M) · patch @{int(AREA * 100)}%")
for i, vv in enumerate(v_adv.values):
    ax.text(i, vv * 100 + 2, f"{vv * 100:.0f}", ha="center", fontsize=10)

fig.suptitle(
    "Defending the banana patch: blind preprocessing barely dents it; only localizing + removing the patch works",
    fontsize=13,
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved", OUT)
