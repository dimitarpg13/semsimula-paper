"""
Figure 8.1 of Improving_the_Fock_Mechanism_to_match_Attention.md: what Fock v2.1 implements.

The register bank is a memory with two ports. Creation is the WRITE port (each register's query
attends over the tokens: softmax over j); the reverse channel is the READ port (each token's query
attends over the active registers: softmax over k). Token i receives token j's content only through
both hops, A_ij = sum_k beta_ik alpha_kj, routed through M registers that persist across layers.

    python3 make_fock_register_read_write.py   -> fock_register_read_write.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

BG = "#FAFAFA"; WRITE = "#1565C0"; READ = "#C62828"; REG = "#2E7D32"; GREY = "#555555"

fig, ax = plt.subplots(figsize=(13.5, 5.6), facecolor=BG)
ax.set_xlim(0, 13.5); ax.set_ylim(0, 5.6); ax.axis("off"); ax.set_facecolor(BG)


def box(x, y, w, h, text, ec, fs=12, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="white", ec=ec, lw=2.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#212121")


def arrow(x0, y0, x1, y1, c):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=c))


ax.text(6.75, 4.75, "Fock v2.1: the registers are a memory with a write port and a read port",
        ha="center", fontsize=14, fontweight="bold", color="#212121")

# token j (source), register k (memory), token i (receiver)
box(0.3, 2.3, 2.0, 1.0, "token  j\n(source)", WRITE)
ax.add_patch(Ellipse((6.75, 2.8), 3.6, 2.3, fc="#F1F8E9", ec=REG, lw=2.0, ls="--"))
ax.text(6.75, 3.35, r"register  $r_k$", ha="center", fontsize=13, fontweight="bold")
ax.text(6.75, 2.78, r"$r_k \leftarrow \sigma_k r_k + (1-\sigma_k)\sum_j \alpha_{kj}\, v_j$",
        ha="center", fontsize=10.5)
ax.text(6.75, 2.2, "persists across layers (salience $\\sigma_k$)", ha="center", fontsize=9.5, color=GREY)
box(11.2, 2.3, 2.0, 1.0, "token  i\n(receiver)", READ)

# write port
arrow(2.35, 2.8, 4.9, 2.8, WRITE)
ax.text(3.45, 3.95, "WRITE  (creation)", ha="center", fontsize=11.5, fontweight="bold", color=WRITE)
ax.text(3.45, 3.45, r"$\alpha_{kj} = \mathrm{softmax}_j\!\left(q_k \cdot k_j / \tau_k\right)$", ha="center", fontsize=11)
ax.text(3.45, 1.98, r"query from the register: $q_k = W_Q^{(k)} r_k$", ha="center", fontsize=9.3, color=GREY)
ax.text(3.45, 2.35, r"payload  $v_j = W_V h_j$", ha="center", fontsize=10)
ax.text(3.45, 3.08, "softmax over tokens j", ha="center", fontsize=9.3, color=GREY)

# read port
arrow(8.6, 2.8, 11.15, 2.8, READ)
ax.text(10.05, 3.95, "READ  (reverse channel)", ha="center", fontsize=11.5, fontweight="bold", color=READ)
ax.text(10.05, 3.45, r"$\beta_{ik} = \mathrm{softmax}_k\!\left(q_i \cdot k^{\mathrm{reg}}_k\right)$", ha="center", fontsize=11)
ax.text(10.05, 1.98, r"query from the token: $q_i = W_Q^{\mathrm{rev}} h_i$", ha="center", fontsize=9.3, color=GREY)
ax.text(10.05, 2.35, r"payload  $W_V^{\mathrm{rev}} r_k$", ha="center", fontsize=10)
ax.text(10.05, 3.08, "softmax over registers k", ha="center", fontsize=9.3, color=GREY)

# bottom: what token i receives, and where it enters the step
ax.text(6.75, 1.25, r"token $i$ receives token $j$'s content only through both hops:   "
        r"$A_{ij} = \sum_{k} \beta_{ik}\,\alpha_{kj}$   (rank $\leq M$; causal: the bank read at $i$ is built from tokens $\leq i$)",
        ha="center", fontsize=10.8, color="#212121")
ax.text(6.75, 0.72, r"applied once per layer, after the integrator step and its LayerNorm:   "
        r"$h_i \leftarrow \mathrm{LN}\!\left(h_i + \frac{dt^2}{m_i}\tanh(s)\,\mathrm{RMSnorm}(Q_i)\right)$,   "
        r"$Q_i = \sum_k \beta_{ik} W_V^{\mathrm{rev}} r_k$",
        ha="center", fontsize=10.3, color=GREY)
ax.text(6.75, 0.25, "Nothing else reads the registers: without the read port they are write-only and cannot affect a prediction.",
        ha="center", fontsize=10.3, color=READ, style="italic")

fig.savefig("fock_register_read_write.png", dpi=150, facecolor=BG, bbox_inches="tight")
print("wrote fock_register_read_write.png")
