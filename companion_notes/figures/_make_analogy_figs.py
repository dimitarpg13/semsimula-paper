"""Figures for semantic_templates/Analogy_Test_Set_Design.md (2026-11-26).

an_overlap.png    real data: token overlap of each kind of second sentence with the clause it restates, current test clauses
                  (template_fresh_clauses.py), against the overlap the analogy levels are designed to have
an_toy_2d.png     toy, two dimensions: one predicate, its repetition, a paraphrase, a near miss and a pronoun, as word vectors
                  (embedding) and as context-mixed states, with the cosines each reading gives
an_context.png    toy, two dimensions: in-context meaning; one token in two senses, and an alias named by a context sentence
an_relation.png   toy, two dimensions: the template as a relation, the same arrows in two unrelated domains
an_toy_curves.png toy simulation: AUROC of restatement against its matched foil along the analogy ladder, for the lexical
                  reading, the embedding and context-mixed states of three strengths
an_solvable.png   the analogy ladder against the readings: which reading can in principle separate each level, from the toy
Toy figures are illustrations of the argument, not measurements or predictions of any trained model.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 150})
INK, MUTED, GREY, BLUE, AMBER, RED, GREEN, LIGHT = "#1F2937", "#6B7280", "#9CA3AF", "#3B82F6", "#F59E0B", "#DC2626", "#059669", "#E5E7EB"
HERE = Path(__file__).resolve().parent; OUT = HERE / "template_matcher"
sys.path.insert(0, str(HERE.parents[1] / "notebooks/conservative_arch/scaleup/debug"))
despine = lambda ax: ax.spines[["top", "right"]].set_visible(False)
save = lambda fig, n: (fig.savefig(OUT / n, bbox_inches="tight", facecolor="white"), plt.close(fig), print("wrote", n))

# 1. token overlap, real clauses
from template_fresh_clauses import FRESH
import tiktoken
enc = tiktoken.get_encoding("gpt2"); tok = lambda s: [enc.decode([t]).strip().lower() for t in enc.encode(" " + s)]
ov = lambda a, b: (lambda A, B: sum(x in B for x in A) / len(A))(tok(a), set(tok(b)))
KINDS = [("exact\n(MP)", lambda P: (P["aff"], P["aff"])), ("negated\n(DS1, MT)", lambda P: (P["neg"], P["aff"])),
         ("paraphrase\n(MPsyn)", lambda P: (P["syn"], P["aff"])), ("near miss", lambda P: (P["near"], P["aff"])), ("near miss,\nnegated", lambda P: (P["near_neg"], P["aff"]))]
vals = [[ov(*f(P)) for P in FRESH] for _, f in KINDS]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
cols = [BLUE, BLUE, GREEN, RED, RED]
bp = a.boxplot(vals, patch_artist=True, widths=0.55, medianprops=dict(color=INK))
for p_, c in zip(bp["boxes"], cols): p_.set_facecolor(c); p_.set_alpha(0.35); p_.set_edgecolor(c)
a.set_xticks(range(1, 6), [k for k, _ in KINDS]); a.set_ylim(0, 1.05); a.set_ylabel("share of the second sentence's tokens\nfound in the restated clause")
a.set_title("Current test set (60 clauses, real)"); despine(a)
a.annotate("paraphrase and near miss\nshare the same overlap:\none word decides", xy=(3.5, 0.70), xytext=(3.2, 0.18), fontsize=9.5, color=INK, ha="center",
           arrowprops=dict(arrowstyle="->", color=MUTED))
lev = ["L0\nverbatim", "L1\none\nword", "L2\nfull\npara-\nphrase", "L3\nentailed\ncase", "L4\nrefer-\nence", "L5\nre-\nstruc-\ntured"]
pos_ov, foil_ov = [1.0, 0.7, 0.0, 0.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]
x = np.arange(len(lev)); b.bar(x - 0.18, pos_ov, 0.34, color=GREEN, alpha=0.8, label="restatement (valid form)")
b.bar(x + 0.18, foil_ov, 0.34, color=RED, alpha=0.8, label="matched foil")
b.set_xticks(x, lev, fontsize=9); b.set_ylim(0, 1.05)
for i_ in range(2, 6): b.text(i_, 0.015, "0", ha="center", fontsize=9, color=MUTED); b.set_title("Analogy levels (design targets)"); b.legend(frameon=False, fontsize=9, loc="upper right"); despine(b)
b.text(3.5, 0.25, "from L2 on, restatement and foil\nshare no content word with the clause:\nstring identity is at chance", ha="center", fontsize=9.5, color=INK)
b.set_ylabel("content-word overlap with the clause")
save(fig, "an_overlap.png")

# 2. toy 2-D: word vectors against context-mixed states
def arrows(ax, V, title, LP, ref="cold (clause)"):
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=LIGHT, lw=1)); C = {"cold (clause)": INK, "cold (repeated)": BLUE, "chilled (paraphrase)": GREEN, "strong (near miss)": RED, "it (pronoun)": AMBER}
    for k, ang in V.items():
        v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C[k], lw=2.2 if "clause" in k else 1.8))
        cs = np.cos(np.radians(ang - V[ref])); lbl = k if "clause" in k else f"{k}: cos {cs:+.2f}"; x_, y_ = LP[k]
        ax.text(x_, y_, lbl, fontsize=9, color=C[k], ha="left"); ax.plot([v[0], x_ - 0.02], [v[1], y_ + 0.03], color=C[k], lw=0.5, alpha=0.6)
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-0.25, 1.35); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
arrows(axs[0], {"cold (clause)": 0, "cold (repeated)": 2, "chilled (paraphrase)": 77, "strong (near miss)": 81, "it (pronoun)": 92}, "Word vectors (embedding): identity, not meaning",
       {"cold (clause)": (1.08, -0.12), "cold (repeated)": (1.08, 0.12), "chilled (paraphrase)": (0.55, 0.95), "strong (near miss)": (0.55, 1.12), "it (pronoun)": (0.55, 1.29)})
# right panel: the states s = 0.6 w + 0.8 m of the worked toy (Analogy_and_Context_Test_Set.md, 4.2); m along "cold" except the near miss's (45 degrees)
st_ang = lambda w, m: float(np.degrees(np.arctan2(0.6 * np.sin(np.radians(w)) + 0.8 * np.sin(np.radians(m)), 0.6 * np.cos(np.radians(w)) + 0.8 * np.cos(np.radians(m)))))
arrows(axs[1], {"cold (clause)": 0, "cold (repeated)": st_ang(2, 0), "chilled (paraphrase)": st_ang(77, 0), "strong (near miss)": st_ang(81, 45), "it (pronoun)": st_ang(92, 0)}, "States s = 0.6 w + 0.8 m: meaning pulls restatements together",
       {"cold (clause)": (1.08, -0.12), "cold (repeated)": (1.08, 0.16), "chilled (paraphrase)": (1.0, 0.45), "strong (near miss)": (0.6, 1.0), "it (pronoun)": (0.95, 0.68)})
fig.text(0.5, 0.0, "Toy angles. Left: the measured SR2-2B embedding cosines (repeated 0.998, paraphrase 0.23, near miss 0.16; a pronoun shares nothing with the predicate).\n"
         "Right: the worked toy of the note (4.2), a state mixing its word with the meaning of its sentence. The paraphrase and the pronoun gain much more than the near miss.",
         ha="center", fontsize=9.5, color=MUTED)
save(fig, "an_toy_2d.png")

# 2b. in-context meaning: one word, two senses; an alias named by the context
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
def ctx_panel(ax, V, title, note):
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=LIGHT, lw=1))
    for k, (ang, c_, dy) in V.items():
        v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c_, lw=2))
        ax.text(v[0] * 1.08, v[1] * 1.08 + dy, k, fontsize=9, color=c_, ha="left" if v[0] >= -0.1 else "right")
    ax.text(0.2, -0.3, note, fontsize=9, color=MUTED, ha="center"); ax.set_xlim(-1.5, 1.9); ax.set_ylim(-0.45, 1.3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
ctx_panel(axs[0], {"bank (river): \"the bank is steep\"": (20, BLUE, 0.06), "bank (money): \"the bank is closed\"": (20.5, RED, -0.16),
                   "the baker": (75, INK, 0.0), "Mr Hale": (150, GREEN, 0.0)},
          "Word vectors: one vector per token", "the two senses of \"bank\" are one vector (cos 1.00);\n\"Mr Hale\" shares nothing with \"the baker\", whatever the context says")
ctx_panel(axs[1], {"bank, after \"we walked by the river\"": (5, BLUE, 0.0), "bank, after \"I needed cash\"": (52, RED, 0.0),
                   "the baker": (95, INK, 0.06), "Mr Hale, after \"Mr Hale is the baker\"": (105, GREEN, -0.04)},
          "States read in context", "a state that carries the context separates the senses\nand pulls the alias onto the person it names")
fig.text(0.5, 0.0, "Toy angles: an illustration of what in-context meaning would look like in the states, not a measurement.", ha="center", fontsize=9.5, color=MUTED)
save(fig, "an_context.png")

# 3. the template as a relation, in two domains
fig, ax = plt.subplots(figsize=(11, 5.2))
dom = [("coffee", np.array([0.0, 0.0]), "the coffee is cold", "it has gone chilly", "the coffee is not cold"),
       ("road", np.array([6.2, 0.6]), "the road is closed", "the street is shut", "the road is not closed")]
neg = np.array([0.0, -1.6]); para = np.array([0.35, 0.25])
for name, o, cl, pa, ng in dom:
    P_ = o; ax.scatter(*P_, s=90, color=INK, zorder=3); ax.text(P_[0] - 0.2, P_[1] - 0.35, f"clause: \"{cl}\"", fontsize=9.5, color=INK, ha="right")
    q = P_ + para; ax.scatter(*q, s=60, color=GREEN, zorder=3); ax.annotate("", xy=q, xytext=P_, arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    ax.text(q[0] + 0.15, q[1] + 0.05, f"MP analogue: \"{pa}\"\n(short arrow: same statement)", fontsize=9, color=GREEN)
    n_ = P_ + neg; ax.scatter(*n_, s=60, color=BLUE, zorder=3); ax.annotate("", xy=n_, xytext=P_, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(n_[0] + 0.12, n_[1] - 0.1, f"denial: \"{ng}\"\n(the negation direction)", fontsize=9, color=BLUE)
ax.annotate("", xy=(5.6, -0.3), xytext=(2.6, -0.5), arrowprops=dict(arrowstyle="<->", color=GREY, lw=1, ls="--"))
ax.text(4.1, -0.25, "different words, no token in common", ha="center", fontsize=9, color=MUTED)
ax.text(3.5, -2.2, "The template is the pattern of arrows, not the points: in a space where meaning is geometry, the arrows repeat\n"
        "across domains, so a template fitted in one domain can fire in another. String identity sees only the points.", ha="center", fontsize=9.5, color=INK)
ax.set_xlim(-3.2, 10.2); ax.set_ylim(-2.5, 1.4); ax.axis("off"); ax.set_title("A template as a relation between parts (toy)")
save(fig, "an_relation.png")

# 4. toy simulation along the ladder
# Calibration: the embedding's word cosines of a paraphrase and of a near miss with the clause's word are drawn so that their medians
# (0.23, 0.07) and their AUROC (about 0.75) match the measured SR2-2B embedding (Template_Matchers_on_the_Dynamics_SR2_2B.md, 5.6).
# A state mixes the word vector, a concept term (the word's meaning, recovered with an item-dependent fidelity), a context term shared
# with the clause's state (the subject and the frame), and noise. phi is the share of the concept term: how much the state carries
# meaning. phi = 0 is a state with context but no meaning, which is what the measured early states look like on this one cosine.
rng = np.random.default_rng(0); D, N = 64, 800
u = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True); G_ = lambda: u(rng.standard_normal((N, D)))
def auroc(a, b):
    v = np.r_[a, b]; _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True); r = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]; return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
mix = lambda base, a: u(np.asarray(a)[:, None] * base + np.sqrt(np.clip(1 - np.asarray(a) ** 2, 0, 1))[:, None] * G_())
def state(w, con, ctx, phi):
    f = rng.uniform(0, 2 * phi, N); con = mix(con, rng.uniform(0, 1, N)); ks, kn = .25, .15
    return u(np.sqrt(1 - f - ks - kn)[:, None] * w + np.sqrt(f)[:, None] * con + np.sqrt(ks) * ctx + np.sqrt(kn) * G_())
def ladder(phi):
    c, cf, cs, ctx = G_(), G_(), G_(), G_(); e_cl = mix(c, np.full(N, .5)); h_cl = state(e_cl, c, ctx, phi); cos = lambda a, b: (a * b).sum(1); out = {}
    syn, foil = lambda: mix(e_cl, rng.uniform(0, .46, N)), lambda: mix(e_cl, rng.uniform(-.1, .25, N))
    for L in ("L1", "L2", "L3"):
        ep, ef = syn(), foil(); cp = c if L != "L3" else mix(c, np.full(N, .75))                       # L3: a narrower concept than the clause's
        if L == "L3": ep = mix(e_cl, rng.uniform(0, .3, N))
        e_p, e_f = cos(ep, e_cl), cos(ef, e_cl); h_p, h_f = cos(state(ep, cp, ctx, phi), h_cl), cos(state(ef, cf, ctx, phi), h_cl)
        if L != "L1":                                                                              # the subject is reworded too: its pair adds the same noise to both
            sub = rng.uniform(0, .46, N); e_p, e_f, h_p, h_f = (e_p + sub) / 2, (e_f + sub) / 2, (h_p + sub) / 2, (h_f + sub) / 2
        out[L] = (0.5, auroc(e_p, e_f), auroc(h_p, h_f))
    out["L0"] = (1.0, 1.0, auroc(cos(state(e_cl, c, ctx, phi), h_cl), cos(state(foil(), cf, ctx, phi), h_cl)))
    e_it = u(np.tile(rng.standard_normal(D), (N, 1)))                                             # L4: one pronoun vector for every premise
    out["L4"] = (0.5, 0.5, auroc(cos(state(e_it, c, ctx, phi), h_cl), cos(state(e_it, cf, ctx, phi), h_cl)))
    al, alf = G_(), G_()                                                                           # C1: an alias named by a context sentence
    out["C1"] = (0.5, auroc(cos(al, e_cl), cos(alf, e_cl)), auroc(cos(state(al, c, ctx, phi), h_cl), cos(state(alf, cf, ctx, phi), h_cl)))
    # C2: restatement and foil both repeat the clause's word, the foil in another sense. String identity and word vectors tie.
    out["C2"] = (0.5, 0.5, auroc(cos(state(e_cl, c, ctx, phi), h_cl), cos(state(e_cl, cs, ctx, phi), h_cl)))
    # C2': the restatement is a paraphrase, the foil repeats the word in another sense (a false friend). String identity ranks it first.
    ep = syn(); out["C2'"] = (0.0, auroc(cos(ep, e_cl), np.ones(N)), auroc(cos(state(ep, c, ctx, phi), h_cl), cos(state(e_cl, cs, ctx, phi), h_cl)))
    return out
fig, ax = plt.subplots(figsize=(12.5, 5.2)); levels = ["L0", "L1", "L2", "L3", "L4", "C1", "C2", "C2'"]
base = ladder(0.0)
ax.plot(levels, [base[L][0] for L in levels], "-o", color=GREY, lw=2, label="lexical (token identity)")
ax.plot(levels, [base[L][1] for L in levels], "-o", color=INK, lw=2, label="embedding (word vectors), calibrated to SR2-2B")
for phi, c_, nm in ((0.0, "#BFDBFE", "state with context, no meaning"), (0.05, "#93C5FD", "state, weak meaning"), (0.12, BLUE, "state, moderate meaning"), (0.25, "#1E3A8A", "state, strong meaning")):
    r = ladder(phi); ax.plot(levels, [r[L][2] for L in levels], "-o", color=c_, lw=2, label=f"{nm} (phi {phi:.2f})"); print(nm, {L: round(r[L][2], 3) for L in levels})
print("embedding", {L: round(base[L][1], 3) for L in levels})
ax.axhline(0.5, color=RED, lw=1, ls=":"); ax.text(7.25, 0.515, "chance", color=RED, fontsize=9); ax.axvspan(4.5, 7.5, color=LIGHT, alpha=0.5, zorder=0)
ax.text(6.0, 0.04, "in-context meaning", ha="center", fontsize=9.5, color=INK)
ax.set_xticks(range(8), ["L0\nverbatim", "L1\none word", "L2\nfull paraphrase", "L3\nentailed case", "L4\nreference", "C1\nalias in context", "C2\nsame word,\nsense differs", "C2\u2032\nparaphrase vs\nfalse friend"])
ax.set_ylim(-0.03, 1.03); ax.set_ylabel("AUROC, restatement against matched foil"); ax.set_title("Toy simulation: who keeps the template as the words go"); despine(ax)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.1))
save(fig, "an_toy_curves.png")

# 5. solvable in principle
rows = ["L0 verbatim", "L1 one word", "L2 full paraphrase", "L3 entailed case", "L4 reference", "L5 restructured", "L6 new domain", "C1 alias defined in context", "C2 same word, sense differs", "C2\u2032 paraphrase vs false friend", "C3 ellipsis", "C4 negation set by context"]
colsR = ["lexical", "embedding", "early state", "late state / path"]
S = np.array([[2, 2, 2, 2], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0.5, 1, 1], [0, 0, 1, 1], [0, 0.5, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [-1, -1, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
txt = [["yes", "yes", "yes", "yes"], ["no", "partly", "partly", "partly"], ["no", "partly", "partly", "partly"], ["no", "weakly", "if the model\nknows it", "if the model\nknows it"],
       ["no", "no", "if the model\nresolves it", "if the model\nresolves it"], ["no", "weakly", "if the model\nknows it", "if the model\nknows it"], ["no", "as L2", "as L2", "as L2"],
       ["no", "no", "if the model\nreads the context", "if the model\nreads the context"], ["no (a tie)", "no (a tie)", "if the model\nseparates senses", "if the model\nseparates senses"],
       ["misled\n(below chance)", "misled\n(below chance)", "only if meaning\noutweighs the word", "only if meaning\noutweighs the word"],
       ["no", "no", "if the model\nresolves it", "if the model\nresolves it"], ["no", "no", "if the model\nreads the context", "if the model\nreads the context"]]
fig, ax = plt.subplots(figsize=(10.5, 7.8)); cm = plt.matplotlib.colors.ListedColormap(["#FCA5A5", "#FEE2E2", "#FEF3C7", "#DBEAFE", "#D1FAE5"])
ax.imshow(np.digitize(S, [-0.5, 0.25, 0.75, 1.5]), cmap=cm, vmin=0, vmax=4, aspect="auto"); ax.axhline(6.5, color=INK, lw=1.2)
for i in range(len(rows)):
    for j in range(len(colsR)): ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=9, color=INK)
ax.set_xticks(range(len(colsR)), colsR); ax.set_yticks(range(len(rows)), rows); ax.tick_params(length=0); ax.xaxis.tick_top()
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title("Which reading can separate each level, in principle", pad=28)
save(fig, "an_solvable.png")
