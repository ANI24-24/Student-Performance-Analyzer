#!/usr/bin/env python
# coding: utf-8

# # Student Performance & Study Group Analyzer
# 
# **Name:** G Anirudh
# **Register No.:** RA2411003012249
# 
# This notebook reads a CSV of student grades and attendance, cleans it, summarises the class with
# statistics, estimates pass and 'A' grade probabilities from the data, and pairs students into
# study buddies using cosine similarity between their grade vectors.
# 
# | Phase | Subject | What happens here |
# |---|---|---|
# | 1. Data Setup | Python & Analytics | Generate and load the CSV, fill missing marks, bar chart of class averages |
# | 2. Stat Analysis | Statistics | Mean, median, standard deviation; attendance vs performance correlation |
# | 3. Pass/Fail Odds | Probability | Empirical probability of passing, scoring an A, scoring above 80% |
# | 4. Study Buddy Match | Linear Algebra | Dot product / cosine similarity between grade vectors, pairing |

# In[1]:


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.precision", 2)
plt.rcParams["figure.dpi"] = 110
SUBJECTS = ["Math_Score", "Science_Score", "English_Score"]


# ## Phase 1: Data Setup
# 
# ### 1.1 Generate the dataset
# This is the script from the project statement. One change: the original seeds NumPy but not
# Python's `random` module, so the rows that lose a Math mark change on every run. Adding
# `random.seed(42)` keeps the results in this notebook reproducible.

# In[2]:


np.random.seed(42)
random.seed(42)   # added so the missing-value rows are the same every run

first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna",
               "Ishaan", "Shaurya", "Ananya", "Aadhya", "Diya", "Saanvi", "Priya", "Neha",
               "Riya", "Kriti", "Pooja", "Kavya", "Rahul", "Rohan", "Amit", "Karan",
               "Vikram", "Sneha", "Nidhi", "Tanvi", "Rashi", "Meera"]

data = []
for name in first_names:
    attendance = np.random.randint(60, 101)
    base_score = attendance * 0.8   # ties grades to attendance
    math_grade = min(100, max(0, int(np.random.normal(base_score, 10))))
    science_grade = min(100, max(0, int(np.random.normal(base_score, 8))))
    english_grade = min(100, max(0, int(np.random.normal(base_score + 5, 7))))
    if random.random() < 0.05:      # simulate a missing Math mark
        math_grade = np.nan
    data.append([name, math_grade, science_grade, english_grade, attendance])

df_gen = pd.DataFrame(data, columns=["Student_Name", "Math_Score", "Science_Score",
                                     "English_Score", "Attendance_Pct"])
df_gen.to_csv("student_data.csv", index=False)
print("Saved student_data.csv with", len(df_gen), "rows")


# ### 1.2 Load and inspect the CSV

# In[3]:


df = pd.read_csv("student_data.csv")
print(df.shape)
df.head(10)


# In[4]:


df.info()
print("\nMissing values per column:")
print(df.isnull().sum())


# ### 1.3 Handle missing values
# A missing mark is filled with that subject's class mean. With 30 students, dropping a row would
# throw away the student's other two subjects and their attendance, which we still need for the
# correlation and the buddy matching.

# In[5]:


missing_rows = df[df[SUBJECTS].isnull().any(axis=1)]
print("Rows with a missing mark:")
print(missing_rows)

for col in SUBJECTS:
    df[col] = df[col].fillna(round(df[col].mean(), 1))

print("Missing values after cleaning:", int(df.isnull().sum().sum()))
df["Overall_Avg"] = df[SUBJECTS].mean(axis=1).round(2)
df.loc[missing_rows.index]


# ### 1.4 Bar chart: class average per subject

# In[6]:


averages = df[SUBJECTS].mean()
labels = [s.replace("_Score", "") for s in SUBJECTS]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, averages.values, color=["#4C72B0", "#55A868", "#C44E52"], width=0.55)
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set_ylim(0, 100)
ax.set_ylabel("Average score")
ax.set_title("Class Average per Subject")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("class_averages.png")
plt.show()


# ## Phase 2: Statistical Analysis
# 
# ### 2.1 Mean, median and standard deviation
# The mean is the class average, the median is the middle score once marks are sorted (it is not
# pulled around by a single very low or very high mark), and the standard deviation measures how
# far scores typically sit from the mean.

# In[7]:


summary = pd.DataFrame({
    "Mean": df[SUBJECTS + ["Overall_Avg"]].mean(),
    "Median": df[SUBJECTS + ["Overall_Avg"]].median(),
    "Std Dev": df[SUBJECTS + ["Overall_Avg"]].std(),     # sample std (n-1)
    "Min": df[SUBJECTS + ["Overall_Avg"]].min(),
    "Max": df[SUBJECTS + ["Overall_Avg"]].max(),
})
summary.index = ["Math", "Science", "English", "Overall"]
summary.round(2)


# To show the formula behind the library call, here is the standard deviation for Math worked out
# by hand: $s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}$

# In[8]:


x = df["Math_Score"].values
mean_x = x.sum() / len(x)
s_manual = np.sqrt(((x - mean_x) ** 2).sum() / (len(x) - 1))
print(f"Math mean (manual)    = {mean_x:.2f}")
print(f"Math std dev (manual) = {s_manual:.2f}")
print(f"Math std dev (pandas) = {df['Math_Score'].std():.2f}")


# In[9]:


fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([df[c] for c in SUBJECTS], tick_labels=labels, patch_artist=True,
           boxprops=dict(facecolor="#DCE6F2"))
ax.set_ylabel("Score")
ax.set_title("Spread of Scores per Subject")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("score_spread.png")
plt.show()


# ### 2.2 Correlation between attendance and performance
# Pearson's r runs from -1 to +1. Values near +1 mean that students who attend more also score
# more.

# In[10]:


corr = df[["Attendance_Pct"] + SUBJECTS + ["Overall_Avg"]].corr()
print("Correlation of each column with attendance:")
print(corr["Attendance_Pct"].drop("Attendance_Pct").round(3))

r = corr.loc["Attendance_Pct", "Overall_Avg"]
strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
print(f"\nAttendance vs overall average: r = {r:.3f} ({strength} positive relationship)")


# In[11]:


m, c = np.polyfit(df["Attendance_Pct"], df["Overall_Avg"], 1)
xs = np.linspace(df["Attendance_Pct"].min(), df["Attendance_Pct"].max(), 50)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter(df["Attendance_Pct"], df["Overall_Avg"], color="#4C72B0", alpha=0.8)
ax.plot(xs, m * xs + c, color="#C44E52", label=f"Best fit: y = {m:.2f}x + {c:.1f}")
ax.set_xlabel("Attendance (%)")
ax.set_ylabel("Overall average score")
ax.set_title(f"Attendance vs Overall Performance (r = {r:.2f})")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("attendance_vs_performance.png")
plt.show()


# ## Phase 3: Pass/Fail Odds (Probability)
# 
# Empirical probability is the share of students in the data who meet a condition:
# 
# $$P(\text{event}) = \frac{\text{number of students who meet the condition}}{\text{total students}}$$
# 
# Thresholds used: **pass = 40 or above**, **A grade = 80 or above**, and the project's
# **above 80%** question is checked on each subject and on the overall average.

# In[12]:


PASS_MARK, A_MARK = 40, 80
n = len(df)

rows = []
for col, lab in zip(SUBJECTS, labels):
    rows.append({
        "Subject": lab,
        "P(Pass, >=40)": (df[col] >= PASS_MARK).sum() / n,
        "P(A grade, >=80)": (df[col] >= A_MARK).sum() / n,
        "P(Above 80, >80)": (df[col] > A_MARK).sum() / n,
    })
prob = pd.DataFrame(rows).set_index("Subject")
prob.round(3)


# In[13]:


p_all_pass = (df[SUBJECTS] >= PASS_MARK).all(axis=1).mean()
p_overall_80 = (df["Overall_Avg"] > 80).mean()
p_any_A = (df[SUBJECTS] >= A_MARK).any(axis=1).mean()

print(f"P(student passes all three subjects)      = {p_all_pass:.3f}")
print(f"P(student scores an A in at least one)    = {p_any_A:.3f}")
print(f"P(random student's overall average > 80%) = {p_overall_80:.3f}")


# **Conditional probability.** Does attendance change the odds? Splitting the class at 80%
# attendance:

# In[14]:


high = df[df["Attendance_Pct"] >= 80]
low = df[df["Attendance_Pct"] < 80]
for label, grp in [("Attendance >= 80%", high), ("Attendance <  80%", low)]:
    p = (grp[SUBJECTS] >= A_MARK).any(axis=1).mean()
    print(f"{label}: {len(grp):2d} students, P(at least one A) = {p:.3f}")


# In[15]:


fig, ax = plt.subplots(figsize=(7, 4))
xpos = np.arange(len(labels))
w = 0.38
b1 = ax.bar(xpos - w/2, prob["P(Pass, >=40)"], w, label="Pass (>=40)", color="#55A868")
b2 = ax.bar(xpos + w/2, prob["P(A grade, >=80)"], w, label="A grade (>=80)", color="#4C72B0")
ax.bar_label(b1, fmt="%.2f", padding=2)
ax.bar_label(b2, fmt="%.2f", padding=2)
ax.set_xticks(xpos, labels)
ax.set_ylim(0, 1.3)
ax.set_ylabel("Empirical probability")
ax.set_title("Probability of Passing vs Scoring an A")
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("probabilities.png")
plt.show()


# ## Phase 4: Study Buddy Match (Linear Algebra)
# 
# Each student becomes a vector of three marks, e.g. $\vec{a} = [\text{Math}, \text{Science}, \text{English}]$.
# Cosine similarity measures the angle between two vectors:
# 
# $$\cos\theta = \frac{\vec{a} \cdot \vec{b}}{\lVert\vec{a}\rVert \, \lVert\vec{b}\rVert}$$
# 
# A value of 1 means the two profiles point the same way.
# 
# ### 4.1 A worked example with two students

# In[16]:


X_raw = df[SUBJECTS].values.astype(float)
names = df["Student_Name"].values

a, b = X_raw[0], X_raw[1]
dot = np.dot(a, b)
na, nb = np.linalg.norm(a), np.linalg.norm(b)
print(f"{names[0]}: {a}")
print(f"{names[1]}: {b}")
print(f"Dot product      = {dot:.1f}")
print(f"|a| = {na:.2f}, |b| = {nb:.2f}")
print(f"Cosine similarity = {dot / (na * nb):.4f}")


# ### 4.2 Why the raw marks need centering first
# All marks are positive and sit in a narrow band, so every raw vector points in nearly the same
# direction and almost every pair scores close to 1. That makes the ranking useless for matching.

# In[17]:


def cosine_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    U = X / norms
    return U @ U.T          # all pairwise dot products of unit vectors

S_raw = cosine_matrix(X_raw)
off = S_raw[~np.eye(len(S_raw), dtype=bool)]
print(f"Raw vectors: similarity ranges from {off.min():.3f} to {off.max():.3f}")


# The fix is to standardise each subject (z-score: subtract the subject mean, divide by its
# standard deviation). Now a positive entry means "above the class in this subject" and a negative
# entry means "below". Cosine similarity on these vectors compares strength profiles, so it
# separates a student who is strong in Math but weak in English from one with the opposite
# pattern.

# In[18]:


Z = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
S = cosine_matrix(Z)
off = S[~np.eye(len(S), dtype=bool)]
print(f"Standardised vectors: similarity ranges from {off.min():.3f} to {off.max():.3f}")

sim_df = pd.DataFrame(S, index=names, columns=names)
sim_df.iloc[:6, :6].round(3)


# In[19]:


fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
ax.set_yticks(range(len(names)), names, fontsize=8)
ax.set_title("Cosine Similarity Between Student Grade Profiles")
fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
plt.tight_layout()
plt.savefig("similarity_heatmap.png")
plt.show()


# ### 4.3 Pair students into study buddies
# Greedy matching: take the most similar unpaired pair, lock it in, repeat until everyone has a
# partner. With 30 students this gives 15 pairs. If the class size were odd, the last student
# joins the pair they are most similar to.

# In[20]:


def make_pairs(S, names):
    n = len(names)
    candidates = sorted(((S[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
                        reverse=True)
    used, pairs = set(), []
    for sim, i, j in candidates:
        if i not in used and j not in used:
            pairs.append([i, j, sim])
            used.update((i, j))
    leftover = [k for k in range(n) if k not in used]
    groups = [[p[0], p[1]] for p in pairs]
    for k in leftover:   # odd class size: join the closest pair
        best = max(range(len(groups)), key=lambda g: np.mean([S[k, m] for m in groups[g]]))
        groups[best].append(k)
    return pairs, groups

pairs, groups = make_pairs(S, names)

def profile(i):
    return " / ".join(f"{int(v)}" for v in X_raw[i])

buddy_df = pd.DataFrame([{
    "Pair": k + 1,
    "Student A": names[i], "A (M/S/E)": profile(i),
    "Student B": names[j], "B (M/S/E)": profile(j),
    "Cosine Similarity": round(sim, 3),
} for k, (i, j, sim) in enumerate(pairs)])
buddy_df.to_csv("study_buddies.csv", index=False)
buddy_df


# Greedy matching pairs the best matches first, so the last few pairs are whoever is left. The
# average similarity below shows how good the pairing is overall.

# In[21]:


print(f"Average similarity across all pairs: {buddy_df['Cosine Similarity'].mean():.3f}")
print(f"Strongest pair: {buddy_df.iloc[0]['Student A']} & {buddy_df.iloc[0]['Student B']} "
      f"({buddy_df.iloc[0]['Cosine Similarity']:.3f})")
last = buddy_df.iloc[-1]
print(f"Weakest pair:   {last['Student A']} & {last['Student B']} ({last['Cosine Similarity']:.3f})")


# ## Final Summary
# Everything above, collected into one printed report.

# In[22]:


print("=" * 60)
print("        STUDENT PERFORMANCE & STUDY GROUP REPORT")
print("=" * 60)
print(f"Students analysed : {len(df)}")
print(f"Missing marks filled with subject mean: {len(missing_rows)}")
print("\n--- Subject statistics ---")
print(summary.loc[["Math", "Science", "English"], ["Mean", "Median", "Std Dev"]].round(2).to_string())
print(f"\nAttendance vs overall score correlation: r = {r:.3f}")
print("\n--- Probabilities ---")
print(prob.round(3).to_string())
print(f"\nP(pass all subjects)       = {p_all_pass:.3f}")
print(f"P(overall average > 80%)   = {p_overall_80:.3f}")
print("\n--- Study buddy pairs ---")
for _, row in buddy_df.iterrows():
    print(f"  {row['Pair']:2d}. {row['Student A']:<8} + {row['Student B']:<8} "
          f"(similarity {row['Cosine Similarity']:.3f})")
print("=" * 60)

