# Student Performance Analyzer

A Python project that reads grades and attendance for 30 students, summarises the class, estimates pass and A-grade odds from the data, and pairs students into study buddies. There is also a standalone HTML dashboard for exploring the results.

By G Anirudh.

## What it does

The analysis runs in four phases:

1. **Data setup.** Generates `student_data.csv` (three subject marks and attendance per student), loads it, and fills the 4 missing Math marks with the class mean. Dropping those rows would lose the students' other two marks and their attendance, which the later steps need.
2. **Statistics.** Mean, median and standard deviation for each subject, with the standard deviation also worked out by hand to check the library. Attendance correlates strongly with overall score (r = 0.917).
3. **Probability.** Empirical odds of passing (40 or above), scoring an A (80 or above) and averaging above 80. 90% of the class passes all three subjects, and 50% scores an A in at least one. Splitting at 80% attendance, 70% of the high-attendance group gets an A somewhere against 10% of the rest.
4. **Study buddies.** Each student becomes a vector of three marks, and cosine similarity between vectors decides who is paired. Raw marks are all positive and close together, so every pair scores between 0.93 and 1 and the ranking tells you nothing. Standardising each subject first (z-scores) fixes that. Pairs are then matched greedily, most similar first, which gives 15 pairs.

The data is synthetic. The script seeds both NumPy and Python's `random` module with 42, so the missing marks fall on the same students every run.

## Files

| File | What it is |
|---|---|
| `Student_Performance_Analyzer.ipynb` | The notebook, with explanations and charts inline |
| `Student_Performance_Analyzer.py` | The same code as a plain script |
| `student_dashboard.html` | Interactive dashboard, opens in any browser |
| `student_data.csv` | The generated dataset |
| `study_buddies.csv` | The 15 pairs and their similarity scores |
| `*.png` | Charts saved by the script |

## Running it

You need Python 3 with numpy, pandas and matplotlib (matplotlib 3.9 or newer).

```
pip install numpy pandas matplotlib
python Student_Performance_Analyzer.py
```

This prints the full report and writes the CSVs and charts into the current folder. To read it step by step, open the notebook instead.

## Dashboard

Open `student_dashboard.html` in a browser. It needs an internet connection the first time to load the chart library and fonts, but the data is built into the file.

- Sliders for minimum attendance, pass mark and A-grade mark update every number and chart.
- Click a student anywhere (chart dot, heatmap cell, table row) to see their report card, rank and study buddy.
- The heatmap switches between standardised and raw marks, so you can see why standardising matters.
- The roster is sortable and searchable.

The dashboard recalculates everything in JavaScript and matches the Python output, including the 15 pairs.
