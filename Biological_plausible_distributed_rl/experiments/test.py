"""Test: kann matplotlib hier speichern?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Gleiche Logik wie compare_agents.py
save_dir = Path(__file__).resolve().parent.parent / "results" / "comparison"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Script location: {Path(__file__).resolve()}")
print(f"Save directory:  {save_dir.resolve()}")
print(f"Directory exists: {save_dir.exists()}")

# Test plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9], "ro-")
ax.set_title("Test Plot - wenn du das siehst, funktioniert es")

filepath = save_dir / "test_plot.png"
fig.savefig(filepath, bbox_inches="tight")
plt.close()

print(f"Saved to: {filepath.resolve()}")
print(f"File exists: {filepath.exists()}")
print(f"File size: {filepath.stat().st_size} bytes")