import pandas as pd
from pathlib import Path

from decision import build_decision
from brief import build_brief
from evaluate import evaluate_system
from impact import estimate_impact, build_impact_summary


DATA_DIR = Path("data")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_decisions(df):
    path = OUTPUT_DIR / "latest_decisions.csv"
    df.tail(20).to_csv(path, index=False)
    print(f"Saved decisions → {path}")


def save_brief(df):
    path = OUTPUT_DIR / "latest_brief.txt"
    brief = build_brief(df)
    path.write_text(brief)
    print(f"Saved brief → {path}")


def save_evaluation(df):
    path = OUTPUT_DIR / "evaluation_summary.txt"
    results = evaluate_system(df)

    lines = []
    lines.append("SCARLET Evaluation Summary\n")

    lines.append("Regime distribution:")
    lines.append(str(results["regime_distribution"]))
    lines.append("")

    lines.append("Action distribution:")
    lines.append(str(results["action_distribution"]))
    lines.append("")

    lines.append(f"Escalation rate: {results['escalation_rate']:.2f}")
    lines.append(f"Stressed rate: {results['stressed_rate']:.2f}")
    lines.append(f"Lead indicator accuracy: {results['lead_indicator_accuracy']:.2f}")
    lines.append(f"Lag warning accuracy: {results['lag_warning_accuracy']:.2f}")

    path.write_text("\n".join(lines))
    print(f"Saved evaluation → {path}")


def save_economic_impact(df):
    path = OUTPUT_DIR / "economic_impact.txt"
    df_impact = estimate_impact(df)
    summary = build_impact_summary(df_impact)

    path.write_text(summary)
    print(f"Saved economic impact → {path}")


def main():
    df = pd.read_csv(DATA_DIR / "jpm_weekly_features.csv")

    df = build_decision(df)

    save_decisions(df)
    save_brief(df)
    save_evaluation(df)
    save_economic_impact(df)


if __name__ == "__main__":
    main()