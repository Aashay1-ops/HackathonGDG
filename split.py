"""The one locked 80/20 split everyone imports. Data owner: replace the source file
here if we switch to main + deduped 20k; everything downstream keeps working."""
import pandas as pd


def load_split(path="data/tickets.csv", seed=42, frac=0.8):
    df = pd.read_csv(path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["text"] = df["subject"].fillna("") + "\n" + df["body"].fillna("")
    cut = int(len(df) * frac)
    kb, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    # Leakage guard: holdout answers never leave this function.
    return kb, test.drop(columns=["answer"])
