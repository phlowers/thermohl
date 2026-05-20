import os.path

import pandas as pd


def get_cable_data(cable_name: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if cable_name in df["conductor"].values:
        return df[df["conductor"] == cable_name].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {cable_name} not found in file {f}.")
