"""
This module implements utilities for printing/outputting
parameters of a given lbm layer.
"""

import torch
import pandas as pd

STYLE = """
<style>
table {
    font-size: 0.9em;
    font-family: sans-serif;
    text-align: center;
    display: inline-table;
    margin: 10px;
    border-collapse: collapse;
    border-top: 2px solid #009879;
    border-bottom: 2px solid #009879;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
}

th, td {
    padding: 15px;
    border: none;
}

tr:nth-child(even) {
    background-color: #f2f2f2;
}
</style>
"""

def tensor_to_html(arr: torch.tensor, caption: str = ""):
    df = pd.DataFrame(arr.numpy())

    if not caption == "":
        return (df.style.set_caption(caption)
                  .hide(axis="index")
                  .hide(axis="columns")
                  .to_html()
        )

    else:
        return df.to_html(index=False, header=False)

def model_to_html(model, filepath: str):
    html = STYLE

    zipped = zip(model.get_current_weights(), model.get_expected_weights())

    for (name, obtained), (_, expected) in zipped:
        html += f"<h1>{name}</h1>"

        html += "<div>"
        html += tensor_to_html(obtained, "obtained")
        html += tensor_to_html(expected, "expected")
        html += "</div>"

    with open(filepath, 'w') as fo:
        fo.write(html)

def print_model(model):
    print("Expected:")
    for name, param in model.get_expected_weights():
        print(f"{name}: {param}")
    print("Obtained:")
    for name, param in model.get_current_weights():
        print(f"{name}: {param}")
