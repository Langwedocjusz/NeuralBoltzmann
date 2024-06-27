"""
This module implements utilities for printing/outputting
parameters of a given lbm layer.
"""

import torch
import pandas as pd

STYLE = """
<style>
body {
    font-family: sans-serif;
    max-width: 100%;
    overflow-x: hidden;
}

table {
    font-size: 0.9em;
    text-align: center;
    display: inline-table;
    overflow-x: auto;
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
    """Utility to convert torch tensor to html table."""

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
    """Saves obtained and expected parameters of the model to an html file at given filepath."""

    html  = "<!DOCTYPE html>\n"
    html += "<html>\n"
    html += "<head>\n"
    html += STYLE
    html += "</head>\n"
    html += "<body>\n"

    zipped = zip(model.get_current_weights(), model.get_classical_weights())

    for (name, obtained), (_, expected) in zipped:
        html += f"<h1>{name}</h1>\n"

        html += "<div>\n"
        html += tensor_to_html(obtained, "obtained")
        html += tensor_to_html(expected, "classical")
        html += "</div>\n"

    html += "</body>\n"
    html += "</html>\n"

    with open(filepath, 'w') as fo:
        fo.write(html)

def print_model(model):
    """Prints expected and obtained parameters of the model to std out."""

    print("Expected:")
    for name, param in model.get_classical_weights():
        print(f"{name}: {param}")
    print("Classical:")
    for name, param in model.get_current_weights():
        print(f"{name}: {param}")
