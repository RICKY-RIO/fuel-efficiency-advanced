import pandas as pd
import webbrowser

# Read predictions from CSV
df = pd.read_csv('new_cars_with_predictions.csv')

# Create stylish HTML
css = """
<style>
body { background: #f5f6fa; font-family: Arial, sans-serif; }
h2 { color: #009879; }
.styled-table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 1.0em;
    min-width: 400px;
    box-shadow: 0 0 20px rgba(0,0,0,0.15);
    width: 80%;
}
.styled-table th, .styled-table td {
    border: 1px solid #dddddd;
    text-align: center;
    padding: 8px;
}
.styled-table th {
    background-color: #009879;
    color: #ffffff;
}
.styled-table tr:nth-child(even) {
    background-color: #f3f3f3;
}
.styled-table tr:hover {
    background-color: #d1e7dd;
}
</style>
"""

html = "<html><head><meta charset='utf-8'>" + css + "</head><body>\n"
html += "<h2>New Car MPG Predictions</h2>\n"
html += df.to_html(index=False, classes='styled-table', border=0)
html += "</body></html>"

with open('new_cars_results.html', 'w', encoding='utf-8') as f:
    f.write(html)

webbrowser.open_new_tab('new_cars_results.html')
