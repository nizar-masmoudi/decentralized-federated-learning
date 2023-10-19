import dash
from dash import Dash, html, dcc


app = Dash(
    __name__,
    external_scripts=['https://cdn.tailwindcss.com'],
    use_pages=True,
    suppress_callback_exceptions=True
)

app.layout = html.Div([
    dcc.Store('local-storage', 'memory'),
    dcc.Location(id='location'),
    dash.page_container
], className='w-screen min-h-screen overflow-hidden')

if __name__ == '__main__':
    app.run_server(debug=True)
