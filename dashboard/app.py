from dash import Dash, html, dcc, callback, Input, Output, State, ALL
from dashboard.components import SideBarAIO, ModalAIO, CytoscapeAIO, DataBlockAIO, ClassDistAIO, LossPlotAIO, AccPlotAIO
from dashboard.plots import ClassDistPlot, LossPlot, AccPlot
from dashboard.utils import format_size, geo_distance

app = Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])

# Layout
app.layout = html.Div([
    # Local storage
    dcc.Store('local-storage', 'local'),

    # Modal
    ModalAIO('modal'),

    # Sidebar
    SideBarAIO('sidebar'),

    # Dashboard
    html.Div([
        # Section 1
        html.Div([
            CytoscapeAIO('cytoscape'),
            DataBlockAIO('datablock'),
            ClassDistAIO('classdist'),
        ], className='grid grid-rows-2 grid-cols-4 grid-flow-col gap-4'),
        html.Div(className='w-full h-0.5 bg-[#444c56] rounded-full my-4'),

        # Section 2
        html.Div([
            LossPlotAIO('lossplot'),
            AccPlotAIO('accplot'),
            # TODO - Confusion matrix (TP, FP, ...) ?
        ], className='grid grid-cols-2 gap-4'),
        html.Div(className='w-full h-0.5 bg-[#444c56] rounded-full my-4'),


        # Section 3
        html.Div([

        ], className='h-96 w-full'),
    ], className='absolute left-[20%] w-4/5 min-h-screen px-10 mt-24')
], className='flex w-screen min-h-screen')


# Callbacks
@callback(
    Output(DataBlockAIO.ID.config(ALL), 'children'),
    Input('local-storage', 'data'),
    Input(CytoscapeAIO.ID.cytoscape('cytoscape'), 'tapNodeData'),
    Input(SideBarAIO.ID.round('sidebar'), 'children'),
    State(DataBlockAIO.ID.config(ALL), 'children'),
)
def update_configs(data: dict, node: dict, round_: int, configs: list):
    round_ -= 1
    if data:
        client = node or data['1']
        configs = [
            'Client {}'.format(node['id'] if node else '1'),
            round(client['locations'][round_][0], 7),
            round(client['locations'][round_][1], 7),
            client['cpu']['fpc'],
            format_size(client['cpu']['frequency'], 'hertz'),
            client['cpu']['kappa'],
            '{} dBm'.format(client['transmitter']['power']),
            format_size(client['transmitter']['bandwidth']),
            format_size(client['transmitter']['signal_frequency'], 'hertz'),
        ]
    return configs


@callback(
    Output(CytoscapeAIO.ID.cytoscape('cytoscape'), 'elements'),
    Input('local-storage', 'data'),
    Input(SideBarAIO.ID.neighbors('sidebar'), 'on'),
    Input(SideBarAIO.ID.peers('sidebar'), 'on'),
    Input(CytoscapeAIO.ID.cytoscape('cytoscape'), 'tapNodeData'),
    State(SideBarAIO.ID.round('sidebar'), 'children'),
    Input(CytoscapeAIO.ID.cytoscape('cytoscape'), 'layout'),
)
def update_cytoscape_elements(data: dict, show_neighbors: bool, show_peers: bool, node: dict, round_: int, _):
    if not data:
        return []
    if node is None:
        node_id = '1'
    else:
        node_id = node['id']

    round_ -= 1
    (min_lat, min_lon), (max_lat, max_lon) = ((36.897092, 10.152086), (36.870453, 10.219636))
    # Set nodes
    nodes = [{
        'data': {'id': id_, **data[id_], 'label': f'Client {id_}',  'active': data[id_]['activity'][round_]},
        'position': {
            'y': 1500 * (data[id_]['locations'][round_][0] - min_lat) / (max_lat - min_lat),
            'x': 1500 * (data[id_]['locations'][round_][1] - min_lon) / (max_lon - min_lon)},
        'grabbable': False} for id_ in data.keys()]
    # Set edges
    edges = []
    if show_neighbors:
        edges += [
            {'data': {
                'id': node_id + neighbor,
                'source': node_id,
                'target': neighbor,
                'opaque': show_peers and (neighbor not in data[node_id]['peers'][round_]),
                'distance': '{:.2f} Km'.format(geo_distance(
                    data[node_id]['locations'][round_],
                    data[neighbor]['locations'][round_]
                ))
            }} for neighbor in data[node_id]['neighbors'][round_]
        ]
    elif show_peers and not show_neighbors:
        edges += [
            {'data': {
                'id': node_id + peer,
                'source': node_id,
                'target': peer,
                'opaque': False,
                'distance': '{:.2f} Km'.format(geo_distance(
                    data[node_id]['locations'][round_],
                    data[peer]['locations'][round_]
                ))
            }} for peer in data[node_id]['peers'][round_]
        ]
    return nodes + edges


@callback(
    Output(CytoscapeAIO.ID.cytoscape('cytoscape'), 'layout'),
    Input(SideBarAIO.ID.round('sidebar'), 'children'),
    State('local-storage', 'data'),
)
def update_layout(round_: int, data: dict):
    round_ -= 1
    (min_lat, min_lon), (max_lat, max_lon) = ((36.897092, 10.152086), (36.870453, 10.219636))
    layout = {
        'name': 'preset',
        'animate': True,
        'animationDuration': 500,
        'positions': {
            id_: {
                'y': 1500 * (data[id_]['locations'][round_][0] - min_lat) / (max_lat - min_lat),
                'x': 1500 * (data[id_]['locations'][round_][1] - min_lon) / (max_lon - min_lon),
            } for id_ in data.keys()
        }
    }
    return layout


@callback(
    Output(ClassDistAIO.ID.graph('classdist'), 'figure'),
    Input(CytoscapeAIO.ID.cytoscape('cytoscape'), 'tapNodeData'),
    Input('local-storage', 'data'),
)
def update_classdist(node: dict, data: dict):
    if data:
        node = node or data['1']
        return ClassDistPlot(
            x=list(range(1, 11)),
            y=node['distribution'],
        )
    else:
        return ClassDistPlot()


@callback(
    Output(LossPlotAIO.ID.graph('lossplot'), 'figure'),
    Input(SideBarAIO.ID.dropdown_button('sidebar'), 'children'),
    Input('local-storage', 'data'),
    State(LossPlotAIO.ID.graph('lossplot'), 'figure'),
)
def update_lossplot(value: str, data: dict, figure):
    if data == {}:
        return LossPlot()
    xs = list(range(1, len(data['1']['tloss']) + 1))
    if value == 'Training':
        key = 'tloss'
    elif value == 'Validation':
        key = 'vloss'
    else:
        key = 'tloss'  # TODO add test

    ys = {id_: data[str(id_)][key] for id_ in range(1, len(data) + 1)}
    figure = LossPlot(x=xs, ys=ys)
    return figure


@callback(
    Output(AccPlotAIO.ID.graph('accplot'), 'figure'),
    Input(SideBarAIO.ID.dropdown_button('sidebar'), 'children'),
    Input('local-storage', 'data'),
    State(AccPlotAIO.ID.graph('accplot'), 'figure'),
)
def update_lossplot(value: str, data: dict, figure):
    if data == {}:
        return AccPlot()
    xs = list(range(1, len(data['1']['tacc']) + 1))
    if value == 'Train':
        key = 'tacc'
    elif value == 'Validation':
        key = 'vacc'
    else:
        key = 'tacc'  # TODO add test

    ys = {id_: data[str(id_)][key] for id_ in range(1, len(data) + 1)}
    figure = AccPlot(x=xs, ys=ys)
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
