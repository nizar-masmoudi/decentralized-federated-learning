import uuid

import dash_cytoscape as cyto
from dash import html, callback, Input, Output, State, ALL
from dash_daq.NumericInput import NumericInput

from dashboard.utils import format_size


# noinspection PyMethodParameters
class TopologyAIO(html.Div):
    class ID:
        client_cfg = lambda index: {
            'type': 'client_cfg',
            'index': index,
        }
        channel_cfg = lambda index: {
            'type': 'channel_cfg',
            'index': index,
        }

    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        super().__init__([
            html.Div([
                html.H2('Network Topolgy', className='font-semibold text-lg'),

                # Client configuration
                html.Div([
                    html.Span([
                        html.P('Client'),
                        html.P('-', self.ID.client_cfg(0)),
                    ], className='flex justify-between'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Latitude'),
                            html.P('-', self.ID.client_cfg(1)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Longitude'),
                            html.P('-', self.ID.client_cfg(2)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('FLOPs per CPU cycle'),
                            html.P('-', self.ID.client_cfg(3)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('CPU clock frequency'),
                            html.P('-', self.ID.client_cfg(4)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P("CPU's effective capacitance"),
                            html.P('-', self.ID.client_cfg(5)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Transmission power'),
                            html.P('-', self.ID.client_cfg(6)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Signal bandwidth'),
                            html.P('-', self.ID.client_cfg(7)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Signal frequency'),
                            html.P('-', self.ID.client_cfg(8)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                ], className='w-full my-7 pr-7 space-y-2'),

                # Channel configuration
                html.Div([
                    html.Span([
                        html.P('Channel'),
                        html.P('-', self.ID.channel_cfg(0)),
                    ], className='flex justify-between'),
                    html.Div(className='w-full h-px bg-[#EFF1F3]'),
                    html.Div([
                        html.Span([
                            html.P('Distance'),
                            html.P('-', self.ID.channel_cfg(1)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Channel gain'),
                            html.P('-', self.ID.channel_cfg(2)),
                        ], className='flex justify-between'),
                        html.Span([
                            html.P('Transmission rate'),
                            html.P('-', self.ID.channel_cfg(3)),
                        ], className='flex justify-between'),
                    ], className='space-y-2'),
                ], className='w-full my-7 pr-7 space-y-2'),

                # Legend
                html.Div([
                    html.Span([
                        html.Div(className='w-3 h-3 bg-[#EF4444] rounded-full'),
                        html.P('Inactive client')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='w-3 h-3 bg-[#00E096] rounded-full'),
                        html.P('Active client')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='border-t border-[#444A6D] border-dashed w-3 h-px'),
                        html.P('Neighbor link')
                    ], className='flex items-center space-x-2 text-xs'),
                    html.Span([
                        html.Div(className='border-t border-[#444A6D] border-solid w-3 h-px'),
                        html.P('Peer link')
                    ], className='flex items-center space-x-2 text-xs'),
                ], className='absolute bottom-0 left-0 flex justify-evenly w-full pr-7'),
            ], className='relative w-1/3 h-full text-[#444A6D] text-sm'),

            # Network
            html.Div([
                cyto.Cytoscape(
                    id='cytoscape',
                    layout={
                        'name': 'preset',
                        'animate': True,
                        'animationDuration': 500,
                    },
                    userZoomingEnabled=False,
                    userPanningEnabled=False,
                    zoomingEnabled=False,
                    panningEnabled=False,
                    pan={'x': 0, 'y': 0},
                    zoom=1,
                    style={'width': '100%', 'height': '100%'},
                    elements=[],
                    stylesheet=[
                        {'selector': 'node', 'style': {
                            'background-color': '#EF4444',
                            'content': 'data(label)',
                            'text-margin-y': '-10px',
                            'width': 20,
                            'height': 20,
                            'color': '#444A6D',
                            'font-size': '12px',
                        }},
                        {'selector': 'node[?active]', 'style': {
                            'background-color': '#00E096',
                        }},
                        {'selector': 'edge', 'style': {
                            'line-color': '#444A6D',
                            'width': 1,
                            'line-style': 'dashed',
                            'curve-style': 'bezier',
                            'target-arrow-color': '#444A6D',
                            'target-arrow-shape': 'triangle',
                            'arrow-scale': .7,
                            'text-halign': 'center',
                            'text-valign': 'center',
                            'color': '#444A6D',
                            'font-size': '12px',
                            'text-background-color': '#FAFBFC',
                            'text-background-opacity': 1,
                        }},
                        {'selector': '[?peer]', 'style': {
                            'line-style': 'solid',
                        }},
                    ]
                ),
                html.Div([
                    html.P('Round'),
                    NumericInput('round', 1, 40, 1, 10)
                ], className='absolute right-4 top-4 flex items-center justify-between bg-white text-sm '
                             'rounded-lg w-36 h-12 p-2 border border-[#EFF1F3]'),
            ], className='relative w-2/3 h-full bg-[#FAFBFC] rounded-lg'),
        ], className='flex items-center justify-center p-7 w-full h-[650px] bg-white '
                     'rounded-lg shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @callback(
        Output('cytoscape', 'elements', True),
        Input('local-storage', 'data'),
        prevent_initial_call=True
    )
    def init_cytoscape(data: dict):
        if data == {}:
            return []

        (min_lat, min_lon), (max_lat, max_lon) = data['config']['geo_limits']

        elements = []
        for client in data['clients']:
            id_ = client['id']
            elements.append({
                'data': {'id': id_, 'label': f'Client {id_}',  'active': client['activity'][0]},
                'position': {
                    'y': 594 * (client['locations'][0][0] - min_lat) / (max_lat - min_lat),
                    'x': 1040 * (client['locations'][0][1] - min_lon) / (max_lon - min_lon)},
                'grabbable': False
            })
        return elements

    @callback(
        Output('round', 'max'),
        Input('local-storage', 'data'),
        prevent_initial_call=True
    )
    def setup_max_rounds(data: dict):
        if data == {}:
            return 10
        return len(data['clients'][0]['activity'])

    @callback(
        Output('cytoscape', 'elements', True),
        Input('cytoscape', 'tapNodeData'),
        State('local-storage', 'data'),
        State('round', 'value'),
        State('cytoscape', 'elements'),
        prevent_initial_call=True
    )
    def display_edges(node: dict, data: dict, current_round: int, elements: list):
        if node is None:
            return elements

        id_ = int(node['id'])
        client = next((item for item in data['clients'] if item['id'] == id_), None)

        nodes = [item for item in elements if 'position' in item.keys()]
        edges = [
            {'data': {
                'id': 1/2 * (id_ + neighbor['id'])*(id_ + neighbor['id'] + 1) + neighbor['id'],  # Unique edge encoding
                'source': id_,
                'target': neighbor['id'],
                'peer': neighbor['id'] in [peer['id'] for peer in client['peers'][current_round - 1]],
            }} for neighbor in client['neighbors'][current_round - 1]
        ]
        return nodes + edges

    @callback(
        Output('cytoscape', 'elements', True),
        Input('round', 'value'),
        State('local-storage', 'data'),
        State('cytoscape', 'elements'),
        prevent_initial_call=True
    )
    def update_cyotscape(current_round: int, data: dict, default_elements: dict):
        if data == {}:
            return default_elements

        (min_lat, min_lon), (max_lat, max_lon) = data['config']['geo_limits']
        elements = []
        for client in data['clients']:
            id_ = client['id']
            elements.append({
                'data': {'id': id_, 'label': f'Client {id_}',  'active': client['activity'][current_round - 1]},
                'position': {
                    'y': 594 * (client['locations'][current_round - 1][0] - min_lat) / (max_lat - min_lat),
                    'x': 1040 * (client['locations'][current_round - 1][1] - min_lon) / (max_lon - min_lon)},
                'grabbable': False
            })
        return elements

    @callback(
        Output('cytoscape', 'layout'),
        Input('round', 'value'),
        State('local-storage', 'data'),
        State('cytoscape', 'layout'),
        prevent_initial_call=True
    )
    def animate_cytoscape(current_round: int, data: dict, layout: dict):
        if data == {}:
            return layout

        (min_lat, min_lon), (max_lat, max_lon) = data['config']['geo_limits']

        layout['positions'] = {}
        for client in data['clients']:
            id_ = client['id']
            layout['positions'][id_] = {
                'y': 594 * (client['locations'][current_round - 1][0] - min_lat) / (max_lat - min_lat),
                'x': 1040 * (client['locations'][current_round - 1][1] - min_lon) / (max_lon - min_lon),
            }
        return layout

    @callback(
        Output(ID.client_cfg(ALL), 'children'),
        Input('local-storage', 'data'),
        Input('cytoscape', 'tapNodeData'),
        Input('round', 'value'),
        State(ID.client_cfg(ALL), 'children'),
    )
    def update_client_config(data: dict, node: dict, current_round: int, default_configs: list):
        if data == {}:
            return default_configs

        id_ = int(node['id']) if node else 1
        client = next((item for item in data['clients'] if item['id'] == id_), None)

        return [
            'Client {}'.format(client['id'] if client else 1),
            round(client['locations'][current_round - 1][0], 7),
            round(client['locations'][current_round - 1][1], 7),
            client['components']['cpu']['fpc'],
            format_size(client['components']['cpu']['frequency'], 'hertz'),
            client['components']['cpu']['kappa'],
            '{} dBm'.format(client['components']['transmitter']['power']),
            format_size(client['components']['transmitter']['bandwidth']),
            format_size(client['components']['transmitter']['signal_frequency'], 'hertz'),
        ]

    @callback(
        Output(ID.channel_cfg(ALL), 'children'),
        Input('local-storage', 'data'),
        Input('cytoscape', 'tapEdgeData'),
        State('round', 'value'),
        State(ID.channel_cfg(ALL), 'children'),
    )
    def update_channel_config(data: dict, edge: dict, current_round: int, configs: list):
        if not data or edge is None:
            return configs
        else:
            source = next((item for item in data['clients'] if item['id'] == int(edge['source'])), None)
            target = next((item for item in data['clients'] if item['id'] == int(edge['target'])), None)
            assert source is not None, 'source was not found in data!'
            assert target is not None, 'target was not found in data!'

            distance = next((item['distance'] for item in source['neighbors'][current_round - 1] if item['id'] == int(target['id'])), None)

            configs = [
                'Client {} - Client {}'.format(source['id'], target['id']),
                '{:.3f} Km'.format(distance),
                '-',
                '-'
            ]
        return configs
