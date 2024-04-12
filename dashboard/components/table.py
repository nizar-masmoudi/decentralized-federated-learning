import json
import os
import uuid

import numpy as np
import pandas as pd
from dash import html, dash_table


class TableAIO(html.Div):
    class ID:
        pass
    ID = ID

    def __init__(self, aio_id: str = None):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        files = [os.path.join('logs/', file) for file in os.listdir('logs/')]
        table = []
        for i, file in enumerate(files):
            with open(file) as json_file:
                data = json.load(json_file)

                avg_globalloss = np.mean([client['test/loss'][-1] for client in data['clients']])
                std_globalloss = np.std([client['test/loss'][-1] for client in data['clients']])
                avg_globalacc = np.mean([client['test/accuracy'][-1] for client in data['clients']])
                std_globalacc = np.std([client['test/accuracy'][-1] for client in data['clients']])
                total_comm_energy = np.sum([sum([item['energy'] for sublist in client['peers'] for item in sublist]) for client in data['clients']])
                total_comp_energy = np.sum([client['computation_energy'] * sum(client['activity']) for client in data['clients']])

                print(file, total_comm_energy)

                table.append([
                    '{:02d}'.format(i + 1),
                    data['name'],
                    data['clients'][0]['dataset']['name'],
                    TableAIO.format_policy_params({'activator': data['config']['activator']}),
                    TableAIO.format_policy_params({'aggregator': data['config']['aggregator']}),
                    TableAIO.format_policy_params({'selector': data['config']['selector']}),
                    round(avg_globalloss, 2),
                    round(std_globalloss, 2),
                    round(avg_globalacc, 2),
                    round(std_globalacc, 2),
                    round(total_comm_energy, 2),
                    round(total_comp_energy / 1e3, 2),
                ])
        df = pd.DataFrame(table, columns=['#', 'Name', 'Dataset', 'Activation policy', 'Aggregation policy',
                                          'Selection policy', 'Average loss', 'Loss std.', 'Average accuracy',
                                          'Accuracy std.', 'Communication energy (J)', 'Computation energy (kJ)'])

        super().__init__([
            html.H2('Runs', className='font-semibold text-lg'),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                sort_action='native',
                style_table={
                    'minWidth': '100%',
                    'overflowX': 'auto',
                    'marginTop': '16px',
                },
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'white',
                    'color': '#96A5B8',
                },
                style_cell={
                    'fontFamily': 'Poppins, sans-serif',
                    'fontSize': '14px',
                    'color': '#444A6D',
                    'border': '1px solid #EFF1F3',
                    'padding': '10px',
                    'textAlign': 'left'
                },
                fixed_columns={'headers': True, 'data': 2},
                style_data_conditional=[
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': 'inherit !important',
                        'border': 'inherit !important',
                    },
                    {
                        'if': {
                            'filter_query': '{{Communication energy (J)}} = {}'.format(
                                df['Communication energy (J)'].max()
                            ),
                            'column_id': 'Communication energy (J)'
                        },
                        'backgroundColor': '#FF6969',
                        'color': 'white'
                    },
                    {
                        'if': {
                            'filter_query': '{{Communication energy (J)}} = {}'.format(
                                df['Communication energy (J)'].min()
                            ),
                            'column_id': 'Communication energy (J)'
                        },
                        'backgroundColor': '#79AC78',
                        'color': 'white'
                    },
                    {
                        'if': {
                            'filter_query': '{{Average loss}} = {}'.format(
                                df['Average loss'].max()
                            ),
                            'column_id': 'Average loss'
                        },
                        'backgroundColor': '#FF6969',
                        'color': 'white'
                    },
                    {
                        'if': {
                            'filter_query': '{{Average loss}} = {}'.format(
                                df['Average loss'].min()
                            ),
                            'column_id': 'Average loss'
                        },
                        'backgroundColor': '#79AC78',
                        'color': 'white'
                    },
                ]
            )
        ], className='w-full p-7 bg-white rounded-lg shadow-[0px_4px_20px_rgba(237,237,237,0.5)]')

    @staticmethod
    def format_policy_params(module: dict):
        name = list(module.keys())[0]
        props = list(module.values())[0]
        if name == 'aggregator':
            return 'FedAvg'
        else:
            if props['policy'].startswith('Full'):
                return 'Full'
            elif props['policy'].startswith('Rand'):
                return 'Random (p = {})'.format(props['p'])
            elif props['policy'].startswith('Efficient'):
                if name == 'activator':
                    return 'Efficient (α = {}, threshold = {})'.format(props['alpha'], props['threshold'])
                elif name == 'selector':
                    return 'Efficient (θ = {})'.format(props['theta'])
        return '-'
