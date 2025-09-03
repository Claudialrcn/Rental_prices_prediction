import json
import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_leaflet as dl
import pandas as pd
import dash_mantine_components as dmc

from utils.app_utils import (
    get_size_options,
    get_location_info_from_coords, 
    get_coords_from_location, 
    get_floor_options, 
    get_price_prediction)
from utils.param_utils import DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS

# Geojson of Malaga province municipalities
# with open("data/distritos_malaga.geojson", "r", encoding="utf-8") as f:
#     malaga_geojson = json.load(f)

# with open("data/malaga_province.geojson", "r", encoding="utf-8") as f:
#      malaga_geojson = json.load(f)

with open("data/app_data/municipios_20250824.geojson", "r", encoding="utf-8") as f:
      malaga_geojson = json.load(f)

df = pd.read_csv("../data/working_data/data_cleaned_20250827.csv")

df_municipalities = pd.read_csv('data/working_data/municipalities.csv')

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True
app.config.prevent_initial_callbacks = True

app.callback_map

#============================================== CONSTANTS ==============================================

PROPERTY_SIZE_LIMITS = get_size_options(df)

DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS = DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS
DIC_MUNICIPALITIES
LOCATION_CENTROIDS = pd.read_pickle("data/working_data/datalocation_centroids.pkl")

#============================================== LAYOUT ==============================================
app.layout = dmc.MantineProvider(
    children =      
        html.Div([
            html.H2("Price Prediction of Rent and Sale Properties in Málaga Province", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([
                                html.Label("Property Type", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="property_type_dropdown",
                                    options=[
                                        {"label": "Flat", "value": "flat"},
                                        {"label": "Country House", "value": "countryHouse"},
                                        {"label": "Studio", "value": "studio"},
                                        {"label": "Chalet", "value": "chalet"},
                                        {"label": "Duplex", "value": "duplex"},
                                        {"label": "Premise", "value": "premise"},
                                        {"label": "Garage", "value": "garage"},
                                        {"label": "Office", "value": "office"},
                                        {"label": "Penthouse", "value": "penthouse"},
                                    ],
                                    placeholder="Property Type",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("Detailed Property Type", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="detailed_type_dropdown",
                                    options=[
                                        {'label': 'Flat', 'value': 'flat'}, 
                                        {'label': 'Penthouse', 'value': 'penthouse'}, 
                                        {'label': 'Studio', 'value': 'studio'}, 
                                        {'label': 'Semi detached House', 'value': 'semidetachedHouse'},
                                        {'label': 'Duplex', 'value': 'duplex'},
                                        {'label': 'Independant House', 'value': 'independantHouse'},
                                        {'label': 'Terraced House', 'value': 'terracedHouse'}, 
                                        {'label': 'Office', 'value': 'office'}, 
                                        {'label': 'Commercial Property', 'value': 'commercialProperty'}, 
                                        {'label': 'Industrial Premise', 'value': 'industrialPremise'}, 
                                        {'label': 'Premise', 'value': 'premise'}, 
                                        {'label': 'Casale', 'value': 'casale'}, 
                                        {'label': 'Chalet', 'value': 'chalet'}, 
                                        {'label': 'Country House', 'value': 'countryHouse'}, 
                                        {'label': 'Cortijo', 'value': 'cortijo'}, 
                                        {'label': 'Village house', 'value': 'casaDePueblo'}, 
                                        {'label': 'Car And Motorcycle', 'value': 'carAndMotorcycle'}, 
                                        {'label': 'compactCar', 'value': 'compactCar'}, 
                                        {'label': 'sedanCar', 'value': 'sedanCar'}, 
                                        {'label': 'garage', 'value': 'garage'}, 
                                        {'label': 'twoCars', 'value': 'twoCars'}, 
                                        {'label': 'caseron', 'value': 'caseron'}
                                    ],
                                    placeholder="Property Type",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("Operation", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="operation_dropdown",
                                    options=[
                                        {"label": "Sale", "value": "sale"},
                                        {"label": "Rent", "value": "rent"}
                                    ],
                                    placeholder="Operation",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("Size (m²)", style={"fontWeight": "bold"}),
                                html.Div(
                                    id='slider_container',
                                    children=[
                                        dmc.Alert(
                                            "Please, select first a Property Type",
                                            id="size_warning_box",
                                            title="Property Type Required",
                                            color="yellow",
                                            hide=False,
                                            style={"marginTop": "10px"}
                                        ),
                                        dcc.Slider(
                                            min=0,
                                            max=1,
                                            value=0,
                                            id='size_slider',
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            included=True
                                        ),
                                    ]
                                )
                            ]),

                            html.Div([
                                html.Label("Number of Rooms", style={"fontWeight": "bold", "marginBottom": "5px"}),
                                dcc.Input(
                                    id='rooms_input',
                                    type='number',
                                    min=0,
                                    max=13,
                                    step=1,
                                    placeholder='Insert number of rooms',
                                    style={
                                        "width": "100%",
                                        "padding": "7px 10px",
                                        "fontSize": "14px",
                                        "borderRadius": "6px",
                                        "border": "1px solid #ccc",
                                        "boxSizing": "border-box"
                                    }
                                )
                            ], style={"display": "flex", "flexDirection": "column", "marginBottom": "10px"}),

                            html.Div([
                                html.Label("Number of Bathrooms", style={"fontWeight": "bold", "marginBottom": "5px"}),
                                dcc.Input(
                                    id='bathrooms_input',
                                    type='number',
                                    min=0,
                                    max=14,
                                    step=1,
                                    placeholder='Insert number of bathrooms',
                                    style={
                                        "width": "100%",
                                        "padding": "7px 10px",
                                        "fontSize": "14px",
                                        "borderRadius": "6px",
                                        "border": "1px solid #ccc",
                                        "boxSizing": "border-box"
                                    }
                                )
                            ], style={"display": "flex", "flexDirection": "column", "marginBottom": "10px"}),


                            html.Div([
                                html.Label("Municipality", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='municipality_dropdown',
                                    options=DIC_MUNICIPALITIES,
                                    placeholder="Select Municipality",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("District", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='district_dropdown',
                                    options=DIC_DISTRICTS,
                                    placeholder="Select district",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("Neighborhood", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='neighborhood_dropdown',
                                    options=DIC_NEIGHBORHOODS,
                                    placeholder="Select neighborhood",
                                    value=None
                                )
                            ]),

                            html.Div([dmc.Alert(
                                id="location_warning_box",
                                title="Location Warning",
                                color="yellow",
                                hide=True,
                                style={
                                    "marginTop": "15px",          
                                    "padding": "10px",        
                                    "boxSizing": "border-box",
                                    "whiteSpace": "normal",   
                                }
                            ),]),

                            html.Div([
                                html.Label("Latitude", style={"fontWeight": "bold"}),
                                html.Div(
                                    "No coordinates selected yet",
                                    id="lat_value_box",
                                    style={
                                        "border": "1px solid #ccc",
                                        "padding": "7px 10px",
                                        "borderRadius": "5px",
                                        "backgroundColor": "#f9f9f9",
                                        "color": "#888",  # gris
                                        "marginTop": "5px"
                                    }
                                )
                            ]),

                            html.Div([
                                html.Label("Longitude", style={"fontWeight": "bold"}),
                                html.Div(
                                    "No coordinates selected yet",
                                    id="lon_value_box",
                                    style={
                                        "border": "1px solid #ccc",
                                        "padding": "7px 10px",
                                        "borderRadius": "5px",
                                        "backgroundColor": "#f9f9f9",
                                        "color": "#888",  # gris
                                        "marginTop": "5px"
                                    }
                                )
                            ]),

                            html.Div([
                                html.Label("Status", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id='status_dropdown',
                                    options=[{'label': 'Good', 'value': 'good'}, 
                                            {'label': 'New development', 'value': 'newdevelopment'}, 
                                            {'label': 'Renew', 'value': 'renew'}],
                                    placeholder="Select status",
                                    value=None
                                )
                            ]),

                            html.Div([
                                html.Label("Lift", style={"fontWeight": "bold"}),
                                dcc.RadioItems(
                                    id='lift_radio',
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False},
                                    ],
                                    value=None,
                                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                )
                            ]),

                            html.Div([
                                html.Label('Floor', style={"fontWeight": "bold"}),
                                dcc.Dropdown( 
                                    id='floor_dropdown',
                                    options=[],
                                    placeholder="Select floor",
                                    value=None
                                ),
                            ]),

                            html.Div([
                                html.Label("Parking Space", style={"fontWeight": "bold"}),
                                dcc.RadioItems(
                                    id='parking_radio',
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False},
                                    ],
                                    value=None,
                                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                                )
                            ]),

                        ], style={
                                'flex': '1',
                                'padding': '20px',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '10px',
                                'height': '80vh',         
                                'overflowY': 'auto',     
                                'boxSizing': 'border-box',
                            }
                    ),
                    html.Div(
                        [
                            dl.Map(
                                center=[36.7, -4.4],  
                                zoom=9,
                                id='map',
                                style={'width': '100%', 'height': '80vh'},
                                children=[
                                    dl.TileLayer(),  
                                    dl.LayerGroup(id="click_marker_layer"),  
                                    dl.GeoJSON(data=malaga_geojson)
                                ]
                            ),
                        ], style={
                            'flex': '3',
                            'height': '80vh',} 
                    ),

                ], style={
                        'display': 'flex', 
                        'width': '100%', 
                        'height': '80vh', 
                        'boxSizing':'border-box'}
            ),
            html.Div([
                html.Button("Calcular Predicción",          
                    id="predict_button", 
                    n_clicks=0, 
                        style={
                        "marginTop": "20px",
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer"
                        }
                    ),
                dmc.Alert(
                    "Fill all the fields before the prediction, please.",
                    id="input_warning_alert",
                    title="INCOMPLETE INPUTS",
                    color="red",
                    hide=True,
                    style={"marginTop": "10px"}
                ),
                html.Div(
                    id="prediction_result",
                    style={"marginTop": "15px", "fontSize": "20px", "fontWeight": "bold"}
                ),
            ]),
        ]),
    
)

#============================================== CALBACKS ==============================================

# Callback del slider
@app.callback(
    Output('size_slider', 'min'),
    Output('size_slider', 'max'),
    Output('size_slider', 'value'),
    Output('size_slider', 'disabled'),
    Output('size_warning_box', 'hide'),
    Input('property_type_dropdown', 'value')
)
def update_slider(property_type):
    if property_type is None:
        return dash.no_update, dash.no_update, dash.no_update, True, {
            "padding": "12px",
            "backgroundColor": "#f0f0f0",
            "border": "1px dashed #bbb",
            "color": "#888",
            "textAlign": "center",
            "borderRadius": "6px",
            "display": "block"
        }

    limits = PROPERTY_SIZE_LIMITS.get(property_type)
    return limits['min'], limits['max'], limits['min'], False, True

@app.callback(
    Output('floor_dropdown', 'options'),
    Output('floor_dropdown', 'value'),
    Input('property_type_dropdown', 'value')
)
def update_floor_dropdown(property_type):
    if property_type:
        options = get_floor_options(property_type)
        if len(options) == 1:
            return options, options[0]['value']
        else:
            return options, None
    return [], None


# Callback to update location dropdowns and handle map clicks
@app.callback(
    Output('municipality_dropdown', 'options'),
    Output('district_dropdown', 'options'),
    Output('neighborhood_dropdown', 'options'),
    Output('municipality_dropdown', 'value'),
    Output('district_dropdown', 'value'),
    Output('neighborhood_dropdown', 'value'),
    Output('lat_value_box', 'children'),
    Output('lon_value_box', 'children'),
    Output('click_marker_layer', 'children'),
    Output('location_warning_box', 'children'),
    Output('location_warning_box', 'hide'),
    Input('municipality_dropdown', 'value'),
    Input('district_dropdown', 'value'),
    Input('neighborhood_dropdown', 'value'),
    Input('map', 'clickData'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def handle_location_changes(municipality, district, neighborhood, click_data):
    triggered = ctx.triggered_id
    df_filtered = df_municipalities.copy()
    warning_msg = ""
    lat_value = None
    lon_value = None

    # === Caso: clic en el mapa ===
    if triggered == 'map' and click_data:
        coordinates = click_data['latlng']
        lat_value, lon_value = round(coordinates['lat'], 5), round(coordinates['lng'], 5)

        municipality, district, neighborhood = get_location_info_from_coords(lat_value, lon_value)

        # Si no se encontró nada
        if not all([municipality, district, neighborhood]):
            warning_msg = "Could not determine location details from the selected point. Please try another location."

        df_filtered = df_filtered[
            (df_filtered['municipality'] == municipality) &
            (df_filtered['district'] == district) &
            (df_filtered['neighborhood'] == neighborhood)
        ]

    elif triggered == 'neighborhood_dropdown' and neighborhood:
        df_filtered = df_filtered[df_filtered['neighborhood'] == neighborhood]
        if not df_filtered.empty:
            district = df_filtered['district'].iloc[0]
            municipality = df_filtered['municipality'].iloc[0]

    elif triggered == 'district_dropdown' and district:
        df_filtered = df_filtered[df_filtered['district'] == district]
        if municipality is None and not df_filtered.empty:
            municipality = df_filtered['municipality'].mode().iloc[0]

    elif triggered == 'municipality_dropdown' and municipality:
        df_filtered = df_filtered[df_filtered['municipality'] == municipality]

    # Filtrar para valores actuales
    if municipality:
        df_filtered = df_filtered[df_filtered['municipality'] == municipality]
    if district:
        df_filtered = df_filtered[df_filtered['district'] == district]
    if neighborhood:
        df_filtered = df_filtered[df_filtered['neighborhood'] == neighborhood]

    # Coordenadas si no las tenemos aún
    if lat_value is None or lon_value is None:
        lat_value, lon_value = get_coords_from_location(municipality, district, neighborhood, LOCATION_CENTROIDS)

    # Dropdowns
    mun_opts = [{'label': m, 'value': m} for m in sorted(df_municipalities['municipality'].unique())]
    dist_opts = [{'label': d, 'value': d} for d in sorted(df_municipalities[df_municipalities['municipality'] == municipality]['district'].dropna().unique())] if municipality else []
    neigh_opts = [{'label': n, 'value': n} for n in sorted(df_municipalities[(df_municipalities['municipality'] == municipality) & (df_municipalities['district'] == district)]['neighborhood'].dropna().unique())] if municipality and district else []

    if lat_value and lon_value:
        marker = dl.Marker(position=[lat_value, lon_value])
    else:
        marker = []

    show_alert = bool(warning_msg)

    return (
        mun_opts,
        dist_opts,
        neigh_opts,
        municipality,
        district,
        neighborhood,
        lat_value,
        lon_value,
        marker,
        warning_msg,
        not show_alert
    )

# Callback para la prediccion del precio
@app.callback(
    Output("input_warning_alert", "hide"),
    Output("prediction_result", "children"),
    Input("predict_button", "n_clicks"),
    State("property_type_dropdown", "value"),
    State("operation_dropdown", "value"),
    State("size_slider", "value"),
    State("rooms_input", "value"),
    State("bathrooms_input", "value"),
    State("municipality_dropdown", "value"),
    State("district_dropdown", "value"),
    State("neighborhood_dropdown", "value"),
    State("lat_value_box", "children"),
    State("lon_value_box", "children"),
    State("status_dropdown", "value"),
    State("lift_radio", "value"),
    State("floor_dropdown", "value"),
    State("parking_radio", "value")
    
)
def handle_prediction(n_clicks, property_type, operation, size, rooms, bathrooms,
                      municipality, district, neighborhood, lat, lon,
                      status, lift, floor, parking):

    if n_clicks == 0:
        return True, dash.no_update

    # Verificación
    inputs = [property_type, operation, size, rooms, bathrooms,
              municipality, district, neighborhood, lat, lon,
              status, lift, floor, parking]
    
    if any(x is None for x in inputs):
        return False, ""  # Mostrar alerta

    # Asegúrate de que lat/lon estén bien formateados (pueden venir como strings)
    try:
        lat = float(lat) if isinstance(lat, (float, int)) else float(str(lat).strip())
        lon = float(lon) if isinstance(lon, (float, int)) else float(str(lon).strip())
    except:
        return False, "Invalid Coordinates."

    try:
        pred = get_price_prediction(
            property_type, operation, size, rooms, bathrooms,
            municipality, district, neighborhood, lat, lon,
            status, lift, floor, parking
        )
        formatted = f"Predicted price: {pred:,.2f} €"
        return True, formatted
    except Exception as e:
        return False, f"Error calculating the prediction: {e}"


#============================================== RUN SERVER ==============================================
if __name__ == '__main__':
    app.run(debug=True)
