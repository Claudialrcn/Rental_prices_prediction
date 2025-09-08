import json
import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_leaflet as dl
import pandas as pd
import dash_mantine_components as dmc
from shapely.geometry import shape, Point
import pickle
import plotly.express as px
import os
from dotenv import load_dotenv

from utils.app_utils import (
    get_size_options,
    get_location_info_from_coords, 
    get_coords_from_location, 
    get_floor_options, 
    get_price_prediction)
from utils.param_utils import DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS

load_dotenv()


#============================================== DATA ==============================================

# Geojson of Malaga province municipalities
# with open("data/distritos_malaga.geojson", "r", encoding="utf-8") as f:
#     malaga_geojson = json.load(f)

# with open("data/malaga_province.geojson", "r", encoding="utf-8") as f:
#      malaga_geojson = json.load(f)

with open("data/app_data/municipios_20250824.geojson", "r", encoding="utf-8") as f:
      malaga_geojson = json.load(f)

with open("data/app_data/malaga_polygon.pkl", "rb") as f:
    malaga_polygon = pickle.load(f)

df = pd.read_csv("data/working_data/data_cleaned_20250827.csv")

df_municipalities = pd.read_csv('data/working_data/municipalities.csv')

PROPERTY_SIZE_LIMITS = get_size_options(df)

DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS = DIC_MUNICIPALITIES, DIC_DISTRICTS, DIC_NEIGHBORHOODS
DIC_MUNICIPALITIES
LOCATION_CENTROIDS = pd.read_pickle("data/working_data/datalocation_centroids.pkl")
with open("data/app_data/detailed_to_property.json", "r") as f:
    DETAILED_TO_PROPERTY_DICT = json.load(f)

with open("data/app_data/property_to_detailed.json", "r") as f:
    PROPERTY_TO_DETAILED_DICT = json.load(f)

#============================================== APP ==============================================

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True
#app.config.prevent_initial_callbacks = True

app.callback_map

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
                                            "Please, select first a Detailed Property Type",
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
                                    dl.GeoJSON(data=malaga_geojson),
                                ]
                            ),

                            html.Div(
                                id="average_prices_box",
                                children=[
                                    html.H4("Average Prices", style={"margin": "0 0 10px 0", "textAlign": "center"}),
                                    html.Div([
                                        html.Div([
                                            html.Div("Rent", style={"fontWeight": "bold", "marginBottom": "5px", "textAlign": "center"}),
                                            html.Div("Avg Price by Area", style={"fontSize": "12px", "color": "#666"}),
                                            html.Div(id="avg_rent_price_by_area", style={"fontSize": "16px", "marginBottom": "5px"}),
                                            html.Div("Avg Price", style={"fontSize": "12px", "color": "#666"}),
                                            html.Div(id="avg_rent_price", style={"fontSize": "16px"})
                                        ], style={"flex": 1, "padding": "8px", "textAlign": "center"}),

                                        html.Div([
                                            html.Div("Sale", style={"fontWeight": "bold", "marginBottom": "5px", "textAlign": "center"}),
                                            html.Div("Avg Price by Area", style={"fontSize": "12px", "color": "#666"}),
                                            html.Div(id="avg_sale_price_by_area", style={"fontSize": "16px", "marginBottom": "5px"}),
                                            html.Div("Avg Price", style={"fontSize": "12px", "color": "#666"}),
                                            html.Div(id="avg_sale_price", style={"fontSize": "16px"})
                                        ], style={"flex": 1, "padding": "8px", "textAlign": "center"}),
                                    ], style={"display": "flex", "gap": "15px"})
                                ],
                                style={
                                    "position": "absolute",
                                    "top": "80px",       
                                    "right": "40px",     
                                    "backgroundColor": "rgba(205, 205, 205, 0.9)", 
                                    "padding": "15px",
                                    "borderRadius": "10px",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.25)",
                                    "zIndex": 1000,
                                    "display": "none",  
                                    "minWidth": "250px"
                                }
                            )

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
                html.Button("Get prediction",          
                    id="predict_button", 
                    n_clicks=0, 
                        style={
                        "marginTop": "20px",
                        "marginLeft": "20px",
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
                    style={"marginTop": "10px", "marginLeft": "20px"}
                ),
                html.Div(
                    id="prediction_result",
                    style={"marginTop": "15px", "marginLeft": "20px", "fontSize": "20px", "fontWeight": "bold"}
                ),
            ]),
            html.Div([
                html.H3("Interactive Property Graphs", style={"marginBottom": "15px"}),

                dmc.Group(
                    [
                        html.Div([
                            html.Label("Property Type", style={"fontWeight": "bold", "marginRight": "5px"}),
                            dcc.Dropdown(
                                id="property_type_graph_dropdown",
                                options=[{"label": pt, "value": pt} for pt in df['propertyType'].unique()],
                                value=None,
                                multi=True,
                                placeholder="Select Property Type",
                                style={"width": "220px"}
                            ),
                        ], style={"marginRight": "20px"}),

                        html.Div([
                            html.Label("Municipality", style={"fontWeight": "bold", "marginRight": "5px"}),
                            dcc.Dropdown(
                                id="municipality_graph_dropdown",
                                options=[{"label": m, "value": m} for m in df['municipality'].unique()],
                                value=None,
                                multi=True,
                                placeholder="Select Municipality",
                                style={"width": "220px"}
                            ),
                        ], style={"marginRight": "20px"}),

                        html.Div([
                            html.Label("Operation", style={"fontWeight": "bold", "marginRight": "10px"}),
                            dcc.RadioItems(
                                id="operation_selector",
                                options=[
                                    {"label": "Sale", "value": "sale"},
                                    {"label": "Rent", "value": "rent"},
                                    {"label": "Both", "value": "both"}
                                ],
                                value="both",
                                inline=True,
                                style={"marginTop": "5px"}
                            ),
                        ]),
                    ],
                    justify="space-between", 
                    align="center",         
                    gap="md",
                    style={"marginBottom": "20px"}  
                ),

                dcc.Graph(id="price_by_area_graph", style={"marginTop": "20px"}),
                dcc.Graph(id="total_price_graph", style={"marginTop": "20px"}),
            ], style={"padding": "20px", "marginLeft":"20px", "marginRight":"20px", })

        ]),
    
)

#============================================== CALLBACKS ==============================================

@app.callback(
    Output("detailed_type_dropdown", "options"),
    Input("property_type_dropdown", "value")
)
def update_detailed_options(selected_property):
    if not selected_property:
        return [{"label": d, "value": d} for d in df["detailedType"].unique()]
    return [{"label": d, "value": d} for d in PROPERTY_TO_DETAILED_DICT[selected_property]]


@app.callback(
    Output("property_type_dropdown", "value"),
    Input("detailed_type_dropdown", "value"),
    prevent_initial_call=True
)
def sync_property_with_detailed(selected_detailed):
    if not selected_detailed:
        return no_update
    return DETAILED_TO_PROPERTY_DICT[selected_detailed]

@app.callback(
    Output('size_slider', 'min'),
    Output('size_slider', 'max'),
    Output('size_slider', 'value'),
    Output('size_slider', 'disabled'),
    Output('size_warning_box', 'hide'),
    Input('detailed_type_dropdown', 'value')
)
def update_slider(detailed_type):
    print(f"[DEBUG update_slider] detailed_type received: {detailed_type}")

    if detailed_type is None:
        print("[DEBUG update_slider] No detailed_type provided, disabling slider")
        return dash.no_update, dash.no_update, dash.no_update, True, {
            "padding": "12px",
            "backgroundColor": "#f0f0f0",
            "border": "1px dashed #bbb",
            "color": "#888",
            "textAlign": "center",
            "borderRadius": "6px",
            "display": "block"
        }

    limits = PROPERTY_SIZE_LIMITS.get(detailed_type)
    print(f"[DEBUG update_slider] Limits found: {limits}")
    return limits['min'], limits['max'], limits['min'], False, True


@app.callback(
    Output('floor_dropdown', 'options'),
    Input('property_type_dropdown', 'value')
)
def update_floor_dropdown(property_type):
    print(f"[DEBUG update_floor_dropdown] property_type received: {property_type}")

    if not property_type:
        print("[DEBUG update_floor_dropdown] No property_type, returning empty list")
        return []

    dic_min_max_floor = get_floor_options(df)
    print(f"[DEBUG update_floor_dropdown] Available property types: {list(dic_min_max_floor.keys())[:10]}...")

    if property_type not in dic_min_max_floor:
        print(f"[DEBUG update_floor_dropdown] {property_type} not found in dictionary")
        return []

    min_floor = int(dic_min_max_floor[property_type]["min"])
    max_floor = int(dic_min_max_floor[property_type]["max"])
    print(f"[DEBUG update_floor_dropdown] min_floor={min_floor}, max_floor={max_floor}")

    if min_floor == max_floor:
        print("[DEBUG update_floor_dropdown] Only one floor option available")
        return [{"label": str(min_floor), "value": min_floor}]
    else:
        return [{"label": str(i), "value": i} for i in range(min_floor, max_floor + 1)]


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
    print(f"[DEBUG handle_location_changes] Triggered by: {triggered}")
    print(f"[DEBUG handle_location_changes] Inputs - municipality: {municipality}, district: {district}, neighborhood: {neighborhood}")

    df_filtered = df_municipalities.copy()
    warning_msg = ""
    lat_value = None
    lon_value = None

    # if map clicked
    if triggered == 'map' and click_data:
        coordinates = click_data['latlng']
        lat_value, lon_value = round(coordinates['lat'], 5), round(coordinates['lng'], 5)
        print(f"[DEBUG handle_location_changes] Map clicked at: lat={lat_value}, lon={lon_value}")

        point = Point(lon_value, lat_value)
        municipality, district, neighborhood = get_location_info_from_coords(lat_value, lon_value)
        print(f"[DEBUG handle_location_changes] Location info resolved: {municipality}, {district}, {neighborhood}")

        if not all([municipality, district, neighborhood]):
            warning_msg = "Could not determine location details from the selected point. Please try another location."
            print("[WARNING handle_location_changes] Location info not found")

        if not malaga_polygon.contains(point):
            warning_msg = "Selected point is outside Málaga province. Please select a valid location."
            print("[WARNING handle_location_changes] Point outside Malaga polygon")

        df_filtered = df_filtered[
            (df_filtered['municipality'] == municipality) &
            (df_filtered['district'] == district) &
            (df_filtered['neighborhood'] == neighborhood)
        ]

    elif triggered == 'neighborhood_dropdown' and neighborhood:
        print(f"[DEBUG handle_location_changes] Neighborhood selected: {neighborhood}")
        df_filtered = df_filtered[df_filtered['neighborhood'] == neighborhood]
        if not df_filtered.empty:
            district = df_filtered['district'].iloc[0]
            municipality = df_filtered['municipality'].iloc[0]

    elif triggered == 'district_dropdown' and district:
        print(f"[DEBUG handle_location_changes] District selected: {district}")
        df_filtered = df_filtered[df_filtered['district'] == district]
        if municipality is None and not df_filtered.empty:
            municipality = df_filtered['municipality'].mode().iloc[0]

    elif triggered == 'municipality_dropdown' and municipality:
        print(f"[DEBUG handle_location_changes] Municipality selected: {municipality}")
        df_filtered = df_filtered[df_filtered['municipality'] == municipality]

    # Filtering again
    if municipality:
        df_filtered = df_filtered[df_filtered['municipality'] == municipality]
    if district:
        df_filtered = df_filtered[df_filtered['district'] == district]
    if neighborhood:
        df_filtered = df_filtered[df_filtered['neighborhood'] == neighborhood]

    if lat_value is None or lon_value is None:
        lat_value, lon_value = get_coords_from_location(municipality, district, neighborhood, LOCATION_CENTROIDS)
        print(f"[DEBUG handle_location_changes] Coordinates resolved from location: {lat_value}, {lon_value}")

    mun_opts = [{'label': m, 'value': m} for m in sorted(df_municipalities['municipality'].unique())]
    dist_opts = [{'label': d, 'value': d} for d in sorted(df_municipalities[df_municipalities['municipality'] == municipality]['district'].dropna().unique())] if municipality else []
    neigh_opts = [{'label': n, 'value': n} for n in sorted(df_municipalities[(df_municipalities['municipality'] == municipality) & (df_municipalities['district'] == district)]['neighborhood'].dropna().unique())] if municipality and district else []

    if lat_value and lon_value:
        marker = dl.Marker(position=[lat_value, lon_value])
    else:
        marker = []

    show_alert = bool(warning_msg)
    print(f"[DEBUG handle_location_changes] Warning message: {warning_msg}")

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

@app.callback(
    Output("average_prices_box", "style"),
    Output("avg_rent_price_by_area", "children"),
    Output("avg_rent_price", "children"),
    Output("avg_sale_price_by_area", "children"),
    Output("avg_sale_price", "children"),
    Input("property_type_dropdown", "value"),
    Input("municipality_dropdown", "value"),
)
def update_average_prices(property_type, municipality):
    if not property_type or not municipality:
        return {"display": "none"}, "", "", "", ""

    df_filtered = df.copy()
    df_filtered = df_filtered[
        (df_filtered["propertyType"] == property_type) &
        (df_filtered["municipality"] == municipality)
    ]

    avg_rent_area = df_filtered[df_filtered["operation"]=="rent"]["priceByArea"].mean()
    avg_rent_price = df_filtered[df_filtered["operation"]=="rent"]["price"].mean()
    avg_sale_area = df_filtered[df_filtered["operation"]=="sale"]["priceByArea"].mean()
    avg_sale_price = df_filtered[df_filtered["operation"]=="sale"]["price"].mean()
    print(avg_rent_area, avg_rent_price, avg_sale_area, avg_sale_price)
    

    style = {
        "position": "absolute",
        "top": "100px",
        "right": "20px",
        "backgroundColor": "rgba(255, 255, 255, 0.9)",
        "padding": "15px",
        "borderRadius": "10px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.25)",
        "zIndex": 1000,
        "display": "block",
        "minWidth": "250px"
    }

    return (
        style,
        f"{avg_rent_area:.2f} €/m²" if not pd.isna(avg_rent_area) else "No data",
        f"{avg_rent_price:,.0f} €" if not pd.isna(avg_rent_price) else "No data",
        f"{avg_sale_area:.2f} €/m²" if not pd.isna(avg_sale_area) else "No data",
        f"{avg_sale_price:,.0f} €" if not pd.isna(avg_sale_price) else "No data",
    )


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
    State("detailed_type_dropdown", "value"),
    State("lift_radio", "value"),
    State("floor_dropdown", "value"),
    State("parking_radio", "value")
)
def handle_prediction(n_clicks, property_type, operation, size, rooms, bathrooms,
                      municipality, district, neighborhood, lat, lon,
                      status, detailed_type, lift, floor, parking):
    print(f"[DEBUG handle_prediction] Button clicked {n_clicks} times")

    if n_clicks == 0:
        print("[DEBUG handle_prediction] First load, no prediction triggered")
        return True, dash.no_update

    inputs = [property_type, operation, size, rooms, bathrooms,
              municipality, district, neighborhood, lat, lon,
              status, detailed_type, lift, floor, parking]
    print(f"[DEBUG handle_prediction] Inputs: {inputs}")

    if any(x is None for x in inputs):
        print("[WARNING handle_prediction] Missing inputs")
        return False, "" 

    try:
        lat = float(lat) if isinstance(lat, (float, int)) else float(str(lat).strip())
        lon = float(lon) if isinstance(lon, (float, int)) else float(str(lon).strip())
        print(f"[DEBUG handle_prediction] Coordinates parsed: lat={lat}, lon={lon}")
    except Exception as e:
        print(f"[ERROR handle_prediction] Invalid coordinates: {e}")
        return False, "Invalid Coordinates."

    try:
        pred = get_price_prediction(
            property_type, operation, size, rooms, bathrooms,
            municipality, district, neighborhood, lat, lon,
            status, detailed_type, lift, floor, parking
        )

        # Construcción adaptada del mensaje
        if property_type.lower() == "garage":
            message = (
                f"The predicted price to {operation} a **{property_type}** "
                f"of {size} m² in {district}, {municipality} is: "
            )
        else:
            message = (
                f"The predicted price to {operation} a {property_type} "
                f"with {rooms} rooms and {bathrooms} bathrooms, "
                f"of {size} m² in {district}, {municipality} is: "
            )

        # Caja estilizada con el resultado
        formatted_result = html.Div([
            html.Div(
                f"{message}{pred:,.2f} €",
                style={
                    "backgroundColor": "#f2f2f2",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "border": "1px solid #ccc",
                    "fontSize": "20px",
                    "fontWeight": "bold",
                    "textAlign": "center"
                }
            )
        ])

        print(f"[DEBUG handle_prediction] Prediction successful")
        return True, formatted_result

    except Exception as e:
        print(f"[ERROR handle_prediction] Prediction failed: {e}")
        return False, f"Error calculating the prediction: {e}"


@app.callback(
    Output("price_by_area_graph", "figure"),
    Input("municipality_graph_dropdown", "value"),
    Input("property_type_graph_dropdown", "value"),
    Input("operation_selector", "value")
)
def update_price_by_area_graph(selected_municipalities, selected_property_type, operation_filter):
    df_filtered = df.copy()

    if operation_filter in ["sale", "rent"]:
        df_filtered = df_filtered[df_filtered["operation"] == operation_filter]

    if selected_property_type:
        df_filtered = df_filtered[df_filtered["propertyType"] == selected_property_type]

    if selected_municipalities:
        df_filtered = df_filtered[df_filtered["municipality"].isin(selected_municipalities)]

    df_grouped = (
        df_filtered.groupby(["municipality", "operation"])["priceByArea"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        df_grouped,
        x="municipality",
        y="priceByArea",
        color="operation",
        barmode="group",
        text="priceByArea",
        labels={"priceByArea": "Price per Area (€)", "municipality": "Municipality"},
        title="Average Price per Area by Municipality and Operation"
    )

    fig.update_traces(texttemplate="%{text:.2f}")
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig



@app.callback(
    Output("total_price_graph", "figure"),
    Input("property_type_graph_dropdown", "value"),
    Input("operation_selector", "value")
)
def update_total_price_graph(property_type, operation_filter):
    df_filtered = df.copy()

    if operation_filter == "sale":
        df_filtered = df_filtered[df_filtered["operation"] == "sale"]
    elif operation_filter == "rent":
        df_filtered = df_filtered[df_filtered["operation"] == "rent"]


    if property_type:
        df_filtered = df_filtered[df_filtered["propertyType"] == property_type]

    df_grouped = df_filtered.groupby("municipality")["price"].mean().reset_index()
    df_top10 = df_grouped.sort_values("price", ascending=False).head(10)

    fig = px.bar(
        df_top10,
        x="municipality",
        y="price",
        text=df_top10["price"].round(2),
        labels={"price": "Average Price (€)"},
        title=f"Top 10 Municipalities by Average Price{f' ({property_type})' if property_type else ''}"
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig




#============================================== RUN SERVER ==============================================
if __name__ == "__main__":
    env = os.getenv("APP_ENV", "prod")

    if env == "dev":
        # Local dev
        app.run(debug=True)  # Usa 127.0.0.1:8050 por defecto
    else:
        # Producción (Docker, despliegue)
        app.run(
            host="0.0.0.0",
            port=8050,
            debug=False
        )
