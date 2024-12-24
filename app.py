import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

df=pd.read_csv("spacexvisual.csv")
dfm=pd.read_csv("spacexcomplete.csv")
dfa=df.copy()


# Rename Launchoutcome for better clarity
dfa['Launchoutcome'] = dfa['Launchoutcome'].map({1: 'Success', 0: 'Failure'})


# Load the trained model
model = joblib.load('models/spacex_landing_model.pkl')

# Dropdown options for landing predictor model
orbit_options = [
    {'label': 'LEO (Low Earth Orbit)', 'value': 1},
    {'label': 'GTO (Geostationary Transfer Orbit)', 'value': 2},
    {'label': 'SSO (Sun-Synchronous Orbit)', 'value': 3},
    {'label': 'Polar LEO', 'value': 4},
    {'label': 'MEO (Medium Earth Orbit)', 'value': 5},
    {'label': 'Others', 'value': 99}
]

launchsite_options = [
    {'label': 'Cape Canaveral', 'value': 'Cape Canaveral'},
    {'label': 'Kennedy', 'value': 'Kennedy'},
    {'label': 'Vandenberg', 'value': 'Vandenberg'}
]

gridfins_options = [
    {'label': 'Yes', 'value': 1},
    {'label': 'No', 'value': 0}
]

legs_options = [
    {'label': 'Yes', 'value': 1},
    {'label': 'No', 'value': 0}
]

block_options = [
    {'label': 'B4', 'value': 'B4'},
    {'label': 'B5', 'value': 'B5'},
    {'label': 'FT', 'value': 'FT'},
    {'label': 'v1.0', 'value': 'v1.0'},
    {'label': 'v1.1', 'value': 'v1.1'}
]

booster_options = [
    {'label': 'B4', 'value': 'B4'},
    {'label': 'B5', 'value': 'B5'},
    {'label': 'FT', 'value': 'FT'},
    {'label': 'v1.0', 'value': 'v1.0'},
    {'label': 'v1.1', 'value': 'v1.1'}
]

# Group by Year and Month, and count launches (Flight_No)
launches_by_month = df.groupby(['Year', 'Month']).agg(launch_count=('Flight_No', 'count')).reset_index()

# Define unique options for dropdowns
filters = {
    "Booster Version": df["Booster_Version"].unique(),
    "Year": sorted(df["Year"].unique()),
    "Orbit": df["Orbit"].unique(),
    "Customer": df["Customer"].unique(),
    "Launchsite": df["Launchsite"].unique(),
}


# Initializing the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                meta_tags=[{"name": "viewport",                         # To make the app responsive
                            "content": "width=device-width, initial-scale=1.0"}]
                )

app.layout = html.Div(
    children=[
        # Banner Section
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="left-wrapper",
                    children=[
                        html.Img(src="assets/SpaceX-white.svg"),  # Logo
                        html.H1("FALCON9 DASHBOARD"),   # Title of the page
                        html.Button(
                            children=[
                            "Open Landing Predictor",
                            ],    # Modal toggle button
                            id="open-ml-modal-btn",
                            className="btn-primary"
                        ),
                        html.Button(
                            children=[
                                "About"
                            ],
                            id="about-button",
                            className="btn-primary",
                        )
                    ],
                ),
                html.Div(
                    className="icon-wrapper",
                    children=[
                        html.A(
                            href="https://github.com/furkanonal06",
                            target="_blank",
                            children=[
                                html.I(className="fa-brands fa-github github-icon"),
                            ],
                        ),
                        html.A(
                            href="https://www.linkedin.com/in/furkan-onal/",
                            target="_blank",
                            children=[
                                html.I(className="fa-brands fa-linkedin linkedin-icon"),
                            ],
                        )
                    ]
                ),
            ],
        ),

        # Full screen modal for about
        html.Div(
            n_clicks=0,
            id="about-modal",
            className="modal-overlay",
            children=[
                # Modal wrapper
                html.Div(
                    className="modal-wrapper",
                    children=[
                        # Modal Content
                        html.Div(
                            className="modal-content",
                            children=[
                                # Modal header
                                html.Div(
                                    className="modal-header",
                                    children=[
                                        html.H2(
                                            "About the Project",
                                            className="modal-title",
                                        ),
                                        html.Button(
                                            "x",
                                            id="close-modal-btn-2",
                                            className="close-modal-btn circle",
                                        ),
                                    ],
                                ),
                                html.Br(),
                                # Text
                                html.Div(
                                    className="text",
                                    children=[
                                        html.Div(children=[
                                        html.P("This project is an exploration of SpaceX Falcon 9 launch and landing data, designed to showcase my capabilities in data science and analysis. By leveraging real-world data, I aimed to demonstrate my ability to preprocess, analyze, and model complex datasets to derive meaningful insights and build predictive tools."),
                                        html.P("An important piece of this project is a predictive model that determines the likelihood of a Falcon 9 booster successfully landing, based on customizable user inputs. I integrated this model into an interactive dashboard using Dash Plotly, allowing users to explore past launch and landing trends, also simulate landing parameters and visualize the results in real-time."),
                                        html.P("This work not only highlights my technical proficiency in data engineering, machine learning, and visualization but also reflects my learning journey in applying these skills to solve practical problems. It serves as a testament to my dedication to learning data science concepts and my commitment to professional growth in this dynamic field."),
                                        html.Br(),
                                        html.P("Furkan Önal", style={"font-weight":"bold"}),
                                        html.P("furkanonl@hotmail.com")
                                        ], style={"padding":"20px"})

                                    ],
                                ),
                            ]      
                        )
                ],
            )
        ]),

        # Full screen modal for the machine learning model
        html.Div(
            n_clicks=0,
            id="ml-model-modal",
            className="modal-overlay",
            children=[
                # Modal wrapper
                html.Div(
                    className="modal-wrapper",
                    children=[
                        # Modal Content
                        html.Div(
                            className="modal-content",
                            children=[
                                # Modal header
                                html.Div(
                                    className="modal-header",
                                    children=[
                                    html.H2(
                                        "Interactive Landing Outcome Predictor",
                                        className="modal-title",
                                    ),
                                    html.Button(
                                        "x",
                                        id="close-ml-modal-btn",
                                        className="close-modal-btn circle",
                                    ),
                                    ],
                                ),

                        # 1st Modal content
                        html.Div(
                            className="input-wrapper",
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("Orbit", className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id="orbit-dropdown",
                                            options=orbit_options,
                                            placeholder="Select an Orbit",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Launch Site', className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id='launchsite-dropdown',
                                            options=launchsite_options,
                                            placeholder="Select a Launch Site",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Grid Fins', className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id='gridfins-dropdown',
                                            options=gridfins_options,
                                            placeholder="Does Your Rocket Has Grid Fins?",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Legs', className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id='legs-dropdown',
                                            options=legs_options,
                                            placeholder="Does Your Rocket Has Legs?",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Block', className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id='block-dropdown',
                                            options=block_options,
                                            placeholder="Select a Rocket Block",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Booster', className="input-label"),
                                        dcc.Dropdown(
                                            className="filter-dropdown",
                                            id='booster-dropdown',
                                            options=booster_options,
                                            placeholder="Select a Rocket Booster Version",
                                        )
                                    ]
                                ),

                                html.Div(
                                    children=[
                                        html.Label('Payload Mass (kg)', className="input-label"),
                                        dcc.Input(id='payload-mass2', 
                                                type='number', 
                                                placeholder='Enter Payload Mass')
                                    ],
                                ),

                                html.Button('Submit', id='predict-btn', className="predict-button"),
                                html.H2(id='prediction-output', className="predict-output")
                                    ]
                                ),
                            ]   
                    )
                ],
            )
        ]),
        
        # Filters section
        html.Div(
                className="right-side-filters-container",
                children=[
                    # Toggle button
                    html.Button(
                        "Hide Data Filters",  # Initial button text
                        id="toggle-button",  # Button ID for the callback
                        className="toggle-button",
                    ),
                    # Filters section
                    html.Div(
                        id="filters-section",  # Filters section ID for toggling
                        className="right-side-filters-section",
                        children=[
                            # Filters section title
                            html.Div(
                                "Filters",  # Title text
                                className="filters-title",  # Title class for styling
                            ),
                            # Generate dropdowns dynamically based on filters
                            *[
                                html.Div(
                                    children=[
                                        html.Label(filter_name, className="filter-label"),
                                        dcc.Dropdown(
                                            id=filter_name.replace(" ", "_").lower(),
                                            options=[{"label": str(option), "value": option} for option in options],
                                            placeholder=f"All {filter_name}",
                                            multi=False,
                                            className="filter-dropdown"
                                        ),
                                    ],
                                    className="filter-container"
                                )
                                for filter_name, options in filters.items()
                            ],
                        ],
                    ),
                ],
            ),

        # KPI Section
        html.Div(
            className="kpi-section",
            children=[
                
                # Title for the KPI section
                html.H1("KEY MEASURES", className="kpi-title"),
                html.H2("Select a data filter to observe the changes in key measures", className="kpi-subtitle"),
                # Cards container
                html.Div(
                    className="cards-container",
                    children=[
                        
                        # Total launches card
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="card-header",
                                    children=[
                                        html.I(className="fas fa-rocket card-icon"), # Card icon
                                        html.Span("Total Launches", className="card-header"), # Card title
                                        
                                    ]
                                ),                       
                                html.Div(id="total-launches", className="card-value"),                                 
                               
                            ]
                        ),
                       
                        # Launch success rate card
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="card-header",
                                    children=[
                                        html.I(className="fas fa-bullseye card-icon"), # Card icon
                                        html.Span("Launch Success Rate", className="card-header"), # Card title
                                        
                                    ]
                                ),                       
                                                          
                                html.Div(id="success-rate", className="card-value"),
                            ]
                        ),
                        
                        # Total payload card
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="card-header",
                                    children=[
                                        html.I(className="fas fa-weight-hanging card-icon"), # Card icon
                                        html.Span("Total Payload Mass", className="card-header"), # Card title
                                        
                                    ]
                                ),                       
                                                          
                               html.Div(id="payload-mass", className="card-value"),
                            ]
                        ),
                        
                        # Reused boosters card
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="card-header",
                                    children=[
                                        html.I(className="fas fa-shuttle-space card-icon"), # Card icon
                                        html.Span("No. of Reused Boosters", className="card-header"), # Card title
                                        
                                    ]
                                ),                       
                                                          
                                html.Div(id="reused-boosters", className="card-value"),
                            ]
                        ),
                        
                        # Landing success rate card
                        html.Div(
                            className="card",
                            children=[
                                html.Div(
                                    className="card-header",
                                    children=[
                                        html.I(className="fas fa-helicopter-symbol  card-icon"), # Card icon
                                        html.Span("Landing Success Rate", className="card-header"), # Card title
                                        
                                    ]
                                ),                       
                                                          
                                html.Div(id="landing-success", className="card-value"),
                            ]
                        ),

                    ],
                ),
            ],
            
        ),

        # Main Content: Graphs and Map
        html.Div(
            className="main-content",
            children=[

                # Left column
                html.Div(
                    className="left-column",
                    children=[

                        # First graph
                        html.Div(
                            className="graph-container",
                            children=[
                                dcc.Graph(
                                    id="line-chart",
                                    config={"displayModeBar": False,},  # Disable toolbar
                                ),
                            ],
                        ),
                       
                        # Second graph
                        html.Div(
                            className="graph-container",
                            children=[
                                dcc.Graph(
                                    id="pie-chart",
                                    config={"displayModeBar": False},  # Disable toolbar
                                ),
                            ],
                        ),
                        # Third graph
                        html.Div(
                            className="graph-container",
                            children=[
                                dcc.Graph(
                                    id="launchseries",
                                    config={"displayModeBar": False},  # Disable toolbar
                                ),
                            ],
                        ),
                    ],
                ),

                # Right column
                html.Div(
                    className="right-column",
                    children=[

                        # First visual
                        html.Div(
                            className="graph-container",
                            children=[
                                html.Div(
                                    className="map-container",
                                    children=[
                                        # Toggle buttons
                                        html.Div(children=["Launch Sites"], className="map-title"),
                                        html.Div(children=["Outcome of the Launches Per Launch Site"], className="map-subtitle"),
                                        html.Div(
                                            className="map-button-container",
                                            children=[
                                                html.Button(
                                                    "All Launches",
                                                    id="all-btn",
                                                    className="tab-button active",
                                                ),
                                                html.Button(
                                                    "Successful Launches",
                                                    id="success-btn",
                                                    className="tab-button",
                                                ),
                                                html.Button(
                                                    "Failed Launches",
                                                    id="failure-btn",
                                                    className="tab-button",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="launch-site-cluster-map",
                                                config={"displayModeBar": False,},
                                            ),
                                        ),
                                    ]
                                ),

                            ],
                        ),

                        # Second visual
                        html.Div(
                            className="graph-container", # container for tabs and graphs
                            children=[
                                # graphs
                                html.Div(
                                    children=[
                                        html.Div(
                                            children=[
                                                # histogram graph
                                                html.Div(
                                                    dcc.Graph(id="histogram-graph"),
                                                    id="histogram-graph-container",
                                                    className="graph-container-animation active",
                                                ),
                                                # timeline graph
                                                html.Div(
                                                    dcc.Graph (id="timeline-graph"),
                                                    id="timeline-graph-container",
                                                    className="graph-container-animation",
                                                    style={"display":"none"},
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                # Buttons for switch
                                html.Div(
                                    className="button-container",
                                    children=[
                                        html.Button(
                                            "Launch Distribution", 
                                            id="btn-histogram", 
                                            n_clicks=1, 
                                            className="tab-button active"
                                        ),
                                        html.Button(
                                            "Launch Timeline",
                                            id="btn-timeline",
                                            n_clicks=0,
                                            className="tab-button",
                                        ),
                                         ]),

                            ],
                        ),

                        html.Div(
                            className="heading-frame",
                            children=[
                                html.Div(
                                    className="overlay-container",
                                    children=[
                                        html.Img(src="assets/FO_Portrett_58-03-min2.JPG", className="overlay-image"),
                                        html.Div(
                                            className="overlay-text",
                                            children=[
                                                html.P("Furkan Önal",className="overlay-text-style"),
                                                html.P("Hi! I am analytical and detail-oriented professional with extensive international experience, currently enhancing skills in data science and analytics. I have hands-on experience in system dynamics modeling, data visualization, machine learning, data cleansing, ETL processes, and statistical analysis. In addition, I have proven ability to transform datasets into actionable insights for stakeholders in busines and academia. I am seeking to leverage data science and business intelligence expertise to solve real-world challenges and contribute to data-driven strategies.",
                                                       className="overlay-text-style2"),
                                            ]
                                        )
                                    ]
                                ),
                                html.Div(
                                    className="borders",
                                    children=[
                                        html.Div(
                                            contentEditable="true",
                                            children=[
                                                html.H2("About"),
                                                html.H2("me")
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ],

                ),
            
            ],
        ),

    ],
)

# Callback for updating KPI cards
@app.callback(
    [
        Output("total-launches", "children"),
        Output("success-rate", "children"),
        Output("payload-mass", "children"),
        Output("reused-boosters", "children"),
        Output("landing-success", "children"),
    ],
    [
        Input("booster_version", "value"),
        Input("year", "value"),
        Input("orbit", "value"),
        Input("customer", "value"),
        Input("launchsite", "value"),
    ],
)
# Function for collapsable filters
def update_cards(booster_version, year, orbit, customer, launchsite):
    # Filter data based on dropdown values
    filtered_data = df.copy()
    if booster_version:
        filtered_data = filtered_data[filtered_data["Booster_Version"] == booster_version]
    if year:
        filtered_data = filtered_data[filtered_data["Year"] == year]
    if orbit:
        filtered_data = filtered_data[filtered_data["Orbit"] == orbit]
    if customer:
        filtered_data = filtered_data[filtered_data["Customer"] == customer]
    if launchsite:
        filtered_data = filtered_data[filtered_data["Launchsite"] == launchsite]

    # Compute KPI values
    total_launches = len(filtered_data)
    success_rate = f"{(filtered_data['Launchoutcome'].mean() * 100):.1f}%" if not filtered_data.empty else "N/A"
    payload_mass = f"{filtered_data['Payload_Mass'].sum():,.0f} kg" if not filtered_data.empty else "N/A"
    reused_boosters = filtered_data["Reused"].sum() if not filtered_data.empty else "N/A"
    landing_success = f"{(filtered_data['Landing'].mean() * 100):.1f}%" if not filtered_data.empty else "N/A"

    return total_launches, success_rate, payload_mass, reused_boosters, landing_success


# Callback for line chart
@app.callback(
    Output("line-chart", "figure"),
    [Input("year", "value")],
)
# Function for line chart
def update_line_chart(year):
    # Group by year and calculate mean success rates for Landing and Launch
    grouped_dfa = df.groupby('Year').agg(
        Landing_Success_Rate=('Landing', 'mean'),
        Launch_Success_Rate=('Launchoutcome', 'mean')
    ).reset_index()
    
    # Convert the success rates to percentages (multiply by 100)
    grouped_dfa["Landing_Success_Rate"] = grouped_dfa["Landing_Success_Rate"] * 100
    grouped_dfa["Launch_Success_Rate"] = grouped_dfa["Launch_Success_Rate"] * 100
    
    # Create the multi-line chart
    fig = px.line(
        grouped_dfa,
        x="Year",
        y=["Landing_Success_Rate", "Launch_Success_Rate"],
        labels={
            "value": "%",
            "Year": "Year",
            "variable": " "
        },
        color_discrete_sequence=["#D97430", "#009F93"],  # Custom colors for the lines
        height=390, 
    )
    
    # Customize lines and hoverover
    fig.update_traces(  
        mode="lines+markers",
        line=dict(
            width=2.5,
        ),
        
        # Update hover template and layout
        hovertemplate=(
            "<b>Success Rate:</b> %{y:.1f}% <br>"
            "<b>Year:</b> %{x}<extra></extra>"
        ),
    )
    
    fig.update_layout(
        title=dict(
            text="Tracking Launch and Landing Success (%)",
            font=dict(
                color="#333",
                weight=800,
                family="Poppins",
                size=18,
            ),
            subtitle=dict(
                text="From Failures to Triumph: The SpaceX Success Story",
                font=dict(
                    color="#333",
                    size=13,
                    family="Poppins",
                )
            )
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins",  
            bordercolor="white"  
        ),
        xaxis_title=None,
        yaxis_title=None,
        dragmode=False,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            ticks="outside",
            tickmode='linear',
            tickfont=dict(
                family='Poppins',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            linewidth=4,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            range=[-5, 105],
            tickformat="%{y:.0f}%",
            showticksuffix="none",
        ),
        template="plotly_white",
        margin=dict(t=80, b=40, l=50, r=50),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            yanchor="bottom",
            y=0.05,
            x=0.65
        )
    )
    
    return fig


# Callback for pie chart
@app.callback(
    Output("pie-chart", "figure"),
    [Input("orbit", "value")],
)
# Function for pie chart
def update_pie_chart(orbit):
    data = df.copy()
    if orbit:
        data = data[data["Orbit"] == orbit]

    orbit_counts = data["Orbit"].value_counts(normalize=True)     # Calculate the value counts
    
    threshold = 0.02     # Set a threshold for grouping small categories
    
    small_categories = orbit_counts[orbit_counts < threshold].index    # Identify "small" categories
    
    # Replace small categories with "Others"
    data["Orbit"] = data["Orbit"].apply(lambda x: "Others" if x in small_categories else x)
    
    # Recalculate the grouped data
    grouped_data = data["Orbit"].value_counts().reset_index()
    grouped_data.columns = ["Orbit", "count"]  # Rename columns for clarity

    # Create the pie chart
    fig = px.pie(
        grouped_data,
        names="Orbit",  
        values="count",  
        title="Orbit Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu,
        hole=.5,
    )
        # Image to the center of the pie chart
    fig.update_layout(
            title=dict(
            text="Where Do Falcon9 Rockets Go?",
            font=dict(
                color="#333",
                weight=800,
                family="Poppins",
                size=18,
            ),
            subtitle=dict(
                text="A Closer Look at SpaceX's Orbital Targets",
                font=dict(
                    color="#333",
                    size=13,
                    family="Poppins",
                )
            )
        ),
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins", 
            bordercolor="white"  
        ),
        margin=dict(t=80, b=80, l=50, r=50),
        images=[
            dict(
                source=r"assets\orbit.png",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                sizex=0.3,
                sizey=0.3,
                xref="paper",
                yref="paper",
                layer="above"
            )
        ]
    )

        # Update the traces for hover formatting and line mode
    fig.update_traces(
        hovertemplate=(
            "<b>Orbit:</b> %{label} <br>"  # Display the value in percentage
            "<b>Number of Launch:</b> %{value}"  # Add bold formatting for the category
            "<extra></extra>"  # Remove default trace information
        ),
    )
    
    return fig


# Timeseries chart
@app.callback(
    Output('launchseries', 'figure'),
    [Input('launchseries', 'id')]
)
# Function for timeseries chart
def update_graph(value):
    # Group the dataset by Booster Version and Year
    grouped_df = df.groupby(['Year', 'Booster_Version']).size().reset_index(name="Launch_Count")

    # Sort the grouped dataframe by Year to ensure cumulative calculation works
    grouped_df = grouped_df.sort_values(['Booster_Version', 'Year'])

    # Calculate cumulative launches for each booster version
    grouped_df['Cumulative_Launches'] = grouped_df.groupby('Booster_Version')['Launch_Count'].cumsum()

    # custom colors for booster versions
    custom_colors = {
        "F9 v1.0": "#38b000",
        "F9 v1.1": "#ee6c4d",
        "F9 FT": "#e09f3e",
        "F9 B4": "#9e2a2b",
        "F9 B5": "#009F93",
    }


    # Create the Plotly Express line chart
    fig = px.line(
        grouped_df,
        x='Year',
        y='Cumulative_Launches',
        color='Booster_Version',
        markers=True,
        color_discrete_map=custom_colors,  
        hover_data={
        'Booster_Version': True,  # Show Booster Version
        'Year': True,
        },
    )

    # Customize lines and hover over
    fig.update_traces(
        mode="lines+markers",
        line=dict(
            width=2.5,
        ),
        
        # Update hover template and layout
        hovertemplate=(
            "<b>Version:</b> %{customdata[0]} <br>"
        ),
    )

    # Customize the layout
    fig.update_layout(
        title=dict(
            text="Tracing the Falcon 9 Booster Lifecycle",
            font=dict(
                color="#333",
                weight=800,
                family="Poppins",
                size=18,
            ),
            subtitle=dict(
                text="The Continuous Development of Reusable Rocket Technology",
                font=dict(
                    color="#333",
                    size=13,
                    family="Poppins",
                )
            )
        ),
        
        yaxis_title='Cumulative Number of Launches',
        xaxis_title=None,
        dragmode=False,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            ticks="outside",
            tickmode='linear',
            tickfont=dict(
                family='Poppins',
                size=12,
                color='rgb(82, 82, 82)',
            ),
            linewidth=4,
        ),
        yaxis=dict(
            range=[-20, grouped_df["Cumulative_Launches"].max() + 50], # Cover all data
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            showticksuffix="none",
        ),
        height=390,
        margin=dict(t=100,b=50,),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins", 
            bordercolor="white" 
        ),
        template="plotly_white",
        showlegend=False,

    )

    return fig


# Callbacks for the 4th and 5th graphs
@app.callback(
    [
        Output("btn-histogram","className"),
        Output("btn-timeline","className"),
        Output("histogram-graph-container","className"),
        Output("timeline-graph-container","className"),
        Output("histogram-graph-container","style"),
        Output("timeline-graph-container","style"),
    ],

    [
        Input("btn-histogram","n_clicks"),
        Input("btn-timeline","n_clicks"),
    ]
)
# Function for histogram
def toggle_tabs(histogram_clicks,timeline_clicks):
    # Determine which tab was clicked last
    ctx=dash.callback_context
    triggered_id=ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "btn-histogram":
        return (
            "tab-button active", "tab-button",
            "graph-container-animation active","graph-container-animation",
            {"display":"block"},{"display":"none"},
        )
    elif triggered_id == "btn-timeline":
        return (
            "tab-button", "tab-button active",
            "graph-container-animation","graph-container-animation active",
            {"display":"none"},{"display":"block"},
        )
    
    # Default state
    return (
        "tab-button active", "tab-button",
        "graph-container-animation active", "graph-container-animation",
        {"display":"block"}, {"display":"none"},
    )


# Callback for timeline graph
@app.callback(
        Output("timeline-graph","figure"),
        [Input("btn-timeline","n_clicks")]
)
# Function for timeline graph
def update_timeline_graph(n_clicks):

    # Create a line chart
    fig_timeline = go.Figure()
    # Aggregate launch count per month in a specific year
    monthly_launches = df.groupby(['Year', 'Month']).size().reset_index(name='launch_count')

    # Loop through unique years to add a line for each year with a specific color
    for year in monthly_launches['Year'].unique():
        year_data = monthly_launches[monthly_launches['Year'] == year]
        fig_timeline.add_trace(go.Scatter(
            x=year_data['Month'],
            y=year_data['launch_count'],
            mode='lines+markers',
            name=f"Year {year}",
            line=dict(color=px.colors.qualitative.Set1[year % len(px.colors.qualitative.Set1)]),  # Color based on the year
        ))

    fig_timeline.update_layout(
        title=dict(
        text="Is There Seasonality in Launches?",
        font=dict(
            color="#333",
            weight=800,
            family="Poppins",
            size=18,
        ),
        subtitle=dict(
            text="Monthly Launch Frequency",
            font=dict(
                color="#333",
                size=13,
                family="Poppins",
                )
            )
        ),
        showlegend=False,
        height=340,
        width=720,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins",  
            bordercolor="white"  
        ),
        xaxis_title=None,
        dragmode=False,
        yaxis_title='Launch Count',
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            showticksuffix="none",
        ),
        xaxis=dict(
            tickfont=dict(
            family="Poppins",
            size=12,
            color='rgb(82, 82, 82)',
            ),
            tickmode='array',
            tickvals=list(range(1, 13)), 
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                   ),
        margin=dict(t=75,b=40,),
        template="plotly_white",

    )
    return fig_timeline


# Callback for histogram
@app.callback(
        Output("histogram-graph","figure"),
        [Input("btn-histogram","n_clicks")],
)
# Function for histogram
def update_histogram_graph(n_clicks):

    # Count launches for each month
    monthly_launches = df.groupby('Month').size().reset_index(name='Launch_Count')

    # Histogram (Launches per Month)
    fig_histogram = px.bar(
        monthly_launches,
        x='Month',
        y='Launch_Count',
        labels={'Month': 'Month', 'Launch_Count': 'Number of Launches'},
        text='Launch_Count',
        width=720,
        height=340,
    )
    fig_histogram.update_traces(
        marker_color="#009F93",
    )

    fig_histogram.update_layout(
        title=dict(
        text="Is There Seasonality in Launches?",
        font=dict(
            color="#333",
            weight=800,
            family="Poppins",
            size=18,
        ),
        subtitle=dict(
            text="Number of Launches by Month",
            font=dict(
                color="#333",
                size=13,
                family="Poppins",
                )
            )
        ),
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins",  
            bordercolor="white"  
        ),
        xaxis_title=None,
        yaxis_title=None,
        dragmode=False,
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False,
            showticksuffix="none",
        ),
        xaxis=dict(
            tickfont=dict(
            family="Poppins",
            size=12,
            color='rgb(82, 82, 82)',
            ),
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        bargap=0.2,
        margin=dict(t=75,b=40,),
        template="plotly_white",

    )
    return fig_histogram


# Callback for the map
@app.callback(
    [Output('launch-site-cluster-map', 'figure'),
     Output('all-btn', 'className'),
     Output('success-btn', 'className'),
     Output('failure-btn', 'className')],
    [Input('all-btn', 'n_clicks'),
     Input('success-btn', 'n_clicks'),
     Input('failure-btn', 'n_clicks')],
    [State('all-btn', 'n_clicks'),
     State('success-btn', 'n_clicks'),
     State('failure-btn', 'n_clicks')]
)
# Function for the map
def update_cluster_map(all_clicks, success_clicks, failure_clicks, all_state, success_state, failure_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'all-btn'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Filter based on button clicks
    if button_id == "success-btn":
        filtered_df = dfa[dfa['Launchoutcome'] == 'Success']
        marker_color="green"
    elif button_id == "failure-btn":
        filtered_df = dfa[dfa['Launchoutcome'] == 'Failure']
        marker_color="red"
    else:
        filtered_df = dfa.copy()
        marker_color="blue"

    # Group data per Launchsite
    grouped = filtered_df.groupby(['Launchsite', 'Longitude', 'Latitude']).size().reset_index(name='Count')

    # Plotly Mapbox figure without native clustering
    fig = px.scatter_mapbox(
        grouped,
        lat="Latitude",
        lon="Longitude",
        size="Count",
        color="Count",
        hover_name="Launchsite",
        hover_data={"Latitude": False, "Longitude": False, "Count": True},
        zoom=3,
        title="Launch Metrics Per Site",
        height=None,
    )
    
    fig.update_traces(
        hovertemplate=(
            "<b>Count:</b> %{marker.size} <br>"  # Display the value in percentage
            "<extra></extra>"  # Remove default trace information  
        ),
        marker=dict(
            color=marker_color
        )
    )
    # Update layout for Mapbox
    fig.update_layout(
        mapbox=dict(
            accesstoken='pk.eyJ1IjoiZnVya2Fub25hbCIsImEiOiJjbTR0cXZ3YmEwYXpnMmlzNGk3ZWUyczk5In0.SqaYxNgjOIf3F5sA9uZoSw',
            center=dict(lat=30, lon=-100.6),  
            zoom=3,
            style="light"
        ),
        hoverlabel=dict(
            bgcolor="white",  
            font_size=12,     
            font_color="black",
            font_family="Poppins", 
            bordercolor="white"
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_showscale=False,
        height=None,
    )
    # Update button styles
    all_class = "tab-button active" if button_id == 'all-btn' else "tab-button"
    success_class = "tab-button active" if button_id == 'success-btn' else "tab-button"
    failure_class = "tab-button active" if button_id == 'failure-btn' else "tab-button"

    return fig, all_class, success_class, failure_class


# Callback to toggle the visibility of the filters section
@app.callback(
    [Output("filters-section", "className"), Output("toggle-button", "children")],
    [Input("toggle-button", "n_clicks")],
    prevent_initial_call=True,
)
# Function for toggle section
def toggle_filters(n_clicks):
    # Check if button is clicked an odd or even number of times
    if n_clicks % 2 == 1:
        return "right-side-filters-section collapsed", "Show Data Filters"
    else:
        return "right-side-filters-section", "Hide Data Filters"


# Callback for modals
@app.callback(
    [Output('ml-model-modal', 'style'),
     Output('about-modal', 'style')],

    [Input('open-ml-modal-btn', 'n_clicks'),
     Input('close-ml-modal-btn', 'n_clicks'),
     Input('about-button', 'n_clicks'),
     Input('close-modal-btn-2', 'n_clicks')]
)
# Function for modals
def toggle_modals(open_clicks_1, close_clicks_1,
                  open_clicks_2, close_clicks_2):
    ctx = dash.callback_context

    # Default styles: both modals hidden
    modal_1_style = {'display': 'none'}
    modal_2_style = {'display': 'none'}

    if not ctx.triggered:
        return modal_1_style, modal_2_style

    # Identify the triggered element
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Logic for the first modal
    if triggered_id == 'open-ml-modal-btn':
        modal_1_style = {
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': '1000',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'overflowY': 'auto'
        }
    elif triggered_id == 'close-ml-modal-btn':
        modal_1_style = {'display': 'none'}

    # Logic for the second modal
    if triggered_id == 'about-button':
        modal_2_style = {
            'position': 'fixed',
            'top': '0',
            'left': '0',
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(0,0,0,0.5)',
            'zIndex': '1000',
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'overflowY': 'auto'
        }
    elif triggered_id == 'close-modal-btn-2':
        modal_2_style = {'display': 'none'}

    # Return styles for both modals
    return modal_1_style, modal_2_style


# Callback for landing predictor
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [
        Input('orbit-dropdown', 'value'),
        Input('launchsite-dropdown', 'value'),
        Input('gridfins-dropdown', 'value'),
        Input('legs-dropdown', 'value'),
        Input('block-dropdown', 'value'),
        Input('booster-dropdown', 'value'),
        Input('payload-mass2', 'value'),
    ],
    prevent_initial_call=True
)
# Function for landing predictor
def predict_landing(n_clicks, orbit, launchsite, gridfins, legs, block, booster, payload_mass):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]  # Identify which input triggered the callback

    if triggered_id in ["orbit-dropdown", "launchsite-dropdown", "gridfins-dropdown", "legs-dropdown", "block-dropdown", "booster-dropdown", "payload_mass2"]:
        return "" # Clear output when inputs change

    if triggered_id == 'predict-btn':
        if payload_mass is None or orbit is None or launchsite is None or gridfins is None or legs is None or block is None or booster is None or payload_mass is None:
            return "🚨 Please Fill All the Sections"
        
    if n_clicks is None:
        return "No prediction yet" 

    input_data = pd.DataFrame([{
        'Orbit': orbit,
        'Payload_Mass': payload_mass,
        'GridFins': gridfins,
        'Legs': legs,
        'Launchsite_Cape Canaveral, SLC-40': 1 if launchsite == "Cape Canaveral" else 0,
        'Launchsite_Kennedy, LC-39A': 1 if launchsite == "Kennedy" else 0,
        'Launchsite_Vandenberg, SLC-4E': 1 if launchsite == "Vandenberg" else 0,
        'Block_B4': 1 if block == 'B4' else 0,
        'Block_B5': 1 if block == 'B5' else 0,
        'Block_FT': 1 if block == 'FT' else 0,
        'Block_v1.0': 1 if block == 'v1.0' else 0,
        'Block_v1.1': 1 if block == 'v1.1' else 0,
        'Booster_Version_F9 B4': 1 if block == 'B4' else 0,
        'Booster_Version_F9 B5': 1 if block == 'B5' else 0,
        'Booster_Version_F9 FT': 1 if booster == 'FT' else 0,
        'Booster_Version_F9 v1.0': 1 if booster == 'v1.0' else 0,
        'Booster_Version_F9 v1.1': 1 if booster == 'v1.1' else 0,
    }])

    pred = model.predict(input_data)[0]

    return f"{'✅ Successful' if pred else '🚫 Failed'}"


# defining font-awesome and fonts -------------------------------------------------
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <link rel="icon" href="/assets/logos/favicon.svg" type="image/x-icon">
    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>

<style>
@import url("https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap");
</style>
"""


if __name__ == '__main__':
    app.run_server(debug=True)
