import dash_bootstrap_components as dbc
from dash import html, dcc

from components.ids import Ids

def create_login_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Login Dashboard Olio d'Oliva", className="text-center mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Input(
                            id=Ids.LOGIN_USERNAME,
                            type="text",
                            placeholder="Username",
                            className="mb-3"
                        ),
                        dbc.Input(
                            id=Ids.LOGIN_PASSWORD,
                            type="password",
                            placeholder="Password",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "Login",
                            id=Ids.LOGIN_BUTTON,
                            color="primary",
                            className="w-100 mb-3"
                        ),
                        html.Div(id=Ids.LOGIN_ERROR),
                        html.Hr(),
                        html.P("Non hai un account?", className="text-center"),
                        dbc.Button(
                            "Registrati",
                            id=Ids.SHOW_REGISTER_BUTTON,
                            color="secondary",
                            className="w-100"
                        )
                    ])
                ])
            ], md=6, className="mx-auto")
        ], className="vh-100 align-items-center")
    ])

def create_register_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Registrazione", className="text-center mb-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Input(
                            id=Ids.REGISTER_USERNAME,
                            type="text",
                            placeholder="Username",
                            className="mb-3"
                        ),
                        dbc.Input(
                            id=Ids.REGISTER_PASSWORD,
                            type="password",
                            placeholder="Password",
                            className="mb-3"
                        ),
                        dbc.Input(
                            id=Ids.REGISTER_CONFIRM,
                            type="password",
                            placeholder="Conferma Password",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "Registrati",
                            id=Ids.REGISTER_BUTTON,
                            color="primary",
                            className="w-100 mb-3"
                        ),
                        html.Div(id=Ids.REGISTER_ERROR),
                        html.Div(id=Ids.REGISTER_SUCCESS),
                        html.Hr(),
                        html.P("Hai già un account?", className="text-center"),
                        dbc.Button(
                            "Torna al Login",
                            id=Ids.SHOW_LOGIN_BUTTON,
                            color="secondary",
                            className="w-100"
                        )
                    ])
                ])
            ], md=6, className="mx-auto")
        ], className="vh-100 align-items-center")
    ])