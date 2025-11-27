import copy
import pandas as pd
import numpy as np
import re
import networkx as nx
import json
import ssl
import traceback
import base64
import io
from io import StringIO

from direct.showbase.PythonUtil import safeReprNotify
# Librerie Scientifiche e di Visualizzazione
from nilearn import datasets, image, surface
import community as community_louvain
import plotly.graph_objects as go
import plotly.express as px

# Librerie per elaborazione Mesh (Necessarie per generare le zone separate)
try:
    from skimage import measure
    from scipy import ndimage

    has_volumetric_libs = True
except ImportError:
    has_volumetric_libs = False
    print("!!! ATTENZIONE: Manca 'scikit-image' o 'scipy'. Le zone anatomiche non potranno essere generate.")

# Librerie Dash e Componenti UI
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State, ALL
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from openai import OpenAI

# Configurazione per usare GROQ (Gratis e Veloce)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1", # Indirizziamo le richieste a Groq
    api_key="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"
)

# =============================================================================
# 1. CARICAMENTO MESH CEREBRALE E GENERAZIONE ZONE (AAL)
# =============================================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

# Variabili Globali per Mesh
REGION_MESHES = {}  # Conterrà le mesh divise per zona (Codice 2)
WHOLE_BRAIN_TRACES = []  # Fallback mesh intera (Codice 1)

# Configurazione Zone (Dal Codice 2)
ZONE_CONFIG = {
    "Frontal": {"keys": ["Frontal", "Precentral", "Rolandic", "Supp_Motor_Area", "Olfactory", "Rectus", "Paracentral"],
                "color": "#E57373"},
    "Parietal": {"keys": ["Parietal", "Postcentral", "SupraMarginal", "Angular", "Precuneus"], "color": "#64B5F6"},
    "Occipital": {"keys": ["Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform"], "color": "#81C784"},
    "Temporal": {"keys": ["Temporal", "Heschl"], "color": "#FFB74D"},
    "Limbic": {"keys": ["Cingulum", "Hippocampus", "ParaHippocampal", "Amygdala"], "color": "#BA68C8"},
    "Insula": {"keys": ["Insula"], "color": "#4DD0E1"},
    "Subcortical": {"keys": ["Caudate", "Putamen", "Pallidum", "Thalamus"], "color": "#90A4AE"},
    "Cerebellum": {"keys": ["Cerebellum", "Vermis"], "color": "#FF8A65"},
}
MACRO_ZONES_MAP = {k: v["keys"] for k, v in ZONE_CONFIG.items()}

print(">>> Inizio generazione Mesh Anatomiche...")

# TENTATIVO 1: Generazione Mesh Separate (Metodo Codice 2)
generated_regions = False
if has_volumetric_libs:
    try:
        print(">>> Scaricamento Atlas AAL...")
        dataset_aal = datasets.fetch_atlas_aal()
        img_aal = image.load_img(dataset_aal.maps)
        data_aal = img_aal.get_fdata()
        affine_aal = img_aal.affine

        print(f">>> Generazione isosuperfici per zone...")
        for zone_name, config in ZONE_CONFIG.items():
            keywords = config["keys"]
            target_indices = []
            for idx, label in zip(dataset_aal.indices, dataset_aal.labels):
                if any(k in label for k in keywords):
                    target_indices.append(int(idx))

            if target_indices:
                mask_zone = np.isin(data_aal, target_indices)
                mask_zone = ndimage.binary_closing(mask_zone, iterations=1)

                if np.sum(mask_zone) > 0:
                    verts, faces, _, _ = measure.marching_cubes(mask_zone, level=0.5, step_size=2)
                    verts_mni = np.c_[verts, np.ones(verts.shape[0])].dot(affine_aal.T)

                    REGION_MESHES[zone_name] = go.Mesh3d(
                        x=verts_mni[:, 0], y=verts_mni[:, 1], z=verts_mni[:, 2],
                        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                        color=config["color"],
                        opacity=0.10,
                        name=zone_name,
                        hoverinfo='name'  # Mostra nome zona al passaggio mouse
                    )
        generated_regions = True
        print(">>> Generazione Mesh Zone completata.")
    except Exception as e:
        print(f"!!! Errore generazione AAL (fallback a fsaverage): {e}")

# TENTATIVO 2: Caricamento SEMPRE della mesh fsaverage per la vista "Grigia"
print(">>> Caricamento mesh standard fsaverage (per vista 'Intero')...")
try:
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    l_verts, l_faces = surface.load_surf_data(fsaverage.pial_left)
    r_verts, r_faces = surface.load_surf_data(fsaverage.pial_right)

    WHOLE_BRAIN_TRACES = [
        go.Mesh3d(x=l_verts[:, 0], y=l_verts[:, 1], z=l_verts[:, 2], i=l_faces[:, 0], j=l_faces[:, 1], k=l_faces[:, 2],
                  color='gray', opacity=0.10, name='H. Sinistro', hoverinfo='skip'),
        go.Mesh3d(x=r_verts[:, 0], y=r_verts[:, 1], z=r_verts[:, 2], i=r_faces[:, 0], j=r_faces[:, 1], k=r_faces[:, 2],
                  color='gray', opacity=0.10, name='H. Destro', hoverinfo='skip')
    ]
    print(">>> Mesh fsaverage caricata.")
except Exception as e:
    print(f"!!! Errore fatale mesh fsaverage: {e}")
    WHOLE_BRAIN_TRACES = []


# =============================================================================
# 2. FUNZIONI UTILI DI PARSING (DEBUG EXCEL ATTIVO)
# =============================================================================

def extract_coords_regex(name_string):
    if not isinstance(name_string, str): return None
    match = re.search(r'[(\[]\s*(-?\d+\.?\d*)[,\s;]+\s*(-?\d+\.?\d*)[,\s;]+\s*(-?\d+\.?\d*)\s*[)\]]', name_string)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return None


def extract_macro_region(name):
    if not isinstance(name, str): return "Other"
    for zone, keywords in MACRO_ZONES_MAP.items():
        for k in keywords:
            if k.lower() in name.lower():
                return zone
    return "Other"


def smart_standardize_columns(df, file_type="edges"):
    df.columns = [str(c).strip().lower() for c in df.columns]

    if file_type == "edges":
        map_source = ['source', 'source_id', 'src', 'id1', 'start', 'from']
        map_target = ['target', 'target_id', 'tgt', 'id2', 'end', 'to']
        map_weight = ['weight', 'w', 'strength', 'peso', 'value']

        new_map = {}
        for col in df.columns:
            if col in map_source:
                new_map[col] = 'source'
            elif col in map_target:
                new_map[col] = 'target'
            elif col in map_weight:
                new_map[col] = 'weight'

        df = df.rename(columns=new_map)

        # Fallback posizionale
        if 'source' not in df.columns and df.shape[1] >= 2:
            df.columns.values[0] = 'source'
            df.columns.values[1] = 'target'
            if df.shape[1] > 2: df.columns.values[2] = 'weight'

    elif file_type == "nodes":
        map_id = ['id', 'roi_id', 'node_id', 'index', 'n']
        map_name = ['name', 'roi_name', 'label', 'region', 'area']
        map_x = ['x', 'mni_x', 'r']
        map_y = ['y', 'mni_y', 'a']
        map_z = ['z', 'mni_z', 's']

        new_map = {}
        for col in df.columns:
            if col in map_id:
                new_map[col] = 'roi_id'
            elif col in map_name:
                new_map[col] = 'roi_name'
            elif col in map_x:
                new_map[col] = 'x'
            elif col in map_y:
                new_map[col] = 'y'
            elif col in map_z:
                new_map[col] = 'z'

        df = df.rename(columns=new_map)

        if 'roi_name' not in df.columns and df.shape[1] >= 2:
            df.columns.values[0] = 'roi_id'
            df.columns.values[1] = 'roi_name'

    return df


def parse_file_contents(contents, file_type="unknown"):
    if not contents: return None

    print(f"DEBUG: Inizio lettura file ({file_type})...")

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
    except Exception as e:
        print(f"ERRORE DECODIFICA BASE64: {e}")
        return None

    df = None

    # --- TENTATIVO 1: EXCEL (Richiede openpyxl) ---
    try:
        # Proviamo a leggere come Excel
        df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        print("DEBUG: Successo! File letto come Excel (.xlsx)")
    except Exception as e:
        # Se fallisce, stampiamo perché (es. manca openpyxl) ma non ci fermiamo
        print(f"DEBUG: Fallita lettura Excel. Motivo: {e}")
        print("DEBUG: Tento lettura come CSV/TXT...")

    # --- TENTATIVO 2: CSV/TXT (Testo) ---
    if df is None:
        text_content = None
        for codec in ["utf-8", "cp1252", "latin1"]:
            try:
                text_content = decoded.decode(codec)
                break
            except:
                continue

        if text_content:
            try:
                df = pd.read_csv(io.StringIO(text_content), sep=None, engine='python')
                print("DEBUG: Successo! File letto come CSV (auto-detect).")
            except:
                # Fallback per matrici pure (spazi)
                try:
                    df = pd.read_csv(io.StringIO(text_content), sep=r'\s+', header=None, engine='python')
                    print("DEBUG: Successo! File letto come Matrice TXT.")
                except:
                    pass

    if df is None:
        print("ERRORE FATALE: Nessun metodo di lettura ha funzionato.")
        raise ValueError("Formato file illeggibile. Assicurati di aver installato 'openpyxl' se usi Excel.")

    # Standardizzazione colonne
    if file_type != "unknown":
        df = smart_standardize_columns(df, file_type)

    return df

# =============================================================================
# 3. DASH INIT
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server=app.server

# Definizione stile globale per i grafici Plotly (Tema Chiaro)
graph_layout_props = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        xaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
        yaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
        zaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
        dragmode='turntable'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", itemsizing="constant")
)

# =============================================================================
# 4. HEADER COMPONENT (CORRETTO SENZA PROP 'COLOR')
# =============================================================================

header_component = dmc.Paper(
    id="header-container",
    shadow="xs", p="md", withBorder=True,
    style={"position": "fixed", "top": 0, "width": "100%", "zIndex": 2000, "backdropFilter": "blur(10px)",
           "opacity": 1},
    children=[
        dmc.Group(
            justify="space-between",
            children=[
                dmc.Anchor(
                    dmc.Group([
                        dmc.ThemeIcon(DashIconify(icon="mdi:brain", width=20), size="lg", radius="xl", color="blue",
                                      variant="light"),
                        dmc.Text("NeuroGraph 3D", size="lg", fw=700, c="dark")
                    ]), href="/", style={"textDecoration": "none"}
                ),

                # --- MENU CENTRALE DI NAVIGAZIONE ---
                dmc.Group([
                    dmc.Anchor(dmc.Button("Grafico 3D", variant="subtle", color="gray",
                                          leftSection=DashIconify(icon="mdi:cube-scan")), href="/graph"),
                    dmc.Anchor(dmc.Button("Simulazione AI", variant="subtle", color="grape",
                                          leftSection=DashIconify(icon="mdi:robot-industrial")), href="/simulation"),
                    dmc.Anchor(dmc.Button("Confronto", variant="subtle", color="orange",
                                          leftSection=DashIconify(icon="mdi:scale-balance")), href="/compare"),
                ]),

                # --- SEZIONE DESTRA: CONTESTO E UTILITY ---
                dmc.Group([
                    # NUOVO: SELETTORE DI RETE ATTIVA
                    dmc.Select(
                        id="network-context-selector",
                        data=[{"value": "main", "label": "📁 Upload Singolo"}],
                        value="main",
                        variant="filled",
                        # RIMOSSO 'color="indigo"' CHE CAUSAVA L'ERRORE
                        leftSection=DashIconify(icon="mdi:source-branch"),
                        style={"width": "180px", "display": "none"},  # Nascosto se non serve
                        allowDeselect=False
                    ),

                    dmc.Divider(orientation="vertical", h=20),

                    dmc.Switch(
                        id="theme-switch", size="lg",
                        onLabel=DashIconify(icon="mdi:weather-night", width=15),
                        offLabel=DashIconify(icon="mdi:weather-sunny", width=15),
                        checked=False
                    ),
                    dmc.Anchor(dmc.Button("Nuovo Upload", variant="filled", color="blue", size="sm",
                                          leftSection=DashIconify(icon="mdi:upload")), href="/")
                ])
            ]
        )
    ]
)
# =============================================================================
# 5. LAYOUT PAGINA UPLOAD (HOME PAGE & SELEZIONE MODALITÀ)
# =============================================================================

upload_box_style = {
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderColor': '#dee2e6',
    'borderRadius': '10px',
    'textAlign': 'center',
    'padding': '20px',
    'cursor': 'pointer',
    'backgroundColor': '#f8f9fa',
    'transition': 'all 0.3s ease',
    'color': '#495057'
}

# --- CONTENUTO TAB 1: VISUALIZZAZIONE (Il tuo vecchio layout upload) ---
tab_visualization_content = dmc.Grid(
    gutter="xl",
    align="center",
    mt="lg",
    children=[
        dmc.GridCol(
            span=7,
            children=[
                dmc.Text("Caricamento Dati Connectome", size="xl", fw=700, c="blue", mb="sm"),
                dmc.Text(
                    "Importa le matrici di adiacenza e le coordinate spaziali per generare il grafo 3D.",
                    size="md", c="dimmed", mb="md"
                ),
                dmc.Alert(
                    title="Specifiche Tecniche",
                    color="blue",
                    variant="outline",
                    icon=DashIconify(icon="mdi:information"),
                    children=[
                        dmc.Text("1. Mapping: Richiesto file CSV con colonne [roi_name, x, y, z]."),
                        dmc.Text("2. Edges: Richiesto file CSV o Matrice di Adiacenza pesata."),
                        dmc.Space(h=10),
                        dmc.Group([
                            dmc.Badge("Supporto AAL/M.Cube", color="green"),
                            dmc.Badge("CSV/TXT", color="gray"),
                        ])
                    ]
                )
            ]
        ),
        dmc.GridCol(
            span=5,
            children=[
                dmc.Card(
                    withBorder=True, shadow="md", radius="md", p="xl",
                    children=[
                        dmc.Text("Pannello Ingestione Dati", size="lg", fw=700, ta="center", mb="lg"),

                        # Upload 1
                        dmc.Text("1. File Mapping (Nodi)", fw=500, mb=5),
                        dcc.Upload(
                            id='upload-mapping',
                            style=upload_box_style,
                            children=html.Div([
                                DashIconify(icon="mdi:map-marker-path", width=30, color="cyan"),
                                html.Div("Trascina Mapping.csv", style={'marginTop': '5px'})
                            ])
                        ),
                        html.Div(id="file-info-mapping", style={"marginTop": "10px"}),
                        dmc.Space(h=15),

                        # Upload 2
                        dmc.Text("2. File Edges (Connessioni)", fw=500, mb=5),
                        dcc.Upload(
                            id='upload-edges',
                            style=upload_box_style,
                            children=html.Div([
                                DashIconify(icon="mdi:vector-line", width=30, color="orange"),
                                html.Div("Trascina Edges.csv", style={'marginTop': '5px'})
                            ])
                        ),
                        html.Div(id="file-info-edges", style={"marginTop": "10px"}),
                        dmc.Space(h=25),

                        dmc.Button(
                            "Genera Modello 3D",
                            id="start-button",
                            fullWidth=True,
                            size="md",
                            disabled=True,
                            color="blue",
                            leftSection=DashIconify(icon="mdi:cube-scan")
                        ),
                    ]
                )
            ]
        )
    ]
)

# --- CONTENUTO TAB 2: AI SIMULATION (Placeholder) ---
tab_ai_content = dmc.Container(
    fluid=True,
    py="xl",
    children=[
        dmc.Grid(
            gutter="xl",
            align="center",
            children=[
                # Lato Sinistro: Descrizione
                dmc.GridCol(span=5, children=[
                    dmc.Stack([
                        DashIconify(icon="mdi:robot-industrial", width=60, color="#868e96"),
                        dmc.Title("Simulazione Neurale AI", order=2),
                        dmc.Text(
                            "Carica i dati qui per saltare la visualizzazione 3D e accedere direttamente "
                            "al modulo di stress-test e previsione lesionale.",
                            size="lg", c="dimmed"
                        ),
                        dmc.Alert(
                            "Questa modalità è ottimizzata per computer meno potenti o per analisi puramente numeriche.",
                            color="grape", variant="light", icon=DashIconify(icon="mdi:flash")
                        )
                    ])
                ]),

                # Lato Destro: Pannello Upload Dedicato
                dmc.GridCol(span=7, children=[
                    dmc.Card(
                        withBorder=True, shadow="sm", radius="md", p="lg",
                        children=[
                            dmc.Text("Caricamento Rapido per Simulazione", fw=700, size="lg", mb="md"),

                            dmc.Group([
                                # Upload Mapping Sim
                                dmc.Stack([
                                    dmc.Text("1. Mapping (Nodi)", size="sm", fw=500),
                                    dcc.Upload(
                                        id='upload-mapping-sim',  # ID DIVERSO
                                        style={
                                            'border': '1px dashed #ced4da', 'borderRadius': '5px',
                                            'padding': '10px', 'textAlign': 'center', 'cursor': 'pointer'
                                        },
                                        children=html.Div([
                                            DashIconify(icon="mdi:map-marker-path", width=20, color="gray"),
                                            dmc.Text("Select Mapping", size="xs")
                                        ])
                                    ),
                                    html.Div(id="file-info-mapping-sim")
                                ], style={"flex": 1}),

                                # Upload Edges Sim
                                dmc.Stack([
                                    dmc.Text("2. Edges (Connessioni)", size="sm", fw=500),
                                    dcc.Upload(
                                        id='upload-edges-sim',  # ID DIVERSO
                                        style={
                                            'border': '1px dashed #ced4da', 'borderRadius': '5px',
                                            'padding': '10px', 'textAlign': 'center', 'cursor': 'pointer'
                                        },
                                        children=html.Div([
                                            DashIconify(icon="mdi:vector-line", width=20, color="gray"),
                                            dmc.Text("Select Edges", size="xs")
                                        ])
                                    ),
                                    html.Div(id="file-info-edges-sim")
                                ], style={"flex": 1}),
                            ], grow=True, mb="xl"),

                            dmc.Button(
                                "Carica & Vai alla Simulazione",
                                id="btn-load-and-go-sim",  # ID PER LA NUOVA AZIONE
                                fullWidth=True,
                                size="lg",
                                color="grape",
                                rightSection=DashIconify(icon="mdi:arrow-right"),
                                disabled=True  # Abilitato via callback
                            )
                        ]
                    )
                ])
            ]
        )
    ]
)

upload_style_home = {
    'border': '1px dashed #ced4da', 'borderRadius': '8px', 'padding': '10px',
    'textAlign': 'center', 'cursor': 'pointer', 'backgroundColor': '#fff', 'transition': '0.2s', 'fontSize': '12px'
}


# --- 1. DEFINIZIONE FUNZIONE HELPER (DEVE STARE QUI, PRIMA DEL LAYOUT) ---
def create_upload_slot(letter, color, label):
    return dmc.Stack(gap="xs", children=[
        dmc.Badge(label, color=color, variant="filled", fullWidth=True),
        dcc.Upload(id=f'home-map-{letter}', style=upload_style_home,
                   children=html.Div([DashIconify(icon="mdi:file"), f" Map {letter.upper()}"])),
        html.Div(id=f"fb-map-{letter}", style={"fontSize": "10px", "color": "green", "minHeight": "15px"}),
        dcc.Upload(id=f'home-edge-{letter}', style=upload_style_home,
                   children=html.Div([DashIconify(icon="mdi:link"), f" Edge {letter.upper()}"])),
        html.Div(id=f"fb-edge-{letter}", style={"fontSize": "10px", "color": "green", "minHeight": "15px"})
    ])


# --- 2. LAYOUT CONTENITORE ---
tab_compare_content = dmc.Container(
    fluid=True, py="xl",
    children=[
        dmc.Stack(gap="lg", children=[
            dmc.Group([
                DashIconify(icon="mdi:scale-balance", width=50, color="#fd7e14"),
                dmc.Stack(gap=0, children=[
                    dmc.Title("Confronto Multi-Rete", order=2),
                    dmc.Text("Carica le reti da confrontare. Il layout si adatterà automaticamente.", c="dimmed",
                             size="sm")
                ])
            ]),

            # CARD UPLOAD (GRIGLIA ELASTICA)
            dmc.Card(withBorder=True, shadow="sm", radius="md", p="lg", children=[
                dmc.SimpleGrid(
                    id="dynamic-grid",
                    cols={"base": 1, "sm": 2},  # Default 2 colonne
                    spacing="md",
                    children=[
                        # A e B sempre visibili
                        create_upload_slot("a", "blue", "RETE A (Baseline)"),
                        create_upload_slot("b", "orange", "RETE B"),
                        # C e D inizialmente nascosti
                        html.Div(id="slot-container-c", style={"display": "none"},
                                 children=create_upload_slot("c", "green", "RETE C")),
                        html.Div(id="slot-container-d", style={"display": "none"},
                                 children=create_upload_slot("d", "red", "RETE D")),
                    ]
                ),
            ]),

            # CONTROLLI DINAMICI
            dmc.Center(
                dmc.Group([
                    dmc.Button("Rimuovi Rete", id="btn-remove-network", variant="light", color="red",
                               leftSection=DashIconify(icon="mdi:minus"), style={"display": "none"}),
                    dmc.Button("Aggiungi Rete", id="btn-add-network", variant="light", color="blue",
                               rightSection=DashIconify(icon="mdi:plus"), style={"display": "block"})
                ])
            ),

            dmc.Divider(),

            # PULSANTE AVVIO
            dmc.Button(
                "Avvia Analisi Comparativa", id="btn-launch-compare",
                fullWidth=True, size="xl", radius="md", color="indigo",
                rightSection=DashIconify(icon="mdi:rocket-launch")
            )
        ])
    ]
)

# --- CONTENUTO TAB 3: CONFRONTO MULTI-SOGGETTO (GRID ELASTICA) ---
tab_compare_content = dmc.Container(
    fluid=True, py="xl",
    children=[
        dmc.Stack(gap="lg", children=[
            dmc.Group([
                DashIconify(icon="mdi:scale-balance", width=50, color="#fd7e14"),
                dmc.Stack(gap=0, children=[
                    dmc.Title("Confronto Multi-Rete", order=2),
                    dmc.Text("Carica le reti da confrontare. Il layout si adatterà automaticamente.", c="dimmed", size="sm")
                ])
            ]),

            # CARD UPLOAD
            dmc.Card(withBorder=True, shadow="sm", radius="md", p="lg", children=[
                # GRIGLIA DINAMICA
                dmc.SimpleGrid(
                    id="dynamic-grid",  # <--- ID IMPORTANTE PER LA CALLBACK
                    cols={"base": 1, "sm": 2}, # Default iniziale: 2 colonne
                    spacing="md",
                    children=[
                        # A e B sempre presenti nel DOM
                        create_upload_slot("a", "blue", "RETE A (Baseline)"),
                        create_upload_slot("b", "orange", "RETE B"),
                        # C e D inizialmente nascosti
                        html.Div(id="slot-container-c", style={"display": "none"}, children=create_upload_slot("c", "green", "RETE C")),
                        html.Div(id="slot-container-d", style={"display": "none"}, children=create_upload_slot("d", "red", "RETE D")),
                    ]
                ),
            ]),

            # CONTROLLI DINAMICI (Pulsanti + e -)
            dmc.Center(
                dmc.Group([
                    dmc.Button("Rimuovi Rete", id="btn-remove-network", variant="light", color="red", leftSection=DashIconify(icon="mdi:minus"), style={"display": "none"}),
                    dmc.Button("Aggiungi Rete", id="btn-add-network", variant="light", color="blue", rightSection=DashIconify(icon="mdi:plus"), style={"display": "block"})
                ])
            ),

            dmc.Divider(),

            # PULSANTE AVVIO
            dmc.Button(
                "Avvia Analisi Comparativa", id="btn-launch-compare",
                fullWidth=True, size="xl", radius="md", color="indigo",
                rightSection=DashIconify(icon="mdi:rocket-launch")
            )
        ])
    ]
)
# --- LAYOUT FINALE HOME PAGE ---
layout_upload = dmc.Container(
    fluid=True,
    children=[
        # 1. HERO SECTION
        dmc.Paper(
            radius="md",
            p="xl",
            withBorder=False,
            style={"background": "linear-gradient(135deg, #e3fafc 0%, #e7f5ff 100%)", "marginTop": "40px"},
            children=[
                dmc.Stack(
                    align="center",
                    gap="xs",
                    children=[
                        dmc.Group([
                            DashIconify(icon="mdi:brain", width=50, color="#1c7ed6"),
                            dmc.Title("Benvenuto in NeuroGraph 3D", order=1, c="dark")
                        ]),
                        dmc.Text(
                            "Piattaforma avanzata per l'esplorazione topologica, l'analisi di connettività "
                            "e la visualizzazione interattiva del connettoma umano.",
                            size="lg", c="dimmed", ta="center", style={"maxWidth": "700px"}
                        ),
                        dmc.Divider(size="sm", style={"width": "100px"}, color="blue")
                    ]
                )
            ]
        ),

        # 2. SELEZIONE MODALITÀ (TABS)
        dmc.Container(
            size="lg",
            mt="xl",
            children=[
                dmc.Tabs(
                    color="blue",
                    variant="outline",
                    radius="md",
                    value="viz",
                    children=[
                        dmc.TabsList(
                            grow=True,
                            children=[
                                dmc.TabsTab(
                                    "Visualizzazione Dati",
                                    value="viz",
                                    leftSection=DashIconify(icon="mdi:chart-bubble", width=20),
                                ),
                                # --- MODIFICA QUI SOTTO ---
                                dmc.TabsTab(
                                    "Simulazione AI",
                                    value="ai",
                                    # HO CAMBIATO L'ICONA QUI:
                                    leftSection=DashIconify(icon="mdi:robot-industrial", width=20),
                                ),
                                # --------------------------
                                dmc.TabsTab(
                                    "Confronto Reti",
                                    value="compare",
                                    leftSection=DashIconify(icon="mdi:compare-horizontal", width=20),
                                ),
                            ]
                        ),

                        # Pannelli Contenuto
                        dmc.TabsPanel(tab_visualization_content, value="viz"),
                        dmc.TabsPanel(tab_ai_content, value="ai"),
                        dmc.TabsPanel(tab_compare_content, value="compare"),
                    ]
                )
            ]
        )
    ]
)


# =============================================================================
# UI CALLBACK: GESTIONE SLOT E GRIGLIA ELASTICA
# =============================================================================

@app.callback(
    Output("slot-container-c", "style"),
    Output("slot-container-d", "style"),
    Output("dynamic-grid", "cols"),  # <--- NUOVO OUTPUT: Modifica le colonne
    Output("btn-add-network", "style"),
    Output("btn-remove-network", "style"),
    Output("visible-networks-count", "data"),

    Input("btn-add-network", "n_clicks"),
    Input("btn-remove-network", "n_clicks"),
    State("visible-networks-count", "data"),
    prevent_initial_call=True
)
def manage_network_slots(n_add, n_remove, current_count):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_count = current_count

    if button_id == "btn-add-network":
        new_count = min(4, current_count + 1)
    elif button_id == "btn-remove-network":
        new_count = max(2, current_count - 1)

    # Logica Visibilità Slot
    style_c = {"display": "block"} if new_count >= 3 else {"display": "none"}
    style_d = {"display": "block"} if new_count >= 4 else {"display": "none"}

    # Logica Griglia Elastica (Responsive)
    # Su mobile (base) sempre 1 colonna. Su desktop (sm) tante colonne quante sono le reti.
    grid_cols = {"base": 1, "sm": new_count}

    # Logica Bottoni
    style_btn_add = {"display": "block"} if new_count < 4 else {"display": "none"}
    style_btn_rem = {"display": "block"} if new_count > 2 else {"display": "none"}

    return style_c, style_d, grid_cols, style_btn_add, style_btn_rem, new_count

# =============================================================================
# 6. LAYOUT PAGINA GRAFICO (CON ANIMAZIONE "SIPARIO")
# =============================================================================

NODE_METRICS_OPTIONS = [
    {"label": "Community", "value": "community"},
    {"label": "Degree (Grado)", "value": "degree"},
    {"label": "Strength (Forza)", "value": "strength"},
    {"label": "Betweenness Centrality", "value": "betweenness_centrality"},
]

info_button_component = html.Div([
    dmc.Affix(
        position={"bottom": 20, "right": 20},
        zIndex=9999,
        children=dmc.ActionIcon(
            DashIconify(icon="mdi:information-variant", width=25),
            id="info-fab-btn",
            size="xl",
            radius="xl",
            variant="filled",
            color="blue",
            style={"boxShadow": "0px 4px 12px rgba(0,0,0,0.2)"}
        )
    ),
    dmc.Affix(
        position={"bottom": 80, "right": 20},
        zIndex=9998,
        children=dmc.Paper(
            id="info-panel-popup",
            shadow="xl",
            radius="md",
            p="md",
            withBorder=True,
            style={
                "width": "380px",  # Leggermente più largo per il testo
                "display": "none",
                "backgroundColor": "rgba(255, 255, 255, 0.98)",
                "color": "#333"
            },
            children=[
                dmc.Group([
                    DashIconify(icon="mdi:school", width=22, color="blue"),
                    dmc.Text("Guida alle Funzionalità", fw=700, size="lg")
                ], mb="md", pb="xs", style={"borderBottom": "1px solid #eee"}),

                dmc.ScrollArea(
                    h=450,  # Altezza aumentata per contenere tutto
                    type="hover",
                    offsetScrollbars=True,
                    children=[
                        dmc.Stack(gap="sm", children=[

                            # 1. NAVIGAZIONE
                            dmc.Text("🖱️ Navigazione 3D", fw=700, size="sm"),
                            dmc.Text("• Ruota: Click Sinistro + Trascina", size="xs", c="dimmed"),
                            dmc.Text("• Sposta: Click Destro + Trascina", size="xs", c="dimmed"),
                            dmc.Text("• Zoom: Rotella del mouse", size="xs", c="dimmed"),

                            dmc.Divider(),

                            # 2. ANATOMIA E FILTRI ZONA
                            dmc.Text("🧠 Anatomia e Regioni", fw=700, size="sm"),
                            dmc.Text(
                                "• Stile Mesh: Scegli tra 'Intero (Grigio)' per il cervello completo o 'Regioni (Colorate)' per vedere le aree AAL.",
                                size="xs", c="dimmed"),
                            dmc.Text(
                                "• Isola Zona: (Solo in modalità Regioni) Seleziona specifiche aree (es. Frontale, Occipitale) per nascondere il resto.",
                                size="xs", c="dimmed"),

                            dmc.Divider(),

                            # 3. METRICHE E COLORI
                            dmc.Text("🎨 Aspetto dei Nodi", fw=700, size="sm"),
                            dmc.Text(
                                "• Colore/Dimensione: Configura i nodi in base a metriche come Grado (Degree), Forza (Strength) o Comunità.",
                                size="xs", c="dimmed"),
                            dmc.Text("• Slider: Regola la scala dei colori e la dimensione dei nodi in tempo reale.",
                                     size="xs", c="dimmed"),

                            dmc.Divider(),

                            # 4. ANALISI RETE
                            dmc.Text("🕸️ Filtri di Rete", fw=700, size="sm"),
                            dmc.Text(
                                "• Risoluzione (Louvain): Sposta lo slider per ricalcolare le comunità (0.5 = poche/grandi, 2.0 = molte/piccole).",
                                size="xs", c="dimmed"),
                            dmc.Text("• Peso Minimo: Nasconde gli archi con peso inferiore alla soglia scelta.",
                                     size="xs", c="dimmed"),

                            dmc.Divider(),

                            # 5. CONFRONTO COMUNITA'
                            dmc.Text("🔗 Confronto A ↔ B", fw=700, size="sm"),
                            dmc.Text(
                                "Seleziona due comunità diverse (es. Comm 1 e Comm 3) e premi il pulsante. Il grafico isolerà ed evidenzierà in arancione solo le connessioni che collegano i due gruppi.",
                                size="xs", c="dimmed"),

                            dmc.Divider(),

                            # 6. INTERAZIONE E DATI
                            dmc.Text("📊 Interazione e Dettagli", fw=700, size="sm"),
                            dmc.Text(
                                "• Click Nodo: Mostra dettagli nel pannello sinistro e la lista delle connessioni nel pannello destro.",
                                size="xs", c="dimmed"),
                            dmc.Text("• Export: Scarica l'immagine (PNG) o i dati del grafico (JSON).", size="xs",
                                     c="dimmed"),
                        ])
                    ]
                )
            ]
        )
    )
])

layout_graph = html.Div(
    id="graph-page-container",
    style={"minHeight": "100vh", "transition": "background-color 0.3s ease"},
    children=[
        # --- 1. ANIMAZIONE DI CARICAMENTO (SPLASH) ---
        dcc.Interval(id="loading-timer", interval=3800, max_intervals=1),

        html.Div(
            id="splash-overlay",
            className="splash-overlay-container",
            children=[
                html.Div(className="splash-curtain"),
                html.Div(className="splash-content-group", children=[
                    html.Div(className="brain-loader-wrapper", children=[
                        html.Div(className="brain-circle-bg"),
                        html.Div(
                            DashIconify(icon="mdi:brain", width=80),
                            className="brain-base",
                            style={"top": "50%", "left": "50%", "transform": "translate(-50%, -50%)"}
                        ),
                        html.Div(className="brain-fill-mask", children=[
                            html.Div(
                                DashIconify(icon="mdi:brain", width=80, className="brain-fill-icon"),
                                style={"position": "absolute", "top": "50%", "left": "50%",
                                       "transform": "translate(-50%, -50%)"}
                            )
                        ])
                    ]),
                    html.Div("NeuroGraph 3D", className="splash-text")
                ])
            ]
        ),

        # --- 2. UI REALE ---
        html.Div(id="final-header-wrapper", style={"opacity": 0}, children=header_component),

        info_button_component,
        dcc.Store(id='info-panel-state', data=False),

        html.Div(
            id="main-graph-content",
            style={"opacity": 0},
            children=dmc.Container(
                style={"paddingTop": 90, "paddingBottom": 40},
                fluid=True,
                children=[
                    dmc.Grid(
                        gutter="md",
                        children=[
                            # ======================
                            # SIDEBAR SINISTRA
                            # ======================
                            dmc.GridCol(
                                span=3,
                                children=[
                                    dmc.Stack([
                                        # 1. Sezione Accordion (Controlli)
                                        dmc.Paper(
                                            withBorder=True,
                                            shadow="sm",
                                            radius="md",
                                            children=[
                                                dmc.Accordion(
                                                    variant="separated",
                                                    radius="md",
                                                    value="filters",  # Aperto su filtri di default
                                                    children=[
                                                        # GRUPPO ASPETTO
                                                        dmc.AccordionItem([
                                                            dmc.AccordionControl(
                                                                dmc.Group([
                                                                    DashIconify(icon="mdi:palette", color="blue"),
                                                                    dmc.Text("Aspetto e Colore")
                                                                ]),
                                                            ),
                                                            dmc.AccordionPanel([
                                                                dmc.Text("Colorazione Nodi", size="sm", fw=500, mb=5),
                                                                dmc.Select(id="node-color-metric",
                                                                           data=NODE_METRICS_OPTIONS, value="community",
                                                                           clearable=False),
                                                                html.Div(id="color-range-controls"),

                                                                dmc.Space(h=15),
                                                                dmc.Text("Dimensione Nodi", size="sm", fw=500, mb=5),
                                                                dmc.Select(
                                                                    id="node-size-metric",
                                                                    data=[{"label": "Dimensione Fissa",
                                                                           "value": "fixed"}] + NODE_METRICS_OPTIONS[
                                                                             1:],
                                                                    value="strength",
                                                                    clearable=False
                                                                ),
                                                                html.Div(id="size-range-controls"),
                                                            ])
                                                        ], value="appearance"),

                                                        # GRUPPO FILTRI
                                                        dmc.AccordionItem([
                                                            dmc.AccordionControl(
                                                                dmc.Group([
                                                                    DashIconify(icon="mdi:filter-variant",
                                                                                color="orange"),
                                                                    dmc.Text("Filtri e Struttura")
                                                                ]),
                                                            ),
                                                            dmc.AccordionPanel([

                                                                # --- NUOVO SELETTORE STILE CERVELLO ---
                                                                dmc.Text("Stile Mesh Cerebrale", size="sm", fw=700,
                                                                         c="blue"),
                                                                dmc.SegmentedControl(
                                                                    id="brain-style-selector",
                                                                    value="gray",
                                                                    data=[
                                                                        {"value": "gray", "label": "Intero (Grigio)"},
                                                                        {"value": "regions",
                                                                         "label": "Regioni (Colorate)"}
                                                                    ],
                                                                    fullWidth=True,
                                                                    mb=10,
                                                                    color="blue"
                                                                ),

                                                                # --- CONTENITORE FILTRO ZONA (Visibile solo se regions) ---
                                                                html.Div(
                                                                    id="region-filter-container",
                                                                    style={"display": "none"},  # Nascosto di default
                                                                    children=[
                                                                        dmc.Text("Isola Zona Anatomica", size="sm",
                                                                                 fw=700, mt=10),
                                                                        dmc.Text("Seleziona per isolare:", size="xs",
                                                                                 c="dimmed", mb=5),
                                                                        dmc.MultiSelect(id="region-filter",
                                                                                        placeholder="Es. Cerebellum...",
                                                                                        clearable=True, data=[],
                                                                                        searchable=True),
                                                                        dmc.Divider(my=15),
                                                                    ]
                                                                ),
                                                                # --------------------------------------------------

                                                                dmc.Text("Risoluzione (Louvain)", size="sm", fw=500),
                                                                dmc.Slider(id="resolution-slider", min=0.5, max=2.0,
                                                                           step=0.1, value=1.0,
                                                                           marks=[{"value": 0.5, "label": "0.5"},
                                                                                  {"value": 1.0, "label": "1.0"},
                                                                                  {"value": 2.0, "label": "2.0"}]),

                                                                dmc.Space(h=15),
                                                                dmc.Text("Filtra Singola Community", size="sm", fw=500),
                                                                dmc.MultiSelect(id="community-filter",
                                                                                placeholder="Seleziona...",
                                                                                clearable=True,
                                                                                data=[]),

                                                                # --- SEZIONE INTER-COMUNITÀ ---
                                                                dmc.Divider(my="sm", label="Connessioni Tra Comunità",
                                                                            labelPosition="center"),
                                                                dmc.Group([
                                                                    dmc.Select(id="select-comm-a", placeholder="Com. A",
                                                                               style={"flex": 1}, data=[],
                                                                               clearable=True),
                                                                    dmc.Select(id="select-comm-b", placeholder="Com. B",
                                                                               style={"flex": 1}, data=[],
                                                                               clearable=True),
                                                                ], grow=True, mb=10),
                                                                dmc.Button(
                                                                    "Mostra A ↔ B",
                                                                    id="btn-inter-comm",
                                                                    variant="outline",
                                                                    color="indigo",
                                                                    fullWidth=True,
                                                                    leftSection=DashIconify(icon="mdi:vector-link")
                                                                ),

                                                                dmc.Space(h=15),
                                                                dmc.Text("Peso Minimo Archi", size="sm", fw=500),
                                                                dmc.Slider(id="weight-threshold", min=0, max=1,
                                                                           step=0.01,
                                                                           value=0.2),
                                                            ])
                                                        ], value="filters"),

                                                        # GRUPPO VISUALIZZAZIONE
                                                        dmc.AccordionItem([
                                                            dmc.AccordionControl(
                                                                dmc.Group([
                                                                    DashIconify(icon="mdi:eye", color="green"),
                                                                    dmc.Text("Opzioni Vista")
                                                                ]),
                                                            ),
                                                            dmc.AccordionPanel([
                                                                dmc.Switch(id="toggle-brain", label="Mostra Cervello",
                                                                           checked=True, mb=5),
                                                                dmc.Switch(id="toggle-default-edges",
                                                                           label="Mostra Connessioni", checked=True,
                                                                           mb=5),
                                                                dmc.Switch(id="toggle-highlight-edges",
                                                                           label="Evidenzia Selezione", checked=True),

                                                                dmc.Divider(my="sm"),

                                                                dmc.Text("Opacità Cervello", size="sm", fw=500),
                                                                dmc.Slider(id="brain-opacity-slider", min=0, max=1,
                                                                           step=0.05, value=0.1),

                                                                dmc.Space(h=10),

                                                                dmc.Text("Opacità Archi", size="sm", fw=500),
                                                                dmc.Slider(id="edge-opacity-slider", min=0, max=1,
                                                                           step=0.05, value=0.15),

                                                                dmc.Space(h=10),

                                                                dmc.Text("Spessore Archi", size="sm", fw=500),
                                                                dmc.Slider(id="edge-width-slider", min=1, max=5,
                                                                           step=0.5,
                                                                           value=1),
                                                            ])
                                                        ], value="view"),
                                                    ]
                                                ),
                                            ]
                                        ),

                                        # 2. Sezione Ricerca e Export
                                        dmc.Paper(
                                            withBorder=True, shadow="sm", radius="md", p="md",
                                            children=[
                                                dmc.Select(
                                                    id="node-selector-dropdown",
                                                    placeholder="🔍 Cerca nodo...",
                                                    data=[], searchable=True, clearable=True,
                                                    leftSection=DashIconify(icon="mdi:magnify"),
                                                    nothingFoundMessage="Nessun nodo trovato"
                                                ),
                                                dmc.Group([
                                                    dmc.Button("PNG", id="export-png", variant="light", size="xs",
                                                               flex=1,
                                                               leftSection=DashIconify(icon="mdi:camera")),
                                                    dmc.Button("JSON", id="export-json", variant="outline", size="xs",
                                                               flex=1, leftSection=DashIconify(icon="mdi:code-json")),
                                                ], mt="sm")
                                            ]
                                        ),

                                        # 3. Card Dettagli Nodo
                                        dmc.Card(
                                            withBorder=True, shadow="sm", radius="md", p="md",
                                            children=[
                                                dmc.Group([
                                                    DashIconify(icon="mdi:chart-box", color="grape"),
                                                    dmc.Text("Dettagli Nodo", fw=700)
                                                ], mb="sm"),
                                                dmc.Divider(mb="md"),
                                                html.Div(id="node-details-output", children=[
                                                    dmc.Center([
                                                        dmc.Stack([
                                                            DashIconify(icon="mdi:cursor-default-click-outline",
                                                                        width=40,
                                                                        color="#dee2e6"),
                                                            dmc.Text("Clicca un nodo", c="dimmed", size="sm")
                                                        ], align="center", gap=5)
                                                    ], style={"height": "100px"})
                                                ])
                                            ]
                                        ),
                                    ])
                                ]
                            ),

                            # ======================
                            # AREA GRAFICO (CENTRALE)
                            # ======================
                            dmc.GridCol(
                                span=6,
                                children=[
                                    # Card Grafico 3D
                                    dmc.Card(
                                        withBorder=True,
                                        shadow="sm",
                                        radius="lg",
                                        padding=0,
                                        style={"overflow": "hidden", "marginBottom": "20px"},
                                        children=[
                                            dmc.Tooltip(
                                                label="Aggiorna Vista",
                                                position="right",
                                                withArrow=True,
                                                children=dmc.ActionIcon(
                                                    DashIconify(icon="mdi:refresh", width=20),
                                                    id="update-graph-btn",
                                                    variant="filled",
                                                    color="indigo",
                                                    size="lg",
                                                    radius="xl",
                                                    style={
                                                        "position": "absolute",
                                                        "top": 15,
                                                        "left": 15,
                                                        "zIndex": 100,
                                                        "boxShadow": "0 2px 5px rgba(0,0,0,0.2)"
                                                    }
                                                )
                                            ),
                                            dcc.Loading(
                                                type="cube",
                                                color="blue",
                                                children=dcc.Graph(
                                                    id="brain-connectome-web",
                                                    style={"height": "65vh"},
                                                    config={"displayModeBar": False}
                                                )
                                            )
                                        ]
                                    ),
                                    # --- CARD UNIFICATA: REPORT + ISTOGRAMMI ---
                                    dmc.Paper(
                                        withBorder=True, shadow="sm", radius="md", p="md",
                                        children=[
                                            dmc.Group([
                                                DashIconify(icon="mdi:google-analytics", color="blue", width=25),
                                                dmc.Text("Report Analisi & Metriche", fw=700, size="lg")
                                            ], mb="sm"),
                                            dmc.Alert(
                                                id="analysis-report-output",
                                                variant="light", color="gray",
                                                style={"padding": "10px", "marginBottom": "20px"}
                                            ),
                                            dmc.Divider(mb="md"),
                                            dmc.Text("Distribuzione Statistica", fw=700, size="sm", mb="sm",
                                                     c="dimmed"),
                                            dmc.SimpleGrid(
                                                cols={"base": 1, "sm": 3},
                                                spacing="xs",
                                                children=[
                                                    dcc.Graph(id="hist-degree", config={"displayModeBar": False},
                                                              style={"height": "150px"}),
                                                    dcc.Graph(id="hist-community", config={"displayModeBar": False},
                                                              style={"height": "150px"}),
                                                    dcc.Graph(id="hist-strength", config={"displayModeBar": False},
                                                              style={"height": "150px"}),
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            ),

                            # ======================
                            # SIDEBAR DESTRA
                            # ======================
                            dmc.GridCol(
                                span=3,
                                children=[
                                    dmc.Stack([
                                        # 1. CARD: STATISTICHE INTER-COMMUNITY
                                        dmc.Card(
                                            id="inter-community-card",
                                            withBorder=True, shadow="sm", radius="md", p="md",
                                            style={"display": "none", "borderColor": "#FFC107", "borderWidth": "2px"},
                                            children=[
                                                dmc.Group([
                                                    DashIconify(icon="mdi:vector-link", color="orange", width=25),
                                                    dmc.Text("Confronto Tra Comunità", fw=700, c="orange")
                                                ], mb="sm"),
                                                html.Div(id="inter-community-content")
                                            ]
                                        ),
                                        # 2. Card Connessioni
                                        dmc.Card(
                                            withBorder=True, shadow="sm", radius="md", p="md",
                                            style={"flex": 1, "minHeight": "400px", "borderColor": "#0CA678",
                                                   "borderWidth": "2px"},
                                            children=[
                                                dmc.Group([
                                                    DashIconify(icon="mdi:hub", color="teal"),
                                                    dmc.Text("Connessioni", fw=700)
                                                ], mb="sm"),
                                                dmc.ScrollArea(
                                                    h=450,
                                                    children=[html.Div(id="node-connections-output")]
                                                )
                                            ]
                                        ),
                                    ])
                                ]
                            ),
                        ]
                    )
                ]
            ))
    ]
)

# 6b. LAYOUT PAGINA SIMULAZIONE

layout_simulation = html.Div(
    id="graph-page-container",
    # Questo stile iniziale verrà gestito dal Theme Switcher, quindi non mettiamo qui il padding
    style={"minHeight": "100vh", "backgroundColor": "#f8f9fa", "transition": "background-color 0.3s ease"},
    children=[
        header_component,

        dmc.Container(
            fluid=True,
            px="xl",
            style={"paddingTop": "120px", "paddingBottom": "50px"},
            children=[
                dmc.Title("Simulazione Neurale & Resilienza", order=2, mb="lg", c="indigo"),

                dmc.Grid(
                    gutter="xl",
                    children=[
                        # --- COLONNA SINISTRA: INPUT ---
                        dmc.GridCol(
                            span=4,
                            children=[
                                dmc.Paper(
                                    withBorder=True, shadow="sm", radius="md", p="lg",
                                    children=[
                                        dmc.Group([
                                            DashIconify(icon="mdi:robot-confused", color="indigo", width=25),
                                            dmc.Text("Configurazione AI", size="lg", fw=700)
                                        ], mb="md"),

                                        dmc.Text("1. Scenario Patologico", size="sm", fw=500, mb=5),
                                        dmc.Select(
                                            id="sim-scenario-select",
                                            placeholder="Seleziona scenario...",
                                            data=[
                                                {"value": "stroke", "label": "Ictus Ischemico (Rimozione Nodo)"},
                                                {"value": "alzheimer", "label": "Neurodegenerazione (Alzheimer)"},
                                                {"value": "random", "label": "Random Failure (Attacco Casuale)"},
                                            ],
                                            value="stroke",
                                            mb=20
                                        ),

                                        dmc.Text("2. Nodo Target (Focale)", size="sm", fw=500, mb=5),
                                        dmc.Select(
                                            id="sim-target-node",
                                            placeholder="Carica i dati per scegliere...",
                                            searchable=True,
                                            data=[],
                                            mb=20
                                        ),

                                        dmc.Text("3. Intensità Danno / Tempo", size="sm", fw=500, mb=5),
                                        dmc.Slider(
                                            id="sim-severity-slider",
                                            min=0, max=100, step=10,
                                            value=50,
                                            marks=[
                                                {'value': 0, 'label': 'Lieve'},
                                                {'value': 50, 'label': 'Moderato'},
                                                {'value': 100, 'label': 'Critico'},
                                            ],
                                            mb=30,
                                            color="red"
                                        ),

                                        dmc.Divider(mb="lg"),

                                        dmc.Button(
                                            "Avvia Simulazione",
                                            id="btn-run-simulation",
                                            fullWidth=True,
                                            color="red",
                                            leftSection=DashIconify(icon="mdi:heart-pulse"),
                                            variant="filled"
                                        )
                                    ]
                                )
                            ]
                        ),

                        # --- COLONNA DESTRA: RISULTATI ---
                        dmc.GridCol(
                            span=8,
                            children=[
                                # 1. KPI CARDS
                                dmc.SimpleGrid(
                                    cols=3,
                                    mb="lg",
                                    children=[
                                        dmc.Card(withBorder=True, shadow="xs", radius="md", p="md", children=[
                                            dmc.Group([
                                                DashIconify(icon="mdi:shield-check", color="green", width=30),
                                                dmc.Text("Resilienza Rete", size="xs", c="dimmed", fw=700,
                                                         tt="uppercase")
                                            ]),
                                            dmc.Text("--", id="kpi-resilience", fw=700, size="xl", mt="xs"),
                                            dmc.Text("In attesa di dati", size="xs", c="dimmed")
                                        ]),
                                        dmc.Card(withBorder=True, shadow="xs", radius="md", p="md", children=[
                                            dmc.Group([
                                                DashIconify(icon="mdi:speedometer", color="blue", width=30),
                                                dmc.Text("Efficienza Globale", size="xs", c="dimmed", fw=700,
                                                         tt="uppercase")
                                            ]),
                                            dmc.Text("--", id="kpi-efficiency", fw=700, size="xl", mt="xs"),
                                            dmc.Text("Critica < 0.30", size="xs", c="dimmed")
                                        ]),
                                        dmc.Card(withBorder=True, shadow="xs", radius="md", p="md", children=[
                                            dmc.Group([
                                                DashIconify(icon="mdi:chart-scatter-plot", color="orange", width=30),
                                                dmc.Text("Frammentazione", size="xs", c="dimmed", fw=700,
                                                         tt="uppercase")
                                            ]),
                                            dmc.Text("--", id="kpi-fragmentation", fw=700, size="xl", mt="xs"),
                                            dmc.Text("Stato connettivo", size="xs", c="dimmed")
                                        ]),
                                    ]
                                ),

                                # 2. GRAFICO
                                dmc.Card(
                                    withBorder=True, shadow="sm", radius="md", p="lg",
                                    children=[
                                        dmc.Text("Curva di Degrado Funzionale", fw=700, size="lg", mb="md"),
                                        dcc.Loading(
                                            children=dcc.Graph(
                                                id="sim-impact-graph",
                                                style={"height": "300px"},
                                                config={"displayModeBar": False},
                                                figure={}
                                            )
                                        )
                                    ]
                                ),

                                dmc.Space(h=20),

                                # 3. BOX AI
                                html.Div(id="ai-insight-box", children=[
                                    dmc.Alert(
                                        "Carica i dati nella Home, poi avvia la simulazione qui.",
                                        title="Pronto per l'analisi",
                                        color="blue",
                                        variant="light"
                                    )
                                ])
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# 6c. LAYOUT PAGINA CONFRONTO (RIDISEGNATA E DETTAGLIATA)

layout_compare = html.Div(
    id="graph-page-container",
    style={"minHeight": "100vh", "backgroundColor": "#f8f9fa", "paddingTop": "120px", "paddingBottom": "50px"},
    children=[
        header_component,
        dmc.Container(fluid=True, px="xl", children=[
            dmc.Group([
                DashIconify(icon="mdi:clipboard-pulse", width=40, color="orange"),
                dmc.Title("Report Analisi Comparativa", order=2, c="dark")
            ], mb="lg"),

            # Qui verranno iniettati i risultati dalla callback
            dcc.Loading(
                type="cube", color="orange",
                children=html.Div(id="stats-output-container", children=[
                    # Placeholder se l'utente arriva qui senza dati
                    dmc.Alert("Carica i dati dalla Home Page > Confronto Reti per vedere i risultati.",
                              title="Nessun dato", color="gray")
                ])
            )
        ])
    ]
)

# =============================================================================
# 7. LAYOUT PRINCIPALE (CON CHATBOT GLOBALE)
# =============================================================================

# Stile del bottone galleggiante (SPOSTATO A SINISTRA)
fab_style = {
    "position": "fixed", "bottom": "30px", "left": "30px", "zIndex": 9999,
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)"
}

# Stile della finestra di chat (SPOSTATA A SINISTRA)
chat_window_style = {
    "position": "fixed", "bottom": "100px", "left": "30px", "width": "380px", "height": "500px",
    "backgroundColor": "white", "borderRadius": "12px", "boxShadow": "0 8px 24px rgba(0,0,0,0.2)",
    "zIndex": 9998, "display": "none", "flexDirection": "column", "overflow": "hidden",
    "border": "1px solid #e9ecef"
}

app.layout = dmc.MantineProvider(
    id="mantine-provider",
    forceColorScheme="light",
    theme={"colorScheme": "light"},
    children=[
        # --- STORES (Tutti quelli di prima) ---
        dcc.Store(id='store-node-data'), dcc.Store(id='store-adj-matrix'), dcc.Store(id='store-edge-coords'),
        dcc.Store(id='store-analysis-report'), dcc.Store(id='store-clicked-node'), dcc.Store(id='store-camera-state'),
        dcc.Store(id='store-valid-color-range'), dcc.Store(id='store-valid-size-scale'),
        dcc.Store(id='store-inter-comm-state', data={"active": False}),
        dcc.Store(id='store-map-a'), dcc.Store(id='store-edge-a'), dcc.Store(id='name-edge-a'),
        dcc.Store(id='store-map-b'), dcc.Store(id='store-edge-b'), dcc.Store(id='name-edge-b'),
        dcc.Store(id='store-map-c'), dcc.Store(id='store-edge-c'), dcc.Store(id='name-edge-c'),
        dcc.Store(id='store-map-d'), dcc.Store(id='store-edge-d'), dcc.Store(id='name-edge-d'),
        dcc.Store(id='visible-networks-count', data=2),

        # STORE CHAT (Memoria conversazione)
        dcc.Store(id="chat-history", data=[]),
        dcc.Store(id="chat-is-open", data=False),

        # --- COMPONENTI DASH ---
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content'),
        dcc.Download(id="download-png"),
        dcc.Download(id="download-json"),

        # --- CHATBOT UI ---
        # 1. Bottone Galleggiante (FAB)
        dmc.ActionIcon(
            DashIconify(icon="mdi:robot-excited-outline", width=30),
            id="btn-toggle-chat", size="xl", radius="xl", color="indigo", variant="filled",
            style=fab_style
        ),

        # 2. Finestra Chat
        html.Div(id="chat-window", style=chat_window_style, children=[
            # Header Chat
            dmc.Group(
                justify="space-between", p="xs", bg="indigo", c="white",
                children=[
                    dmc.Group([DashIconify(icon="mdi:brain"), dmc.Text("NeuroAssistant AI", fw=700, size="sm")]),
                    dmc.ActionIcon(DashIconify(icon="mdi:close"), id="btn-close-chat", variant="transparent", c="white")
                ]
            ),

            # Area Messaggi (Scrollabile)
            dmc.ScrollArea(
                id="chat-scroll-area",
                style={"flex": 1, "padding": "15px", "backgroundColor": "#f8f9fa"},
                type="always",
                children=dmc.Stack(id="chat-messages-container", gap="sm")
            ),

            # Input Area
            dmc.Group(
                p="xs", bg="white", style={"borderTop": "1px solid #eee"},
                children=[
                    dmc.TextInput(
                        id="chat-input", placeholder="Chiedi qualcosa sulle reti...",
                        style={"flex": 1}, radius="xl", rightSection=None
                    ),
                    dmc.ActionIcon(
                        DashIconify(icon="mdi:send"), id="btn-send-chat",
                        variant="filled", color="indigo", radius="xl", size="lg"
                    )
                ]
            )
        ])
    ]
)

# =============================================================================
# 8. CALLBACK: Routing e Navigazione
# =============================================================================

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(path):
    if path == "/graph":
        return layout_graph
    elif path == "/simulation":
        return layout_simulation
    elif path == "/compare":   # <--- NUOVA ROTTA
        return layout_compare
    else:
        return layout_upload

# =============================================================================
# 8b. CALLBACK: Gestione TEMA
# =============================================================================
@app.callback(
    Output("mantine-provider", "forceColorScheme"),
    Output("graph-page-container", "style"),
    Output("header-container", "style", allow_duplicate=True),
    Input("theme-switch", "checked"),
    prevent_initial_call=True
)
def toggle_theme(is_dark):
    if is_dark:
        theme_str = "dark"
        bg_color = "#1A1B1E"
        header_bg = "rgba(26, 27, 30, 0.85)"
    else:
        theme_str = "light"
        bg_color = "#f8f9fa"
        header_bg = "rgba(255, 255, 255, 0.85)"

    page_style = {"minHeight": "100vh", "backgroundColor": bg_color, "transition": "background-color 0.3s ease"}
    header_style = {
        "position": "fixed", "top": 0, "width": "100%", "zIndex": 2000,
        "backgroundColor": header_bg, "backdropFilter": "blur(10px)",
        "transition": "background-color 0.3s ease, opacity 0.5s ease",
    }
    return theme_str, page_style, header_style


# =============================================================================
# 8c. CALLBACK: ANIMAZIONE SPLASH SCREEN
# =============================================================================
@app.callback(
    Output("splash-overlay", "style"),
    Output("main-graph-content", "style"),
    Output("final-header-wrapper", "style"),
    Input("loading-timer", "n_intervals")
)
def handle_loading_transition(n):
    if n:
        return {"display": "none"}, \
            {"opacity": 1, "transition": "opacity 1.5s ease", "animation": "fadeInContent 1s forwards"}, \
            {"opacity": 1, "transition": "opacity 1s ease"}
    return dash.no_update, {"opacity": 0}, {"opacity": 0}


# =============================================================================
# 9. CALLBACK: Gestione Upload
# =============================================================================
# 9a. Feedback nomi file (Funziona per ENTRAMBE le schede)
@app.callback(
    Output("file-info-mapping", "children"),
    Output("file-info-edges", "children"),
    Output("file-info-mapping-sim", "children"),
    Output("file-info-edges-sim", "children"),
    Input("upload-mapping", "filename"),
    Input("upload-edges", "filename"),
    Input("upload-mapping-sim", "filename"),
    Input("upload-edges-sim", "filename")
)
def update_file_feedback(name_map, name_edge, name_map_sim, name_edge_sim):
    # Funzione helper per creare l'alert
    def make_alert(name):
        if not name: return None
        return dmc.Text(f"✓ {name}", size="xs", c="green", fw=500, mt=5)

    return make_alert(name_map), make_alert(name_edge), make_alert(name_map_sim), make_alert(name_edge_sim)


# 9b. Abilitazione Bottoni (Funziona per ENTRAMBE le schede)
@app.callback(
    Output("start-button", "disabled"),
    Output("btn-load-and-go-sim", "disabled"),
    Input("upload-mapping", "contents"),
    Input("upload-edges", "contents"),
    Input("upload-mapping-sim", "contents"),
    Input("upload-edges-sim", "contents")
)
def enable_start_buttons(map_c, edge_c, map_sim_c, edge_sim_c):
    # Abilita il bottone Visualizzazione se ci sono i file viz
    viz_disabled = not (map_c and edge_c)
    # Abilita il bottone Simulazione se ci sono i file sim
    sim_disabled = not (map_sim_c and edge_sim_c)

    return viz_disabled, sim_disabled


# =============================================================================
# 10. CALLBACK: Elaborazione Dati (Core Analysis - DUAL MODE)
# =============================================================================

@app.callback(
    Output("store-node-data", "data"),
    Output("store-adj-matrix", "data"),
    Output("store-edge-coords", "data"),
    Output("store-analysis-report", "data"),
    Output("url", "pathname"),

    # Input dai DUE bottoni di avvio
    Input("start-button", "n_clicks"),
    Input("btn-load-and-go-sim", "n_clicks"),

    # State dai file Visualizzazione
    State("upload-mapping", "contents"),
    State("upload-edges", "contents"),

    # State dai file Simulazione
    State("upload-mapping-sim", "contents"),
    State("upload-edges-sim", "contents"),

    prevent_initial_call=True
)
def run_analysis(n_viz, n_sim, map_c, edge_c, map_sim_c, edge_sim_c):
    # Determina chi ha scatenato la callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Logica di Selezione Sorgente
    target_url = "/graph"  # Default
    final_map = None
    final_edge = None

    if button_id == "start-button":
        # Modalità Visualizzazione
        final_map = map_c
        final_edge = edge_c
        target_url = "/graph"
        print(">>> Avvio Analisi per Visualizzazione 3D...")

    elif button_id == "btn-load-and-go-sim":
        # Modalità Simulazione Diretta
        final_map = map_sim_c
        final_edge = edge_sim_c
        target_url = "/simulation"  # Reindirizza qui!
        print(">>> Avvio Analisi per Simulazione AI...")

    if not final_map or not final_edge:
        return dash.no_update

    try:
        # --- PARSING DEI FILE (Identico a prima) ---
        df_map = parse_file_contents(final_map, file_type="nodes")
        df_edges = parse_file_contents(final_edge, file_type="edges")
        # Standardizzazione nomi colonne
        col_mapping_map = {c.lower(): c for c in df_map.columns}
        name_col = col_mapping_map.get('roi_name',
                                       col_mapping_map.get('name', col_mapping_map.get('label', df_map.columns[1])))
        id_col = col_mapping_map.get('roi_id', col_mapping_map.get('id', df_map.columns[0]))
        df_map = df_map.rename(columns={name_col: 'roi_name', id_col: 'roi_id'})

        # Estrazione Regioni e Coordinate
        df_map['region'] = df_map['roi_name'].apply(extract_macro_region)
        coords_list = []
        for idx, row in df_map.iterrows():
            coords = extract_coords_regex(str(row['roi_name']))
            if coords is None:
                cols_lower = [c.lower() for c in df_map.columns]
                if 'x' in cols_lower and 'y' in cols_lower and 'z' in cols_lower:
                    try:
                        coords = [float(row.get('x', row.get('X'))), float(row.get('y', row.get('Y'))),
                                  float(row.get('z', row.get('Z')))]
                    except:
                        coords = None
            coords_list.append(coords)

        df_map["coords"] = coords_list
        df_map = df_map.dropna(subset=["coords"]).reset_index(drop=True)
        coords = np.array(df_map["coords"].tolist())
        n_nodes = len(df_map)

        # Costruzione Grafo
        id_to_idx = {row["roi_id"]: i for i, (idx, row) in enumerate(df_map.iterrows())}
        adj = np.zeros((n_nodes, n_nodes))
        G = nx.Graph()

        edge_cols = {c.lower(): c for c in df_edges.columns}
        src_col = edge_cols.get('source_id', edge_cols.get('source', df_edges.columns[0]))
        tgt_col = edge_cols.get('target_id', edge_cols.get('target', df_edges.columns[1]))
        w_col = edge_cols.get('weight', df_edges.columns[2] if len(df_edges.columns) > 2 else None)

        for _, row in df_edges.iterrows():
            i = id_to_idx.get(row[src_col])
            j = id_to_idx.get(row[tgt_col])
            weight = float(row[w_col]) if w_col else 1.0
            if i is not None and j is not None:
                adj[i, j] = weight
                adj[j, i] = weight
                G.add_edge(i, j, weight=weight)

        # Calcolo Metriche Base
        part = community_louvain.best_partition(G) if len(G) > 0 else {}
        df_map["community"] = df_map.index.map(part.get).fillna(-1).astype(int)
        df_map["degree"] = pd.Series(dict(G.degree())).reindex(df_map.index).fillna(0).astype(int)
        df_map["strength"] = pd.Series(dict(G.degree(weight='weight'))).reindex(df_map.index).fillna(0.0).astype(float)

        if len(G) > 0:
            bet_dict = nx.betweenness_centrality(G, weight='weight')
            df_map["betweenness_centrality"] = df_map.index.map(bet_dict).fillna(0.0).astype(float)
        else:
            df_map["betweenness_centrality"] = 0.0

        # Normalizzazioni
        for metric in ["degree", "strength", "betweenness_centrality"]:
            max_val = df_map[metric].max()
            df_map[f"{metric}_norm"] = df_map[metric] / max_val if max_val > 0 else 0

        # Generazione Edge Coords (Solo se serve, ma lo calcoliamo cmq per consistenza)
        edge_coords = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = adj[i, j]
                if w > 0:
                    edge_coords.append({
                        'source': int(i), 'target': int(j), 'weight': w,
                        'coords': [coords[i].tolist(), coords[j].tolist()],
                        'comm_i': int(df_map.loc[i, "community"]),
                        'comm_j': int(df_map.loc[j, "community"]),
                    })

        avg_degree = df_map["degree"].mean()
        report_text = f"**Analisi Completata**\n\n* **Nodi:** {n_nodes}\n* **Connessioni:** {G.number_of_edges()}"

        return df_map.to_json(orient="split"), \
            pd.DataFrame(adj).to_json(orient="split"), \
            json.dumps(edge_coords), \
            report_text, \
            target_url  # <--- QUI AVVIENE IL REINDIRIZZAMENTO CORRETTO

    except Exception as e:
        print("ERRORE:", e)
        traceback.print_exc()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# 10b. Ricalcolo Dinamico Comunità
@app.callback(
    Output("store-node-data", "data", allow_duplicate=True),
    Input("resolution-slider", "value"),
    State("store-node-data", "data"),
    State("store-adj-matrix", "data"),
    prevent_initial_call=True
)
def recalculate_communities(resolution, node_json, adj_json):
    if not node_json or not adj_json: return dash.no_update
    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        if df.empty: return dash.no_update
        adj = pd.read_json(StringIO(adj_json), orient="split").values
        G = nx.Graph()
        rows, cols = np.where(adj > 0)
        edges = zip(rows.tolist(), cols.tolist())
        for r, c in edges:
            if r < c: G.add_edge(r, c, weight=adj[r, c])
        if G.number_of_edges() == 0: return dash.no_update
        partition = community_louvain.best_partition(G, resolution=float(resolution), weight='weight')
        df["community"] = df.index.map(partition.get).fillna(-1).astype(int)
        return df.to_json(orient="split")
    except Exception as e:
        print(f"Errore ricalcolo: {e}")
        return dash.no_update


# 10c. Aggiornamento Report Dinamico
@app.callback(
    Output("analysis-report-output", "children"),
    Input("store-node-data", "data"),
    State("store-analysis-report", "data")
)
def update_dynamic_report(node_json, initial_report):
    if not node_json: return dmc.Text("In attesa di dati...", c="dimmed", size="sm")
    ctx = callback_context
    if not ctx.triggered and initial_report and not node_json: return dmc.Text("Analisi pronta.", size="sm")
    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        n_comms = len(df["community"].unique())
        avg_deg = df['degree'].mean()
        return dmc.Group([
            dmc.Text("Analisi completata:", fw=700, c="blue", size="sm"),
            dmc.Text(f"Nodi: {len(df)}", size="sm"),
            dmc.Text("•", c="dimmed", size="sm"),
            dmc.Text(f"Comunità attuali: {n_comms}", size="sm"),
            dmc.Text("•", c="dimmed", size="sm"),
            dmc.Text(f"Grado medio: {avg_deg:.2f}", size="sm"),
            dmc.Text("(Ricalcolato)", c="dimmed", size="xs", style={"fontStyle": "italic"})
        ], gap="xs", align="center")
    except Exception as e:
        return dmc.Text("Dati non disponibili.", c="dimmed", size="sm")


# =============================================================================
# 11. CALLBACK: Controlli UI (Filtri e Selettori)
# =============================================================================

# --- 11a. NUOVA CALLBACK: VISIBILITA' FILTRO REGIONI ---
@app.callback(
    Output("region-filter-container", "style"),
    Input("brain-style-selector", "value")
)
def toggle_region_filter_visibility(style_value):
    if style_value == "regions":
        return {"display": "block"}
    return {"display": "none"}


# 11b. Popola Filtro Community E NUOVO FILTRO REGIONI
@app.callback(
    Output("community-filter", "data"),
    Output("select-comm-a", "data"),
    Output("select-comm-b", "data"),
    Output("region-filter", "data"),  # <--- NUOVO OUTPUT PER REGIONI
    Input("store-node-data", "data")
)
def populate_selectors(node_json):
    if not node_json: return [], [], [], []
    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        if "community" not in df.columns: return [], [], [], []

        # Popola Community
        comms = sorted(df["community"].unique())
        comm_options = [{"value": str(c), "label": f"Community {c}"} for c in comms]

        # Popola Regioni (Dal dataframe parsato)
        regions = sorted(df["region"].astype(str).unique())
        region_options = [{"value": r, "label": r} for r in regions]

        return comm_options, comm_options, comm_options, region_options
    except:
        return [], [], [], []


# 11c. Popola Selettore Nodo
@app.callback(
    Output("node-selector-dropdown", "data"),
    Input("store-node-data", "data")
)
def populate_node_selector(node_json):
    if not node_json: return []
    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        df["roi_name"] = df["roi_name"].astype(str)
        df_sorted = df.sort_values(by="roi_name")
        return [{"value": str(row.name), "label": str(row["roi_name"])} for idx, row in df_sorted.iterrows()]
    except:
        return []


# 11d. Renderizza Controlli Range Colore
@app.callback(
    Output("color-range-controls", "children"),
    Input("node-color-metric", "value"),
    Input("store-node-data", "data")
)
def render_color_controls(metric, node_json):
    if not metric or metric == "community" or not node_json: return html.Div()
    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        if metric not in df.columns: return html.Div()
        min_val = df[metric].min()
        max_val = df[metric].max()
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val: return html.Div()
        step_val = (max_val - min_val) / 100
        if step_val == 0: step_val = 0.1
        return dmc.Stack([
            dmc.Text(f"Range {metric.replace('_', ' ').title()}", size="xs", c="dimmed"),
            dmc.RangeSlider(id={"type": "color-slider", "index": 1}, min=min_val, max=max_val, step=step_val,
                            value=[min_val, max_val], persistence=False, marks=None, labelAlwaysOn=True, size="sm")
        ], spacing="xs", my=5)
    except Exception:
        return html.Div()


# 11e. Renderizza Controlli Dimensione
@app.callback(
    Output("size-range-controls", "children"),
    Input("node-size-metric", "value"),
    Input("store-node-data", "data")
)
def render_size_controls(metric, node_json):
    if not metric or metric == "fixed" or not node_json: return html.Div()
    try:
        return dmc.Stack([
            dmc.Text("Fattore di Scala (Esponente)", size="xs", c="dimmed"),
            dmc.Slider(id={"type": "size-slider", "index": 1}, min=0.1, max=3, step=0.1, value=1.0,
                       marks=[{"value": 1, "label": "1x"}, {"value": 2, "label": "2x"}, {"value": 3, "label": "3x"}],
                       size="sm")
        ], spacing="xs", my=5)
    except Exception:
        return html.Div()


# 11f. Sync Store Color
@app.callback(Output("store-valid-color-range", "data"), Input({"type": "color-slider", "index": dash.ALL}, "value"),
              prevent_initial_call=True)
def sync_color_range(slider_values): return slider_values[0] if slider_values else dash.no_update


# 11g. Sync Store Size
@app.callback(Output("store-valid-size-scale", "data"), Input({"type": "size-slider", "index": dash.ALL}, "value"),
              prevent_initial_call=True)
def sync_size_scale(slider_values): return slider_values[0] if slider_values else dash.no_update


# =============================================================================
# 12. CALLBACK: Interazione e Selezione
# =============================================================================

# 12a. Aggiorna Store da Click su Grafico
@app.callback(
    Output("store-clicked-node", "data"),
    Output("node-selector-dropdown", "value"),
    Input("brain-connectome-web", "clickData"),
    prevent_initial_call=True
)
def update_store_from_graph_click(cd):
    if cd and "points" in cd and cd["points"]:
        p = cd["points"][0]
        if "customdata" in p and len(p["customdata"]) > 0:
            node_index = int(p["customdata"][0])
            return node_index, str(node_index)
    return None, None


# 12b. Aggiorna Store da Dropdown
@app.callback(
    Output("store-clicked-node", "data", allow_duplicate=True),
    Input("node-selector-dropdown", "value"),
    State("store-clicked-node", "data"),
    prevent_initial_call=True
)
def update_store_from_dropdown_select(dropdown_value, current_store_val):
    if dropdown_value is None:
        new_store_val = None
    else:
        new_store_val = int(dropdown_value)
    if new_store_val == current_store_val: return dash.no_update
    return new_store_val


# 12c. Mostra Dettagli Nodo (AGGIORNATA CON REGIONE)
@app.callback(
    Output("node-details-output", "children"),
    Input("store-clicked-node", "data"),
    State("store-node-data", "data")
)
def display_node_details(clicked_index, node_json):
    if clicked_index is None or not node_json:
        return dmc.Text("Clicca su un nodo o selezionalo dalla lista.", c="dimmed", mt="md")

    try:
        idx = int(clicked_index)
        df = pd.read_json(StringIO(node_json), orient="split")
        if idx not in df.index: raise IndexError("Indice non trovato.")
        node_data = df.loc[idx]

        details_md = f"""
        **ROI:** {node_data['roi_name']}

        **Zona:** {node_data.get('region', 'N/A')}

        ---

        * **Indice (ID interno):** {idx}
        * **ID Atlas:** {node_data['roi_id']}
        * **Community:** {int(node_data['community'])}

        ---

        **METRICHE DI RETE**

        * **Grado (Degree):** {int(node_data['degree'])}
        * **Forza (Strength):** {node_data['strength']:.2f}
        * **Betweenness Centrality:** {node_data['betweenness_centrality']:.4f}
        """
        return dcc.Markdown(details_md)

    except Exception as e:
        return dmc.Text(f"Errore: Impossibile trovare i dati per l'indice selezionato.", c="red", mt="md")


# 12d. Mostra Connessioni Nodo
@app.callback(
    Output("node-connections-output", "children"),
    Input("store-clicked-node", "data"),
    Input("store-inter-comm-state", "data"),
    State("store-node-data", "data"),
    State("store-adj-matrix", "data")
)
def display_node_connections(clicked_index, inter_comm_state, node_json, adj_json):
    if clicked_index is None or not node_json or not adj_json:
        return dmc.Text("Seleziona un nodo per vedere le sue connessioni.", c="dimmed")

    try:
        idx = int(clicked_index)
        df_map = pd.read_json(StringIO(node_json), orient="split")
        df_adj = pd.read_json(StringIO(adj_json), orient="split")
        adj = df_adj.values

        node_name = df_map.loc[idx, 'roi_name']
        node_comm = int(df_map.loc[idx, 'community'])

        inter_mode_active = False
        target_comm_id = None
        if inter_comm_state and inter_comm_state.get("active"):
            inter_mode_active = True
            comm_a = int(inter_comm_state.get("comm_a"))
            comm_b = int(inter_comm_state.get("comm_b"))
            if node_comm == comm_a:
                target_comm_id = comm_b
            elif node_comm == comm_b:
                target_comm_id = comm_a

        node_connections = adj[idx]
        neighbor_indices = np.where(node_connections > 0)[0]

        total_degree = 0
        inter_degree = 0
        output_list = []

        for neighbor_idx in neighbor_indices:
            if neighbor_idx == idx: continue
            weight = node_connections[neighbor_idx]
            total_degree += 1
            neighbor_comm = int(df_map.loc[neighbor_idx, "community"])
            neighbor_name_str = df_map.loc[neighbor_idx, 'roi_name']

            is_target_conn = False
            if inter_mode_active and neighbor_comm == target_comm_id:
                inter_degree += 1
                is_target_conn = True

            if is_target_conn:
                icon_color = "orange"
                icon_name = "mdi:bullseye-arrow"
                extra_badge = dmc.Badge("MATCH", size="xs", color="orange", variant="filled", ml=5)
                row_content = dmc.Paper(
                    children=[html.Span([
                        dmc.Text(neighbor_name_str, size="sm", c="orange", fw=700, style={"display": "inline"}),
                        " ", dmc.Badge(f"C{neighbor_comm}", size="xs", variant="outline", color="dark"),
                        extra_badge, html.Br(),
                        dmc.Text(f"Peso: {weight:.3f}", size="xs", c="dimmed")
                    ])],
                    withBorder=True,
                    style={"backgroundColor": "#fff4e6", "borderColor": "#ffcc80", "padding": "8px",
                           "borderRadius": "6px"}
                )
            else:
                icon_color = "gray"
                icon_name = "mdi:arrow-right-thin"
                row_content = html.Span([
                    f"{neighbor_name_str} ",
                    dmc.Badge(f"C{neighbor_comm}", size="xs", variant="outline", color="gray"),
                    html.Br(), dmc.Text(f"Peso: {weight:.3f}", size="xs", c="dimmed")
                ], style={"padding": "5px", "display": "block"})

            output_list.append({
                "component": dmc.ListItem(row_content, icon=DashIconify(icon=icon_name, color=icon_color, width=20),
                                          style={"marginBottom": "5px"}),
                "weight": weight, "is_target": is_target_conn
            })

        output_list.sort(key=lambda x: (not x["is_target"], -x["weight"]))
        final_list_children = [item["component"] for item in output_list]

        if inter_mode_active and target_comm_id is not None:
            header = dmc.Paper(withBorder=True, p="xs", mb="md", radius="md", bg="gray.1",
                               children=[
                                   dmc.Text(f"Nodo: {node_name}", fw=700, size="sm"), dmc.Divider(my=5),
                                   dmc.Group([
                                       dmc.Stack([dmc.Text(f"{total_degree}", fw=700, size="sm"),
                                                  dmc.Text("Totali", size="xs", c="dimmed")], gap=0),
                                       dmc.Divider(orientation="vertical"),
                                       dmc.Stack([dmc.Text(f"{inter_degree}", fw=700, size="sm", c="orange"),
                                                  dmc.Text(f"Verso C{target_comm_id}", size="xs", c="orange")], gap=0)
                                   ], grow=True)
                               ])
            summary_text = dmc.Text("Le connessioni del confronto sono evidenziate in arancione:", size="xs", mb="xs",
                                    c="dimmed")
        else:
            header = dmc.Text(f"Trovate **{len(final_list_children)} connessioni** totali:", mb="md", fw=700)
            summary_text = html.Div()

        if not final_list_children: return dmc.Stack([header, dmc.Text("Nessuna connessione.", c="dimmed", size="sm")])
        return dmc.Stack([header, summary_text, dmc.List(children=final_list_children, spacing="xs", withPadding=True)])

    except Exception as e:
        traceback.print_exc()
        return dmc.Text(f"Errore visualizzazione: {str(e)}", c="red")


# =============================================================================
# 13. CALLBACK: Gestione UI Ausiliaria
# =============================================================================

# 13a. Toggle Pannello Info (LIGHT THEME)
@app.callback(
    Output("info-panel-popup", "style"),
    Output("info-panel-state", "data"),
    Input("info-fab-btn", "n_clicks"),
    Input("brain-connectome-web", "clickData"),
    State("info-panel-state", "data"),
    prevent_initial_call=True
)
def toggle_info_panel(n_clicks, graph_click, is_open):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == "info-fab-btn":
        new_state = not is_open
    elif triggered_id == "brain-connectome-web" and is_open:
        new_state = False
    else:
        return dash.no_update, dash.no_update
    new_style = {
        "width": "350px", "backgroundColor": "rgba(255, 255, 255, 0.95)", "color": "#333",
        "display": "block" if new_state else "none", "border": "1px solid #e9ecef"
    }
    return new_style, new_state


# 13b. CALLBACK: Logica Pulsante Inter-Comunità (CLICK)
@app.callback(
    Output("store-inter-comm-state", "data"),
    Output("btn-inter-comm", "variant"),
    Output("btn-inter-comm", "children"),
    Input("btn-inter-comm", "n_clicks"),
    State("select-comm-a", "value"),
    State("select-comm-b", "value"),
    State("store-inter-comm-state", "data"),
    prevent_initial_call=True
)
def toggle_inter_comm_filter(n_clicks, comm_a, comm_b, current_state):
    if not n_clicks: return dash.no_update, dash.no_update, dash.no_update
    if current_state is None: current_state = {"active": False}
    is_active = current_state.get("active", False)
    if is_active:
        return {"active": False}, "outline", "Mostra A ↔ B"
    else:
        if comm_a is not None and comm_b is not None and comm_a != comm_b:
            return {"active": True, "comm_a": comm_a, "comm_b": comm_b}, "filled", "Reset Vista"
        else:
            return dash.no_update, "outline", "Seleziona 2 Comunità diverse"


# 13c. CALLBACK: Popola Card Inter-Community (Analisi Macro A vs B)
@app.callback(
    Output("inter-community-card", "style"),
    Output("inter-community-content", "children"),
    Input("store-inter-comm-state", "data"),
    State("store-node-data", "data"),
    State("store-adj-matrix", "data")
)
def update_inter_community_card(inter_comm_state, node_json, adj_json):
    hidden_style = {"display": "none"}
    if not inter_comm_state or not inter_comm_state.get("active"): return hidden_style, dash.no_update
    if not node_json or not adj_json: return hidden_style, dash.no_update
    try:
        comm_a = int(inter_comm_state.get("comm_a"))
        comm_b = int(inter_comm_state.get("comm_b"))
        df_map = pd.read_json(StringIO(node_json), orient="split")
        df_adj = pd.read_json(StringIO(adj_json), orient="split")
        adj = df_adj.values

        indices_a = df_map[df_map["community"] == comm_a].index.tolist()
        indices_b = df_map[df_map["community"] == comm_b].index.tolist()

        total_edges = 0
        total_weight = 0.0
        strongest_link = {"weight": 0, "pair": "N/A"}

        for i in indices_a:
            for j in indices_b:
                w = adj[i, j]
                if w > 0:
                    total_edges += 1
                    total_weight += w
                    if w > strongest_link["weight"]:
                        name_i = df_map.loc[i, "roi_name"]
                        name_j = df_map.loc[j, "roi_name"]
                        strongest_link = {"weight": w, "pair": f"{name_i} ↔ {name_j}"}

        content = dmc.Stack([
            dmc.Text(f"Confronto: Community {comm_a} vs {comm_b}", size="xs", c="dimmed"),
            dmc.Group([
                dmc.Stack([dmc.Text(f"{total_edges}", fw=700, size="xl", lh=1),
                           dmc.Text("Archi Totali", size="xs", c="dimmed")], gap=2),
                dmc.Divider(orientation="vertical"),
                dmc.Stack([dmc.Text(f"{total_weight:.1f}", fw=700, size="xl", lh=1),
                           dmc.Text("Peso Totale", size="xs", c="dimmed")], gap=2),
            ], grow=True, mt="sm"),
            dmc.Divider(my="sm"),
            dmc.Text("Collegamento più forte:", size="xs", fw=700),
            dmc.Text(strongest_link["pair"], size="xs"),
            dmc.Badge(f"Peso: {strongest_link['weight']:.2f}", color="orange", variant="light")
        ])
        visible_style = {"display": "block", "borderColor": "#FFC107", "borderWidth": "2px"}
        return visible_style, content
    except Exception as e:
        print(f"Err inter card: {e}")
        return hidden_style, dash.no_update


# =============================================================================
# 14. CALLBACK: Rendering Grafico (AGGIORNAMENTO AUTOMATICO)
# =============================================================================

@app.callback(
    Output("brain-connectome-web", "figure"),

    # --- INPUTS (Scatenano l'aggiornamento automatico) ---
    Input("update-graph-btn", "n_clicks"),
    Input("store-inter-comm-state", "data"),
    Input("theme-switch", "checked"),

    # -- Filtri che triggerano update --
    Input("brain-style-selector", "value"),  # <--- NUOVO INPUT: SELETTORE STILE CERVELLO
    Input("region-filter", "value"),
    Input("community-filter", "value"),
    Input("weight-threshold", "value"),
    Input("toggle-brain", "checked"),
    Input("toggle-default-edges", "checked"),
    Input("toggle-highlight-edges", "checked"),
    Input("node-color-metric", "value"),
    Input("node-size-metric", "value"),
    Input("edge-width-slider", "value"),
    Input("brain-opacity-slider", "value"),
    Input("edge-opacity-slider", "value"),

    # --- STATES ---
    State("store-clicked-node", "data"),
    State("store-node-data", "data"),
    State("store-adj-matrix", "data"),
    State("store-edge-coords", "data"),
    State("store-camera-state", "data"),
    State("store-valid-color-range", "data"),
    State("store-valid-size-scale", "data"),
)
def update_figure(
        # --- INPUTS ---
        n_clicks_update, inter_comm_state, is_dark_theme,
        brain_style, selected_regions, selected_communities, weight_thr,  # brain_style aggiunto
        show_brain, show_edges, show_highlight,
        color_metric, size_metric, default_edge_width, brain_opacity, edge_opacity,

        # --- STATES ---
        clicked_node,
        node_json, adj_json, edge_coords_json, camera_state, color_range_val_store, size_scale_val_store
):
    text_color = "white" if is_dark_theme else "black"
    layout_copy = copy.deepcopy(graph_layout_props)
    layout_copy["title_font_color"] = text_color
    if "legend" in layout_copy: layout_copy["legend"]["font"] = dict(color=text_color, size=12)

    if not node_json or not adj_json or not edge_coords_json:
        empty_fig = go.Figure()
        layout_copy["scene"]["xaxis"]["visible"] = False
        layout_copy["scene"]["yaxis"]["visible"] = False
        layout_copy["scene"]["zaxis"]["visible"] = False
        layout_copy["showlegend"] = False
        empty_fig.update_layout(**layout_copy)
        return empty_fig

    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        edge_coords_list = json.loads(edge_coords_json)
        n_nodes = len(df)
        if n_nodes == 0: raise ValueError("Empty Dataframe")
    except Exception as e:
        print(f"Errore dati grafico: {e}")
        return go.Figure()

    # --- LOGICA DI FILTRO REGIONI ---
    # Se siamo in modalità "regions", applichiamo il filtro.
    # Se siamo in modalità "gray" (intero), ignoriamo il filtro regioni per i NODI e per la MESH,
    # per garantire che si veda "tutto".

    valid_indices = set(df.index)
    if brain_style == "regions" and selected_regions:
        # Se ho selezionato regioni e sono in modalita regioni, tengo solo i nodi che matchano
        valid_indices = set(df[df['region'].isin(selected_regions)].index)
    # -------------------------------------------

    inter_mode_active = False
    comm_a_id, comm_b_id = -1, -1
    target_comms_inter = set()

    if inter_comm_state and inter_comm_state.get("active"):
        try:
            inter_mode_active = True
            comm_a_id = int(inter_comm_state.get("comm_a"))
            comm_b_id = int(inter_comm_state.get("comm_b"))
            target_comms_inter = {comm_a_id, comm_b_id}
        except:
            inter_mode_active = False

    visible_comms = target_comms_inter if inter_mode_active else (
        set(int(c) for c in selected_communities) if selected_communities else None)

    # 5. COSTRUZIONE FIGURA
    fig = go.Figure()

    # --- DISEGNO CERVELLO ---
    if show_brain:
        # LOGICA SCELTA TRA GRAY E REGIONS

        if brain_style == "gray":
            # MODALITÀ INTERO GRIGIO: Usa le mesh fsaverage (WHOLE_BRAIN_TRACES)
            if WHOLE_BRAIN_TRACES:
                for trace in WHOLE_BRAIN_TRACES:
                    t = copy.deepcopy(trace)
                    t.opacity = brain_opacity
                    fig.add_trace(t)
            # Se fsaverage fallisce, prova a usare le mesh regioni ma forzando il colore grigio (fallback)
            elif REGION_MESHES:
                for zone_name, trace in REGION_MESHES.items():
                    t = copy.deepcopy(trace)
                    t.color = "gray"
                    t.opacity = brain_opacity
                    fig.add_trace(t)

        elif brain_style == "regions":
            # MODALITÀ REGIONI COLORATE: Usa le mesh AAL (REGION_MESHES)
            if REGION_MESHES:
                # Se l'utente ha selezionato regioni specifiche, mostra solo quelle
                zones_to_draw = selected_regions if selected_regions else REGION_MESHES.keys()
                for zone_name in zones_to_draw:
                    if zone_name in REGION_MESHES:
                        trace = copy.deepcopy(REGION_MESHES[zone_name])
                        trace.opacity = brain_opacity
                        fig.add_trace(trace)
            else:
                # Se non ho le librerie per le regioni, mostro intero con warning (o nulla)
                pass

    # --- FILTRO DATI PLOTTING ---
    # Applica filtro community e filtro region (se attivo)
    if visible_comms:
        comm_indices = set(df[df["community"].isin(visible_comms)].index)
        valid_indices = valid_indices.intersection(comm_indices)

    df_plot = df[df.index.isin(valid_indices)]

    # Calcolo scale colori/dimensioni
    if color_metric != 'community' and color_metric not in df.columns: color_metric = 'degree'
    metric_min = df[color_metric].min() if color_metric in df.columns else 0
    metric_max = df[color_metric].max() if color_metric in df.columns else 1

    use_store_color = (color_range_val_store and len(color_range_val_store) == 2)
    if use_store_color:
        lower_bound, upper_bound = color_range_val_store
    else:
        lower_bound, upper_bound = metric_min, metric_max

    size_scale_val = size_scale_val_store if size_scale_val_store is not None else 1.0
    min_size, max_size = 4, 15
    size_data = [min_size] * n_nodes
    if size_metric == 'fixed':
        size_data = [8] * n_nodes
    elif size_metric and f"{size_metric}_norm" in df.columns:
        norm_data = df[f"{size_metric}_norm"].pow(size_scale_val)
        max_norm = norm_data.max()
        if max_norm > 0: norm_data = norm_data / max_norm
        size_data = (norm_data * (max_size - min_size) + min_size).tolist()

    # --- DISEGNO NODI ---
    if not df_plot.empty:
        if color_metric == 'community':
            colors = px.colors.qualitative.G10
            for c in sorted(df_plot["community"].unique()):
                sub_df = df_plot[df_plot["community"] == c]
                safe_c_idx = int(c) % len(colors)
                trace_color = colors[safe_c_idx]
                comm_indices = sub_df.index.tolist()

                fig.add_trace(go.Scatter3d(
                    x=sub_df["coords"].apply(lambda x: x[0]),
                    y=sub_df["coords"].apply(lambda x: x[1]),
                    z=sub_df["coords"].apply(lambda x: x[2]),
                    mode="markers",
                    marker=dict(size=[size_data[i] for i in comm_indices], color=trace_color, opacity=0.9,
                                line=dict(width=0)),
                    text=sub_df["roi_name"],
                    customdata=np.stack((sub_df.index, sub_df['region']), axis=-1),  # Custom data esteso
                    hoverinfo="text", name=f"Comm. {c}",
                    hovertemplate='%{text}<br>Zone: %{customdata[1]}<extra></extra>'
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=df_plot["coords"].apply(lambda x: x[0]),
                y=df_plot["coords"].apply(lambda x: x[1]),
                z=df_plot["coords"].apply(lambda x: x[2]),
                mode="markers",
                marker=dict(
                    size=[size_data[i] for i in df_plot.index],
                    color=df_plot[color_metric],
                    colorscale='Viridis',
                    cmin=lower_bound, cmax=upper_bound,
                    colorbar=dict(title=dict(text=color_metric, font=dict(color=text_color)),
                                  tickfont=dict(color=text_color), thickness=10, len=0.5, x=1.0, y=0.5),
                    opacity=1
                ),
                text=df_plot["roi_name"],
                customdata=np.stack((df_plot.index, df_plot['region']), axis=-1),
                hoverinfo="text", name="Nodi",
                hovertemplate='%{text}<br>Zone: %{customdata[1]}<extra></extra>'
            ))

    # --- DISEGNO ARCHI ---
    x, y, z, xh, yh, zh = [], [], [], [], [], []
    sel = {clicked_node} if clicked_node is not None else set()

    for e in edge_coords_list:
        if e['weight'] < weight_thr: continue
        source_idx = int(e['source'])
        target_idx = int(e['target'])
        if source_idx >= n_nodes or target_idx >= n_nodes: continue

        # Filtro Region (Importante: entrambi i nodi devono essere validi)
        if source_idx not in valid_indices or target_idx not in valid_indices: continue

        ci = int(df.at[source_idx, "community"])
        cj = int(df.at[target_idx, "community"])

        should_draw = False
        if inter_mode_active:
            if (ci == comm_a_id and cj == comm_b_id) or (ci == comm_b_id and cj == comm_a_id): should_draw = True
        else:
            if visible_comms:
                if ci in visible_comms and cj in visible_comms: should_draw = True
            else:
                should_draw = True  # Se non ci sono filtri community attivi, disegna (poiché filtro region è già passato)

        if not should_draw: continue

        c = e['coords']
        is_hi = (source_idx in sel or target_idx in sel)
        if show_highlight and is_hi:
            xh.extend([c[0][0], c[1][0], None])
            yh.extend([c[0][1], c[1][1], None])
            zh.extend([c[0][2], c[1][2], None])
        elif show_edges:
            x.extend([c[0][0], c[1][0], None])
            y.extend([c[0][1], c[1][1], None])
            z.extend([c[0][2], c[1][2], None])

    final_edge_opacity = 0.8 if inter_mode_active else edge_opacity
    edge_color = "#FFC107" if inter_mode_active else "#29B6F6"
    edge_w = default_edge_width * 2 if inter_mode_active else default_edge_width

    if x: fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=edge_color, width=edge_w), opacity=final_edge_opacity,
                     hoverinfo="none", name="Connessioni"))
    if xh: fig.add_trace(
        go.Scatter3d(x=xh, y=yh, z=zh, mode="lines", line=dict(color="#FF5252", width=5), opacity=1, hoverinfo="none",
                     name="Selezione"))

    if camera_state: fig.update_layout(scene_camera=camera_state)
    fig.update_layout(**layout_copy)
    fig.update_layout(showlegend=True, uirevision="constant")

    return fig


# =============================================================================
# 14b. CALLBACK: Aggiornamento Istogrammi (Dashboard Inferiore)
# =============================================================================
@app.callback(
    Output("hist-degree", "figure"),
    Output("hist-community", "figure"),
    Output("hist-strength", "figure"),
    Input("store-node-data", "data"),
    Input("theme-switch", "checked")
)
def update_histograms(node_json, is_dark):
    empty_layout = dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0))

    if not node_json: return go.Figure(layout=empty_layout), go.Figure(layout=empty_layout), go.Figure(
        layout=empty_layout)

    try:
        df = pd.read_json(StringIO(node_json), orient="split")
        text_color = "#C1C2C5" if is_dark else "#333"
        grid_color = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"
        bar_color_1 = "#4dabf7"
        bar_color_2 = "#ff922b"
        bar_color_3 = "#20c997"

        def create_layout(title, x_title):
            return dict(title=dict(text=title, font=dict(size=12, color=text_color), x=0.5, xanchor='center'),
                        xaxis=dict(title=x_title, showgrid=False, color=text_color, title_font=dict(size=10)),
                        yaxis=dict(showgrid=True, gridcolor=grid_color, showticklabels=True, color=text_color,
                                   title_font=dict(size=10)), paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=30, r=10, b=30, t=30), showlegend=False)

        fig_deg = go.Figure(data=[go.Histogram(x=df["degree"], marker_color=bar_color_1, opacity=0.8)])
        fig_deg.update_layout(create_layout("Distribuzione Grado", "Grado"))

        comm_counts = df["community"].value_counts().sort_index()
        fig_comm = go.Figure(
            data=[go.Bar(x=comm_counts.index, y=comm_counts.values, marker_color=bar_color_2, opacity=0.8)])
        fig_comm.update_layout(create_layout("Dimensione Comunità", "Community ID"))
        fig_comm.update_xaxes(type='category')

        fig_str = go.Figure(data=[go.Histogram(x=df["strength"], marker_color=bar_color_3, opacity=0.8, nbinsx=20)])
        fig_str.update_layout(create_layout("Distribuzione Forza", "Strength"))

        return fig_deg, fig_comm, fig_str
    except:
        return go.Figure(layout=empty_layout), go.Figure(layout=empty_layout), go.Figure(layout=empty_layout)


# =============================================================================
# 15. CALLBACK: Salvataggio Stato (Camera)
# =============================================================================

# 15a. Salva posizione camera
@app.callback(
    Output("store-camera-state", "data"),
    Input("brain-connectome-web", "relayoutData"),
    State("store-camera-state", "data")
)
def save_camera(relayout, current):
    if relayout and "scene.camera" in relayout:
        return relayout["scene.camera"]
    return current


# =============================================================================
# 18. CALLBACK: LOGICA SIMULAZIONE AI (VERSIONE BLINDATA & DEBUG)
# =============================================================================

import numpy as np
import random


# =============================================================================
# 18. CALLBACK: LOGICA SIMULAZIONE AI (VERSIONE FINALE CON INDICATORI VISIVI)
# =============================================================================

# 18a. Popola il menu a tendina evidenziando i nodi CRITICI (🔴)
@app.callback(
    Output("sim-target-node", "data"),
    Input("store-node-data", "data"),
    State("store-adj-matrix", "data")  # <--- AGGIUNTO: Serve per calcolare chi è importante
)
def populate_simulation_dropdown(node_json, adj_json):
    if not node_json: return []

    try:
        # 1. Caricamento Dati
        df = pd.read_json(StringIO(node_json), orient="split")

        # Se abbiamo la matrice, calcoliamo i nodi critici
        critical_indices = set()

        if adj_json:
            adj = pd.read_json(StringIO(adj_json), orient="split").values
            G = nx.Graph()
            n_nodes = len(df)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adj[i, j] > 0:
                        G.add_edge(i, j, weight=adj[i, j])

            # Calcolo Centralità (Hub Detection)
            if len(G) > 0:
                centrality = nx.betweenness_centrality(G, weight='weight')
                # Identifica il Top 15% dei nodi più importanti
                threshold = np.percentile(list(centrality.values()), 85)
                critical_indices = {k for k, v in centrality.items() if v >= threshold}

        # 2. Creazione Lista con Etichette Visive
        options = []
        for idx, row in df.iterrows():
            idx_int = int(idx)
            original_name = str(row["roi_name"])

            if idx_int in critical_indices:
                # Nodo Critico: Aggiungi Pallino Rosso e sposta in alto
                label = f"🔴 {original_name} (HUB CRITICO)"
                # Usiamo una tupla per ordinare: (0, nome) mette i critici prima degli (1, nome)
                sort_key = (0, original_name)
            else:
                label = original_name
                sort_key = (1, original_name)

            options.append({
                "value": str(idx),
                "label": label,
                "sort_key": sort_key  # Chiave temporanea per ordinamento
            })

        # 3. Ordinamento: Prima i rossi, poi alfabetico
        options.sort(key=lambda x: x["sort_key"])

        # Pulizia chiave temporanea
        final_options = [{"value": o["value"], "label": o["label"]} for o in options]

        return final_options

    except Exception as e:
        print("Errore popolamento dropdown:", e)
        return []

# 18b. CORE ENGINE: Esegue la simulazione
@app.callback(
    Output("sim-impact-graph", "figure"),
    Output("ai-insight-box", "children"),
    Output("kpi-resilience", "children"),
    Output("kpi-efficiency", "children"),
    Output("kpi-fragmentation", "children"),
    Input("btn-run-simulation", "n_clicks"),
    State("sim-scenario-select", "value"),
    State("sim-target-node", "value"),
    State("sim-severity-slider", "value"),
    State("store-node-data", "data"),
    State("store-adj-matrix", "data"),
    prevent_initial_call=True
)
def run_real_ai_simulation(n_clicks, scenario, target_node_idx, severity, node_json, adj_json):
    # Controllo Dati Iniziale
    if not node_json or not adj_json:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    print(f"\n--- AVVIO SIMULAZIONE (Click {n_clicks}) ---")
    print(f"Scenario: {scenario} | Nodo Target ID: {target_node_idx} | Severità: {severity}%")

    try:
        # 1. Ricostruzione Grafo
        df = pd.read_json(StringIO(node_json), orient="split")
        adj = pd.read_json(StringIO(adj_json), orient="split").values
        G_original = nx.Graph()
        n_nodes = len(df)

        weights_list = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = adj[i, j]
                if w > 0:
                    G_original.add_edge(i, j, weight=w)
                    weights_list.append(w)

        initial_efficiency = nx.global_efficiency(G_original)
        print(f"Efficienza Iniziale: {initial_efficiency}")

        # 2. Applicazione Danno
        G_damaged = G_original.copy()
        damaged_desc = ""
        error_flag = False

        # --- LOGICA ICTUS ---
        if scenario == "stroke":
            if target_node_idx is None:
                damaged_desc = "⚠️ ERRORE: Nessun nodo selezionato! Seleziona un nodo dal menu."
                error_flag = True
            else:
                idx = int(target_node_idx)
                node_name = df.loc[idx, "roi_name"]

                # Se severità è alta (>80%), rimuoviamo il nodo.
                # Altrimenti rimuoviamo archi casuali del nodo.
                if severity >= 80:
                    G_damaged.remove_node(idx)
                    damaged_desc = f"Ictus: Rimozione totale nodo **{node_name}**."
                else:
                    edges = list(G_damaged.edges(idx))
                    if not edges:
                        damaged_desc = f"Il nodo **{node_name}** era già isolato."
                    else:
                        # Rimuoviamo almeno 1 arco se la severità è > 0
                        k_cut = max(1, int(len(edges) * (severity / 100.0)))
                        edges_to_remove = random.sample(edges, k_cut)
                        G_damaged.remove_edges_from(edges_to_remove)
                        damaged_desc = f"Ictus parziale: rotti {k_cut} collegamenti su {len(edges)} di **{node_name}**."

        # --- LOGICA RANDOM ---
        elif scenario == "random":
            # Rimuoviamo nodi in base alla severità (Max 50% dei nodi totali per non svuotare tutto)
            percent_remove = (severity / 100.0) * 0.5
            k_remove = int(n_nodes * percent_remove)
            k_remove = max(1, k_remove) if severity > 10 else k_remove  # Assicura almeno 1 rimozione se severity > 10

            nodes = list(G_damaged.nodes())
            if len(nodes) >= k_remove:
                nodes_to_remove = random.sample(nodes, k_remove)
                G_damaged.remove_nodes_from(nodes_to_remove)
                damaged_desc = f"Attacco Random: rimossi {k_remove} nodi a caso."
            else:
                damaged_desc = "Grafo troppo piccolo per rimozione random."

            # --- LOGICA ALZHEIMER (Neurodegenerazione) ---
        elif scenario == "alzheimer":
            if not weights_list or min(weights_list) == max(weights_list):
                all_edges = list(G_damaged.edges())
                if all_edges:
                    # Calcoliamo quanti archi tagliare in base allo slider (severity)
                    k_cut = int(len(all_edges) * (severity / 100.0))

                if k_cut > 0:
                    edges_to_remove = random.sample(all_edges, k_cut)
                    G_damaged.remove_edges_from(edges_to_remove)

                damaged_desc = f"Degenerazione (Grafo Binario): rimossi {k_cut} collegamenti casuali ({severity}%)."
            else:
                damaged_desc = "Grafo vuoto, impossibile simulare."

        # Caso B: Grafo Pesato (Normale)
        else:
            # Calcoliamo la soglia reale basata sulla distribuzione dei pesi
            threshold = np.percentile(weights_list, severity)

            # Identifica gli archi sotto la soglia
            edges_to_cut = [(u, v) for u, v, d in G_damaged.edges(data=True) if d['weight'] <= threshold]

            # PROTEZIONE CRASH: Evitiamo di cancellare il 100% degli archi se la soglia è troppo alta
            # (Succede se molti archi hanno lo stesso peso massimo)
            total_edges = G_damaged.number_of_edges()
            if len(edges_to_cut) == total_edges and severity < 100:
                # Se stiamo per cancellare tutto ma lo slider non è al 100%,
                # salviamo un 10% di archi a caso per non azzerare il grafico
                edges_to_cut = edges_to_cut[:int(total_edges * 0.9)]

            G_damaged.remove_edges_from(edges_to_cut)
            damaged_desc = f"Neurodegenerazione: persi {len(edges_to_cut)} collegamenti deboli (Peso < {threshold:.2f})."

        # 3. Calcolo KPI Finali
        final_efficiency = nx.global_efficiency(G_damaged)
        print(f"Efficienza Finale: {final_efficiency}")

        # Gestione divisione per zero
        if initial_efficiency > 0:
            resilience_val = (final_efficiency / initial_efficiency) * 100
            delta_eff = ((final_efficiency - initial_efficiency) / initial_efficiency) * 100
        else:
            resilience_val = 0
            delta_eff = 0

        resilience_text = f"{resilience_val:.1f}%"

        n_components = nx.number_connected_components(G_damaged)
        if n_components == 1:
            frag_text = "Intatta (1)"
            frag_color = "green"
        elif n_components < 10:
            frag_text = f"Lieve ({n_components})"
            frag_color = "orange"
        else:
            frag_text = f"Critica ({n_components})"
            frag_color = "red"

        # 4. Insight AI
        try:
            bet_cen = nx.betweenness_centrality(G_original, weight='weight')
            critical_node_idx = max(bet_cen, key=bet_cen.get)
            critical_node_name = df.loc[critical_node_idx, "roi_name"]
        except:
            critical_node_name = "N/A"

        # 5. Grafico Aggiornato
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Baseline", "Post-Simulazione"],
            y=[initial_efficiency, final_efficiency],
            marker_color=["#20c997", "#fa5252"],
            text=[f"{initial_efficiency:.3f}", f"{final_efficiency:.3f}"],
            textposition='auto',
            width=0.5
        ))

        fig.add_hline(y=initial_efficiency * 0.5, line_dash="dot", line_color="gray", annotation_text="Soglia Critica")

        fig.update_layout(
            title="Efficienza Globale",
            yaxis=dict(title="Efficienza (0-1)", range=[0, max(initial_efficiency * 1.2, 0.1)]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=50, b=40)
        )

        # 6. Messaggio Dinamico (Con gestione Errore Utente)
        alert_color = "red" if error_flag else ("red" if delta_eff < -15 else "orange")

        ai_message = dmc.Stack([
            dmc.Alert(
                title="Diagnosi",
                color=alert_color,
                variant="light",
                children=[
                    dmc.Text(f"{damaged_desc}"),
                    dmc.Divider(my=5),
                    dmc.Text(f"Variazione: {delta_eff:.2f}%", fw=700)
                ]
            ),
            dmc.Alert(
                title="Analisi Strutturale",
                color="blue",
                variant="outline",
                children=[
                    dmc.Text(f"Hub Critico rilevato: {critical_node_name}"),
                ]
            )
        ])

        return fig, ai_message, resilience_text, f"{final_efficiency:.3f}", dmc.Text(frag_text, c=frag_color)

    except Exception as e:
        print("!!! CRASH SIMULAZIONE:", e)
        traceback.print_exc()
        return dash.no_update, dmc.Alert(f"Errore: {e}", color="red"), "Err", "Err", "Err"


# =============================================================================
# XX. CALLBACK INTERMEDIA: SALVATAGGIO DATI E CAMBIO PAGINA
# =============================================================================

# 1. Feedback Visivo (Spunte Verdi)
@app.callback(
    [Output(f"fb-map-{x}", "children") for x in ['a', 'b', 'c', 'd']] +
    [Output(f"fb-edge-{x}", "children") for x in ['a', 'b', 'c', 'd']],
    [Input(f"home-map-{x}", "filename") for x in ['a', 'b', 'c', 'd']] +
    [Input(f"home-edge-{x}", "filename") for x in ['a', 'b', 'c', 'd']]
)
def update_home_compare_filenames(*args):
    return [f"✓ {n}" if n else "" for n in args]


# 2. Salvataggio e Routing
@app.callback(
    # Output DATI
    Output("store-map-a", "data"), Output("store-edge-a", "data"),
    Output("store-map-b", "data"), Output("store-edge-b", "data"),
    Output("store-map-c", "data"), Output("store-edge-c", "data"),
    Output("store-map-d", "data"), Output("store-edge-d", "data"),
    # Output NOMI
    Output("name-edge-a", "data"), Output("name-edge-b", "data"),
    Output("name-edge-c", "data"), Output("name-edge-d", "data"),
    # Output URL
    Output("url", "pathname", allow_duplicate=True),

    # Input
    Input("btn-launch-compare", "n_clicks"),

    # State (Contenuti)
    State("home-map-a", "contents"), State("home-edge-a", "contents"),
    State("home-map-b", "contents"), State("home-edge-b", "contents"),
    State("home-map-c", "contents"), State("home-edge-c", "contents"),
    State("home-map-d", "contents"), State("home-edge-d", "contents"),

    # State (Nomi File)
    State("home-edge-a", "filename"), State("home-edge-b", "filename"),
    State("home-edge-c", "filename"), State("home-edge-d", "filename"),

    prevent_initial_call=True
)
def launch_comparison(n, ma, ea, mb, eb, mc, ec, md, ed, na, nb, nc, nd):
    if not n: return [dash.no_update] * 13

    # Controllo: Almeno A e B devono esserci
    if not (ma and ea and mb and eb):
        # Opzionale: potresti ritornare un Alert qui, ma per ora non facciamo nulla
        return [dash.no_update] * 13

    # Ritorna: 8 contenuti + 4 nomi + 1 url
    return ma, ea, mb, eb, mc, ec, md, ed, na, nb, nc, nd, "/compare"

# =============================================================================
# 19. CALLBACK: LOGICA CONFRONTO RETI (COMPARATIVE CONNECTOMICS)
# =============================================================================

# 19a. Feedback visivo caricamento file confronto
@app.callback(
    Output("info-map-a", "children"), Output("info-edge-a", "children"),
    Output("info-map-b", "children"), Output("info-edge-b", "children"),
    Input("up-map-a", "filename"), Input("up-edge-a", "filename"),
    Input("up-map-b", "filename"), Input("up-edge-b", "filename")
)
def update_compare_feedback(ma, ea, mb, eb):
    def mk_icon(n): return DashIconify(icon="mdi:check-bold", color="green") if n else None

    return mk_icon(ma), mk_icon(ea), mk_icon(mb), mk_icon(eb)


# =============================================================================
# 19b. CORE ENGINE MULTI-RETE: VERSIONE DEFINITIVA (Nomi Reali + Top 10 Nodi)
# =============================================================================
@app.callback(
    Output("stats-output-container", "children", allow_duplicate=True),
    Input("url", "pathname"),

    # Dati dagli Store
    State("store-map-a", "data"), State("store-edge-a", "data"),
    State("store-map-b", "data"), State("store-edge-b", "data"),
    State("store-map-c", "data"), State("store-edge-c", "data"),
    State("store-map-d", "data"), State("store-edge-d", "data"),

    # Nomi File
    State("name-edge-a", "data"), State("name-edge-b", "data"),
    State("name-edge-c", "data"), State("name-edge-d", "data"),

    prevent_initial_call='initial_duplicate'
)
def render_comparison_results(path, ma, ea, mb, eb, mc, ec, md, ed, na, nb, nc, nd):
    if path != "/compare": return dash.no_update

    # Verifica Dati Minimi
    if not (ma and ea and mb and eb):
        return dmc.Center(dmc.Alert("Dati mancanti. Torna alla Home.", color="red", variant="outline"), pt=50)

    print(">>> [DEBUG] Analisi Comparativa: Calcolo Metriche e Nomi Anatomici...")

    try:
        # --- 1. SETUP ---
        raw_inputs = [
            {'m': ma, 'e': ea, 'n': na, 'color': '#228be6', 'id': 'A'},
            {'m': mb, 'e': eb, 'n': nb, 'color': '#fd7e14', 'id': 'B'},
            {'m': mc, 'e': ec, 'n': nc, 'color': '#40c057', 'id': 'C'},
            {'m': md, 'e': ed, 'n': nd, 'color': '#fa5252', 'id': 'D'}
        ]

        networks = []
        errors = []

        # --- 2. ELABORAZIONE DATI ---
        for item in raw_inputs:
            if item['m'] and item['e']:
                try:
                    # A. Parsing
                    df_m = parse_file_contents(item['m'], file_type="nodes")
                    try:
                        df_e = parse_file_contents(item['e'], file_type="edges")
                    except:
                        raise ValueError("Errore lettura file Edges")

                    if df_m is None or df_e is None: raise ValueError("File non validi o vuoti")

                    # B. Nome Rete
                    clean_name = item['n'].rsplit('.', 1)[0] if item['n'] else f"Rete {item['id']}"
                    clean_name = clean_name[:20] + "..." if len(clean_name) > 22 else clean_name

                    # C. Mappatura Nomi (IMPORTANTE PER IL GRAFICO)
                    # Cerchiamo la colonna del nome anatomico
                    name_col = 'roi_name' if 'roi_name' in df_m.columns else df_m.columns[1]
                    id_col = 'roi_id' if 'roi_id' in df_m.columns else df_m.columns[0]

                    # Creiamo un dizionario {Indice_Grafo: "Nome Anatomico"}
                    # NetworkX usa indici 0,1,2... noi mappiamo questi indici ai nomi reali
                    node_names_map = {}
                    id_map = {}  # Mappa ID file -> Indice 0,1,2

                    for i, row in df_m.iterrows():
                        real_id = row[id_col]
                        real_name = str(row[name_col])
                        id_map[real_id] = i
                        node_names_map[i] = real_name

                    # D. Costruzione Grafo
                    G = nx.Graph()
                    src_col = 'source' if 'source' in df_e.columns else df_e.columns[0]
                    tgt_col = 'target' if 'target' in df_e.columns else df_e.columns[1]
                    w_col = 'weight' if 'weight' in df_e.columns else None

                    for _, row in df_e.iterrows():
                        try:
                            u = id_map.get(row[src_col])
                            v = id_map.get(row[tgt_col])
                            w = float(row[w_col]) if w_col else 1.0
                            if u is not None and v is not None: G.add_edge(u, v, weight=w)
                        except:
                            continue

                    # E. Calcolo Metriche
                    n_nodes = G.number_of_nodes()
                    dens = nx.density(G)
                    eff = nx.global_efficiency(G)

                    if nx.is_connected(G):
                        avg_path = nx.average_shortest_path_length(G, weight='weight')
                    else:
                        Gc = G.subgraph(max(nx.connected_components(G), key=len))
                        avg_path = nx.average_shortest_path_length(Gc, weight='weight')

                    clust = nx.average_clustering(G, weight='weight')
                    trans = nx.transitivity(G)

                    try:
                        part = community_louvain.best_partition(G, weight='weight')
                        mod = community_louvain.modularity(part, G, weight='weight')
                        n_comm = len(set(part.values()))
                    except:
                        mod, n_comm = 0, 0

                    # Centralità (media)
                    bet_vals = list(nx.betweenness_centrality(G, weight='weight').values())
                    clo_vals = list(nx.closeness_centrality(G).values())
                    avg_bet = np.mean(bet_vals) if bet_vals else 0
                    avg_clo = np.mean(clo_vals) if clo_vals else 0

                    try:
                        assort = nx.degree_assortativity_coefficient(G, weight='weight')
                    except:
                        assort = 0

                    networks.append({
                        'id': item['id'],
                        'name': clean_name,
                        'color': item['color'],
                        'G': G,
                        'node_names': node_names_map,  # Salviamo i nomi per il grafico!
                        'metrics': {
                            "Nodi": n_nodes, "Connessioni": G.number_of_edges(), "Densità": dens,
                            "Efficienza Globale": eff, "Cammino Medio (L)": avg_path,
                            "Clustering Coeff.": clust, "Transitivity": trans,
                            "Modularità (Q)": mod, "Num. Comunità": n_comm,
                            "Betweenness (avg)": avg_bet, "Closeness (avg)": avg_clo,
                            "Assortatività": assort
                        }
                    })
                except Exception as ex:
                    print(f"Err Rete {item['id']}: {ex}")
                    errors.append(f"{item['id']}: {str(ex)}")
                    continue

        if len(networks) < 2:
            return dmc.Alert(f"Errore: servono almeno 2 reti valide. {errors}", color="red")

        base = networks[0]  # Rete A

        # --- 3. REPORT CLINICO NARRATIVO ---
        report_cards = []
        for net in networks[1:]:
            mA = base['metrics']
            mB = net['metrics']

            def get_d(k):
                va, vb = mA.get(k, 0), mB.get(k, 0)
                if va == 0: return 0.0
                return ((vb - va) / va) * 100

            d_eff = get_d("Efficienza Globale")
            d_mod = get_d("Modularità (Q)")
            d_dens = get_d("Densità")
            d_path = get_d("Cammino Medio (L)")

            if d_eff < -10:
                status, color_st, icon_st = "Deterioramento Significativo", "red", "mdi:alert-octagon"
                intro = f"Si osserva una **compromissione critica** dell'architettura di rete in {net['name']}."
            elif d_eff < -2:
                status, color_st, icon_st = "Lieve Calo Funzionale", "orange", "mdi:alert"
                intro = f"Si nota una **leggera riduzione** della capacità integrativa in {net['name']}."
            elif d_eff > 5:
                status, color_st, icon_st = "Miglioramento / Iper-Connessione", "green", "mdi:check-decagram"
                intro = f"La rete {net['name']} mostra un **aumento dell'efficienza globale**."
            else:
                status, color_st, icon_st = "Stabilità Topologica", "blue", "mdi:equal"
                intro = f"Il profilo topologico di {net['name']} appare **stabile** rispetto alla baseline."

            p_int = f"**Integrazione:** L'Efficienza è variata del **{d_eff:+.2f}%**."
            if d_path > 5: p_int += f" Il cammino medio è aumentato (+{d_path:.2f}%), indicando rallentamenti."

            p_seg = f"**Segregazione:** La Modularità varia del **{d_mod:+.2f}%**."
            if d_mod < -5: p_seg += " Questo suggerisce una perdita di specializzazione funzionale."

            report_cards.append(
                dmc.Card(withBorder=True, shadow="sm", mb="md", style={"borderLeft": f"5px solid {color_st}"},
                         children=[
                             dmc.Group([DashIconify(icon=icon_st, color=color_st, width=25), dmc.Text(status, fw=700)],
                                       mb="xs"),
                             dcc.Markdown(f"{intro}\n\n{p_int}\n\n{p_seg}\n\n**Densità:** {d_dens:+.2f}%")
                         ])
            )

        # --- 4. GRAFICI ---
        # A. RADAR
        radar_keys = ["Efficienza Globale", "Clustering Coeff.", "Modularità (Q)", "Transitivity", "Densità"]
        fig_radar = go.Figure()
        for k in radar_keys:
            vals = [n['metrics'][k] for n in networks]
            mx = max(vals) if vals and max(vals) > 0 else 1
            for n in networks: n['metrics'][f"{k}_n"] = n['metrics'][k] / mx

        for n in networks:
            vals = [n['metrics'][f"{k}_n"] for k in radar_keys]
            fig_radar.add_trace(
                go.Scatterpolar(r=vals, theta=radar_keys, fill='toself', name=n['name'], line_color=n['color']))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(t=30, b=20), height=350,
                                template="plotly_white")

        # B. BAR CHART - TOP 10 NODI (CON NOMI REALI)
        target = networks[1] if len(networks) > 1 else networks[0]

        # Calcolo Strength per nodo
        degA = dict(base['G'].degree(weight='weight'))
        degB = dict(target['G'].degree(weight='weight'))

        diffs = []
        # Iteriamo sugli indici presenti nel grafo A
        for idx in dict(base['G'].nodes()):
            val_a = degA.get(idx, 0)
            val_b = degB.get(idx, 0)
            delta = val_b - val_a

            # Recuperiamo il nome reale dalla mappa salvata
            real_name = base['node_names'].get(idx, f"Nodo {idx}")

            diffs.append({'name': real_name, 'val_a': val_a, 'val_b': val_b, 'abs_delta': abs(delta)})

        # Ordina per cambiamento maggiore
        diffs.sort(key=lambda x: x['abs_delta'], reverse=True)
        top_10 = diffs[:10]

        fig_bar = go.Figure()
        # Barre Rete A
        fig_bar.add_trace(go.Bar(
            x=[d['val_a'] for d in top_10], y=[d['name'] for d in top_10],
            name=base['name'], orientation='h', marker_color=base['color']
        ))
        # Barre Rete B
        fig_bar.add_trace(go.Bar(
            x=[d['val_b'] for d in top_10], y=[d['name'] for d in top_10],
            name=target['name'], orientation='h', marker_color=target['color']
        ))

        fig_bar.update_layout(
            title=f"Top 10 Nodi più variati ({target['name']} vs {base['name']})",
            margin=dict(t=40, b=40, l=150),  # Margine sinistro per i nomi lunghi
            height=350,
            template="plotly_white",
            barmode='group',
            xaxis_title="Strength (Connettività Totale)",
            yaxis=dict(autorange="reversed")  # Nodi più importanti in alto
        )

        # --- 5. TABELLE COMPLETE ---
        def make_table(net):
            rows = []
            keys_ordered = ["Nodi", "Connessioni", "Densità", "Efficienza Globale", "Cammino Medio (L)",
                            "Clustering Coeff.",
                            "Modularità (Q)", "Num. Comunità", "Betweenness (avg)", "Closeness (avg)", "Assortatività"]
            for k in keys_ordered:
                v = net['metrics'].get(k, 0)
                fmt = "{:.4f}" if isinstance(v, float) else "{}"
                rows.append(html.Tr([html.Td(k, style={"fontSize": "11px"}), html.Td(fmt.format(v),
                                                                                     style={"textAlign": "right",
                                                                                            "fontWeight": "bold",
                                                                                            "fontSize": "11px"})]))

            return dmc.Card([
                dmc.Group([DashIconify(icon="mdi:file-document-outline", color=net['color']),
                           dmc.Text(net['name'], fw=700, c=net['color'], size="sm")], mb="xs"),
                dmc.Table(html.Tbody(rows), striped=True, withTableBorder=True)
            ], withBorder=True, p="xs")

        table_grid = dmc.SimpleGrid(
            cols={"base": 1, "sm": 2, "md": len(networks)},
            spacing="md", children=[make_table(n) for n in networks]
        )

        # --- 6. OUTPUT FINALE ---
        return dmc.Stack(gap="lg", children=[
            dmc.SimpleGrid(cols={"base": 1, "sm": 2}, spacing="lg", children=[
                dmc.Card([dcc.Graph(figure=fig_radar, config={"displayModeBar": False})], withBorder=True, p="sm"),
                dmc.Card([dcc.Graph(figure=fig_bar, config={"displayModeBar": False})], withBorder=True, p="sm")
            ]),
            table_grid,
            dmc.Card([
                dmc.Group([DashIconify(icon="mdi:text-box-search-outline", color="blue"),
                           dmc.Text("Analisi Clinica Narrativa", fw=700, size="lg")], mb="md"),
                dmc.Stack(gap="xs", children=report_cards)
            ], withBorder=True, p="lg", shadow="sm", bg="#f8f9fa")
        ])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return dmc.Alert(f"Errore Critico: {str(e)}", color="red")


# =============================================================================
# 20. CALLBACK: GESTIONE CONTESTO (SYNC TRA TAB)
# =============================================================================

# 20a. GESTIONE CONTESTO INTELLIGENTE (AUTO-SELECT)
@app.callback(
    Output("network-context-selector", "data"),
    Output("network-context-selector", "style"),
    Output("network-context-selector", "value"),  # <--- NUOVO OUTPUT: Cambia valore automaticamente

    Input("store-node-data", "data"),  # Trigger upload singolo
    Input("store-map-a", "data"),  # Trigger confronto

    State("network-context-selector", "value"),
    State("store-map-b", "data"), State("store-map-c", "data"), State("store-map-d", "data")
)
def update_context_options(main_data, ma, current_value, mb, mc, md):
    # 1. Costruisci le opzioni disponibili
    options = []

    # Controlla se esiste l'upload singolo
    has_main = False
    if main_data:
        try:
            # Verifica rapida che sia un JSON valido e non vuoto
            if len(main_data) > 10:
                options.append({"value": "main", "label": "📁 Upload Singolo"})
                has_main = True
        except:
            pass

    # Aggiungi opzioni confronto
    if ma: options.append({"value": "A", "label": "🔵 Rete A (Base)"})
    if mb: options.append({"value": "B", "label": "🟠 Rete B"})
    if mc: options.append({"value": "C", "label": "🟢 Rete C"})
    if md: options.append({"value": "D", "label": "🔴 Rete D"})

    # 2. Determina la Visibilità del Menu
    # Mostra il menu solo se c'è più di una scelta possibile, OPPURE se siamo in modalità confronto senza upload singolo
    visible = len(options) > 0
    style = {"width": "180px", "display": "block"} if visible else {"display": "none"}

    # 3. LOGICA DI AUTO-SELEZIONE (UX FIX)
    new_value = current_value

    # Se non c'è nulla selezionato o la selezione non è più valida
    valid_values = [o['value'] for o in options]
    if current_value not in valid_values:
        if has_main:
            new_value = "main"  # Priorità all'upload singolo se esiste
        elif ma:
            new_value = "A"  # Altrimenti fallback sulla Rete A
        elif options:
            new_value = options[0]['value']  # Altrimenti il primo disponibile
        else:
            new_value = None  # Nessun dato

    # Se l'utente non ha caricato nulla in "Singolo" ma ha caricato "Confronto",
    # forziamo il valore a "A" per evitare la schermata bianca
    if not has_main and ma and current_value == "main":
        new_value = "A"

    return options, style, new_value


# 20b. ENGINE DI SCAMBIO RETI (Il Ponte tra Confronto e Visualizzazione)
@app.callback(
    Output("store-node-data", "data", allow_duplicate=True),
    Output("store-adj-matrix", "data", allow_duplicate=True),
    Output("store-edge-coords", "data", allow_duplicate=True),
    Output("store-analysis-report", "data", allow_duplicate=True),

    Input("network-context-selector", "value"),

    # Dati Confronto (Sorgenti)
    State("store-map-a", "data"), State("store-edge-a", "data"),
    State("store-map-b", "data"), State("store-edge-b", "data"),
    State("store-map-c", "data"), State("store-edge-c", "data"),
    State("store-map-d", "data"), State("store-edge-d", "data"),

    # Dati Main (Per revert) - Opzionale, qui ricalcoliamo per sicurezza
    prevent_initial_call=True
)
def switch_active_network(selection, ma, ea, mb, eb, mc, ec, md, ed):
    if not selection or selection == "main":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    print(f">>> Cambio Contesto Attivo: Caricamento Rete {selection}...")

    # 1. Seleziona la sorgente corretta
    target_map = None
    target_edge = None

    if selection == "A":
        target_map, target_edge = ma, ea
    elif selection == "B":
        target_map, target_edge = mb, eb
    elif selection == "C":
        target_map, target_edge = mc, ec
    elif selection == "D":
        target_map, target_edge = md, ed

    if not target_map or not target_edge:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        # 2. PROCESSO DI ELABORAZIONE (Copia della logica di run_analysis)
        # Parsing
        df_map = parse_file_contents(target_map, file_type="nodes")
        df_edges = parse_file_contents(target_edge, file_type="edges")

        # Standardizza colonne
        id_col = 'roi_id' if 'roi_id' in df_map.columns else df_map.columns[0]
        name_col = 'roi_name' if 'roi_name' in df_map.columns else df_map.columns[1]
        df_map = df_map.rename(columns={name_col: 'roi_name', id_col: 'roi_id'})
        df_map['region'] = df_map['roi_name'].apply(extract_macro_region)

        # Coordinate
        coords_list = []
        for idx, row in df_map.iterrows():
            coords = extract_coords_regex(str(row['roi_name']))
            if coords is None:
                # Fallback su colonne x,y,z se esistono
                if 'x' in df_map.columns and 'y' in df_map.columns and 'z' in df_map.columns:
                    try:
                        coords = [float(row['x']), float(row['y']), float(row['z'])]
                    except:
                        coords = None
            coords_list.append(coords)

        df_map["coords"] = coords_list
        df_map = df_map.dropna(subset=["coords"]).reset_index(drop=True)
        coords = np.array(df_map["coords"].tolist())
        n_nodes = len(df_map)

        # Grafo
        id_to_idx = {row["roi_id"]: i for i, (idx, row) in enumerate(df_map.iterrows())}
        adj = np.zeros((n_nodes, n_nodes))
        G = nx.Graph()

        src_col = 'source' if 'source' in df_edges.columns else df_edges.columns[0]
        tgt_col = 'target' if 'target' in df_edges.columns else df_edges.columns[1]
        w_col = 'weight' if 'weight' in df_edges.columns else None

        for _, row in df_edges.iterrows():
            try:
                i = id_to_idx.get(row[src_col])
                j = id_to_idx.get(row[tgt_col])
                weight = float(row[w_col]) if w_col else 1.0
                if i is not None and j is not None:
                    adj[i, j] = weight
                    adj[j, i] = weight
                    G.add_edge(i, j, weight=weight)
            except:
                continue

        # Metriche & Clustering
        try:
            part = community_louvain.best_partition(G) if len(G) > 0 else {}
        except:
            part = {}
        df_map["community"] = df_map.index.map(part.get).fillna(-1).astype(int)

        df_map["degree"] = pd.Series(dict(G.degree())).reindex(df_map.index).fillna(0).astype(int)
        df_map["strength"] = pd.Series(dict(G.degree(weight='weight'))).reindex(df_map.index).fillna(0.0).astype(float)

        # Centralità (opzionale per velocità nello switch)
        if len(G) < 500:
            bet = nx.betweenness_centrality(G, weight='weight')
            df_map["betweenness_centrality"] = df_map.index.map(bet).fillna(0.0)
        else:
            df_map["betweenness_centrality"] = 0.0

        # Edge Coords per visualizzazione 3D
        edge_coords = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = adj[i, j]
                if w > 0:
                    edge_coords.append({
                        'source': int(i), 'target': int(j), 'weight': w,
                        'coords': [coords[i].tolist(), coords[j].tolist()]
                    })

        report_text = f"**Rete {selection} Attiva**\n\n* **Nodi:** {n_nodes}\n* **Archi:** {G.number_of_edges()}"

        return df_map.to_json(orient="split"), \
            pd.DataFrame(adj).to_json(orient="split"), \
            json.dumps(edge_coords), \
            report_text

    except Exception as e:
        print(f"Errore switch rete: {e}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# =============================================================================
# 21. CALLBACK: CHATBOT AI CONTESTUALE
# =============================================================================

# 21a. Apertura/Chiusura Finestra
@app.callback(
    Output("chat-window", "style"),
    Output("chat-is-open", "data"),
    Input("btn-toggle-chat", "n_clicks"),
    Input("btn-close-chat", "n_clicks"),
    State("chat-is-open", "data"),
    prevent_initial_call=True
)
def toggle_chat(n1, n2, is_open):
    new_state = not is_open
    display = "flex" if new_state else "none"

    # Aggiorniamo lo stile mantenendo le proprietà fisse
    style = chat_window_style.copy()
    style["display"] = display

    return style, new_state


# 21b. CHATBOT CON INTELLIGENZA REALE (OpenAI GPT)
@app.callback(
    Output("chat-messages-container", "children"),
    Output("chat-input", "value"),
    Output("chat-history", "data"),

    Input("btn-send-chat", "n_clicks"),
    Input("chat-input", "n_submit"),

    State("chat-input", "value"),
    State("chat-history", "data"),

    # CONTESTO: Leggiamo i dati per "istruire" l'AI
    State("store-node-data", "data"),  # Rete Singola
    State("store-map-a", "data"),  # Rete A (Confronto)
    State("store-map-b", "data"),  # Rete B (Confronto)

    prevent_initial_call=True
)
def chat_interaction(n_click, n_sub, user_text, history, single_net, map_a, map_b):
    if not user_text: return dash.no_update, dash.no_update, dash.no_update
    if history is None: history = []

    # 1. Aggiungi messaggio utente alla storia locale
    history.append({"role": "user", "content": user_text})

    # --- A. COSTRUZIONE DEL CONTESTO (Cosa sa l'AI dei dati?) ---
    # Questo è il "System Prompt": istruisce l'AI su come comportarsi e cosa sta guardando.
    system_instruction = "Sei un assistente esperto in Neuroscienze Computazionali e Teoria dei Grafi per l'applicazione 'NeuroGraph 3D'. Rispondi in modo tecnico ma conciso."

    data_context = ""
    try:
        if map_a and map_b:
            # Modalità Confronto
            df_a = pd.read_json(StringIO(map_a), orient="split")
            df_b = pd.read_json(StringIO(map_b), orient="split")
            data_context = f"""
            CONTESTO DATI ATTUALE:
            L'utente sta analizzando un CONFRONTO tra due reti cerebrali:
            1. Rete A (Baseline): {len(df_a)} nodi.
            2. Rete B (Target): {len(df_b)} nodi.
            L'utente potrebbe chiederti differenze di Efficienza Globale, Modularità o Robustezza.
            """
        elif single_net:
            # Modalità Singola
            df = pd.read_json(StringIO(single_net), orient="split")
            cols = ", ".join(df.columns)
            data_context = f"""
            CONTESTO DATI ATTUALE:
            L'utente sta analizzando una SINGOLA rete neurale.
            - Numero Nodi: {len(df)}.
            - Metriche disponibili nel dataset: {cols}.
            Se l'utente chiede consigli, suggerisci di analizzare la 'Betweenness Centrality' per trovare gli Hub o la 'Modularità'.
            """
        else:
            data_context = "CONTESTO DATI: Nessun dato è stato ancora caricato dall'utente. Se chiede di analizzare qualcosa, invitalo a caricare i file nella Home Page."
    except Exception as e:
        data_context = f"Errore lettura contesto dati: {str(e)}"

    # --- B. CHIAMATA A CHATGPT ---
    ai_response = "Errore di connessione."

    try:
        # Prepariamo i messaggi per l'API
        # 1. Istruzione di sistema (invisibile all'utente)
        messages_payload = [
            {"role": "system", "content": system_instruction + "\n\n" + data_context}
        ]

        # 2. Aggiungiamo la storia recente (ultimi 4 scambi) per dare memoria alla chat
        # (Escludiamo i messaggi troppo vecchi per risparmiare token e costi)
        recent_history = history[-5:]
        for msg in recent_history:
            # Mappiamo 'bot' in 'assistant' per l'API di OpenAI
            role = "assistant" if msg["role"] == "bot" else "user"
            messages_payload.append({"role": role, "content": msg["content"]})

        # 3. Invio richiesta
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_payload,  # Limita la lunghezza della risposta
            temperature=0.7  # Creatività (0=Robotico, 1=Creativo)
        )

        ai_response = completion.choices[0].message.content

    except Exception as e:
        ai_response = f"⚠️ Errore API OpenAI: {str(e)}. Controlla la tua API Key o la connessione internet."

    # 4. Aggiorna storia con la risposta del bot
    history.append({"role": "bot", "content": ai_response})

    # 5. Renderizza i messaggi (UI)
    bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubbles.append(
            dmc.Group(
                justify="flex-end" if is_user else "flex-start",
                children=[
                    dmc.Paper(
                        p="xs", radius="md",
                        bg="indigo" if is_user else "gray.1",
                        c="white" if is_user else "dark",
                        style={"maxWidth": "85%", "fontSize": "14px"},
                        children=dcc.Markdown(msg["content"])
                    )
                ]
            )
        )

    return bubbles, "", history

# =============================================================================
# 16. CALLBACK: Export Dati
# =============================================================================

# 16a. Export PNG
@app.callback(
    Output("download-png", "data"),
    Input("export-png", "n_clicks"),
    State("brain-connectome-web", "figure"),
    prevent_initial_call=True
)
def export_png(n_clicks, fig):
    try:
        # Richiede kaleido installato
        fig_bytes = go.Figure(fig).to_image(format="png", scale=3)
        content_string = base64.b64encode(fig_bytes).decode('utf-8')
        if not n_clicks: return dash.no_update
        return dict(content=content_string,
                    filename="connectome_snapshot.png",
                    base64=True)

    except Exception as e:
        print(f"Errore esportazione PNG: {e}")
        return dash.no_update


# 16b. Export JSON
@app.callback(
    Output("download-json", "data"),
    Input("export-json", "n_clicks"),
    State("brain-connectome-web", "figure"),
    prevent_initial_call=True
)
def export_json(n_clicks, fig):
    if not n_clicks: return dash.no_update
    return dict(
        content=json.dumps(fig, indent=2),
        filename="connectome_snapshot.json"
    )


# =============================================================================
# 17. RUN APP
# =============================================================================

if __name__ == "__main__":
    print("Server avviato. Apri http://127.0.0.1:8050/")
    app.run(debug=True, dev_tools_hot_reload=False)
