from operator import truth
from tkinter import X
import pandas as pd
from sklearn import cluster
import streamlit as st
import numpy as np
import mpld3
import matplotlib
from mpld3 import plugins
import logging
import seaborn as sns
from sklearn.decomposition import PCA
import streamlit.components.v1 as components
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import io
from scipy import interpolate
import pymde
import hdbscan
import gseapy as gp
import time
import umap
from sklearn.manifold import TSNE
import hydralit_components as hc
from PIL import Image
from scipy.spatial import ConvexHull
from streamlit_elements import elements, mui, html, nivo
import plotly.graph_objects as go
from multiprocessing import Process
from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
import pyspark.pandas as pypd
import dask.dataframe as dd
import os, signal, psutil
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
#from google.oauth2 import service_account
#import pandas_gbq
import pyarrow.parquet as pq


#credentials = service_account.Credentials.from_service_account_info(
#    {
#    "type": "service_account",
#    "project_id": "gwperturbseq",
#    "private_key_id": "086ba68d0133d5b4e587c727bb0ef379a20e14af",
#    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC+m2nfKmOUpXye\neMZuTPgfu5Y91IynqDUOeUnoJr1gTzhEmp963ELn9ajgO6NYi7lp7gHQJY7gpp8j\nlDpQdcowU5w/emkJ/hpgRLPk6NvnAGh+kAEud+RvLvEIoAKU4Fg1n+NG6B/LpQUj\nkFSI/8vPmzOe1vLI8E/KX50BUWZnPxVW43mtxUeWfha2aUVkKiURhjtfZ90LQiRs\nO1/YY2wYk2b2/9D4NJHzLsDgsdrcLfy93Go8c2Qrl9GaR5eSmXdhaj5LrTf8jDCs\nv0n+VQRSG8eFgpd210LurhRZVcW21Jtd96cAza3ffn+P/r44aukPgVYMt6eZ1oYZ\nDYYwsyFlAgMBAAECggEAA6c7REggK9c5JetrnMkYqabA9C5/jKYWlNBmqqDQ4i0R\nYunNV3+sCVQKga8op4D9OD04jAhLrm+I1fNYiFrCLCLg+Bl/RvdJ8MXGCwzj8KtC\nFXYPMXPGmvfvQDzkBX2Jb9llM/S/pLCb5tsiN6b+BqXOxKfIqvszSpdLn4qFruaq\nVEn9hrPwQ+Yp7X2CenutjqLOqJO200+DExPWQSXJkGSjRzTzkn/ccwreAD+55YXs\nzsafFScPqrfvWMqvH57qS8/tjNhEkgIK5HJWBZE4fc9K+TcElrl00JVuAsguh5CR\nPtCy22dW5T3s3WN2tvAGWKS2yAX606CSPs5HmgHmSQKBgQDnuCmCzVqe+baZAqzv\nFh2pLdv1a8jclPip3wwXHr1EmTnXmhRuFG4+17MLXEsH3XPdca9F0udKkNsMq5eX\nvj4cqOn/8TzZk0NjlKxNrQW44BmrOkck8E5ubpbzm+cEwDi4OeU/GXvf8UC8YIoL\n+0TAeRtM1+CEOR30M0z3dW8WOQKBgQDSlGuNS7ZT1XVmhjwOVZY2Bbpi9YhA+59p\ntHE94g8sGOPI0uH0VCKojNQZrYnTuRlGbGYtlB7NmA14P50DCRkOC0AybvGc0FP+\nimXwvgiZW56CS9wks/WHotrKMkwFjwjnLOe1zS2RdPwyHzkMRIZJ/d4vx2jnQ0rE\naGK3n4MEjQKBgQDXe8JWmkNIjW3J8twA5m8k0bm4C8jZoEtyJTLoGTTnIxrQLcAL\n8kHnfM1KpkQ8BytlZgAZjZx7EiQyLywk98xo+IfK9Hqi1riXVT26aklk+DnGqsZY\nSoSVW5BS8Msv9tWINIbEXB1/TqnlxFz78WmpF7ZUiniGXVtGqaJMiU+JuQKBgCGX\nF0jkqOLai+2cv/uxX1Kiu0aJ0AA9owIjjGTIqI7qMXuZV/23Zgfo3vRxoCirwAJh\nzDK9C+TTumJSE2Omkd6RoN3qoJPZD2zs+rHaPljoKnKyUx1vr1zBd5EFUOcaF9t4\naEWxwvEO7iO4o0r3q0wmOMj6UBuDumt2EzkjMq0JAoGAHIuITeoilhBTgAEYRbtd\nLtrx7PXVKeKF1VNSrbsgR31l+IsOCTrNcd6FxJeXQi/uCiRYq0iVzbT87IV5g9Gf\n3zKvk9dA8OU2/nZpajRqPTKqm0o9PELjZ3uLrOcea09n+g14cACneqOf/O0QwYhu\nceJIEBV23WnZCAV1gXKi6tA=\n-----END PRIVATE KEY-----\n",
#    "client_email": "perturbseq@gwperturbseq.iam.gserviceaccount.com",
#    "client_id": "114403865856498936028",
#    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#    "token_uri": "https://oauth2.googleapis.com/token",
#    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
#    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/perturbseq%40gwperturbseq.iam.gserviceaccount.com"
#    },
#)
#pandas_gbq.context.credentials = credentials
#pandas_gbq.context.project = "gwperturbseq"


#region GENERAL
Image.MAX_IMAGE_PIXELS = None
# Check for GPU and try to import TSNECuda if possible. Otherwise just use sklearn
gpu_avail = False
try :
    from tsnecuda import TSNE as TSNECuda
    gpu_avail = True
except :
    print('TSNE Cuda could not be imported, not using GPU.')

global clusterLoc, time2, header

distance_metric = ["correlation", "euclidean", "jaccard","braycurtis", "canberra", "chebyshev", "cityblock", "cosine", "dice",  "hamming",  "jensenshannon", "kulsinski", "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]

pd.set_option("display.precision", 2)

st.set_page_config(
        layout = "wide",
        page_title="Perturb-Seq Analyzer",
        page_icon="🎈",
        #initial_sidebar_state='collapsed'
)

def set_page_container_style(
        max_width: int = 1200, max_width_100_percent: bool = False,
        padding_top: int = 5, padding_right: int = 2, padding_left: int = 2, padding_bottom: int = 2,
        #color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}               
            </style>
            ''',
            unsafe_allow_html=True,
        )


# specify the primary menu definition
menu_data = [
    {'id':'Correlation','icon': "far fa-copy", 'label':"Correlation"},
    {'id':'PCA','icon':"🐙",'label':"PCA"},
    {'id':'MDE','icon':"🐙",'label':"MDE"},
    {'id':'UMAP','icon':"🐙",'label':"UMAP"},
    {'id':'tSNE','icon':"🐙",'label':"tSNE"},
    {'id':'biCluster','icon':"🐙",'label':"Bi-Clustering"},
    {'id':'Regulation','icon':"🐙",'label':"GeneRegulation"},
    {'id':'HeatMap','icon':"🗺️",'label':"HeatMap"}
    #{'icon': "fa-solid fa-radar",'label':"Dropdown1", 'submenu':[{'id':' subid11','icon': "fa fa-paperclip", 'label':"Sub-item 1"},{'id':'subid12','icon': "💀", 'label':"Sub-item 2"},{'id':'subid13','icon': "fa fa-database", 'label':"Sub-item 3"}]},
    #{'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
    #{'id':' Crazy return value 💀','icon': "", 'label':"Calendar"},
    #{'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
    #{'icon': "far fa-copy", 'label':"Right End"},
    #{'icon': "fa-solid fa-radar",'label':"Dropdown2", 'submenu':[{'label':"Sub-item 1", 'icon': "fa fa-meh"},{'label':"Sub-item 2"},{'icon':'🙉','label':"Sub-item 3",}]},
]

over_theme = {'txc_inactive': '#FFFFFF', 'menu_background':'blue','txc_active':'black','option_active':'white'
              
              }


#if "tab" in st.session_state and not st.session_state["tab"] == None:
#    st.session_state["tab"] 

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    #login_name='Logout',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

if  "tabs" not in st.session_state:
    st.session_state["tabs"] =  menu_id
elif st.session_state["tabs"] !=  menu_id:
    st.session_state["tabs"] =  menu_id
    st    
    st.session_state.rerun = 1
    print("Rerun") 
    st.experimental_rerun()
print(st.session_state["tabs"])     


st.markdown(
        """<style>
     block-container{{
        width: 700px;
        padding-top: 1rem;
        padding-right: 5%;
        padding-left: 2rem;
        padding-bottom: 5rem;
    }}
   
</style>
""",
        unsafe_allow_html=True,
    )

 #.reportview-container .main {{
    #    color: {COLOR};
    #    background-color: {BACKGROUND_COLOR};
    #}}
#endregion
def cancel():
    print("I am in cancel")    
    st.session_state.rerun = 0
    st.session_state["isRunning"] = False
    st.warning("Stopped calculations")    
    st.stop()
    #st.session_state["drawButton"] = False
def runNow():
    print("I am in run now")       
    st.session_state.rerun = 1
    st.session_state["isRunning"] = True

def updateGeneList():
    genes = st.session_state.geneList2.replace(';',',').replace(' ', ',').replace('\n',',').split(',')
    st.session_state["geneList"] =[]       
    for gene in genes:
        st.session_state["geneList"].append(gene)
        st.session_state["geneList"].append(gene + "_2")
    
    

def getSessionID():
    '''
    Streamlit hack to use report server debugging to get unique session IDs.
    These session IDs are used in order to separate data from different users
    and allows for easy sharing of plots using session IDs.
    Raises
    ------
    RuntimeError
        If the streamlit session object is not available.
    Returns
    -------
    sessionID
        The ID of the current streamlit user session.
    '''
    ctx = ReportThread.get_report_ctx()
    this_session = None    
    current_server = Server.get_current()
    session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:
        s = session_info.session
        if (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue) :
            this_session = s
            
    if this_session is None: raise RuntimeError("Oh noes. Couldn't get your Streamlit Session object")
    return id(this_session)
    
def main(): 
    c30, c31, c32 = st.columns([2.5, 1, 3])
    start = time.time()
    # st.image("logo.png", width=400)
    global header
    #header = st.title(menu_id)
        
    #menu_id
    time1 = time.time()
    print("Step1:"  + str(time.time() - time1))
    #with st.expander("ℹ️ - About this app", expanded=True):
    #    st.write("""-   The *Correlation Analyzer* generates a correlation map among samples or genes.""")
    #    st.markdown("")
    st.markdown("""<style> div.stButton > button:first-child {width: 19rem;} </style>""", unsafe_allow_html=True)   
         
    with st.sidebar:
        #ce, c1, ce, = st.columns([0.01, 2, 0.05])
        
       # with c1: 
            if "isRunning" not in st.session_state or st.session_state.rerun == 1:
                st.session_state["isRunning"] = True           
            
                
            butLoc = st.empty() 
            #st.session_state["isRunning"] ==True  
            print("I am checking" + str(st.session_state["isRunning"])) 
            with butLoc:     
                if st.session_state["isRunning"]:                
                    st.button("CANCEL", key = "cancelButton", on_click= cancel)            
                else:
                    st.button("DRAW", key = "drawButton", on_click= runNow)
                
            if 'tabs' not in st.session_state:
                st.session_state['tabs'] = 'Main'  
         
            if 'tabs' not in st.session_state or st.session_state['tabs'] != "Regulation":   
                
                with st.expander('Core Settings', expanded = True):
                    
                    
                        
                    st.selectbox('Cell Line', ["K562-Whole Genome", "K562-Essential", "RPE1-Whole Genome", "RPE1-Essential",],0, key = "dataSource", on_change= rerun)
                    st.selectbox('Data Type', ["Perturbation", "Gene Expression"],0, key = "dataType", on_change= rerun)

                    list1 = st.text_area(
                        st.session_state.dataType.split(' ')[0] + " list",
                        height=100,placeholder="Please enter gene list seperated by comma, new line, space, or semicolon!",
                        help="Missing genes in the database will be ignored.",
                        key = "geneList2",
                        on_change= updateGeneList
                    ).replace(';',',').replace(' ', ',').replace('\n',',').split(',')
                         
                    if "geneList" in st.session_state and len(st.session_state.geneList)>0:
                        str(int((len(st.session_state.geneList)-1)/2)) + " genes"                        
                    else:
                        "No genes entered!"
                    
                    
                    st.selectbox('Graph Type', ["2D", "3D", "Both"],0, key = "graphType", on_change= rerun)
                    st.text_area(
                            "Highlight Genes",
                            height=100,placeholder="Please enter gene list seperated by comma, new line, space, or semicolon!",
                            help="These gene will be highlighted in the plots.",
                            key = "highlightGeneList"
                        ) 
        
            if 'tabs' in st.session_state and st.session_state['tabs'] == "Correlation":
                with st.expander('Correlation Settings'):
                    st.selectbox('Correlation Method',('Spearman', 'Pearson', 'Kendall'), key = "corrMethod", on_change= rerun)
                    st.checkbox('Remove low correlation', key = "removeLowCorr", on_change = rerun)
                    if "removeLowCorr" not in st.session_state:
                        st.session_state["removeLowCorr"] = True
                    
                    st.slider('Min Correlation',0.00,0.99,0.30,0.05,disabled= not st.session_state.removeLowCorr, key = "corrLimitLow", on_change= rerun)
                    
                    linkage_method = ["single","complete", "average", "weighted", "centroid", "median", "ward"]
                    st.selectbox('Linkage Method', linkage_method,1, key ="linkageSelection", on_change= rerun)
                    st.selectbox('Distance Metric', distance_metric,1, key ="distanceMetric", on_change= rerun)                                    
                    colors = ["bwr","BuPu",  "Accent", "Accent_r", "Blues", "Blues_r", "BrBG", "BrBG_r", "BuGn", "BuGn_r", "BuPu_r", "CMRmap", "CMRmap_r", "Dark2", "Dark2_r", "GnBu", "GnBu_r", "Greens", "Greens_r", "Greys", "Greys_r", "OrRd", "OrRd_r", "Oranges", "Oranges_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1", "Pastel1_r", "Pastel2", "Pastel2_r", "PiYG", "PiYG_r", "PuBu", "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr", "PuOr_r", "PuRd", "PuRd_r", "Purples", "Purples_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r", "RdPu", "RdPu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds", "Reds_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3", "Set3_r", "Spectral", "Spectral_r", "Wistia", "Wistia_r", "YlGn", "YlGnBu", "YlGnBu_r", "YlGn_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r", "afmhot", "afmhot_r", "autumn", "autumn_r", "binary", "binary_r", "bone", "bone_r", "brg", "brg_r", "bwr", "bwr_r", "cividis", "cividis_r", "cool", "cool_r", "coolwarm", "coolwarm_r", "copper", "copper_r", "crest", "crest_r", "cubehelix", "cubehelix_r", "flag", "flag_r", "flare", "flare_r", "gist_earth", "gist_earth_r", "gist_gray", "gist_gray_r", "gist_heat", "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r", "gist_yarg", "gist_yarg_r", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray", "gray_r", "hot", "hot_r", "hsv", "hsv_r", "icefire", "icefire_r", "inferno", "inferno_r", "jet", "jet_r", "magma", "magma_r", "mako", "mako_r", "nipy_spectral", "nipy_spectral_r", "ocean", "ocean_r", "pink", "pink_r", "plasma", "plasma_r", "prism", "prism_r", "rainbow", "rainbow_r", "rocket", "rocket_r", "seismic", "seismic_r", "spring", "spring_r", "summer", "summer_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain", "terrain_r", "turbo", "turbo_r", "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r", "viridis", "viridis_r", "vlag", "vlag_r", "winter", "winter_r"]
                    st.selectbox('Map Color', colors, 0, key ="colorx", on_change= rerun)                        
                    st.selectbox('Z Score Normalization', ['None', 'Rows', 'Columns'],0, key ="zScore", on_change= rerun)
                    st.selectbox('Standardization', ['None', 'Rows', 'Columns'], 0, key ="sscale", on_change= rerun)
                    st.slider('Coloring range', -0.8, 0.8, (-0.5,0.5), 0.01, key ="cRanger",on_change= rerun)
                    st.slider('Size',1,20,5,1, key ="zoomFactor",on_change= rerun)
            selection =0
            if 'tabs' in st.session_state and st.session_state['tabs'] == "PCA":
                with st.expander('PCA Settings'):
                    st.selectbox('PCA Source', ["Raw Data", "Correlation Data"],1, key ="pcaSource", on_change= rerun)
                    st.slider('Number of componenets',2,100,10,1, key ="numOfPCAComponents",on_change= rerun)
                    selection = st.checkbox('HDB Scan Clustering',value = True, key = "hdbClusteringPCA", on_change = rerun)
            
            if 'tabs' in st.session_state and st.session_state['tabs'] == "MDE":
                with st.expander('Embedding Settings'):
                    st.selectbox('Embeding Source', ["PCA Data", "Raw Data", "Correlation Data"],0, key ="embedSource", on_change= rerun)
                    
                    maxSize =4
                    if st.session_state.embedSource ==  "PCA Data":
                        if "numOfPCAComponents" not in st.session_state:
                            "You need to first perform PCA to use it in embeding!"
                        else:
                            maxSize = st.session_state.numOfPCAComponents -1
                    elif st.session_state.embedSource ==  "Raw Data":
                        maxSize = len(pd.read_parquet("K562_Orginal_Zscore.parquet").columns)-1
                    else:
                        data = calculateCorrelations()
                        if len(data.columns)>3:
                            maxSize = len(data.columns)-1                    
                        data = ""  
                                 
                    st.slider('Dimension Count',2,maxSize,int(maxSize/2),1, key ="embeddingDim",on_change= rerun)

                    
                    st.selectbox('MDE Contrsaint', ['Standardized','Centered'],0, key = "pyMdeConstraint",on_change= rerun)
                    #st.slider('Number of Neighbours',3,100,10,1, key ="numOfEmbedDimension",on_change= rerun)
                    st.slider('Repulsive Fraction',0.1,5.0,0.5,0.1, key ="repulsiveFraction",on_change= rerun)
                    selection = st.checkbox('HDB Scan Clustering',value = True, key = "hdbClusteringMDE", on_change = rerun)
                    
            if 'tabs' in st.session_state and st.session_state['tabs'] == "UMAP":
                with st.expander('UMAP Settings'):
                    st.selectbox('UMAP Source', ["PCA Data", "Raw Data", "Correlation Data"],0, key ="UMAPembedSource", on_change= rerun)
                    maxSize =4
                    if st.session_state.UMAPembedSource ==  "PCA Data":
                        if "numOfPCAComponents" not in st.session_state:
                            "You need to first perform PCA to use it in embeding!"
                        else:
                            maxSize = st.session_state.numOfPCAComponents -1
                    elif st.session_state.UMAPembedSource ==  "Raw Data":
                        maxSize = len(rawDataFile.columns)-1
                    else:
                        data = calculateCorrelations()
                        if len(data.columns)>3:
                            maxSize = len( data.columns)-1                    
                        data = ""            
                    st.slider('Dimension Count',2,maxSize,int(maxSize/2),1, key ="UMAPembeddingDim",on_change= rerun)
                    st.select_slider('Minimum Distance', [0,0.1,0.25, 0.5, 0.8, 0.99],value =(0.1), key = "UMAPmin_dist",on_change= rerun)
                    st.select_slider('Number of Neighbours', [2, 5, 10, 20, 50, 100, 200],value =(5), key ="UMAPn_neighbors",on_change= rerun)
                    selection = st.checkbox('HDB Scan Clustering',value = True, key = "hdbClusteringUMAP", on_change = rerun)
            
            if 'tabs' in st.session_state and st.session_state['tabs'] == "tSNE":
                with st.expander('tSNE Settings'):
                    st.selectbox('tSNE Source', ["PCA Data", "Raw Data", "Correlation Data"],0, key ="tSNEembedSource", on_change= rerun)
                    

                    maxSize =4
                    if st.session_state.tSNEembedSource ==  "PCA Data":
                        if "numOfPCAComponents" not in st.session_state:
                            "You need to first perform PCA to use it in embeding!"
                        else:
                            maxSize = st.session_state.numOfPCAComponents -1
                    elif st.session_state.tSNEembedSource ==  "Raw Data":
                        maxSize = len(rawDataFile.columns)-1
                    else:
                        data = calculateCorrelations()
                        if len(data.columns)>3:
                            maxSize = len( data.columns)-1                    
                        data = ""
                                   
                    st.select_slider('Perplexity', [1,2,5, 10, 20, 40,60,80,100,150,300],value =(5), key = "tSNEPerplexity",on_change= rerun)
                    st.select_slider('Learning Rate', [10,20,50, 100, 200, 500,1000,],value =(200), key = "tSNElearning_rate",on_change= rerun)
                    st.select_slider('Number of Iterations', [250,500, 1000, 2000, 5000],value =(1000), key = "tSNEn_iter",on_change= rerun)
                    st.select_slider('Early Exaggeration %', [1,2, 5,10,15,25],value =(5), key = "tSNEearly_exaggeration",on_change= rerun)
                    selection = st.checkbox('HDB Scan Clustering',value = True, key = "hdbClusteringtSNE", on_change = rerun)
            if 'tabs' in st.session_state and st.session_state['tabs'] == "biCluster":
                with st.expander('biClustering Settings'):
                    st.selectbox('biClustering Source', ["PCA Data", "Raw Data", "Correlation Data"],0, key ="biClusteringembedSource", on_change= rerun)
                    st.number_input("Cluster Count",2,2000,10,1,key ="biClusteringCount", on_change= rerun)
            if 'tabs' in st.session_state and st.session_state['tabs'] == "HeatMap":                    
                targetList = st.text_area("Target Gene List", height=100,
                    placeholder="Please enter gene list seperated by comma, new line, space, or semicolon!",
                    help="Missing genes in the database will be ignored.", key = "targetList")
                with st.expander('HeatMap Settings'):                   
                    
                    linkage_method = ["single","complete", "average", "weighted", "centroid", "median", "ward"]
                    st.selectbox('Linkage Method', linkage_method,1, key ="linkageSelection2", on_change= rerun)
                    st.selectbox('Distance Metric', distance_metric,1, key ="distanceMetric2", on_change= rerun)                                    
                    colors = ["bwr","BuPu",  "Accent", "Accent_r", "Blues", "Blues_r", "BrBG", "BrBG_r", "BuGn", "BuGn_r", "BuPu_r", "CMRmap", "CMRmap_r", "Dark2", "Dark2_r", "GnBu", "GnBu_r", "Greens", "Greens_r", "Greys", "Greys_r", "OrRd", "OrRd_r", "Oranges", "Oranges_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1", "Pastel1_r", "Pastel2", "Pastel2_r", "PiYG", "PiYG_r", "PuBu", "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr", "PuOr_r", "PuRd", "PuRd_r", "Purples", "Purples_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r", "RdPu", "RdPu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds", "Reds_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3", "Set3_r", "Spectral", "Spectral_r", "Wistia", "Wistia_r", "YlGn", "YlGnBu", "YlGnBu_r", "YlGn_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r", "afmhot", "afmhot_r", "autumn", "autumn_r", "binary", "binary_r", "bone", "bone_r", "brg", "brg_r", "bwr", "bwr_r", "cividis", "cividis_r", "cool", "cool_r", "coolwarm", "coolwarm_r", "copper", "copper_r", "crest", "crest_r", "cubehelix", "cubehelix_r", "flag", "flag_r", "flare", "flare_r", "gist_earth", "gist_earth_r", "gist_gray", "gist_gray_r", "gist_heat", "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r", "gist_yarg", "gist_yarg_r", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray", "gray_r", "hot", "hot_r", "hsv", "hsv_r", "icefire", "icefire_r", "inferno", "inferno_r", "jet", "jet_r", "magma", "magma_r", "mako", "mako_r", "nipy_spectral", "nipy_spectral_r", "ocean", "ocean_r", "pink", "pink_r", "plasma", "plasma_r", "prism", "prism_r", "rainbow", "rainbow_r", "rocket", "rocket_r", "seismic", "seismic_r", "spring", "spring_r", "summer", "summer_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain", "terrain_r", "turbo", "turbo_r", "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r", "viridis", "viridis_r", "vlag", "vlag_r", "winter", "winter_r"]
                    st.selectbox('Map Color', colors, 0, key ="colorx2", on_change= rerun)                        
                    st.selectbox('Z Score Normalization', ['None', 'Rows', 'Columns'],0, key ="zScore2", on_change= rerun)
                    st.selectbox('Standardization', ['None', 'Rows', 'Columns'], 0, key ="sscale2", on_change= rerun)
                    st.slider('Coloring range', -0.8, 0.8, (-0.5,0.5), 0.01, key ="cRanger2",on_change= rerun)
                    st.slider('Size',1,20,5,1, key ="zoomFactor2",on_change= rerun)    
                         
            if 'tabs' in st.session_state and st.session_state['tabs'] == "Regulation":                             
                with st.expander('Core Settings'):
                    
                    list1 = pd.read_parquet("K562_Orginal_Zscore.parquet").index #.tolist()                
                    print(len(list1))                                      
                    #list1.extend(rawDataFile.columns.values.tolist())
                    #print(len(list1))
                    geneselection = st.selectbox('Select a gene', list1,0, key = "geneSelection",on_change= rerun)               
                    st.slider("Absolute Z Score / Correlation r", 0.0,1.0,0.1,0.05, key = "AbsZScore", on_change= rerun)
                if geneselection != "":
                    
                    st.button("Find expressional alterations upon KD of "+ geneselection,key ='type1',on_click= rerun)
                    st.button("Find perturbations that regulate " + geneselection,on_click= rerun, key ='type2')
                    st.button("Find perturbations that correlate with KD of "+ geneselection,key ='type3',on_click= rerun)
                    st.button("Find genes that correlate with " + geneselection + " upon various perturbations",on_click= rerun, key ='type4')
                    
            if selection != "" and selection == True :
                
                with st.expander('Clustering Settings'):
                    st.checkbox('Show legend',True,key = "clusteringLegend", on_change= rerun)
                    st.checkbox('Show cluster centers',True,key = "clusteringCenter", on_change= rerun)
                    st.checkbox('Highlight clusters',True,key = "clusteringHighlight", on_change= rerun)                    
                    st.slider('Minimum Cluster Size',3,100,10,1, key ="min_cluster_size_", disabled = not selection,on_change= rerun)
                    st.selectbox('Clustering Metric', distance_metric,1, key = "metric_",disabled = not selection,on_change= rerun)
                    st.selectbox('Clustering Method', ["EOM", "LEAF"],0, key = "cluster_selection_method_",disabled = not selection,on_change= rerun)
                    st.slider('Minimum Samples',3,100,10,1, key ="min_samples_",disabled = not selection,on_change= rerun)
                    st.slider('Cluster Selection Epsilon',0.0,1.0,0.0,0.1, key ="cluster_selection_epsilon_",disabled = not selection,on_change= rerun)
                with st.expander('Geneset Enrichment'):
                    global clusterLoc
                    clusterLoc = st.container()                
              
    #st.write(st.session_state)
    global time2
    time2 = time.time()
    print("Step2:"  + str(time2 - time1))
    #buttons = st.container() 
    if st.session_state.rerun == 1:
        st.info("Started calculations")
        st.session_state["isRunning"] = True
        runCalculations()
        st.session_state["isRunning"] = False
        st.info("Finished calculations")              
        with butLoc:     
            st.button("DRAW", key = "drawButton", on_click= runNow)
        #buttonLoc.empty()
        #with buttonLoc:                   
        #    st.button("DRAW", key = "drawButton")
        #st.experimental_rerun()
        #runCalculations()
        #success = st.success('Done!')
        #time.sleep(0.1)
        #success.empty()
   
        
        

def runCalculations():
   
    if "geneList2" not in st.session_state and st.session_state['tabs'] != "Regulation":
        "Please enter the genes that you are interested in to the text area!"
        cancel()
    
    if st.session_state['tabs'] == "Correlation":        
        #Get correlations for perturbations        
        data = calculateCorrelations()
        time3 = time.time()
        print("Step3 Calc Corr:"  + str(time3 - time2))
        #styles = [dict(selector="th", props=[('width', '20px')]), dict(selector="th.col_heading",props=[("writing-mode", "vertical-rl"),('transform', 'rotateZ(-90deg)'),('height', '290px'),('vertical-align', 'top')])]
        #st.dataframe(st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] ))
        #st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] )
        #st.write(st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] ))
                
        if data is not None:
            st.title("Correlation Table") 
            try:
                st.write(data.style.format("{:.2f}"))
            except:
                "This is a huge table. Please download it to view!"
                "Map generation may take some time. Please wait..."
                pass
            csv = convert_df(data)           
            st.download_button(label="Download Corr. Table",data=csv,file_name='CorrelationTable.csv', mime='text/csv', key = "downloadTableBtn", on_click = rerun)
        
                        
        time4 = time.time()
        print("Step4 Corr Table:"  + str(time4 - time3))
    
        if data is not None:
            plotCorrMap(data)   
            if "g" in st.session_state:
                st.title("Correlation map of " + st.session_state.dataType)             
                plt.rcParams['figure.dpi'] = 100
                st.pyplot(st.session_state.g._figure)
                img = io.BytesIO()       
                st.session_state.g._figure.savefig(img, format='png')
                st.download_button(label="Download image",data=img,file_name="Corr.png",mime="image/png", key = "btn1", on_click = rerun)
            
        time5 = time.time()
        print("Step5 PlotCorrmAP1:"  + str(time5 - time4))       
        data = None        
        st.session_state.g = None 
        st.session_state.rerun = 0        
              
   
    if st.session_state['tabs'] == "PCA":
        time6 = time.time()
        result =calculatePCA()
        
        if "geneList" not in st.session_state:            
            cancel()
        else:
            st.session_state["PcaResult"] = result           
            if not st.session_state.hdbClusteringPCA:
                st.title("PCA")
                plotGraph(st.session_state["PcaResult"])            
            if st.session_state.hdbClusteringPCA:
                runHDBClustering(0)         
        time7 = time.time()
        print("Step5B pca RESULTS:"  + str(time7 - time6))        
        
          
 
    if st.session_state['tabs'] == "MDE":      
        time7 = time.time()       
        
        st.session_state["pymdeResult"] = pymdeEmbeding()
        if st.session_state["pymdeResult"] is None:
            "Something went wrong with MDE embeding!"
            return 
                     
        time9 = time.time()
        print("Step6 MDE Embeding b:"  + str(time9 - time7))
        
        if st.session_state.hdbClusteringMDE:
            try:
                runHDBClustering(1)
            except Exception as e:                
                st.warning("HDB clustering has failed!")                
                plotGraph(st.session_state["pymdeResult"]) 
        else:
            st.title("MDE Embeding")
            plotGraph(st.session_state["pymdeResult"])      
    
    if st.session_state['tabs'] == "UMAP":      
        time8 = time.time()   
        st.session_state["UMAPResult"] = umapPlot()
        if st.session_state["UMAPResult"] is None:
            return 
               
        time9 = time.time()
        print("Step6 UMAP Embeding:"  + str(time9 - time8))
        
        #2D 3D graphs
        
        if not st.session_state.hdbClusteringUMAP:
            st.title("UMAP Embeding")
            plotGraph(st.session_state["UMAPResult"])
        
        time10 = time.time()
        print("Step7 UMAP Embeding b:"  + str(time10 - time9))
        if st.session_state.hdbClusteringUMAP:
            runHDBClustering(2)  
            
    if st.session_state['tabs'] == "tSNE":      
        time10 = time.time()
        
        st.session_state["tSNEResult"] = tsnePlot()
        if st.session_state["tSNEResult"] is None:
            return 
               
        time11 = time.time()
        print("Step8 tSNE Embeding:"  + str(time11 - time10))        
       
        if not st.session_state.hdbClusteringtSNE:
            st.title("tSNE Embeding")
            plotGraph(st.session_state["tSNEResult"])   

    
        time12 = time.time()
        print("Step9 tSNE Embeding:"  + str(time12 - time11))
        if st.session_state.hdbClusteringtSNE:
            runHDBClustering(3)     
    
    if st.session_state['tabs'] == "biCluster":      
        time13 = time.time()
        print("SpectralCoClustering 1")
        st.session_state["biClusteringResult"] = SpectralCoClustering()
        if st.session_state["biClusteringResult"] is None:
            return 
               
        time14 = time.time()
        print("Step10 biClustering:"  + str(time14 - time13))        
       
        if not st.session_state.hdbClusteringtSNE:
            st.title("Spectral Co-Clustering (Bi-clustering)")
            plotGraph(st.session_state["biClusteringResult"])   

    
        #time15 = time.time()
        #print("Step11 biClustering:"  + str(time15 - time14))
        #if st.session_state.hdbClusteringtSNE:
        #runHDBClustering(3)  
        
    if st.session_state['tabs'] == "HeatMap":  
       print("HeatMap Generation")
       drawHeatMap()
                  
    if st.session_state['tabs'] == "Regulation":
        styles = [dict(selector="th", props=[('width', '20px')]),
                  dict(selector="th.col_heading",
                   props=[("writing-mode", "vertical-rl"),
                          ('transform', 'rotateZ(-90deg)'), 
                          ('height', '290px'),
                          ('vertical-align', 'top')])]
        
        if st.session_state['type2'] or st.session_state['type2Check'] and not st.session_state['type1']:
            st.session_state['type2Check'] = True
            st.session_state['type1Check'] = False
            st.session_state['type3Check'] = False
            st.session_state['type4Check'] = False
            
            #rawDataFile
            gene = st.session_state['geneSelection']
            dreg = pd.read_parquet("K562_Orginal_Zscore.parquet").transpose()  
            dreg.loc['Population'] = dreg.mean()
            dreg = dreg.filter(items = [gene, "Population"], axis=0)
            
            if dreg.shape[1]<1:
                st.title("Unfortunately, " + gene + " doesn't exists in the data file!")
                cancel()
            
            st.title("Genes that deregulate " + st.session_state['geneSelection'] + " upon their knockdown") 

            dreg = dreg.transpose()
            fig, ax = plt.subplots(figsize=(10, 1))           
            ax.hist(dreg, bins=80)
            st.pyplot(fig)
            res = None
            
            try:
                res = dreg.loc[[gene],[gene]] #rawdatafile.loc
                if res.shape[0]>0:
                    res            
            except:
                pass
            
            figx = go.Figure()
            figx.add_trace(go.Histogram(x = dreg[gene],bingroup=100))
            figx.update_layout(title_text="Check Up")  
            
            figx.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                label="1m",
                                step="month",
                                stepmode="backward"),
                            dict(count=6,
                                label="6m",
                                step="month",
                                stepmode="backward"),
                            dict(count=1,
                                label="YTD",
                                step="year",
                                stepmode="todate"),
                            dict(count=1,
                                label="1y",
                                step="year",
                                stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                    type="linear"
                )
            )
            
            st.plotly_chart(figx)
             
            #hist = dreg.hist(bins=10)
            col1, col2, col3 = st.columns(3,)
            with col1:
                st.markdown("<h3 style='text-align: justify; color: Black;'>All genes</h3>", unsafe_allow_html=True)
                
                st.dataframe(dreg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

                 
            with col2:                
                st.markdown("<h3 style='text-align: justify; color: Blue;'>Targets that downregulate</h3>", unsafe_allow_html=True)

                dreg2 = dreg.loc[dreg[gene] <= (-1 * st.session_state['AbsZScore'])]
                if len(dreg2)>0:
                    dreg2.sort_values(by=gene,inplace = True)
                    #dreg2
                    st.dataframe(dreg2.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

                
            with col3:                
                st.markdown("<h3 style='text-align: justify; color: Red;'>Targets that upregulate</h3>", unsafe_allow_html=True)

                ureg = dreg.loc[dreg[gene] >= st.session_state['AbsZScore']]
                if len(ureg)>0:
                    ureg.sort_values(ascending = False, by=gene,inplace = True )
                    st.dataframe(ureg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
            #st.dataframe(st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] ))
            dreg = None
            dreg2 = None
            ureg = None
        
        if st.session_state['type1'] or st.session_state['type1Check'] and not st.session_state['type2']:
            st.session_state['type1Check'] = True
            st.session_state['type2Check'] = False
            st.session_state['type3Check'] = False
            st.session_state['type4Check'] = False

            RawDataFile = pd.read_parquet("K562_Orginal_Zscore.parquet")
            
            
            gene = st.session_state['geneSelection']          
            dreg = RawDataFile.filter(items = [gene, gene + "_2"], axis=0)
            dreg = dreg.transpose()     #we need this      
            dreg = dreg.rename(columns={gene: 'sg' + gene})
            
            if dreg.shape[1]<1:
                st.title("Unfortunately, sg"  + gene + " doesn't exists in the data file!")
                cancel()
            
            try:
                res = RawDataFile.loc[[gene],[gene]]
                if res.shape[0]>0:
                    res                               
            except:
                pass
            
            st.title("Genes that are deregulated upon knockdown of " + st.session_state['geneSelection']) 

            
            
            fig, ax = plt.subplots(figsize=(10, 1))           
            ax.hist(dreg, bins=80)
            st.pyplot(fig)
            
            if dreg.shape[1]>1:                
                dreg['Average'] = dreg.mean(numeric_only=True, axis=1)
                dreg.sort_values(by="Average",inplace = True)
                dreg = dreg.rename(columns={gene+ "_2": 'sg' + gene + " #2"})
                
           
            #hist = dreg.hist(bins=10)
            col4, col5, col6 = st.columns(3)
            gene = 'sg' + gene  
            with col4:
                st.markdown("<h3 style='text-align: justify; color: Black;'>All genes</h3>", unsafe_allow_html=True)
                dreg.sort_values(by=gene,inplace = True)
                st.dataframe(dreg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
                #st.dataframe(dreg)
            
             
            with col5:                
                st.markdown("<h3 style='text-align: justify; color: Blue;'>Down regulated genes</h3>", unsafe_allow_html=True)
                if len(dreg[gene])>0:              
                    dreg2 = dreg.loc[dreg[gene] <= (-1 * st.session_state['AbsZScore'])].sort_values(by=[gene])              
                    if len(dreg2[gene])>0:    
                        st.dataframe(dreg2.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

                
            with col6:
                st.markdown("<h3 style='text-align: justify; color: Red;'>Up regulated genes</h3>", unsafe_allow_html=True)
                ureg = dreg.loc[dreg[gene] >= st.session_state['AbsZScore']].sort_values(ascending = False, by=[gene])
                
                if len(ureg)>0:
                    if ureg.shape[1]>1:
                        ureg.sort_values(ascending = False, by="Average",inplace = True )
                    else:
                        ureg.sort_values(ascending = False, by=[gene])
                    st.dataframe(ureg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
            
            RawDataFile = None
            dreg2 = None
            ureg = None
            
        if st.session_state['type3'] or st.session_state['type3Check']:
            st.session_state['type3Check'] = True
            st.session_state['type1Check'] = False
            st.session_state['type2Check'] = False
            st.session_state['type4Check'] = False          

            WGCorrMatrix =  pd.read_parquet("WGCorrMatrix.parquet")  #loadData(1).transpose()
            WGCorrMatrix.shape
            print("WGCorrMatrix loaded!") 
            gene = st.session_state['geneSelection']
            dreg = WGCorrMatrix.filter(items = [gene, gene + "_2"], axis=0)                    
 
            
            if dreg.shape[1]<1:
                st.title("Unfortunately, sg"  + gene + " doesn't exists in the data file!")
                cancel()
            
            
          
            st.title("Perturbations that are correlate with the knockdown of " + st.session_state['geneSelection']) 

            
            #fig, ax = plt.subplots(figsize=(10, 1))           
            #ax.hist(dreg, bins=80)
            #st.pyplot(fig)
            dreg = dreg.transpose()           
            dreg = dreg.loc[(dreg[gene] <= -0.05) | (dreg[gene] >= 0.05)].sort_values(by=[gene])
            dreg = dreg.rename(columns={gene: 'sg' + gene})
            
            
            #
            #if dreg.shape[1]>1:                
            #    dreg['Average'] = dreg.mean(numeric_only=True, axis=1)
            #     dreg.sort_values(by="Average",inplace = True)
            #    dreg = dreg.rename(columns={gene+ "_2": 'sg' + gene + " #2"})
            
                
            print("WGCorrMatrix Check Point 2!") 
            #hist = dreg.hist(bins=10)
            col4, col5, col6 = st.columns(3)
            gene = 'sg' + gene  
            with col4:
                st.markdown("<h3 style='text-align: left; color: Black;'>All perturbations</h3>", unsafe_allow_html=True)
                dreg.sort_values(by=gene,inplace = True)
                st.dataframe(dreg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

            
             
            with col5:                
                st.markdown("<h3 style='text-align: left; color: Blue;'>Negatively correlated perturbations</h3>", unsafe_allow_html=True)
                if len(dreg[gene])>0:              
                    dreg2 = dreg.loc[dreg[gene] <= (-1 * st.session_state['AbsZScore'])].sort_values(by=[gene])              
                    if len(dreg2[gene])>0:  
                        st.dataframe(dreg2.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

                
            with col6:
                st.markdown("<h3 style='text-align: left; color: Red;'>Positively correlated perturbations</h3>", unsafe_allow_html=True)
                ureg = dreg.loc[dreg[gene] >= st.session_state['AbsZScore']].sort_values(ascending = False, by=[gene])
                
                if len(ureg)>0:
                    if ureg.shape[1]>1:
                        ureg.sort_values(ascending = False, by="Average",inplace = True )
                    else:
                        ureg.sort_values(ascending = False, by=[gene])
                    st.dataframe(ureg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
            ureg = None
            dreg= None
            dreg2 = None
            WGCorrMatrix = None
            
        if st.session_state['type4'] or st.session_state['type4Check'] and not st.session_state['type2']:
            st.session_state['type4Check'] = True
            st.session_state['type1Check'] = False
            st.session_state['type2Check'] = False
            st.session_state['type3Check'] = False
            
            GECorrMatrix = pd.read_parquet("GeneExpressionBasedCorrMatrix.parquet")
            
            gene = st.session_state['geneSelection']          
            dreg = GECorrMatrix.filter(items = [gene, gene + "_2"], axis=0) 
            dreg = dreg.transpose()                     
            dreg = dreg.loc[(dreg[gene] <= -0.05) | (dreg[gene] >= 0.05)].sort_values(by=[gene])
            dreg = dreg.rename(columns={gene: 'sg' + gene})            
            
            print("GECORR CP1")
            if dreg.shape[1]<1:
                st.title("Unfortunately, sg"  + gene + " doesn't exists in the data file!")
                cancel()
            
            
           
            dreg = dreg.rename(columns={gene: 'sg' + gene})
            st.title("Genes that show similar response to genetic perturbations and " + st.session_state['geneSelection'] + " ") 

            fig, ax = plt.subplots(figsize=(10, 1))           
            ax.hist(dreg, bins=80)
            #st.pyplot(fig)
            print("GECORR CP3")
            print(dreg.shape)
            if dreg.shape[1]>1:                
                dreg['Average'] = dreg.mean(numeric_only=True, axis=1)
                dreg.sort_values(by="Average",inplace = True)
                dreg = dreg.rename(columns={gene+ "_2": 'sg' + gene + " #2"})
                
           
            #hist = dreg.hist(bins=10)
            col4, col5, col6 = st.columns(3)
            gene = 'sg' + gene  
            with col4:
                st.markdown("<h3 style='text-align: left; color: Black;'>All correlations</h3>", unsafe_allow_html=True)
                dreg.sort_values(by=gene,inplace = True)
                
                copy_button = Button(label="Copy DF", width = 10)
                copy_genes = Button(label="Copy Genes", width = 10)
                copy_button.js_on_event("button_click", CustomJS(args=dict(df=dreg.to_csv(sep='\t')), code="""
                navigator.clipboard.writeText(df);
                """))
                
                copy_genes.js_on_event("button_click", CustomJS(args=dict(df=dreg.index), code="""
                navigator.clipboard.writeText(df);
                """))                
               
                st.dataframe(dreg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
                no_event = streamlit_bokeh_events(copy_button,events="GET_TEXT",key="get_text",refresh_on_update=True,override_height=38,debounce_time=0)
                no_event2 = streamlit_bokeh_events(copy_genes,events="GET_TEXT",key="get_genes",refresh_on_update=True,override_height=38,debounce_time=0)
                
            
             
            with col5:                
                st.markdown("<h3 style='text-align: left; color: Blue;'>Negatively correlated genes</h3>", unsafe_allow_html=True)
                if len(dreg[gene])>0:              
                    dreg2 = dreg.loc[dreg[gene] <= (-1 * st.session_state['AbsZScore'])].sort_values(by=[gene])              
                    if len(dreg2[gene])>0: 
                        st.dataframe(dreg2.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))

                
            with col6:
                st.markdown("<h3 style='text-align: left; color: Red;'>Positively correlated genes</h3>", unsafe_allow_html=True)
                ureg = dreg.loc[dreg[gene] >= st.session_state['AbsZScore']].sort_values(ascending = False, by=[gene])
                
                if len(ureg)>0:
                    if ureg.shape[1]>1:
                        ureg.sort_values(ascending = False, by="Average",inplace = True )
                    else:
                        ureg.sort_values(ascending = False, by=[gene])
                    st.dataframe(ureg.style.format("{:.2f}").background_gradient(cmap='bwr', vmin =-1,vmax = 1 ))
            print("GECORR CP3")
            ureg = None
            dreg= None
            dreg2 = None
            GECorrMatrix = None
        #st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] )
        #st.write(st.session_state.corrResult.style.format("{:.2f}").background_gradient(cmap=st.session_state.colorx, vmin =st.session_state.cRanger[0],vmax = st.session_state.cRanger[1] ))
    st.session_state.rerun = 0    

def drawHeatMap():        
    
    genes = st.session_state.geneList
    genes2=[]
    targets =[]
       
    for x in genes:
        genes2.append(x)
        genes2.append(x + "_2")
    if "targetList" not in st.session_state:
        st.warning("Please enter genes to target list")
        return
    
    targetList = st.session_state.targetList.replace(';',',').replace(' ', ',').replace('\n',',').split(',') 
    for x in targetList:
        targets.append(x)
            
    if len(genes2)>1 and len(targets)>1:
        
        results = pd.read_parquet("K562_Orginal_Zscore.parquet").filter(items = genes2, axis =0)
        results = results.filter(items = targets, axis =1)       
        
        zScore = None
        if st.session_state.zScore2 =="None":
            zScore = None
        elif st.session_state.zScore2 =="Rows":
            zScore = 0
        else:
            zScore = 1        
        
        sscale = None   
        if st.session_state.sscale2 =="None":
            sscale = None
        elif st.session_state.sscale2 =="Rows":
            sscale = 0
        else:
            sscale = 1
        
        #fig = plt.Figure(figsize=(1*int(zoomFactor), 1*int(zoomFactor)))      
        #3 + 0.25 * len(gids_found)  
        st.write(len(results))      
        g = sns.clustermap(results, z_score = zScore,        
        standard_scale = sscale,  metric = st.session_state.distanceMetric2,method = st.session_state.linkageSelection2,
        
        vmin=st.session_state.cRanger2[0], vmax=st.session_state.cRanger2[1],cmap = st.session_state.colorx2, figsize=(3 + 0.26 * len(results.index),3 + 0.26 * len(results)))
        plt.rcParams['figure.dpi'] = 100                
        st.pyplot(g._figure)
        img2 = io.BytesIO()       
        g._figure.savefig(img2, format='png')
        st.download_button(label="Download image",data=img2,file_name="HeatMap.png",mime="image/png", key = "btn2", on_click = rerun)

        #st.altair_chart(g._figure)
        
        
        
        
                
        # corrResults = corrResults[corrResults.nlargest(2)>0.6]        
    else:
        st.error("Please check your gene list, or click DRAW button!") 
        return None             
    
    return results  

def plotGraph(data):
       
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    fig = plt.figure()
    
        
    if  st.session_state.graphType == "3D":
        if len(data)>2:
            ax = fig.add_subplot(projection='3d') 
            ax.scatter(*data.T[0:3:], s=20, linewidth=0, c='b', alpha=0.45)
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')
            fig
    elif st.session_state.graphType == "2D":
        if len(data)>1:
            ax = fig.add_subplot()  
            ax.scatter(*data.T[0:2:], s=20, linewidth=0, c='b', alpha=0.45)
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')
            fig
    elif st.session_state.graphType == "Both":
        if len(data)>1:
            ax = fig.add_subplot()
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')  
            ax.scatter(*data.T[0:2:], s=20, linewidth=0, c='b', alpha=0.45)
        if len(data)>2:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(projection='3d') 
            ax2.scatter(*data.T[0:3:], s=20, linewidth=0, c='b', alpha=0.45)
            plt.xlabel('First Dimension')
            plt.ylabel('Second Dimension')            
            ax2.set_zlabel('Third Dimension')
            fig
            fig2
        
def runHDBClustering(x):
    fig2D = None
    fig3D = None
    currentHDBScan= None
    currenthdbClustering = None
    data  = None
    scatter = None    
  
    if x==0:
        st.title("PCA & HDB Scan Clustering")
        fig2D, fig3D , scatter,st.session_state["HDBScanResultPCA"] = HDBScanClustering("PcaResult",st.session_state["PcaResult"])
        currentHDBScan = st.session_state["HDBScanResultPCA"]
        currenthdbClustering = st.session_state["hdbClusteringPCA"]
        data  =st.session_state["PcaResult"]
    elif x==1:
        st.title("MDE Embeding & HDB Scan Clustering")        
        fig2D, fig3D ,scatter, st.session_state["HDBScanResultMDE"] = HDBScanClustering("pymdeResult",st.session_state["pymdeResult"])
        currentHDBScan = st.session_state["HDBScanResultMDE"]
        currenthdbClustering = st.session_state["hdbClusteringMDE"]
        data  =st.session_state["pymdeResult"]
    elif x==2:
        st.title("UMAP & HDB Scan Clustering")
        fig2D, fig3D ,scatter, st.session_state["HDBScanResultUMAP"] = HDBScanClustering("UMAPResult",st.session_state["UMAPResult"])
        currentHDBScan = st.session_state["HDBScanResultUMAP"]
        currenthdbClustering = st.session_state["hdbClusteringUMAP"]
        data  =st.session_state["UMAPResult"]
    elif x==3:
        st.title("tSNE Reduction & HDB Scan Clustering")
        fig2D, fig3D ,scatter, st.session_state["HDBScanResulttSNE"] = HDBScanClustering("tSNEResult",st.session_state["tSNEResult"])
        currentHDBScan = st.session_state["HDBScanResulttSNE"]
        currenthdbClustering = st.session_state["hdbClusteringtSNE"]
        data  =st.session_state["tSNEResult"]
            
    time10 = time.time()        
    if currentHDBScan.labels_.max()>-1:
        st.write(str(currentHDBScan.labels_.max() + 1) + " clusters were identified!")
    else:
        st.write("No clusters were identified!")
        plotGraph(data) 
        return
       
    if  st.session_state.graphType == "3D":
        fig3D      
        
    elif st.session_state.graphType == "2D": 
        fig2D
    elif st.session_state.graphType == "Both": 
               
        # CODE TO ADD
        # Define some CSS to control our custom labels
        css = """
        table{border-collapse: collapse;}
        th{color: #ffffff;background-color: #000000;}
        td{background-color: #cccccc;}
        table, th, td{font-family:Arial, Helvetica, sans-serif;border: 1px solid black;text-align: right;}
        """
        for axes in fig2D.axes:  
            labels = []                   
            for label in st.session_state["Labels"]:               
                # Create a label for each point with the x and y coords
                html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>{label}</th></tr></tbody> </table>'
                labels.append(html_label)
            # Create the tooltip with the labels (x and y coords) and attach it to each line with the css specified
            tooltip = plugins.PointHTMLTooltip(scatter, labels=labels, css=css)
            # Since this is a separate plugin, you have to connect it
            plugins.connect(fig2D, tooltip)
        fig_html = mpld3.fig_to_html(fig2D)
        fig_html = fig_html.replace('"fontsize": 10.0', '"fontsize": 18.0')     
        #print(fig_html)
        components.html(fig_html,width =1200, height=900)         
        fig2D     
        fig3D
         
                     
    gseaDatasets =  getDatasets()
    clusterLoc.multiselect('GSEA Datasets',gseaDatasets, key ="gseaDataset",disabled = not currenthdbClustering,on_change= rerun, )
    if "gseaDataset" not in st.session_state:
        "Please select a geneset enrcihment dataset!"
        return
    
    clusterLoc.multiselect('Select Cluster(s)', st.session_state.clusterNames, key ="clusterSelection",disabled = not currenthdbClustering,on_change= rerun)
    clusterLoc.select_slider('p Value Treshold',['1.0','0.05','0.01','0.001','0,0001'],"0.05", key ="pValTresh",on_change= rerun)
    clusterLoc.select_slider('q Value Treshold',['1','0.05','0.01','0.001','0,0001'],"0.05", key ="qValTresh",on_change= rerun)
    clusterLoc.slider('Min Number Of Genes',1,20,3,1, key ="minGenes",on_change= rerun)
                

    st.title("Enrichment Results")
    if len(st.session_state.clusterSelection)==0:
        st.write("For enrichment analyses you need to select one or more cluster(s) from the sidebar!")
    else:
        st.session_state["EnrichResults"] = PerformEnrichR(x)
    
        if st.session_state["EnrichResults"] is None or st.session_state["EnrichResults"].empty:
            st.write("Nothing to show!")                    
        else:
            st.session_state["EnrichResults"]                
    
            
    time11 = time.time()
    print("Step8 EnrichR:"  + str(time11 - time10))   

@st.experimental_memo
def convert_df(df):
     return df.to_csv().encode('utf-8')


def calculateCorrelations():   
    #method2 =  st.session_state.corrMethod  
    if "geneList" not in st.session_state:
        "Upps. Gene list is empty."
        return     
      
    if len(st.session_state.geneList)>2: 
        if st.session_state.dataType == "Perturbation":
            corrResults = getData("2")            
        else:
            corrResults = getData("3")    
      
        corrResults = corrResults.filter(items = st.session_state.geneList, axis =0)
             
        if "removeLowCorr" in st.session_state and st.session_state["removeLowCorr"]:          
            corrResults = corrResults.replace(1,-99)
            dfMax = corrResults.max(axis=1)            
            dfMax = dfMax[dfMax>st.session_state["corrLimitLow"]]
            
            corrResults = corrResults.replace(-99,1)
            corrResults = corrResults.replace(-1,99)
            dfMin = corrResults.min(axis=1)            
            dfMin = dfMin[dfMin<(st.session_state["corrLimitLow"]*-1)]            
            combined= list(dfMin.index.union(dfMax.index.values).values)            
            corrResults = corrResults.replace(99,-1)
            
            corrResults = corrResults.filter(items = combined)
            corrResults = corrResults.filter(items = combined, axis =0)           
    else:
        st.error("Please check your gene list, or click DRAW button!") 
        cancel()           
    
    return corrResults  


def plotCorrMap(data):        
        if data.empty:
            return
        
        if st.session_state.zScore =="None":
            zScore = None
        elif st.session_state.zScore =="Rows":
            zScore = 0
        else:
            zScore = 1        
            
        if st.session_state.sscale =="None":
            sscale = None
        elif st.session_state.sscale =="Rows":
            sscale = 0
        else:
            sscale = 1
        
        #fig = plt.Figure(figsize=(1*int(zoomFactor), 1*int(zoomFactor)))      
        #3 + 0.25 * len(gids_found)        
        st.session_state["g"] = sns.clustermap(data, z_score = zScore,        
        standard_scale = sscale,  metric = st.session_state.distanceMetric,method = st.session_state.linkageSelection,
        vmin=st.session_state.cRanger[0], vmax=st.session_state.cRanger[1],cmap = st.session_state.colorx, figsize=(3 + 0.26 * len(data),3 + 0.26 * len(data)))


def hasher(data):
    cdata = [str(numeric) for numeric in data]
    s = "-"
    s = s.join(cdata)
    hashed = hash(s)
    if hashed not in st.session_state['dataStore']:         
        return False, hashed
    else:
        return True, st.session_state.dataStore[hashed]

@st.cache
def calculatePCA():
    if "geneList" not in st.session_state:
        "Upps. Gene list is empty."
        return None
    
    PCASource = None
    if st.session_state.pcaSource == "Raw Data":  # else "Correlation Data"   
        if st.session_state.dataType == "Perturbation":  
            PCASource = getData("1").transpose() 
        else:
            PCASource = getData("0").transpose()  
    else:
        PCASource = calculateCorrelations()
                   
                  
            
    n_components_ = min (st.session_state.numOfPCAComponents, len(PCASource.columns)-1,len(PCASource)-1)
   
    pca = PCA(n_components=n_components_)
    st.session_state["Labels"] = np.array(PCASource.index)
    result =pca.fit_transform(PCASource) 
    return result

@st.cache
def pymdeEmbeding():
    MdeEmbedSource = None  
    if st.session_state.embedSource == "Raw Data":  # else "Correlation Data"
        if st.session_state.dataType == "Perturbation":  
            MdeEmbedSource = getData("1") 
        else:
            MdeEmbedSource = getData("0")        
        st.session_state["Labels"] = np.array(MdeEmbedSource.index)
        MdeEmbedSource =MdeEmbedSource.to_numpy()       
    elif st.session_state.embedSource == "PCA Data":
        if "PcaResult" in st.session_state:
            MdeEmbedSource = st.session_state["PcaResult"]
        if MdeEmbedSource is None:
            "PCA calculation was not performed! Please go to PCA tab to perform PCA or change the Embeding source from the sidebar!" 
            return                 
    else:
        "Here it is 1"
        if "corrResult" in st.session_state:
            "Here it is"
            MdeEmbedSource = calculateCorrelations()
            st.session_state["Labels"] = np.array(MdeEmbedSource.index) 
            MdeEmbedSource = MdeEmbedSource.to_numpy()
    
   
        
    if st.session_state.pyMdeConstraint == "Standardized":
        constraint_ = pymde.Standardized()    
    else:
        constraint_ = pymde.Centered()    
    
    #exist, value = hasher (["pyMDE", st.session_state.embedSource,st.session_state.pyMdeConstraint,len(MdeEmbedSource), st.session_state.repulsiveFraction , st.session_state.embeddingDim]) 
    #if exist:
        #return value
    #else:        
    pymde.seed(0)        
    result =  pymde.preserve_neighbors(MdeEmbedSource,constraint=constraint_,
    repulsive_fraction=st.session_state.repulsiveFraction , embedding_dim=st.session_state.embeddingDim, verbose=True).embed()
        #st.session_state.dataStore[value] =result
    return result

def umapPlot():
    UMAPEmbedSource = None  
    if st.session_state.UMAPembedSource == "Raw Data":  # else "Correlation Data"
        UMAPEmbedSource = rawDataFile.to_numpy()
        st.session_state["Labels"] = np.array(rawDataFile.index)       
    elif st.session_state.UMAPembedSource == "PCA Data":
        if "PcaResult" in st.session_state:
            UMAPEmbedSource = st.session_state["PcaResult"]
        if UMAPEmbedSource is None:
            "PCA calculation was not performed! Please go to PCA tab to perform PCA or change the UMAP source from the sidebar!" 
            return                 
    else:
        if "corrResult" in st.session_state:
            UMAPEmbedSource = calculateCorrelations()
            st.session_state["Labels"] = np.array(UMAPEmbedSource.index) 
            UMAPEmbedSource = UMAPEmbedSource.to_numpy()

 
    exist, value = hasher (["uMAP",st.session_state.UMAPembedSource,st.session_state.UMAPembeddingDim
                            ,st.session_state.UMAPmin_dist,st.session_state.UMAPn_neighbors ,len(UMAPEmbedSource)]) 
    if exist:
        return value
    else:       
        result =  umapReduce(UMAPEmbedSource, ncomponents =st.session_state.UMAPembeddingDim, n_neighbors=st.session_state.UMAPn_neighbors, min_dist=st.session_state.UMAPmin_dist, metric='euclidean')
        st.session_state.dataStore[value] =result
        return result   

@st.cache
def tsnePlot():
    
    tSNEEmbedSource = None  
    if st.session_state.tSNEembedSource == "Raw Data":  # else "Correlation Data"
        tSNEEmbedSource = rawDataFile.to_numpy()
        st.session_state["Labels"] = np.array(rawDataFile.index)       
    elif st.session_state.tSNEembedSource == "PCA Data":
        if "PcaResult" in st.session_state:
            tSNEEmbedSource = st.session_state["PcaResult"]
        if tSNEEmbedSource is None:
            "PCA calculation was not performed! Please go to PCA tab to perform PCA or change the tSNE source from the sidebar!" 
            return                 
    else:
        if "corrResult" in st.session_state:
            tSNEEmbedSource = calculateCorrelations()
            st.session_state["Labels"] = np.array(tSNEEmbedSource.index)
            tSNEEmbedSource = tSNEEmbedSource.to_numpy()     
   
    #exist, value = hasher (["tSNE",st.session_state.tSNEembedSource,st.session_state.tSNEPerplexity,len(tSNEEmbedSource), st.session_state.tSNElearning_rate , st.session_state.tSNEn_iter, st.session_state.tSNEearly_exaggeration]) 
    #if exist:
        #return value
    #else:  
    tSNEEmbedSource     
    result = tsneReduce(tSNEEmbedSource, perp=st.session_state.tSNEPerplexity, learning_rate=st.session_state.tSNElearning_rate, n_iter=st.session_state.tSNEn_iter, early_exaggeration=st.session_state.tSNEearly_exaggeration)
    #st.session_state.dataStore[value] =result
    return result

def SpectralCoClustering():
    print("SpectralCoClustering")
    clusterCount =5
    if "biClusteringCount" in st.session_state:
        clusterCount = st.session_state.biClusteringCount
    print("Clustering count " + str(clusterCount))
    biClusteringSource = None  
    if st.session_state.biClusteringembedSource == "Raw Data":  # else "Correlation Data"
        biClusteringSource = rawDataFile.to_numpy()
        st.session_state["Labels"] = np.array(rawDataFile.index)       
    elif st.session_state.biClusteringembedSource == "PCA Data":
        if "PcaResult" in st.session_state:
            biClusteringSource = st.session_state["PcaResult"]
        if biClusteringSource is None:
            "PCA calculation was not performed! Please go to PCA tab to perform PCA or change the clustering source from the sidebar!" 
            return                 
    else:
        if "corrResult" in st.session_state:
            
            biClusteringSource = calculateCorrelations()
            st.session_state["Labels"] = np.array(biClusteringSource.index) 

   
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    fig = plt.figure()   
   
            
    model = SpectralCoclustering(n_clusters=int(clusterCount), random_state=0)
    model.fit(biClusteringSource)    
    idx = np.argsort(model.row_labels_)
    newIndex = [biClusteringSource.index[x] for x in idx]
        
    fit_data = biClusteringSource.reindex(newIndex)
    idx = np.argsort(model.column_labels_)
    columns = biClusteringSource.columns.tolist()
    newIndex = [columns[x] for x in idx]
    fit_data = fit_data[newIndex]
      
    plt.title("After biclustering; rearranged to show biclusters")    
    
    grph = sns.clustermap(fit_data, z_score = None,        
        standard_scale = None,        
        cmap = plt.cm.bwr,row_cluster = False,col_cluster = False,  
        figsize=(3 + 0.24 * len(fit_data),
                 3 + 0.24 * len(fit_data)), vmin=-0.5, vmax=0.5)
    st.pyplot(grph._figure)

    
    
    if "g" in st.session_state:
            #st.pyplot(st.session_state.g._figure)
            plt.rcParams['figure.dpi'] = 100
            st.pyplot(st.session_state.g._figure)

def HDBScanClustering(datasource,data):
    if datasource not in st.session_state:
        return
    
    #Clear previous cluster names
    st.session_state.clusterNames.clear()
    
   
    
    #exist, value = hasher ([datasource, st.session_state.min_cluster_size_,st.session_state.metric_ ,st.session_state.cluster_selection_method_, 
    #                        st.session_state.min_samples_,st.session_state.cluster_selection_epsilon_, len(data), data.size]) 
    #if exist:
    #    clusterer = value
    #else:        
    pymde.seed(0)        
    clusterer = hdbscan.HDBSCAN(min_cluster_size=st.session_state.min_cluster_size_,metric=st.session_state.metric_ ,
                cluster_selection_method =st.session_state.cluster_selection_method_.lower(), 
                min_samples=st.session_state.min_samples_, cluster_selection_epsilon=st.session_state.cluster_selection_epsilon_).fit(data)
    #st.session_state.dataStore[value] =clusterer
        
    clusterCount = clusterer.labels_.max()+1
    if clusterCount>0:
        st.session_state.clusterNames.append("Unclustered")
    for x in range(1, clusterCount+1):
        st.session_state.clusterNames.append("Cluster " + str(x))            
 
    color_palette = sns.color_palette('deep', clusterer.labels_.max()+1)
    cluster_colors = [color_palette[x%20] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
   
    
    clabels = clusterer.labels_.view().reshape(clusterer.labels_.shape[0], -1)    
    
    clusterCenters=[]
    # create a list of legend elemntes
    legend_elements =[]
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
                    markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(color_palette)]
    # If there are unclustered elements.
    if clusterer.labels_.min()<0:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Unclustered', 
                    markerfacecolor= (0.5, 0.5, 0.5), markersize=5))
    i=0
    
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
    interp_x =[]
    interp_y =[]
    
    
                    
    while i < clusterCount:        
        x=0
        y=0
        count =0
        points = []
        for row in range(0,len(clabels)):            
            if int(clabels[row]) ==  i:          
               x=x+ float(data[row][0])
               y=y+ float(data[row][1])
               points.append([float(data[row][0]),float(data[row][1])])
               count+=1               
        points = np.array(points, np.float16)
        
        
           
        if count>0:
            clusterCenters.append([x/count,y/count])            
            try:
                hull = ConvexHull(points)
                # get x and y coordinates
                # repeat last point to close the polygon
                x_hull = np.append(points[hull.vertices,0],
                                points[hull.vertices,0][0])
                y_hull = np.append(points[hull.vertices,1],
                                points[hull.vertices,1][0])
                # interpolate
                dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
                dist_along = np.concatenate(([0], dist.cumsum()))
                spline, u = interpolate.splprep([x_hull, y_hull],u=dist_along, s=0, per=1)
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interpx, interpy = interpolate.splev(interp_d, spline)
                interp_x.append(interpx) 
                interp_y.append(interpy) 
            except:
                pass                      
        i+=1
    fig3D = None
    fig2D = None
    scatter = None
    if len(data)>2:   
        fig3D = plt.figure()    
        ax = fig3D.add_subplot(projection='3d') 
        ax.scatter(*data.T[0:3:], s=20, linewidth=0, c=cluster_member_colors, alpha=0.98)  
        if st.session_state.clusteringLegend:
            plt.legend(handles=legend_elements, loc='best')           
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')            
        ax.set_zlabel('Third Dimension')        
        
    if len(data)>1:  
        fig2D = plt.figure()   
        ax2 = fig2D.add_subplot() 
        scatter = ax2.scatter(*data.T[:2,:], s=20, linewidth=0, c=cluster_member_colors, alpha=0.9)
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')            
        # plot legend
        if st.session_state.clusteringLegend:
            plt.legend(handles=legend_elements, loc='best')
        
        # Highlight clusters
        if st.session_state.clusteringHighlight:   
            for i in range(0,len(interp_x)):       
                plt.fill(interp_x[i], interp_y[i], alpha=0.3, c=color_palette[i])
    
    # Show cluster centers
    cl =1    
    if st.session_state.clusteringCenter:   
        for xv in clusterCenters:
            plt.text(xv[0],xv[1], s= str(cl), bbox= dict(boxstyle="round",ec=(0.3, 0.5, 0.5), fc=(0.8, 0.9, 0.9), alpha=0.8))
            cl=cl+1
    #plt.ion()
    return fig2D, fig3D, scatter, clusterer

def PerformEnrichR(x):
    embedding = None
    clusterer = None
    if x ==0:
        if "HDBScanResultPCA" not in st.session_state or "PcaResult" not in st.session_state:
            return
        embedding = st.session_state["PcaResult"]   
        clusterer = st.session_state["HDBScanResultPCA"] 
    if x ==1:
        if "HDBScanResultMDE" not in st.session_state or "pymdeResult" not in st.session_state:
            return
        embedding = st.session_state["pymdeResult"]   
        clusterer = st.session_state["HDBScanResultMDE"] 
    if x ==2:
        if "HDBScanResultUMAP" not in st.session_state or "UMAPResult" not in st.session_state:
            return
        embedding = st.session_state["UMAPResult"]   
        clusterer = st.session_state["HDBScanResultUMAP"] 
    if x ==3:
        if "HDBScanResulttSNE" not in st.session_state or "tSNEResult" not in st.session_state:
            return
        embedding = st.session_state["tSNEResult"]   
        clusterer = st.session_state["HDBScanResulttSNE"]  
  
    
    if "gseaDataset" not in st.session_state or len(st.session_state.gseaDataset)<1:
        return 
    
    
    
    exist, value = hasher ([len(embedding), len(clusterer.labels_), st.session_state.Labels,st.session_state.clusterSelection]) 
    if exist:
        return value
    else:        
        clabels = clusterer.labels_.view().reshape(clusterer.labels_.shape[0], -1)
        cprobs= clusterer.probabilities_.view().reshape(clusterer.probabilities_.shape[0], -1)
        
        merged = np.concatenate([clabels, cprobs, embedding], axis=1)    
        geneLabels = st.session_state.Labels   
        
        
        tempArray = []
        #Cluster genes fil
        clusterGenes = "" #open("/cluster/work/users/omerfk/scRNAseq/Temp/" + outputPrefix + "cluster_genes" + GeneExp +".txt", 'w+')
        #gseaResults = open("/cluster/work/users/omerfk/scRNAseq/Temp/"+ outputPrefix + "GSEA_results" + GeneExp +".txt", 'w+')
        
        x=float(0)
        y=float(0)
        clusterCenters=[]
        
        selected_clusters = st.session_state.clusterSelection
        
        for cluster in selected_clusters:
            x=0
            y=0
            clusterNumber = ""
            if cluster == "Unclustered":
                clusterNumber = -1
            else:
                clusterNumber = int(cluster.replace("Cluster ", ""))-1
                
            tempArray.clear()
            #Collect gene names in the cluster      
            
            for row in range(0,len(merged)):            
                if int(merged[row][0]) == clusterNumber:            
                    tempArray.append(geneLabels[row].split('_')[0])     # tempArray.append([row[0],row[2]])  
                    x=x+ float(merged[row][2])
                    y=y+ float(merged[row][3])
            
            #if there is no gene for this cluster continue
            if len(tempArray)==0:
                continue
            
            #Collect the gene list for the cluster and calculate cluster center
            clusterGenes += "\nCluster " + str(tempArray)          
            clusterCenters.append([x/len(tempArray),y/len(tempArray)]);
            
            # Perform clustering
            enr = gp.enrichr(gene_list=tempArray,
                gene_sets= st.session_state.gseaDataset,
                organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                description='test_name',
                outdir='test/enrichr_kegg',
                # no_plot=True,
                cutoff=0.5 # test dataset, use lower value from range(0,1)
                )
            enr.results.drop('Old P-value', axis =1,inplace=True)
            enr.results.drop('Old Adjusted P-value',axis =1,inplace=True)
            enr.results.insert(7,'Gene Count', None )
            enr.results.reset_index()
            for index, y in enr.results.iterrows():
                #print(y['Genes'])            
                enr.results.at[index, 'Gene Count'] = len(str(y['Genes']).split(';'))
                
            st.session_state.dataStore[value] =enr.results
            return enr.results

@st.cache
def getData(dataSetName, genes = None):
    data_file=""
    if genes == None:
        if "geneList" in st.session_state and len(st.session_state.geneList)>0:        
            genes = st.session_state.geneList
        else:
            return None
                  
    timeStart = time.time()  
    if dataSetName == "0" or dataSetName == "RAW":            
        data_file = "K562_Orginal_Zscore.parquet"
    elif dataSetName == "1" or dataSetName == "RAWTransposed":            
        data_file = "K562_Orginal_ZscoreTransposed.parquet"            
    elif dataSetName == "2" or dataSetName == "WGCOR":   
        data_file = "WGCorrMatrix.parquet"  
    elif dataSetName == "3" or dataSetName == "GECOR":         
        data_file = "GeneExpressionBasedCorrMatrix.parquet"  
    else: 
        return None
        
    parquet_file = pq.ParquetFile(data_file)
    columns_in_file = [c for c in genes if c in parquet_file.schema.names]
    filex = pd.read_parquet(data_file,columns = columns_in_file)
    print("Dataset Name:" + dataSetName + "  File:" +  data_file + "  Data load time:"  + str(time.time() - timeStart))
    return filex
        
def loadData(dataSetName):#dataSet =  pd.read_parquet("LRLG_FilteredExpressionFile.pq")
    
    if dataSetName == "0" or dataSetName == "RAW":
        return pd.read_parquet("K562_Orginal_Zscore.parquet")    
    elif dataSetName == "1" or "WGCOR":   
        return pd.read_parquet("WGCorrMatrix.parquet")    
    elif dataSetName == "2" or "GECOR":   
         return pd.read_parquet("GeneExpressionBasedCorrMatrix.parquet") 
    else: 
        return None
 
    time1 = time.time()   
    dataSet = ""
    dataSet =  dd.read_parquet("K562_Orginal_Zscore.parquet", npartitions=10 )  
    print("ds 1b: " + str(time.time()-time1))
    #print(dataSet.index)  
    print(dataSet.columns)
    st.session_state['transposedRawDataFile'] = dataSet
    dataSet =  dd.read_parquet("WGCorrMatrix.parquet")    
    print("ds 2: " + str(time.time()-time1))   
    st.session_state['WGCorrMatrix'] = dataSet
    dataSet =  dd.read_parquet("GeneExpressionBasedCorrMatrix.parquet") 
    st.session_state['GECorrMatrix'] = dataSet   
    print("ds 3: " + str(time.time()-time1))   
    time1 = time.time()       
    

@st.experimental_memo
def getDatasets():
    return gp.get_library_name()
    
    #logging.info('rawDataFile.shape: {}'.format(rawDataFile.shape))   
def rerun():   
    #st.session_state['type1Check'] = st.session_state['type1']
    #st.session_state['type2Check'] = st.session_state['type2']
    
    st.session_state.rerun = 1

def umapReduce(npall, ncomponents =2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    '''
    Use UMAP to reduce the data according to the input parameters.
    Parameters
    ----------
    npall : numpy array
        Numpy array of the input vectors that have been uploaded.
    n_neighbors : int, optional
        Number of nearest neighbors to consider when calculating distance metrics.
        The default is 15.
    min_dist : float, optional
        The minimum distance between points allowed in the output.
        The default is 0.1.
    metric : string, optional
        The distance metric to use. The default is 'euclidean'.
    Returns
    -------
    vects : numpy array
        The reduced vectors in the same order and rows as the input but with a
        dimension of only 2
    '''
    print('Running UMAP...')
   
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors, n_components=ncomponents, min_dist=min_dist, metric=metric)
    vects = reducer.fit_transform(npall)
    return vects

def tsneReduce(npall, perp=40, learning_rate=200, n_iter=1000, early_exaggeration=12) :
    '''
    Use TSNE to reduce the input data. Takes in arguments and passes them 
    directly to the TSNE algorithm.
    Parameters
    ----------
    npall : numpy array
        Numpy array of the input vectors that have been uploaded
   
    perp : float, optional
        Perplexity parameter for TSNE (basically how many nearest neighbors are 
        considered for computing distances). The default is 40.
    learning_rate : float, optional
        Learning rate per iteration. The default is 200.
    n_iter : int, optional
        Max number of iterations allowed. The default is 1000.
    early_exaggeration : float, optional
        Amount of early exaggeration to use in the TSNE algorithm. This
        increases the error in the first several iterations to facailitate
        convergence. The default is 12.
    Returns
    -------
    out_vects numpy array
        The reduced vectors in the same order and rows as the input but with a
        dimension of only 2
    '''
   
    

    if gpu_avail :
        try :
            tsne_vects_out = TSNECuda(n_components=2, early_exaggeration=early_exaggeration, perplexity=perp, learning_rate=learning_rate).fit_transform(npall)
            return tsne_vects_out
        except Exception as e:
            print('Error using GPU, defaulting to CPU TSNE implemenation. Error message:\n\n' + str(e))

    tsne = TSNE(n_components=2, n_iter=n_iter, verbose=3, perplexity=perp, method='barnes_hut', angle=0.5, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_jobs=-1)
    out_vects = tsne.fit_transform(npall[1:,1:])
    return out_vects

def genData() :
    '''
    This main function is the tab for generating the reduced data.
    It controls the flow of the user uploading the data, reducing the data,
    plotting it quickly to check, and saving the data to be fully plotted.
    Returns
    -------
    None.
    '''
    header.title('Generate reduced dimensionality data')
    #dfgene, glen = getDataRaw()
    #if glen == 0 : return
    #try :
    #    dfgene, all_types, sample_names, typecols, avg_cols, warnings = processRawData(dfgene)
    #    if len(warnings) > 0 :
    #        st.warning('\n\n'.join(warnings))
    #except Exception as e:
    #    st.error('Error processing data, please check file requirements in read me and reupload. Error:\n\n' + str(e))
    #    return

    # dfprint = pd.DataFrame(typecols).T.fillna('')
    # dfprint.columns = all_types
    # dfprint.index = ['Sample #{}'.format(i) for i in range(1, len(dfprint)+1)]
    # st.info('Review inferred samples (rows) and types (columns) table below:')
    # st.dataframe(dfprint)

    st.sidebar.header('Reduction Run Parameters')
    ralgo = st.sidebar.selectbox('Reduction algorithm:', ['UMAP', 'TSNE'])
    useUmap = ralgo == 'UMAP'

    param_guide_links = ['https://distill.pub/2016/misread-tsne/', 'https://pair-code.github.io/understanding-umap/']
    st.sidebar.markdown('<a href="{}">Guide on setting {} parameters</a>'.format(param_guide_links[useUmap], ralgo), unsafe_allow_html=True)
    remove_zeros = st.sidebar.checkbox('Remove entries with all zeros?', value=True)
    
    #if len(avg_cols) > 2 :
    #    norm_per_row = st.sidebar.checkbox('Normalize per gene (row)?', value=True) 
    #    norm_control = st.sidebar.checkbox('Normalize to type?', value=False)
    #else :
    #    norm_per_row, norm_control = False, False
    #    st.sidebar.text('Cannot normalize per row or control with only 2 types')    
    
    #if norm_control :
    #    control = st.sidebar.selectbox('Select type for normalization:', all_types)
        
    hyperParamDescs = {'TSNE_PCA' : 'Specifies how many PCA components to reduce the input data to prior to running TSNE to speed up convergence. Enter 0 to disable PCA preprocessing.',
                       'TSNE_Perp' : 'Number of nearest neighbors considered when calculating clusters. Large values focus on capturing global structure at the cost of local details.',
                       'TSNE_LR' : 'Controls how fast TSNE creates clusters. Large values lead to faster convergence but too much can cause inaccuracies and divergence instead.',
                       'TSNE_EE' : 'Early exaggeration of the input points. Larger values force clusters to be more distinct by initially exaggerating their relative distances.',
                       'TSNE_MaxIt' : 'Maximum number of iterations for the TSNE algorithm. Larger values will increase accuracy but will take longer to compute.',
                       'UMAP_NN' : 'The number of nearest neighbors to consider for each input point. Large values focus on capturing global structure at the cost of local details.',
                       'UMAP_MinDist' : 'Minimum distance between output points. Controls how tightly points cluster together with low values leading to more tightly packed clusters.',
                       'UMAP_DistMetric' : 'Distance metric used to determine distance between points.'}

    if not useUmap :
        st.sidebar.markdown('''TSNE PCA Preprocess:''')
        pca_comp = st.sidebar.number_input(hyperParamDescs['TSNE_PCA'], value=0, min_value=0, max_value=len(all_types)-1, step=1)
        st.sidebar.markdown('''TSNE Perplexity:''')
        perp = st.sidebar.number_input(hyperParamDescs['TSNE_Perp'], value=50, min_value= 40 if gpu_avail else 2, max_value=10000, step=10)
        st.sidebar.markdown('''TSNE Learning Rate:''')
        learning_rate = st.sidebar.number_input(hyperParamDescs['TSNE_LR'], value=200, min_value=50, max_value=10000, step=25)
        st.sidebar.markdown('''TSNE Early Exaggeration:''')
        exagg = st.sidebar.number_input(hyperParamDescs['TSNE_EE'], value=12, min_value=0, max_value=10000, step=25)
        if not gpu_avail : 
            st.sidebar.markdown('''TSNE Max Iterations:''')
            max_iterations = st.sidebar.number_input(hyperParamDescs['TSNE_MaxIt'], value=1000, min_value=500, max_value=2000, step=100)
        else :
            max_iterations = 1000
    else :
        st.sidebar.markdown('''UMAP Number of Neighbors:''')
        n_neighbors = st.sidebar.number_input(hyperParamDescs['UMAP_NN'], value=15, min_value=2, max_value=10000, step=10)
        st.sidebar.markdown('''UMAP Minimum Distance:''')
        min_dist = st.sidebar.number_input(hyperParamDescs['UMAP_MinDist'], value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        st.sidebar.markdown('''UMAP Distance Metric:''')
        umap_metrics = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','mahalanobis','cosine','correlation']
        umap_metric = st.sidebar.selectbox(hyperParamDescs['UMAP_DistMetric'], umap_metrics)
        
    if st.sidebar.button('Run {} reduction'.format(ralgo)) :
        status = st.header('Running {} reduction'.format(ralgo))
        dfreduce = dfgene.copy(deep=True)

        if remove_zeros :
            dfreduce = dfreduce.loc[(dfreduce[avg_cols]!=0).any(axis=1)]
        if norm_control or norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols] + sys.float_info.epsilon
        if norm_control :
            dfreduce[avg_cols] = dfreduce[avg_cols].div(dfreduce['avg_'+control], axis=0)
        if norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols].div(dfreduce[avg_cols].sum(axis=1), axis=0)
        if norm_control or norm_per_row :
            dfreduce[avg_cols] = dfreduce[avg_cols].round(decimals=4)

        if (dfreduce[avg_cols].isna().sum().sum() > 0) :
            st.write('!Warning! Some NA values found in data, removed all entries with NAs, see below:', dfreduce[avg_cols].isna().sum())
            dfreduce = dfreduce.dropna()

        data_vects_in = dfreduce[avg_cols].values + sys.float_info.epsilon

        start = time.time()
        if not useUmap :
            lvects = tsneReduce(data_vects_in, pca_components=pca_comp, perp=perp, learning_rate=learning_rate, n_iter=max_iterations, early_exaggeration=exagg)
        else :
            lvects = umapReduce(data_vects_in, n_neighbors, min_dist, umap_metric)
        st.write('Reduction took {:0.3f} seconds'.format((time.time()-start) * 1))

        dfreduce['red_x'] = lvects[:,0]
        dfreduce['red_y'] = lvects[:,1]
        checkMakeDataDir()
        dfreduce.round(decimals=4).to_csv(datadir / 'temp_dfreduce.csv', index=False)
    elif not os.path.exists(datadir / 'temp_dfreduce.csv') :
        return
    else :
        status = st.header('Loading previous vectors')
        dfreduce = pd.read_csv(datadir / 'temp_dfreduce.csv')

    st.sidebar.header('Plot Quick View Options')
    form_func = lambda typ : 'Expression of {}'.format(typ) if typ != 'Type' else typ
    chosen_color = st.sidebar.selectbox('Color data', ['Type'] + all_types, format_func=form_func)
    hue = 'type' if chosen_color == 'Type' else 'avg_' + chosen_color

    if chosen_color == 'Type' :
        ax = sns.scatterplot(data=dfreduce, x='red_x', y='red_y', s=5, linewidth=0.01, hue=hue)
        ax.set(xticklabels=[], yticklabels=[], xlabel='{}_x'.format(ralgo), ylabel='{}_y'.format(ralgo))
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
    else :
        fig, ax = plt.subplots(1)
        plt.scatter(x=dfreduce.red_x.values, y=dfreduce.red_y.values, s=5, linewidth=0.01, c=dfreduce[hue].values, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Expression Level')
        plt.subplots_adjust(top=0.98, left=0.05, right=1, bottom=0.1, hspace=0.0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    status.header('Data plot quick view:')
    st.pyplot()
    st.write('Total number of points: ', len(dfreduce))
    st.sidebar.header('Reduced Data Save')

    if not useUmap :
        suggested_fn = '{}p'.format(perp)
        suggested_fn += '_No0' if remove_zeros else ''
        suggested_fn += '_NR' if norm_per_row else ''
        suggested_fn += '_NC-'+control if norm_control else ''
    else :
        suggested_fn = '{}n{}m'.format(n_neighbors, int(100*np.round(min_dist, 2)))
        suggested_fn += '_met-{}'.format(umap_metric) if not umap_metric == 'euclidean' else ''
        suggested_fn += '_No0' if remove_zeros else ''
        suggested_fn += '_NR' if norm_per_row else ''
        suggested_fn += '_NC-'+control if norm_control else ''

    file_name = removeinvalidchars(st.sidebar.text_input('Data file name:', value=suggested_fn, max_chars=150))
    if len(file_name) > 0 and st.sidebar.button('Save data file') :
        dfsave = dfgene.copy()
        dfsave = pd.merge(dfsave, dfreduce[['geneid', 'red_x', 'red_y']], on='geneid', how='right')
        dfsave = dfsave.round(decimals=3)
        checkMakeDataDir()
        dfsave.to_csv( str(datadir / 'dfreduce_') + file_name + '.csv', index=False)
        st.sidebar.success('File \'{}\' saved!'.format(file_name))



def loadData2():#dataSet =  pd.read_parquet("LRLG_FilteredExpressionFile.pq")   
  filename = "K562_full_from_raw.txtTransposed.parquet"
  
  dt1 = pd.read_parquet(filename)  
  dt1 = dt1.reindex(sorted(dt1.columns), axis=1)

  for col in list(dt1.columns):
      if col.find(".")>-1:         
          dt1.rename({col:col.replace(".", "_")}, axis = 1,inplace = True)      
  for ind in list(dt1.index):
      if ind.find(".")>-1:
          dt1.rename(index={ind:ind.replace(".", "_")}, inplace = True)  
 
  dt1.to_parquet("f" + filename) 
    
    

def findPerturbationsRegulating(geneX):
    #df = pandas_gbq.read_gbq("SELECT GeneSymbol," + geneX + "  FROM `gwperturbseq.NewData.TableNew` ORDER BY " + geneX)
    #df
    return df
    
    
if __name__ == '__main__':
    #loadData2()
    #findPerturbationsRegulating("SLC39A10")   
    if 'rerun' not in st.session_state:
        st.session_state['rerun'] = 1 
        
    if 'gseaResults' not in st.session_state:
        st.session_state['gseaResults'] = {}
         
    if 'clusterNames' not in st.session_state:
        st.session_state['clusterNames'] = []     
        
    if 'EnrichResults' not in st.session_state:
        st.session_state['EnrichResults'] = pd.DataFrame()
        
    if 'dataStore' not in st.session_state:
        st.session_state['dataStore'] = {}
        
    
        
    if 'type2Check' not in st.session_state or 'type1Check' not in st.session_state:
        st.session_state['type1Check'] = False
        st.session_state['type2Check'] = False
        st.session_state['type3Check'] = False
        st.session_state['type4Check'] = False
        
    logging.basicConfig(level=logging.CRITICAL)
    set_page_container_style()
      
    
    main()   
    
    