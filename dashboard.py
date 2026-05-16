import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv', sep=';')
df['holiday'] = df['holiday'].fillna('None')
df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month
df['is_holiday'] = (df['holiday'] != 'None').astype(int)
df['temp_celsius'] = df['temp'] - 273.15
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df = df[df['temp'] > 200]
df.drop_duplicates(inplace=True)
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather_main'])
model = joblib.load('modele_trafic_xgboost.pkl')
jours = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']
meteos = sorted(df['weather_main'].unique())
def analyser_question(question):
    q = question.lower().strip()
    trafic_moyen = int(df['traffic_volume'].mean())
    trafic_max = int(df['traffic_volume'].max())
    trafic_min = int(df['traffic_volume'].min())
    meilleure_heure = df.groupby('hour')['traffic_volume'].mean().idxmin()
    pire_heure = df.groupby('hour')['traffic_volume'].mean().idxmax()
    meilleur_jour_idx = df.groupby('day_of_week')['traffic_volume'].mean().idxmin()
    pire_jour_idx = df.groupby('day_of_week')['traffic_volume'].mean().idxmax()
    meilleur_jour = jours[meilleur_jour_idx]
    pire_jour = jours[pire_jour_idx]
    heure_detectee = None
    for h in range(24):
        if str(h)+'h' in q or str(h)+':00' in q:
            heure_detectee = h
            break
    jour_detecte = None
    for i, j in enumerate(jours):
        if j.lower() in q:
            jour_detecte = i
            break
    meteo_detectee = None
    for m in meteos:
        if m.lower() in q:
            meteo_detectee = m
            break
    if any(x in q for x in ['meilleur moment','quand partir','eviter']):
        trafic_h = df.groupby('hour')['traffic_volume'].mean()
        top3 = trafic_h.nsmallest(3).index.tolist()
        r = 'Recommandations:\n'
        r += 'Meilleures heures : '+str(top3[0])+'h, '+str(top3[1])+'h, '+str(top3[2])+'h\n'
        r += 'A eviter : '+str(pire_heure)+'h\n'
        r += 'Conseil : Partez avant 6h ou apres 20h!'
        return r
    elif any(x in q for x in ['calme','tranquille','faible']):
        r = 'Moments calmes:\n'
        r += 'Meilleure heure : '+str(meilleure_heure)+'h\n'
        r += 'Meilleur jour : '+meilleur_jour
        return r
    elif any(x in q for x in ['pire','maximum','bouchon']):
        r = 'Trafic maximum:\n'
        r += 'Pire heure : '+str(pire_heure)+'h\n'
        r += 'Pire jour : '+pire_jour
        return r
    elif any(x in q for x in ['weekend','week-end','samedi','dimanche']):
        trafic_we = int(df[df['is_weekend']==1]['traffic_volume'].mean())
        trafic_sem = int(df[df['is_weekend']==0]['traffic_volume'].mean())
        diff = int(((trafic_sem-trafic_we)/trafic_sem)*100)
        r = 'Weekend vs Semaine:\n'
        r += 'Semaine : '+str(trafic_sem)+' veh/h\n'
        r += 'Weekend : '+str(trafic_we)+' veh/h\n'
        r += 'Weekend '+str(diff)+'% moins charge!'
        return r
    elif heure_detectee is not None:
        trafic_h = int(df[df['hour']==heure_detectee]['traffic_volume'].mean())
        niveau = 'eleve' if trafic_h > 4000 else 'modere' if trafic_h > 2000 else 'faible'
        r = 'Trafic a '+str(heure_detectee)+'h:\n'
        r += 'Volume : '+str(trafic_h)+' veh/h\n'
        r += 'Niveau : '+niveau
        return r
    elif jour_detecte is not None:
        trafic_j = int(df[df['day_of_week']==jour_detecte]['traffic_volume'].mean())
        r = 'Trafic le '+jours[jour_detecte]+':\n'
        r += 'Volume moyen : '+str(trafic_j)+' veh/h'
        return r
    elif any(x in q for x in ['bonjour','salut','hello']):
        return 'Bonjour! Posez-moi une question sur le trafic!'
    else:
        return 'Essayez: meilleur moment, trafic a 8h, jour calme, weekend'
app = dash.Dash(__name__)
jours2 = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']
app.layout = html.Div(style={'fontFamily':'Arial','padding':'20px','backgroundColor':'#f8f9fa'}, children=[
    html.H1('Dashboard Trafic Routier', style={'textAlign':'center','color':'#2c3e50'}),
    html.Hr(),
    html.H2('Analyse du trafic'),
    html.Div([html.Label('Filtrer par meteo :'),dcc.Dropdown(id='meteo-filter',options=[{'label':m,'value':m} for m in meteos],value=None,placeholder='Toutes les conditions',clearable=True)],style={'width':'40%','marginBottom':'20px'}),
    html.Div([dcc.Graph(id='graph-heure',style={'width':'50%','display':'inline-block'}),dcc.Graph(id='graph-jour',style={'width':'50%','display':'inline-block'})]),
    dcc.Graph(id='graph-heatmap'),
    html.Hr(),
    html.H2('Prediction en temps reel'),
    html.Div([
        html.Label('Heure :'),dcc.Slider(0,23,1,value=8,marks={i:str(i) for i in range(0,24,2)},id='input-hour'),html.Br(),
        html.Label('Jour :'),dcc.Dropdown(id='input-day',options=[{'label':jours2[i],'value':i} for i in range(7)],value=0,style={'width':'30%'}),html.Br(),
        html.Label('Mois :'),dcc.Slider(1,12,1,value=6,marks={i:str(i) for i in range(1,13)},id='input-month'),html.Br(),
        html.Label('Temperature (C) :'),dcc.Slider(-20,40,1,value=15,marks={i:str(i) for i in range(-20,41,10)},id='input-temp'),html.Br(),
        html.Label('Meteo :'),dcc.Dropdown(id='input-meteo',options=[{'label':m,'value':m} for m in meteos],value='Clear',style={'width':'30%'}),html.Br(),
        html.Label('Jour ferie ?'),dcc.RadioItems(id='input-holiday',options=[{'label':'Non','value':0},{'label':'Oui','value':1}],value=0,inline=True),html.Br(),
        html.Div(id='prediction-output',style={'fontSize':'24px','fontWeight':'bold','color':'#27ae60','textAlign':'center','padding':'20px','backgroundColor':'white','borderRadius':'10px','border':'2px solid #27ae60'})
    ],style={'backgroundColor':'white','padding':'20px','borderRadius':'10px'}),
    html.Hr(),
    html.H2('Assistant IA',style={'color':'#8e44ad'}),
    html.Div([
        html.Div(id='chat-history',style={'height':'350px','overflowY':'auto','padding':'15px','backgroundColor':'#f8f9fa','borderRadius':'10px','border':'1px solid #ddd','marginBottom':'15px'}),
        html.Div([
            dcc.Input(id='chat-input',type='text',placeholder='Ex: Quel est le meilleur moment pour voyager ?',style={'width':'80%','padding':'10px','borderRadius':'8px','border':'1px solid #ddd','fontSize':'14px'}),
            html.Button('Envoyer',id='chat-btn',style={'width':'18%','marginLeft':'2%','padding':'10px','backgroundColor':'#8e44ad','color':'white','border':'none','borderRadius':'8px','cursor':'pointer'})
        ],style={'display':'flex'})
    ],style={'backgroundColor':'white','padding':'20px','borderRadius':'10px'}),
])
@app.callback(Output('graph-heure','figure'),Output('graph-jour','figure'),Output('graph-heatmap','figure'),Input('meteo-filter','value'))
def update_graphs(meteo):
    dff = df[df['weather_main']==meteo] if meteo else df
    fig1 = px.bar(dff.groupby('hour')['traffic_volume'].mean().reset_index(),x='hour',y='traffic_volume',title='Trafic moyen par heure',color_discrete_sequence=['#3498db'])
    fig2 = px.bar(dff.groupby('day_of_week')['traffic_volume'].mean().reset_index(),x='day_of_week',y='traffic_volume',title='Trafic moyen par jour',color_discrete_sequence=['#e74c3c'])
    fig2.update_xaxes(tickvals=list(range(7)),ticktext=jours)
    pivot = dff.pivot_table(values='traffic_volume',index='day_of_week',columns='hour',aggfunc='mean')
    fig3 = px.imshow(pivot,title='Carte thermique',color_continuous_scale='YlOrRd',y=jours)
    return fig1,fig2,fig3
@app.callback(Output('prediction-output','children'),Input('input-hour','value'),Input('input-day','value'),Input('input-month','value'),Input('input-temp','value'),Input('input-meteo','value'),Input('input-holiday','value'))
def predict(hour,day,month,temp,meteo,holiday):
    is_weekend = 1 if day >= 5 else 0
    weather_enc = le.transform([meteo])[0] if meteo in le.classes_ else 0
    X_input = pd.DataFrame([[hour,day,month,temp,0,0,50,holiday,is_weekend,weather_enc]],columns=['hour','day_of_week','month','temp_celsius','rain_1h','snow_1h','clouds_all','is_holiday','is_weekend','weather_encoded'])
    pred = model.predict(X_input)[0]
    return 'Trafic predit : '+str(int(pred))+' vehicules/heure'
@app.callback(Output('chat-history','children'),Input('chat-btn','n_clicks'),State('chat-input','value'),State('chat-history','children'),prevent_initial_call=True)
def update_chat(n_clicks,message,historique):
    if not message: return historique or []
    historique = historique or []
    reponse = analyser_question(message)
    msg_user = html.Div([html.Span('Vous : ',style={'fontWeight':'bold','color':'#2c3e50'}),html.Span(message)],style={'backgroundColor':'#e8f4f8','padding':'10px','borderRadius':'8px','marginBottom':'8px','borderLeft':'4px solid #3498db'})
    msg_agent = html.Div([html.Span('Agent IA : ',style={'fontWeight':'bold','color':'#8e44ad'}),html.Span(reponse,style={'whiteSpace':'pre-line'})],style={'backgroundColor':'#f5eef8','padding':'10px','borderRadius':'8px','marginBottom':'8px','borderLeft':'4px solid #8e44ad'})
    return list(historique)+[msg_user,msg_agent]
if __name__ == '__main__':
    app.run(debug=False,port=8050)