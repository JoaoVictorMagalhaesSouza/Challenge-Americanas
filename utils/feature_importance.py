import pandas as pd
import plotly.express as px
def plot_importance(model, features):
    feature_important = model.feature_importances_
    keys = list(features)
    values = list(feature_important)

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    fig = px.bar(data, x="score", y=data.index, orientation='h', title='Importância das Features',
    labels={'score': 'Importância','index':'Feature'}, color='score')
    
    fig.update_layout(
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=0, b=50, t=50),
        height=800,
        width=800
    )
    fig.show()
    fig.write_image('figures/feature_importance.png')