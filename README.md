# Desafio Técnico - Americanas SA
A seguir, está documentado todo o <em>pipeline</em> para a resolução do desafio técnico proposto para a vaga de Cientista de Dados Jr., da Americanas SA, realizado pelo candidato João Victor Magalhães Souza.

## 1) Análise Exploratória
Primeiramente, busquei entender, através de gráficos, o comportamento das minhas variáveis de entrada. Na minha visão, através da análise gráfica podemos entender mais facilmente sobre as variáveis, principalmente nesse caso onde não sei o que é cada variável.
### 1.1) Gráfico de série temporal
Como não tenho a informação de que essas variáveis estão organizadas cronologicamente, crio inicialmente gráficos de séries temporais para cada uma das <em>features</em> para <strong>simular</strong> como se essas variáveis estivessem dispostas no tempo. Entretanto, optei por utilizar o conceito de variáveis amostralmente (amostras) organizadas. Por exemplo, para a Feature 0, temos:
![Screenshot](figures/time_series_feature0.png)
Neste tipo de gráfico eu já consigo ter alguns <em>insights</em> iniciais como: qual é o <em>range</em> operacional, a grosso modo, dessas variáveis; consigo observar também se essa variável possui <em>outliers</em>, como mostrado pelos picos na imagem acima. Eu considerei que tais picos são de fato <em>outliers</em>, embora eu não saiba sobre a natureza das variáveis mas, comumente, essas grandes oscilações (subidas e descidas) são consideradas como pontos fora da curva.
### 1.2) Histogramas
Já possuindo uma visão macro do comportamento das variáveis, busquei plotar histogramas para ver a distribuição dos valores dessas variáveis.
![Screenshot](figures/histogram_feature0.png)