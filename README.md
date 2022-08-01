# Desafio Técnico - Americanas SA
A seguir, está documentado todo o <em>pipeline</em> para a resolução do desafio técnico proposto para a vaga de Cientista de Dados Jr., da Americanas SA, realizado pelo candidato <strong>João Victor Magalhães Souza</strong>.

## 1) Análise Exploratória
Primeiramente, busquei entender, através de gráficos, o comportamento das minhas variáveis de entrada. Na minha visão, através da análise gráfica podemos entender mais facilmente sobre as variáveis, principalmente nesse caso onde não sei o que é cada variável. Nas análises a seguir, serão mostrados apenas alguns gráficos como exemplo. Porém, todos os gráficos plotados para a análise exploratória estão na pasta <strong>"figures"</strong>
### 1.1) Gráfico de série temporal
Como não tenho a informação de que essas variáveis estão organizadas cronologicamente, crio inicialmente gráficos de séries temporais para cada uma das <em>features</em> para <strong>simular</strong> como se essas variáveis estivessem dispostas no tempo. Entretanto, optei por utilizar o conceito de variáveis amostralmente (amostras) organizadas. Por exemplo, para a Feature 0, temos:
<br>
![Screenshot](figures/time_series_feature0.png)
<br>
Neste tipo de gráfico eu já consigo ter alguns <em>insights</em> iniciais como: qual é o <em>range</em> operacional, a grosso modo, dessas variáveis; consigo observar também se essa variável possui <em>outliers</em>, como mostrado pelos picos na imagem acima. Eu considerei que tais picos são de fato <em>outliers</em>, embora eu não saiba sobre a natureza das variáveis mas, comumente, essas grandes oscilações (subidas e descidas) são consideradas como pontos fora da curva.

### 1.2) Histogramas
Já possuindo uma visão macro do comportamento das variáveis, busquei plotar histogramas para ver a distribuição dos valores dessas variáveis.
<br>
![Screenshot](figures/histogram_feature0.png)
<br>
Com os histogramas, eu consigo entender um pouco mais sobre os valores assumidos por essas variáveis, sua distribuição e, indiretamente, se elas variam muito ou pouco ao longo das amostras. Essa é uma análise interessante, principalmente em aplicações reais visto que conseguimos imaginar se determinada variávei será útil ou não pois, geralmente, se uma variável varia muito pouco então provavelmente ela não trará muita informação sobre a <em>target</em>. Deixando claro que isso não se aplica para todos os casos.
Com esses gráficos, eu já percebi que haviam muitas variáveis que variavam muito pouco e, sendo assim, já comecei a imaginar que teria que arrumar alguma estratégia para agregar mais informação ao conjunto de dados.

### 1.3) <em>Boxplots</em>
Com os <em>boxplots</em> meu intuito é justamente ver algumas medidas estatísticas como mediana, min, max, IQR e também sobre os <em>outliers</em>:
![Screenshot](figures/box_plot_feature0.png)
Como podemos observar, a maior parte dos valores, para a Feature 0, está contida entre 0 e aproximadamente 1500 e, graficamente, podemos confirmar alguns <em>outliers</em> que já haviam sido apontados nos gráficos de Séries Temporais (1.1).

Essas três primeiras análises são muito úteis pois já me fizeram perceber que uma limpeza nos dados seria extremamente relevante.

### 1.4) Distribuição da <em>target</em>
Como se trata de um problema de classificação binária, a minha última análise exploratória é pautada na verificação do balanceamento do meu problema: o meu intuito aqui é observar se era um problema balanceado ou desbalanceado:
![Screenshot](figures/target_distribuition.png)
Observo que temos um leve desbalanceamento entre as classes e, inicialmente, na minha visão, não será algo problemático para este desafio.

### 1.5) Correlação das variáveis de entrada com a <em>target</em>
Por fim, outra análise que julguei interessante foi analisar as correlações das variáveis preditoras com a <em>target</em> na tentativa de inspecionar o nível de informação que elas me dão sobre o que quero prever. Aqui, utilizei apenas a correlação de Pearson:
![Screenshot](figures/correlation_plot.png)
Como podemos observar, as correlações lineares entre as variáveis preditoras e a nossa variável-alvo estão bem baixas. Isso implica que teremos que enriquecer ainda mais os nossos dados se quisermos relizar boas predições.

## 2) Pré processamento dos dados (<em>Data Preparation</em>)
Uma vez que já entendo melhor tanto sobre meus dados de entrada quanto sobre minha <em>target</em>, é hora de fazer as correções necessárias.

### 2.1) Limpeza
Como mostrado na seção 1, existem alguns <em>outliers</em> no nosso conjunto de dados. Sendo assim, optei por tratá-los ao invés de excluí-los, visto o pequeno número de amostras de dados. A limpeza utilizada nesta etapa foi por <strong>+- 1.5*IQR</strong>, já que as variáveis de entrada não apresentam distribuição normal e, sendo assim, uma limpeza por desvio-padrão não seria adequada. Optei por implementar uma substituição dos <em>outliers</em> pela mediana ao invés da média, visto que obtive melhores resultados com a primeira abordagem.

### 2.2) <em>Feature Engineering</em>
Evidenciado na seção de Análise Exporatória, foi possível observar que algumas variáveis variam bem pouco e que elas possuem uma correlação muito baixa com a nossa <em>target</em>. Uma boa estratégia para agregar mais informação ao dados de entrada é através do processo de <em>Feature Engineering</em>. Como todas as variáveis de entrada são numéricas, as estratégias que pensei para este desafio foram:
- <strong>Derivada:</strong> a minha ideia aqui é calcular a taxa de variação das variáveis preditoras. Assim, eu consigo explicitar para meu modelo de <em>Machine Learning</em> o quanto uma variável variou de uma amostra para a outra ao longo de todas as suas amostras.
- <strong>Integral:</strong> nessa nova <em>feature</em>, meu objetivo é calcular o somatório de uma variável em uma janela móvel de 5 amostras. É parecido com uma média móvel mas ao invés de calcular a média, calculo o somatório dessa variável. 
- <strong>Momentos Estatísticos Móveis:</strong> outras novas variáveis que achei que poderiam ser interessantes são a média e desvio-padrão móveis. Ambas foram calculadas em uma janela móvel de 5 amostras.
- <strong>Combinações Polinomiais:</strong> por fim, criei também as combinações polinomiais que são, basicamente todas as possíveis multiplicações entre pares de variáveis e, além disso, a potência de 2 de todas as variáveis de entrada. Assim como a Integral, essas propostas não têm uma semântica clara, ou seja, possuem apenas um viés matemático. Mesmo assim, na grande maioria dos casos, essas propostas conseguem agregar um nível significativo de informação nos conjuntos de dados nas quais são aplicadas.

Ao término deste processo, de 15 variáveis de entrada vamos para 201 variáveis de entrada.