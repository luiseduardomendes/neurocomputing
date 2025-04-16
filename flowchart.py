from graphviz import Digraph

# Criação do fluxograma
dot = Digraph(comment='Fluxograma do Algoritmo')

# Nós principais
dot.node('A', 'Início')
dot.node('B', 'Carregar Dataset Iris')
dot.node('C', 'Normalizar Features para Spike Times')
dot.node('D', 'Dividir Dados em Treino e Teste')
dot.node('E', 'Inicializar Modelo com RBF')
dot.node('F', 'Construir Rede Neural')
dot.node('G', 'Treinar Modelo (10 Épocas)')
dot.node('H', 'Predizer Rótulos no Conjunto de Teste')
dot.node('I', 'Calcular Acurácia')
dot.node('J', 'Plotar Resultados (RBF)')
dot.node('K', 'Inicializar Modelo com RCE')
dot.node('L', 'Construir Rede Neural')
dot.node('M', 'Treinar Modelo (10 Épocas)')
dot.node('N', 'Predizer Rótulos no Conjunto de Teste')
dot.node('O', 'Calcular Acurácia')
dot.node('P', 'Plotar Resultados (RCE)')
dot.node('Q', 'Fim')

# Conexões
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('J', 'K')
dot.edge('K', 'L')
dot.edge('L', 'M')
dot.edge('M', 'N')
dot.edge('N', 'O')
dot.edge('O', 'P')
dot.edge('P', 'Q')

# Salvar e renderizar o fluxograma
dot.render('flowchart', format='png', cleanup=True)  # Salva como 'flowchart.png'
print("Fluxograma salvo como 'flowchart.png'")