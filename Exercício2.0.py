import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, entradas=3, saida=1, bias=0.5):
        self.entradas = np.zeros(entradas)
        self.saida = np.zeros(saida)
        #self.pesos = np.random.uniform(low = -1, high = +1, size = entradas)
        self.pesos = [0.7, 0.3]
        self.bias = np.random.uniform(low = -1, high = +1, size = 1)
        self.saida_registro = []  # Lista para armazenar os valores de saída
        self.bias_registro = []   # Lista para armazenar os valores de bias
        self.pesos_registro = []  # Lista para armazenar os valores de pesos
        self.entrada_registro = []# Lista para armazenar os valores de entradas

    def input(self, array_in):
        for i, x_i in enumerate(array_in):
            self.entradas[i] = x_i

    def ativa(self):
        self.saida = np.dot(self.entradas, self.pesos) + self.bias
        self.entrada_registro.append(self.entradas)
        self.saida_registro.append(self.saida)  # Registra o valor de saída
        return self.saida, self.bias, self.pesos

    def calculaCusto(self, entradas, saidas, alvo):
        delta = alvo - self.saida
        custo = np.dot(delta, delta)
        return custo

    def atualizaPesos(self, entradas, taxa_de_aprendizado, alvo):
        erro = alvo - self.saida
        for i in range(len(self.pesos)):
            self.pesos[i] += taxa_de_aprendizado * erro * entradas[i]
            self.bias[0] += taxa_de_aprendizado * erro
        self.bias_registro.append(self.bias[0])    # Registra o valor do bias
        self.pesos_registro.append(self.pesos)  # Registra os valores de pesos (cópia)


    def __str__(self):
        return f'Entradas: {self.entradas}, Pesos: {self.pesos}, Saída: {self.saida}, Bias: {self.bias}'

# -

# Informações do treino
taxa_de_aprendizado = 0.001

# Informações do perceptron criado
num_entradas = 2
num_saidas = 1
perceptron = Perceptron(num_entradas, num_saidas)

# Coisas pra plotar
x_dados = []
custo_dados = []
bias_dados = []

lista_de_entradas = []
# Cria lista de entradas
for i1 in range(10):
    for i2 in range(10):
        lista_de_entradas.append([i1*1.0,i2*1.0])
print(f'Lista de entradas:{lista_de_entradas}')

# Iterando sobre várias entradas
for k in range(1):
    for i, entrada in enumerate(lista_de_entradas):
        perceptron.input(entrada)
        saida, bias, pesos = perceptron.ativa()
        alvo = entrada[0]+entrada[1]
        custo = perceptron.calculaCusto(entrada,saida,alvo)
            
        print(f'\n# Lote ({k}) # Rodada ({i+1}) #\n')
        print(f'Entrada: {entrada}')
        print(f'Pesos: {pesos[:]}, Bias: {bias[0]}')
        print(f'Saida: {saida}, Alvo: {alvo}')
        print(f'\nPesos antigos: {perceptron.pesos}')
        
        perceptron.atualizaPesos(entrada,taxa_de_aprendizado,alvo)
        
        print(f'Pesos atualizados: {perceptron.pesos[0]}\n')
        
        # Adicionando os dados para o plot
        x_dados.append(i+1 + k*len(lista_de_entradas))
        custo_dados.append(custo)
        bias_dados.append(bias[0])

plt.plot(x_dados, custo_dados)
plt.xlabel('Época #')  # Rotulo do eixo x
plt.ylabel('')  # Rotulo do eixo y
plt.title(f'{len(lista_de_entradas)} execuções da ativação do perceptron')  # Título do gráfico
plt.show()

plt.plot(x_dados, bias_dados)
plt.xlabel('Época #')  # Rotulo do eixo x
plt.ylabel('Bias')  # Rotulo do eixo y
plt.title(f'{len(lista_de_entradas)} execuções da ativação do perceptron')  # Título do gráfico
plt.show()