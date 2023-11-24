# Bruna Mortari e Maria Luiza Barbosa

import pandas as pd
from sklearn.model_selection import train_test_split

# Importar dados do excel para python
df = pd.read_excel('dados_lagarto.xlsx')
# Dados base e sexo F/M
X = df[df.columns[0:5]]

def proporcao_sexo(p):
  # É necessário manter a proporção de sexo entre os conjuntos
  proporcao = df['sexo'].value_counts(normalize=True)
  # Divide os dados em conjunto de teste e conjunto de treinamento dado p
  [conj_treinamento, conj_teste] = train_test_split(X, test_size=1 - p, stratify=df['sexo'])
  return conj_treinamento, conj_teste

while True:
    try:
        p = float(input("Digite um número entre 0 e 1 (p): "))
        if 0 <= p <= 1:
            break  # A entrada está dentro do intervalo desejado
        else:
            print("Fora do intervalo permitido. Tente novamente.")
    except ValueError:
        print("Entrada inválida. Por favor, insira um número válido.")
        
conj_treinamento, conj_teste = proporcao_sexo(p)

print(conj_treinamento)
print(conj_teste)

min_colMassa = conj_treinamento['massa em gramas'].min()
max_colMassa = conj_treinamento['massa em gramas'].max()


min_colComprimento = conj_treinamento['comprimento da passagem de ar do focinho em milímetros'].min()
max_colComprimento = conj_treinamento['comprimento da passagem de ar do focinho em milímetros'].max()

min_colDimensão = conj_treinamento['dimensão da pata posterior em milímetros'].min()
max_colDimensão = conj_treinamento['dimensão da pata posterior em milímetros'].max()

print(conj_treinamento) # Mostra o conjunto de treinamento

ymin = -1
ymax = 1

#Fórmula para normalizar y = ((ymax - ymin) / (xmax - xmin)) * (x - xmin) + ymin
print("formula de normalizacao: y = ((ymax - ymin) / (xmax - xmin)) * (x - xmin) + ymin")
print(f"ymin = {ymin} e ymax = {ymax} \d" )
print()

def normalize_column(df, col_name,min_col, max_col):
    df[col_name] = ((ymax - ymin) / (max_col - min_col)) * (df[col_name] - min_col) + ymin

normalize_column(conj_treinamento, 'massa em gramas', min_colMassa, max_colMassa)
normalize_column(conj_treinamento, 'comprimento da passagem de ar do focinho em milímetros', min_colComprimento, max_colComprimento)
normalize_column(conj_treinamento, 'dimensão da pata posterior em milímetros', min_colDimensão, max_colDimensão)

# Mostra o conjunto de treinamento com valores normalizados
print(conj_treinamento)

normalize_column(conj_teste, 'massa em gramas', min_colMassa, max_colMassa)
normalize_column(conj_teste, 'comprimento da passagem de ar do focinho em milímetros', min_colComprimento, max_colComprimento)
normalize_column(conj_teste, 'dimensão da pata posterior em milímetros', min_colDimensão, max_colDimensão)

print(conj_teste)

def normalize_value(value, min_col, max_col):
    normalized_value = ((ymax - ymin) / (max_col - min_col)) * (value - min_col) + ymin
    return normalized_value

#try:
   # massa = float(input("Digite a massa em gramas: "))
    #comp = float(input("Digite o comprimento da passagem de ar do focinho em milímetros: "))
    #dim = float(input("Digite a dimensão da pata posterior em milímetros: "))
#except ValueError:
    #print("Entrada inválida. Por favor, insira um número válido.")

#Faz a normalização dos dados fornecidos
#massa_normalizada = normalize_value(massa, min_colMassa, max_colMassa)
#comprimento_normalizado = normalize_value(comp, min_colComprimento, max_colComprimento)
#dimensao_normalizada = normalize_value(dim, min_colDimensão, max_colDimensão)

#Mostra os dados normalizados
#print(f"Massa do espécime dado normalizada: {massa_normalizada}")
#print(f"Comprimento da passagem de ar do focinho do espécime dado normalizado: {comprimento_normalizado}")
#print(f"Dimensão da pata posterior do espécime dado normalizada: {dimensao_normalizada}")

# Definir pesos
w0 = -1.5
w1 = 1
w2 = 1
w3 = 0

# Definir taxa de treinamento
taxaTreinamento = 0.5

num_epocas = 100

# Treinar o perceptron com o conjunto de treinamento
for x in range(num_epocas):
    erros = 0
    print(f"epoca:{x}")
    for i in range(len(conj_treinamento)):
        # Calcular discriminante linear
        amostra = conj_treinamento.iloc[i]
        u1 = amostra['massa em gramas']*w1
        u2 = amostra['comprimento da passagem de ar do focinho em milímetros']*w2
        u3 = amostra['dimensão da pata posterior em milímetros']*w3
        u = u1 + u2 + u3 + w0

        # Calcular saída usando função sinal
        if u > 0:
            y = 1
        else:
            y = -1

        # Decisão de classificação
        if y == 1:
            sexo = 'F'
        else:
            sexo = 'M'

        # Comparar valor da saída e ajustar pesos
        if sexo != amostra['sexo']:
            erros += 1
            if u > 0:
                w1 = w1 - taxaTreinamento * amostra['massa em gramas']
                w2 = w2 - taxaTreinamento * amostra['comprimento da passagem de ar do focinho em milímetros']
                w3 = w3 - taxaTreinamento * amostra['dimensão da pata posterior em milímetros']
            else:
                w1 = w1 + taxaTreinamento * amostra['massa em gramas']
                w2 = w2 + taxaTreinamento * amostra['comprimento da passagem de ar do focinho em milímetros']
                w3 = w3 + taxaTreinamento * amostra['dimensão da pata posterior em milímetros']
    print(f"erros: {erros}")
    if erros == 0:
      break

# Criamos um contador para rastrear todas as vezes que o perceptron acertar a classificação
count = 0

# Testar o perceptron com o conjunto de teste usando os pesos obtidos no treinamento
for i in range(len(conj_teste)):
    amostra = conj_teste.iloc[i]
    u1 = amostra['massa em gramas']*w1
    u2 = amostra['comprimento da passagem de ar do focinho em milímetros']*w2
    u3 = amostra['dimensão da pata posterior em milímetros']*w3
    u = u1 + u2 + u3 + w0
    
    # Calcular saída usando função sinal
    if u > 0:
        y = 1
    else:
        y = -1

    # Decisão de classificação
    if y == 1:
        sexo = 'F'
    else:
        sexo = 'M'
    
    # Checar se acertou e incrementar o contador count para cada acerto
    if sexo == amostra['sexo']:
        count = count + 1

print(f"Quantidade de exemplos do conjunto de teste que o perceptron acertou: {count} de {i+1}") 

# Inicializar variáveis para formar a matriz de confusão
# Verdadeiros positivos
VP = 0
# Verdadeiros negativos
VN = 0
#Falsos positivos
FP = 0
# Falsos negativos
FN = 0

# Criando a matriz de confusão
for i in range(len(conj_teste)):
    amostra = conj_teste.iloc[i]
    u1 = amostra['massa em gramas']*w1
    u2 = amostra['comprimento da passagem de ar do focinho em milímetros']*w2
    u3 = amostra['dimensão da pata posterior em milímetros']*w3

    u = u1 + u2 + u3 + w0

    if u <= 0:
        y = 1
    else:
        y = -1

    # Decisão da classificação
    if y == 1:
        sexo = 'F'
    else:
        sexo = 'M'

    # Comparar o sexo previsto com o sexo real da espécime
    conj_teste['sexo'] = amostra['sexo']

    if sexo == 'F' and amostra['sexo'] == 'F':
        VP += 1
    elif sexo == 'M' and amostra['sexo'] == 'M':
        VN += 1
    elif sexo == 'F' and amostra['sexo'] == 'M':
        FP += 1
    else:
        FN += 1

# Imprimir a matriz de confusão
print("Verdadeiros positivos:", VP)
print("Verdadeiros negativos:", VN)
print("Falsos positivos:", FP)
print("Falsos negativos:", FN)