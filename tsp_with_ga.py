import random
import numpy as np
import matplotlib.pyplot as plt
import statistics # Usado para média e desvio padrão

# --- Seu Código Base (Atividade 5) ---
# --- Matriz de Distâncias (Instância USA13) ---
# 0: New York, 1: Los Angeles, 2: Chicago, 3: Minneapolis, 4: Denver,
# 5: Dallas, 6: Seattle, 7: Boston, 8: San Francisco, 9: St. Louis,
# 10: Houston, 11: Phoenix, 12: Salt Lake City
USA13 = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]

NUM_CIDADES = len(USA13)

def eh_valida_rota(rota):
    """
    Verifica se a rota é válida (contém 13 cidades únicas, de 0 a 12).
    """
    if len(rota) != NUM_CIDADES:
        return False
    if len(set(rota)) != NUM_CIDADES: # Maneira mais rápida de verificar duplicatas
        return False
    if min(rota) < 0 or max(rota) >= NUM_CIDADES:
        return False
    return True

def Distancia_total(rota):
    """
    Calcula a distância total de uma rota (FUNÇÃO FITNESS).
    """
    if not eh_valida_rota(rota):
        return float('inf') # Retorna infinito se a rota for inválida

    total = 0
    for i in range(NUM_CIDADES - 1):
        total += USA13[rota[i]][rota[i+1]]
    
    # Adiciona a distância da última cidade de volta para a primeira
    total += USA13[rota[-1]][rota[0]]
    
    return total

# --- Componentes do Algoritmo Genético (Atividade 6) ---

# Parâmetros da Atividade [cite: 3]
TAM_POPULACAO = 50       # 
NUM_GERACOES = 400       # 
TAM_TORNEIO = 3          # 
TAXA_CROSSOVER = 0.90    # 
TAXA_MUTACAO = 0.05      # 
TAM_ELITE = 5            # 

def criar_individuo():
    """ Cria uma rota aleatória (permutação de cidades) """
    # np.random.permutation cria uma permutação aleatória de 0 a (NUM_CIDADES-1)
    return list(np.random.permutation(NUM_CIDADES))

def criar_populacao_inicial():
    """ Cria a população inicial com TAM_POPULACAO indivíduos """
    return [criar_individuo() for _ in range(TAM_POPULACAO)]

def selecao_torneio(populacao_com_fitness):
    """
    Seleciona um indivíduo usando torneio de tamanho TAM_TORNEIO. 
    """
    # Seleciona TAM_TORNEIO indivíduos aleatórios da população
    competidores = random.sample(populacao_com_fitness, TAM_TORNEIO)
    
    # Encontra o vencedor (menor fitness, pois queremos minimizar a distância)
    vencedor = min(competidores, key=lambda item: item['fitness'])
    
    # Retorna a rota (cromossomo) do vencedor
    return vencedor['rota']

def crossover_ox(pai1, pai2): # 
    """
    Implementa o Order Crossover (OX). 
    """
    tamanho = len(pai1)
    filho = [None] * tamanho
    
    # 1. Seleciona dois pontos de corte aleatórios
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    
    # 2. Copia o segmento do pai1 para o filho
    segmento_pai1 = pai1[inicio:fim+1]
    filho[inicio:fim+1] = segmento_pai1
    
    # 3. Preenche o restante do filho com os genes do pai2
    idx_filho = (fim + 1) % tamanho
    idx_pai2 = (fim + 1) % tamanho
    
    # Cidades que já estão no filho (do segmento do pai1)
    cidades_no_filho = set(segmento_pai1)
    
    while None in filho:
        # Se a cidade do pai2 ainda não está no filho
        if pai2[idx_pai2] not in cidades_no_filho:
            filho[idx_filho] = pai2[idx_pai2]
            idx_filho = (idx_filho + 1) % tamanho
        
        # Move para a próxima cidade do pai2
        idx_pai2 = (idx_pai2 + 1) % tamanho
        
    return filho

def mutacao_swap(rota): # 
    """
    Implementa a Mutação Swap (troca duas cidades de posição). 
    """
    # Seleciona duas posições aleatórias
    idx1, idx2 = random.sample(range(len(rota)), 2)
    
    # Troca as cidades nessas posições
    rota[idx1], rota[idx2] = rota[idx2], rota[idx1]
    
    return rota

def executar_ag():
    """
    Executa uma rodada completa do Algoritmo Genético (400 gerações).
    Retorna o melhor indivíduo encontrado e o histórico de fitness.
    """
    # 1. Inicialização
    populacao_rotas = criar_populacao_inicial()
    historico_melhor_fitness = []

    for geracao in range(NUM_GERACOES): # 
        # 2. Avaliação (Fitness)
        # Calcula o fitness (distância) para cada indivíduo (rota)
        pop_com_fitness = [
            {'rota': rota, 'fitness': Distancia_total(rota)} 
            for rota in populacao_rotas
        ]
        
        # Ordena a população pelo fitness (menor distância é melhor)
        pop_com_fitness.sort(key=lambda item: item['fitness'])
        
        # Salva o melhor fitness desta geração para o gráfico de convergência
        melhor_fitness_geracao = pop_com_fitness[0]['fitness']
        historico_melhor_fitness.append(melhor_fitness_geracao)
        
        # 3. Criação da Nova Geração
        nova_populacao_rotas = []
        
        # 3.1 Elitismo: Mantém os 5 melhores 
        for i in range(TAM_ELITE):
            nova_populacao_rotas.append(pop_com_fitness[i]['rota'])
            
        # 3.2 Preenche o restante da população (50 - 5 = 45)
        while len(nova_populacao_rotas) < TAM_POPULACAO:
            # Seleção 
            pai1 = selecao_torneio(pop_com_fitness)
            pai2 = selecao_torneio(pop_com_fitness)
            
            # Crossover (OX) 
            if random.random() < TAXA_CROSSOVER:
                filho = crossover_ox(pai1, pai2)
            else:
                # Se não houver crossover, um dos pais passa (clone)
                filho = pai1[:] 
            
            # Mutação (Swap) 
            if random.random() < TAXA_MUTACAO:
                filho = mutacao_swap(filho)
                
            nova_populacao_rotas.append(filho)
            
        # A nova geração substitui a antiga
        populacao_rotas = nova_populacao_rotas

    # Fim das gerações, calcula o fitness final da última população
    pop_final_com_fitness = [
        {'rota': rota, 'fitness': Distancia_total(rota)} 
        for rota in populacao_rotas
    ]
    pop_final_com_fitness.sort(key=lambda item: item['fitness'])
    
    # Retorna o melhor indivíduo (rota e fitness) e o histórico
    return pop_final_com_fitness[0], historico_melhor_fitness

# --- Execução Principal e Análise (Atividade 6) ---
if __name__ == "__main__":
    
    print(f"Iniciando {30} execuções do AG para TSP (USA13)...")
    print(f"Configuração: Pop={TAM_POPULACAO}, Gerações={NUM_GERACOES}, Elite={TAM_ELITE}, Torneio={TAM_TORNEIO}")
    print(f"Taxas: Crossover={TAXA_CROSSOVER*100}%, Mutação={TAXA_MUTACAO*100}%")
    print("-" * 30)
    
    resultados_finais_fitness = []
    historico_convergencia_primeira_run = None
    
    # Executar 30 vezes 
    for i in range(30):
        # Executa uma rodada completa do AG
        melhor_individuo_final, historico_fitness = executar_ag()
        
        # Armazena o fitness final (distância)
        fitness_final = melhor_individuo_final['fitness']
        resultados_finais_fitness.append(fitness_final)
        
        # Salva o histórico da primeira execução para o Gráfico de Convergência
        if i == 0:
            historico_convergencia_primeira_run = historico_fitness
            
        print(f"Execução {i+1}/30 - Melhor Fitness (Distância): {fitness_final:.2f}")

    print("\n--- Análise Estatística (Resultados Finais) ---")
    
    # 1. Calcular a média e o desvio padrão 
    media = statistics.mean(resultados_finais_fitness)
    desvio_padrao = statistics.stdev(resultados_finais_fitness)
    
    print(f"Média do Fitness (Distância): {media:.2f}")
    print(f"Desvio Padrão do Fitness: {desvio_padrao:.2f}")
    
    # --- Geração dos Gráficos ---

    # 2. Gráfico de Convergência do AG (melhor fitness por iteração) 
    # (Usando os dados da primeira execução)
    plt.figure(figsize=(10, 6))
    plt.plot(historico_convergencia_primeira_run)
    plt.title('Gráfico de Convergência do AG (1ª Execução)')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Fitness (Distância)')
    plt.grid(True)
    plt.savefig("grafico_convergencia_ag_tsp.png")
    print("\nGráfico de convergência salvo como 'grafico_convergencia_ag_tsp.png'")

    # 3. Criar boxplot com os resultados finais 
    plt.figure(figsize=(8, 6))
    plt.boxplot(resultados_finais_fitness, patch_artist=True, vert=False)
    plt.title('Boxplot dos Resultados Finais (30 Execuções)')
    plt.xlabel('Fitness Final (Distância)')
    plt.yticks([1], ['Resultados']) # Remove o "tick" do eixo Y
    plt.grid(True, axis='x')
    plt.savefig("boxplot_resultados_ag_tsp.png")
    print("Boxplot dos resultados salvo como 'boxplot_resultados_ag_tsp.png'")
    
    # Exibe os gráficos 
    plt.show()