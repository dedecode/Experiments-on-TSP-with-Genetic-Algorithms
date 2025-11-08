import random
import numpy as np
import matplotlib.pyplot as plt
import time 
import os 
import traceback # Importamos para ver erros detalhados

# --- Matriz de Distâncias (Instância USA13) ---
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

# --- Funções do AG (Validação, Fitness, Componentes) ---

def eh_valida_rota(rota):
    # Verifica se a rota é válida
    return (len(rota) == NUM_CIDADES and 
            len(set(rota)) == NUM_CIDADES and 
            min(rota) >= 0 and 
            max(rota) < NUM_CIDADES)

def Distancia_total(rota):
    # Calcula a distância total de uma rota (FUNÇÃO FITNESS)
    if not eh_valida_rota(rota): return float('inf')
    total = sum(USA13[rota[i]][rota[i+1]] for i in range(NUM_CIDADES - 1))
    total += USA13[rota[-1]][rota[0]]
    return total

def criar_individuo():
    # Cria uma rota aleatória (permutação de cidades)
    return list(np.random.permutation(NUM_CIDADES))

def selecao_torneio(pop_com_fitness, TAM_TORNEIO):
    # Seleciona um indivíduo usando torneio
    competidores = random.sample(pop_com_fitness, TAM_TORNEIO)
    vencedor = min(competidores, key=lambda item: item['fitness'])
    return vencedor['rota']

def crossover_ox(pai1, pai2):
    # Implementa o Order Crossover (OX)
    tamanho = len(pai1)
    filho = [None] * tamanho
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    segmento_pai1 = pai1[inicio:fim+1]
    filho[inicio:fim+1] = segmento_pai1
    idx_filho = (fim + 1) % tamanho
    idx_pai2 = (fim + 1) % tamanho
    cidades_no_filho = set(segmento_pai1)
    while None in filho:
        if pai2[idx_pai2] not in cidades_no_filho:
            filho[idx_filho] = pai2[idx_pai2]
            idx_filho = (idx_filho + 1) % tamanho
        idx_pai2 = (idx_pai2 + 1) % tamanho
    return filho

def mutacao_swap(rota):
    # Implementa a Mutação Swap (troca duas cidades)
    idx1, idx2 = random.sample(range(len(rota)), 2)
    rota[idx1], rota[idx2] = rota[idx2], rota[idx1]
    return rota

# --- Função Principal do AG (Robusta) ---

def executar_ag(TAM_POPULACAO, NUM_GERACOES, TAM_TORNEIO, TAXA_CROSSOVER, TAXA_MUTACAO, TAM_ELITE):
    # Executa o AG com os parâmetros fornecidos
    populacao_rotas = [criar_individuo() for _ in range(TAM_POPULACAO)]
    historico_melhor_fitness = []
    historico_diversidade = [] 

    for geracao in range(NUM_GERACOES):
        # Avaliação
        pop_com_fitness = [
            {'rota': rota, 'fitness': Distancia_total(rota)} 
            for rota in populacao_rotas
        ]
        pop_com_fitness.sort(key=lambda item: item['fitness'])
        
        # Salva Históricos
        historico_melhor_fitness.append(pop_com_fitness[0]['fitness'])
        rotas_unicas = set(tuple(item['rota']) for item in pop_com_fitness)
        historico_diversidade.append(len(rotas_unicas))
        
        # Nova Geração
        nova_populacao_rotas = []
        
        # Elitismo
        for i in range(TAM_ELITE):
            nova_populacao_rotas.append(pop_com_fitness[i]['rota'])
            
        # Restante da população
        while len(nova_populacao_rotas) < TAM_POPULACAO:
            pai1 = selecao_torneio(pop_com_fitness, TAM_TORNEIO)
            pai2 = selecao_torneio(pop_com_fitness, TAM_TORNEIO)
            
            filho = crossover_ox(pai1, pai2) if random.random() < TAXA_CROSSOVER else pai1[:]
            
            if random.random() < TAXA_MUTACAO:
                filho = mutacao_swap(filho)
                
            nova_populacao_rotas.append(filho)
            
        populacao_rotas = nova_populacao_rotas

    # Fim
    pop_final_com_fitness = [
        {'rota': rota, 'fitness': Distancia_total(rota)} 
        for rota in populacao_rotas
    ]
    pop_final_com_fitness.sort(key=lambda item: item['fitness'])
    
    return pop_final_com_fitness[0], historico_melhor_fitness, historico_diversidade

# --- Funções de Plotagem ---

def plotar_convergencia_comparativa(resultados_por_config, titulo, config_labels, pasta_graficos, y_label='Melhor Fitness (Distância)'):
    # Gera gráfico de linhas comparativo
    plt.figure(figsize=(10, 6))
    for i, config in enumerate(config_labels):
        historicos = resultados_por_config[config]['historico_fitness']
        if not historicos: 
            print(f"AVISO: Sem dados de convergência para {config}")
            continue
        # Calcula a média por geração
        media_historico = np.mean(historicos, axis=0)
        plt.plot(media_historico, label=f'{config_labels[i]}')
        
    plt.title(titulo)
    plt.xlabel('Geração')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pasta_graficos, f"{titulo.lower().replace(' ', '_').replace(':', '')}.png"))
    plt.close()

def plotar_boxplot_comparativo(resultados_por_config, titulo, config_labels, pasta_graficos):
    # Gera boxplot comparativo dos resultados finais
    dados_boxplot = []
    labels_com_dados = []
    for config in config_labels:
        dados = resultados_por_config[config]['fitness_final']
        if dados: 
            dados_boxplot.append(dados)
            labels_com_dados.append(config)
        else:
            print(f"AVISO: Sem dados de fitness final para {config}")
    
    if not dados_boxplot: 
        print(f"ERRO CRÍTICO: Não foi possível gerar boxplot '{titulo}' por falta de dados.")
        return

    plt.figure(figsize=(10, 6))
    # CORREÇÃO AQUI: 'labels=' foi trocado para 'tick_labels='
    plt.boxplot(dados_boxplot, tick_labels=labels_com_dados, patch_artist=True)
    plt.title(titulo)
    plt.ylabel('Fitness Final (Distância)')
    plt.xlabel('Configuração')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(pasta_graficos, f"{titulo.lower().replace(' ', '_').replace(':', '')}.png"))
    plt.close()

# --- Execução Principal (Atividade 7) ---

def main():
    # Parâmetros Padrão
    NUM_GERACOES_PADRAO = 400
    TAXA_CROSSOVER_PADRAO = 0.90
    POP_PADRAO = 50
    MUTACAO_PADRAO = 0.05
    TORNEIO_PADRAO = 3
    ELITE_PADRAO = 5
    
    NUM_EXECUCOES = 30 # Para testar rápido, mude para 2. Para a atividade, use 30.
    PASTA_GRAFICOS = "graficos_atividade_7"
    
    # Cria a pasta para salvar os gráficos
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    
    print("Iniciando Análise de Parâmetros do AG (Atividade 7)")
    print(f"Total de {NUM_EXECUCOES} execuções por configuração.")
    print(f"Os gráficos serão salvos na pasta: {PASTA_GRAFICOS}")

    # Dicionário para armazenar todos os resultados
    resultados_finais = {
        "exp1_pop": {},
        "exp2_mut": {},
        "exp3_tor": {},
        "exp4_eli": {},
    }

    # --- Experimento 1: Tamanho da População ---
    POP_VALORES = [20, 50, 100]
    config_labels_exp1 = [f"Pop={p}" for p in POP_VALORES]
    
    print("\n--- Iniciando Experimento 1: Tamanho da População ---")
    for i, pop_size in enumerate(POP_VALORES):
        label = config_labels_exp1[i]
        print(f"Testando Configuração: {label}")
        
        # Inicializa os dicionários de resultado
        resultados_finais["exp1_pop"][label] = { 'fitness_final': [], 'tempo_execucao': [], 'historico_fitness': [] }
        
        for i_exec in range(NUM_EXECUCOES):
            try:
                # PRINT DE STATUS (IMPORTANTE!)
                print(f"  ... {label}: Execução {i_exec+1}/{NUM_EXECUCOES}", end='\r') 
                start_time = time.time()
                
                melhor, hist_fit, _ = executar_ag(
                    TAM_POPULACAO=pop_size, NUM_GERACOES=NUM_GERACOES_PADRAO, 
                    TAM_TORNEIO=TORNEIO_PADRAO, TAXA_CROSSOVER=TAXA_CROSSOVER_PADRAO, 
                    TAXA_MUTACAO=MUTACAO_PADRAO, TAM_ELITE=round(0.10 * pop_size)
                )
                
                end_time = time.time()
                
                # Salva os resultados desta execução
                resultados_finais["exp1_pop"][label]['fitness_final'].append(melhor['fitness'])
                resultados_finais["exp1_pop"][label]['tempo_execucao'].append(end_time - start_time)
                resultados_finais["exp1_pop"][label]['historico_fitness'].append(hist_fit)
            
            except Exception as e: # Pega qualquer erro que acontecer
                print(f"\n    !!!! ERRO NA EXECUÇÃO {i_exec+1} ({label}): {e} !!!!")
                traceback.print_exc() # Mostra o erro detalhado
        
        print(f"\n  -> Concluído {label}. {len(resultados_finais['exp1_pop'][label]['fitness_final'])}/{NUM_EXECUCOES} execuções com sucesso.")

    # --- Experimento 2: Taxa de Mutação ---
    MUTACAO_VALORES = [0.01, 0.05, 0.10, 0.20]
    config_labels_exp2 = [f"Mut={m*100}%" for m in MUTACAO_VALORES]

    print("\n--- Iniciando Experimento 2: Taxa de Mutação ---")
    for i, mut_rate in enumerate(MUTACAO_VALORES):
        label = config_labels_exp2[i]
        print(f"Testando Configuração: {label}")
        
        resultados_finais["exp2_mut"][label] = { 'fitness_final': [], 'historico_fitness': [], 'historico_diversidade': [] }
        
        for i_exec in range(NUM_EXECUCOES):
            try:
                print(f"  ... {label}: Execução {i_exec+1}/{NUM_EXECUCOES}", end='\r')
                melhor, hist_fit, hist_div = executar_ag(
                    TAM_POPULACAO=POP_PADRAO, NUM_GERACOES=NUM_GERACOES_PADRAO, 
                    TAM_TORNEIO=TORNEIO_PADRAO, TAXA_CROSSOVER=TAXA_CROSSOVER_PADRAO, 
                    TAXA_MUTACAO=mut_rate, TAM_ELITE=ELITE_PADRAO
                )
                resultados_finais["exp2_mut"][label]['fitness_final'].append(melhor['fitness'])
                resultados_finais["exp2_mut"][label]['historico_fitness'].append(hist_fit)
                resultados_finais["exp2_mut"][label]['historico_diversidade'].append(hist_div)
            except Exception as e:
                print(f"\n    !!!! ERRO NA EXECUÇÃO {i_exec+1} ({label}): {e} !!!!")
                traceback.print_exc()
        
        print(f"\n  -> Concluído {label}. {len(resultados_finais['exp2_mut'][label]['fitness_final'])}/{NUM_EXECUCOES} execuções com sucesso.")

    # --- Experimento 3: Tamanho do Torneio ---
    TORNEIO_VALORES = [2, 3, 5, 7]
    config_labels_exp3 = [f"Torneio={t}" for t in TORNEIO_VALORES]

    print("\n--- Iniciando Experimento 3: Tamanho do Torneio ---")
    for i, torn_size in enumerate(TORNEIO_VALORES):
        label = config_labels_exp3[i]
        print(f"Testando Configuração: {label}")
        
        resultados_finais["exp3_tor"][label] = { 'fitness_final': [], 'historico_fitness': [] }
        
        for i_exec in range(NUM_EXECUCOES):
            try:
                print(f"  ... {label}: Execução {i_exec+1}/{NUM_EXECUCOES}", end='\r')
                melhor, hist_fit, _ = executar_ag(
                    TAM_POPULACAO=POP_PADRAO, NUM_GERACOES=NUM_GERACOES_PADRAO, 
                    TAM_TORNEIO=torn_size, TAXA_CROSSOVER=TAXA_CROSSOVER_PADRAO, 
                    TAXA_MUTACAO=MUTACAO_PADRAO, TAM_ELITE=ELITE_PADRAO
                )
                resultados_finais["exp3_tor"][label]['fitness_final'].append(melhor['fitness'])
                resultados_finais["exp3_tor"][label]['historico_fitness'].append(hist_fit)
            except Exception as e:
                print(f"\n    !!!! ERRO NA EXECUÇÃO {i_exec+1} ({label}): {e} !!!!")
                traceback.print_exc()
        
        print(f"\n  -> Concluído {label}. {len(resultados_finais['exp3_tor'][label]['fitness_final'])}/{NUM_EXECUCOES} execuções com sucesso.")
            
    # --- Experimento 4: Elitismo ---
    ELITE_VALORES = [0, round(0.01 * POP_PADRAO), round(0.05 * POP_PADRAO), round(0.10 * POP_PADRAO)] # [0, 1, 3, 5]
    config_labels_exp4 = [f"Elite={e} ({e*100/POP_PADRAO}%)" for e in ELITE_VALORES]

    print("\n--- Iniciando Experimento 4: Elitismo ---")
    for i, elite_size in enumerate(ELITE_VALORES):
        label = config_labels_exp4[i]
        print(f"Testando Configuração: {label}")
        
        resultados_finais["exp4_eli"][label] = { 'fitness_final': [], 'historico_fitness': [] }
        
        for i_exec in range(NUM_EXECUCOES):
            try:
                print(f"  ... {label}: Execução {i_exec+1}/{NUM_EXECUCOES}", end='\r')
                melhor, hist_fit, _ = executar_ag(
                    TAM_POPULACAO=POP_PADRAO, NUM_GERACOES=NUM_GERACOES_PADRAO, 
                    TAM_TORNEIO=TORNEIO_PADRAO, TAXA_CROSSOVER=TAXA_CROSSOVER_PADRAO, 
                    TAXA_MUTACAO=MUTACAO_PADRAO, TAM_ELITE=elite_size
                )
                resultados_finais["exp4_eli"][label]['fitness_final'].append(melhor['fitness'])
                resultados_finais["exp4_eli"][label]['historico_fitness'].append(hist_fit)
            except Exception as e:
                print(f"\n    !!!! ERRO NA EXECUÇÃO {i_exec+1} ({label}): {e} !!!!")
                traceback.print_exc()
        
        print(f"\n  -> Concluído {label}. {len(resultados_finais['exp4_eli'][label]['fitness_final'])}/{NUM_EXECUCOES} execuções com sucesso.")

    # --- Geração de Gráficos ---
    print("\n--- Todas as execuções concluídas. Gerando gráficos comparativos... ---")

    try:
        # Gráficos Exp 1
        plotar_boxplot_comparativo(resultados_finais["exp1_pop"], "Exp 1: Boxplot Fitness Final por Tamanho da População", config_labels_exp1, PASTA_GRAFICOS)
        plotar_convergencia_comparativa(resultados_finais["exp1_pop"], "Exp 1: Convergência Média por Tamanho da População", config_labels_exp1, PASTA_GRAFICOS)
        dados_tempo = [resultados_finais["exp1_pop"][label]['tempo_execucao'] for label in config_labels_exp1 if resultados_finais["exp1_pop"][label]['tempo_execucao']]
        if dados_tempo:
            labels_tempo = [label for label in config_labels_exp1 if resultados_finais["exp1_pop"][label]['tempo_execucao']]
            plt.figure(figsize=(10, 6))
            # CORREÇÃO AQUI: 'labels=' foi trocado para 'tick_labels='
            plt.boxplot(dados_tempo, tick_labels=labels_tempo, patch_artist=True)
            plt.title("Exp 1: Boxplot Tempo de Execução por Tamanho da População")
            plt.ylabel('Tempo (segundos)')
            plt.xlabel('Configuração')
            plt.grid(True, axis='y')
            plt.savefig(os.path.join(PASTA_GRAFICOS, "exp1_boxplot_tempo.png"))
            plt.close()

        # Gráficos Exp 2
        plotar_boxplot_comparativo(resultados_finais["exp2_mut"], "Exp 2: Boxplot Fitness Final por Taxa de Mutação", config_labels_exp2, PASTA_GRAFICOS)
        plotar_convergencia_comparativa(resultados_finais["exp2_mut"], "Exp 2: Convergência Média por Taxa de Mutação", config_labels_exp2, PASTA_GRAFICOS)
        plotar_convergencia_comparativa(resultados_finais["exp2_mut"], "Exp 2: Diversidade Média (Indivíduos Únicos)", config_labels_exp2, PASTA_GRAFICOS, y_label='Nº de Indivíduos Únicos')

        # Gráficos Exp 3
        plotar_boxplot_comparativo(resultados_finais["exp3_tor"], "Exp 3: Boxplot Fitness Final por Tamanho do Torneio", config_labels_exp3, PASTA_GRAFICOS)
        plotar_convergencia_comparativa(resultados_finais["exp3_tor"], "Exp 3: Convergência Média por Tamanho do Torneio", config_labels_exp3, PASTA_GRAFICOS)

        # Gráficos Exp 4
        plotar_boxplot_comparativo(resultados_finais["exp4_eli"], "Exp 4: Boxplot Fitness Final por Tamanho do Elitismo", config_labels_exp4, PASTA_GRAFICOS)
        plotar_convergencia_comparativa(resultados_finais["exp4_eli"], "Exp 4: Convergência Média por Tamanho do Elitismo", config_labels_exp4, PASTA_GRAFICOS)

        print("--- Gráficos gerados com sucesso! ---")
        print(f"Verifique a pasta '{PASTA_GRAFICOS}'. Você deve ter 10 arquivos .png.")
    
    except Exception as e:
        print(f"!!!! ERRO AO GERAR GRÁFICOS: {e} !!!!")
        traceback.print_exc()

# Ponto de entrada do script
if __name__ == "__main__":
    main()