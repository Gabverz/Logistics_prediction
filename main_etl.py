import pandas as pd
import os
import zipfile

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
from kaggle.api.kaggle_api_extended import KaggleApi

def conectar_api():
    try:
        api = KaggleApi()
        api.authenticate()
        print("Sucesso: Autenticação concluída!")
        return api
    except Exception as e:
        print(f"Erro ao autenticar: {e}")
        return None

# Testar conexão
api = conectar_api()

if api:
    # Exemplo: listar arquivos do dataset do Olist para testar
    files = api.dataset_list_files('olistbr/brazilian-ecommerce').files
    print("Conexão ativa. Arquivos encontrados:", [f.name for f in files])
def extrair_dados():
    path = './data'
    if not os.path.exists(path):
        os.makedirs(path)
    
    print("Baixando dados do Kaggle...")
    api.dataset_download_files('olistbr/brazilian-ecommerce', path=path, unzip=True)
    print("Arquivos extraídos com sucesso.")

# 2. Carga e Merge (Transformação)
def processar_base_mestre():
    path = './data'
    
    # Carregando as tabelas essenciais
    orders = pd.read_csv(f'{path}/olist_orders_dataset.csv')
    items = pd.read_csv(f'{path}/olist_order_items_dataset.csv')
    reviews = pd.read_csv(f'{path}/olist_order_reviews_dataset.csv')
    customers = pd.read_csv(f'{path}/olist_customers_dataset.csv')
    
    print("Iniciando Merges...")
    
    # Unindo Pedidos com Itens (Inner join para garantir que temos o produto)
    df = pd.merge(orders, items, on='order_id', how='inner')
    
    # Unindo com Clientes (Para obter localização)
    df = pd.merge(df, customers, on='customer_id', how='inner')
    
    # Unindo com Reviews (Essencial para a etapa de NLP com Redes Neurais)
    # Usamos left join para manter pedidos mesmo sem comentário (para o modelo supervisionado)
    df = pd.merge(df, reviews[['order_id', 'review_comment_message', 'review_score']], on='order_id', how='left')
    
    # 3. Tratamento de Datas
    cols_data = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in cols_data:
        df[col] = pd.to_datetime(df[col])
        
    # Criando a variável alvo (Atraso)
    # 1 se a entrega real foi após a estimada, 0 se foi no prazo
    df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)
    
    # 4. Limpeza Inicial
    # Focaremos em reviews que possuem texto para a futura etapa de NLP
    print(f"Dataset consolidado com {df.shape[0]} linhas.")
    return df

# Execução
extrair_dados()
df_final = processar_base_mestre()

# Salvando a base intermediária
df_final.to_parquet('base_consolidada_logistica.parquet', index=False)
print("Arquivo base_consolidada_logistica.parquet gerado.")