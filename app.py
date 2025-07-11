import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pickle
import logging
import sys

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recommender.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store loaded data and models
articles_df = None
transactions_df = None
articles_sample_df = None
transactions_sample_df = None
embeddings = None
nn_model = None
article_id_to_idx = {}
idx_to_article_id = {}
is_initialized = False

# Cache file paths
CACHE_DIR = 'cache'
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'embeddings.pkl')
NN_MODEL_CACHE = os.path.join(CACHE_DIR, 'nn_model.pkl')
ARTICLES_CACHE = os.path.join(CACHE_DIR, 'articles_sample.pkl')
TRANSACTIONS_CACHE = os.path.join(CACHE_DIR, 'transactions_sample.pkl')

# Data file paths - using absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_CSV = os.path.join(CURRENT_DIR, 'articles.csv')
TRANSACTIONS_CSV = os.path.join(CURRENT_DIR, 'transactions_train.csv')

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info("Created cache directory")

def verify_data_files():
    """Verify that the required data files exist and are readable"""
    if not os.path.exists(ARTICLES_CSV):
        logger.error(f"Articles CSV file not found at: {ARTICLES_CSV}")
        return False
    if not os.path.exists(TRANSACTIONS_CSV):
        logger.error(f"Transactions CSV file not found at: {TRANSACTIONS_CSV}")
        return False
    
    # Check if files are readable and have content
    try:
        articles_size = os.path.getsize(ARTICLES_CSV)
        transactions_size = os.path.getsize(TRANSACTIONS_CSV)
        
        if articles_size == 0:
            logger.error("Articles CSV file is empty")
            return False
        if transactions_size == 0:
            logger.error("Transactions CSV file is empty")
            return False
            
        logger.info(f"Found articles.csv ({articles_size/1024/1024:.2f} MB)")
        logger.info(f"Found transactions_train.csv ({transactions_size/1024/1024:.2f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"Error checking data files: {str(e)}")
        return False

def load_from_cache():
    """Try to load preprocessed data from cache"""
    global articles_sample_df, transactions_sample_df, embeddings, nn_model, article_id_to_idx, idx_to_article_id
    
    try:
        logger.info("Attempting to load data from cache...")
        with open(ARTICLES_CACHE, 'rb') as f:
            articles_sample_df = pickle.load(f)
        with open(TRANSACTIONS_CACHE, 'rb') as f:
            transactions_sample_df = pickle.load(f)
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(NN_MODEL_CACHE, 'rb') as f:
            nn_model = pickle.load(f)
            
        # Recreate the mappings
        article_id_to_idx = {article_id: i for i, article_id in enumerate(articles_sample_df['article_id'].astype(int))}
        idx_to_article_id = {i: article_id for article_id, i in article_id_to_idx.items()}
        
        logger.info(f"Successfully loaded from cache! Articles shape: {articles_sample_df.shape}, Transactions shape: {transactions_sample_df.shape}")
        return True
    except Exception as e:
        logger.error(f"Cache loading failed: {str(e)}")
        return False

def save_to_cache():
    """Save preprocessed data to cache"""
    ensure_cache_dir()
    try:
        logger.info("Saving data to cache...")
        with open(ARTICLES_CACHE, 'wb') as f:
            pickle.dump(articles_sample_df, f)
        with open(TRANSACTIONS_CACHE, 'wb') as f:
            pickle.dump(transactions_sample_df, f)
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(NN_MODEL_CACHE, 'wb') as f:
            pickle.dump(nn_model, f)
        logger.info("Successfully saved data to cache!")
    except Exception as e:
        logger.error(f"Cache saving failed: {str(e)}")

def load_and_prepare_data():
    """
    Loads data from local CSV files, preprocesses it, and trains the BERT and NearestNeighbors models.
    This function is called once when the Flask application starts.
    """
    global articles_df, transactions_df, articles_sample_df, transactions_sample_df
    global embeddings, nn_model, article_id_to_idx, idx_to_article_id, is_initialized

    # Verify data files exist and are readable
    if not verify_data_files():
        return False

    # Try loading from cache first
    if load_from_cache():
        is_initialized = True
        return True

    logger.info("Cache not found or invalid. Processing data from scratch...")
    logger.info("Step 1: Loading and preprocessing data...")

    try:
        # Load CSV files with explicit encoding and error handling
        articles_df = pd.read_csv(ARTICLES_CSV, encoding='utf-8')
        logger.info(f"Successfully loaded articles.csv with shape: {articles_df.shape}")
        
        transactions_df = pd.read_csv(TRANSACTIONS_CSV, encoding='utf-8')
        logger.info(f"Successfully loaded transactions_train.csv with shape: {transactions_df.shape}")
        
        # Verify the data has the expected columns
        required_article_cols = ['article_id', 'prod_name', 'product_type_name', 'product_group_name', 'detail_desc']
        required_transaction_cols = ['customer_id', 'article_id']
        
        missing_article_cols = [col for col in required_article_cols if col not in articles_df.columns]
        missing_transaction_cols = [col for col in required_transaction_cols if col not in transactions_df.columns]
        
        if missing_article_cols:
            logger.error(f"Missing required columns in articles.csv: {missing_article_cols}")
            return False
        if missing_transaction_cols:
            logger.error(f"Missing required columns in transactions_train.csv: {missing_transaction_cols}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load CSV files: {str(e)}")
        logger.error("Make sure the CSV files are not corrupted and have the correct format")
        return False

    try:
        # First, sample customers to ensure we have enough transaction history
        sample_customers = transactions_df['customer_id'].unique()
        sample_size = min(len(sample_customers), 50000)  # Increased sample size
        sampled_customers = np.random.choice(sample_customers, size=sample_size, replace=False)
        logger.info(f"Sampled {sample_size} customers from {len(sample_customers)} total customers")
        
        # Get all transactions for sampled customers
        transactions_sample_df = transactions_df[transactions_df['customer_id'].isin(sampled_customers)]
        logger.info(f"Got {len(transactions_sample_df)} transactions for sampled customers")
        
        # Get all unique articles from these transactions
        sample_article_ids = transactions_sample_df['article_id'].unique()
        logger.info(f"Found {len(sample_article_ids)} unique articles in sampled transactions")
        
        # Get article details for these articles
        articles_sample_df = articles_df[articles_df['article_id'].isin(sample_article_ids)]
        logger.info(f"Got details for {len(articles_sample_df)} articles")
        
        # Convert article_ids to int
        articles_sample_df['article_id'] = articles_sample_df['article_id'].astype(int)
        transactions_sample_df['article_id'] = transactions_sample_df['article_id'].astype(int)

        # Prepare text data for embeddings
        articles_sample_df['detail_desc'] = articles_sample_df['detail_desc'].fillna('')
        articles_sample_df['text'] = (
            articles_sample_df['prod_name'] + '. ' +
            articles_sample_df['product_type_name'] + '. ' +
            articles_sample_df['product_group_name'] + '. ' +
            articles_sample_df['detail_desc']
        )
        articles_sample_df['text'] = articles_sample_df['text'].str.lower()

        logger.info("Step 2: Generating text embeddings with BERT...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(articles_sample_df['text'].tolist(), show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        logger.info("Step 3: Training the Nearest Neighbor model...")
        nn_model = NearestNeighbors(n_neighbors=13, metric='cosine', algorithm='brute')
        nn_model.fit(embeddings)
        logger.info("Nearest Neighbors model trained successfully")

        article_id_to_idx = {article_id: i for i, article_id in enumerate(articles_sample_df['article_id'].astype(int))}
        idx_to_article_id = {i: article_id for article_id, i in article_id_to_idx.items()}
        
        # Save processed data to cache
        save_to_cache()
        
        logger.info("Recommendation system initialized and ready.")
        is_initialized = True
        return True
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return False

@app.route('/')
def home():
    """Serve the main HTML page"""
    if not is_initialized:
        if not load_and_prepare_data():
            return jsonify({"error": "Server is initializing. Please try again later."}), 503
    return send_file('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint to receive a customer ID and return product recommendations."""
    if not is_initialized:
        if not load_and_prepare_data():
            return jsonify({"error": "Server is initializing. Please try again later."}), 503

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        customer_id = request.json.get('customer_id')
        num_recs = request.json.get('num_recs', 12)

        if not customer_id:
            return jsonify({"error": "Customer ID is required."}), 400

        if not isinstance(transactions_sample_df, pd.DataFrame):
            logger.error("transactions_sample_df is not properly initialized")
            return jsonify({"error": "Server is not properly initialized. Please try again later."}), 500

        # Get all transactions for this customer
        customer_transactions = transactions_sample_df[transactions_sample_df['customer_id'] == customer_id]

        if customer_transactions.empty:
            return jsonify({"error": f"No transaction history found for customer {customer_id} in the sample data. Please try a different customer ID."}), 404

        # Get the last purchase
        last_purchase_article_id = customer_transactions.iloc[-1]['article_id']
        last_purchase_info = articles_sample_df[articles_sample_df['article_id'] == last_purchase_article_id]
        
        if last_purchase_info.empty:
            return jsonify({"error": f"Details for last purchased article {last_purchase_article_id} not found in articles sample."}), 404

        last_purchase_details = {
            'article_id': str(last_purchase_info['article_id'].iloc[0]),
            'prod_name': last_purchase_info['prod_name'].iloc[0],
            'product_type_name': last_purchase_info['product_type_name'].iloc[0]
        }

        try:
            item_idx = article_id_to_idx[last_purchase_article_id]
            item_embedding = embeddings[item_idx].reshape(1, -1)
        except KeyError:
            return jsonify({"error": f"Article {last_purchase_article_id} from customer's last purchase not found in the trained articles sample."}), 404

        # Get recommendations
        distances, indices = nn_model.kneighbors(item_embedding)
        similar_item_indices = indices.flatten()[1:]  # Exclude the first item (itself)
        recommended_article_ids_raw = [idx_to_article_id[i] for i in similar_item_indices]

        # Filter out already purchased items
        purchased_ids = set(customer_transactions['article_id'])
        final_recommendations_ids = [rec_id for rec_id in recommended_article_ids_raw if rec_id not in purchased_ids]
        
        # Get details for recommended products
        recommended_products_details = []
        for rec_id in final_recommendations_ids[:num_recs]:
            rec_info = articles_sample_df[articles_sample_df['article_id'] == rec_id]
            if not rec_info.empty:
                recommended_products_details.append({
                    'article_id': str(rec_info['article_id'].iloc[0]),
                    'prod_name': rec_info['prod_name'].iloc[0],
                    'product_type_name': rec_info['product_type_name'].iloc[0]
                })

        if not recommended_products_details:
            return jsonify({
                "error": "No new recommendations could be generated (all similar items might have been purchased or no sufficient similar items found after filtering).",
                "last_purchase": last_purchase_details,
                "recommendations": []
            }), 200

        return jsonify({
            "last_purchase": last_purchase_details,
            "recommendations": recommended_products_details
        })
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

if __name__ == '__main__':
    # Initialize data and model when the app starts
    if not load_and_prepare_data():
        logger.error("Application startup failed due to data loading issues. Please resolve the data file paths.")
        exit()
    app.run(debug=True, host='0.0.0.0', port=5000)