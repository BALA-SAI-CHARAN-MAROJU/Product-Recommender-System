import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info("Created cache directory")

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
        
        logger.info("Successfully loaded data from cache!")
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

    # Try loading from cache first
    if load_from_cache():
        is_initialized = True
        return True

    logger.info("Cache not found or invalid. Processing data from scratch...")
    logger.info("Step 1: Loading and preprocessing data...")

    try:
        articles_df = pd.read_csv('articles.csv')
        transactions_df = pd.read_csv('transactions_train.csv')
        logger.info("Data loaded from local CSV files successfully.")
    except Exception as e:
        logger.error(f"Failed to load CSV files: {str(e)}")
        logger.error("Make sure 'articles.csv' and 'transactions_train.csv' are in the same directory as 'app.py'")
        return False

    try:
        # Sample data for faster processing
        articles_sample_df = articles_df.sample(n=20000, random_state=42)
        articles_sample_df['article_id'] = articles_sample_df['article_id'].astype(int)
        
        sample_article_ids = articles_sample_df['article_id'].tolist()  # Convert to list to avoid type issues
        transactions_sample_df = transactions_df[transactions_df['article_id'].isin(sample_article_ids)]
        transactions_sample_df['article_id'] = transactions_sample_df['article_id'].astype(int)

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

        logger.info("Step 3: Training the Nearest Neighbor model...")
        nn_model = NearestNeighbors(n_neighbors=13, metric='cosine', algorithm='brute')
        nn_model.fit(embeddings)

        article_id_to_idx = {article_id: i for i, article_id in enumerate(articles_sample_df['article_id'].astype(int))}
        idx_to_article_id = {i: article_id for article_id, i in article_id_to_idx.items()}
        
        # Save processed data to cache
        save_to_cache()
        
        logger.info("Recommendation system initialized and ready.")
        is_initialized = True
        return True
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
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

        customer_transactions = transactions_sample_df[transactions_sample_df['customer_id'] == customer_id]

        if customer_transactions.empty:
            return jsonify({"error": f"No transaction history found for customer {customer_id} in the sample data. Please try a different customer ID."}), 404

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

        distances, indices = nn_model.kneighbors(item_embedding)
        similar_item_indices = indices.flatten()[1:]
        recommended_article_ids_raw = [idx_to_article_id[i] for i in similar_item_indices]

        purchased_ids = set(customer_transactions['article_id'])
        final_recommendations_ids = [rec_id for rec_id in recommended_article_ids_raw if rec_id not in purchased_ids]
        
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